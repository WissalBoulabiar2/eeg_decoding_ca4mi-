import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import wandb
from methods.ca4mi_mp import CA4MI_MP
from networks import base_ca4mi as network
from trainer import EEGTrainer
import utils


class EEGTrainerMP(EEGTrainer):
    """EEGTrainer for CA4MI-MP. The original EEGTrainer / CA4MI pipeline is left untouched."""

    def initialize_dataloader(self):
        from dataloaders import dataloader_ca4mi as factory
        return factory.IncrementalDataStreaming(self.args)

    def run_single_experiment(self, run_index):
        self.args.seed = run_index
        self.set_seed(self.args.seed)
        return self.run_ca4mi_mp(run_index)

    def run_ca4mi_mp(self, run_index):
        dataloader = self.initialize_dataloader()
        net = network.Cls_HEAD(self.args).to(self.args.device)
        net.print_model_size()
        ca4mi = CA4MI_MP(net, self.args, network=network)

        num_subjects = self.args.n_subjects
        acc = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        lss = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        f1_mat = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        inf_time_mat = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        updated_refEA = None

        for t in range(num_subjects):
            print("-" * 50)
            dataset = dataloader.load_data(t)
            print(f"{' '} Dataset {t + 1:2d} ({dataset[t]['name']})")
            print("-" * 50)

            # IEA alignment, identical to CA4MI
            dataset[t]['train'], updated_refEA = self.align_data(
                dataset[t]['train'], ca4mi, updated_refEA, self.args.device,
                update_ref=True, gamma=0.1, batch_size=self.args.batch_size)
            dataset[t]['valid'], _ = self.align_data(
                dataset[t]['valid'], ca4mi, updated_refEA, self.args.device,
                update_ref=False, batch_size=self.args.batch_size)
            dataset[t]['test'], _ = self.align_data(
                dataset[t]['test'], ca4mi, updated_refEA, self.args.device,
                update_ref=False, batch_size=self.args.batch_size)

            wandb.init(
                project="CA4MI-MP-EEG",
                name=f"ca4mi-mp-phase{self.args.get('phase', 'custom')}-sub{t + 1}",
                config=self.filter_wandb_config(),
                mode=self.args.get('wandb_mode', 'online')
            )

            if self.args.use_prototypes == 'yes':
                anchors = ca4mi.get_anchors()
                ca4mi.adversarial_training(t, dataset[t], prototypes=anchors)
                current_model = utils.load_current_models(
                    network_class=ca4mi.network.Cls_HEAD, args=ca4mi.args,
                    checkpoint_dir=ca4mi.checkpoint, sub_id=t)
                ca4mi.update_memory(dataset[t]['train'], current_model, t)
                self.save_memory(ca4mi, t)
            else:
                ca4mi.adversarial_training(sub_id=t, dataset=dataset[t], prototypes=None)

            self.evaluate_all_previous_subjects(ca4mi, dataset, t, acc, lss, f1_mat, inf_time_mat)

        output_path = os.path.join(self.args.checkpoint,
                                   f"{self.args.experiment}_{self.args.n_subjects}_subjects_seed_{self.args.seed}.txt")
        print(f"\nSaved accuracies at {output_path}")
        np.savetxt(output_path, acc, '%.6f')

        avg_acc, gem_bwt, avg_f1, inf_time_mat = utils.print_log_acc_bwt_f1_inference_time(
            acc, lss, f1_mat, inf_time_mat, output_path=output_path, run_id=run_index)
        self.save_matrices_ca4mi(acc, f1_mat)

        return {'avg_acc': avg_acc, 'bwt': gem_bwt, 'avg_f1': avg_f1, 'inf_time_mat': inf_time_mat}

    def evaluate_all_previous_subjects(self, ca4mi, dataset, t, acc, lss, f1_mat, inf_time_mat):
        """
        Same as EEGTrainer, except that with eval_with_current_shared='yes' the shared encoder
        evaluated on past subjects is the one actually trained so far (ca4mi.model).
        The original passes ca4mi.load_current_models(u), which builds a freshly initialised network.
        """
        if self.args.get('eval_with_current_shared', 'yes') != 'yes':
            return super().evaluate_all_previous_subjects(ca4mi, dataset, t, acc, lss, f1_mat, inf_time_mat)

        for u in range(t + 1):
            model = utils.load_model_with_shared_update(
                network_class=ca4mi.network.Cls_HEAD, args=ca4mi.args,
                current_model=ca4mi.model, checkpoint_dir=ca4mi.checkpoint, sub_id=u)
            res = ca4mi.test_previous_subjects(dataset[u]['test'], u, model=model)
            print(
                f">>> Test on {dataset[u]['name']:15s}: "
                f"loss={res['loss_cls']:.3f}, "
                f"acc={res['acc_cls']:5.2f}% "
                f"f1_macro={res['f1_macro']:.3f}, "
                f"avg_inf_time={res['avg_inf_time'] * 1000:.2f} ms <<<"
            )
            acc[t, u] = res['acc_cls']
            lss[t, u] = res['loss_cls']
            f1_mat[t, u] = res['f1_macro']
            inf_time_mat[t, u] = res['avg_inf_time']

    def save_memory(self, ca4mi, t):
        """Save memory content after each subject (to analyse / visualise the modes)."""
        m = ca4mi.memory
        torch.save({'mu': m.mu.cpu(), 'var': m.var.cpu(), 'count': m.count.cpu(),
                    'labels': m.labels.cpu(), 'subjects': m.subjects.cpu()},
                   os.path.join(self.args.checkpoint, f'prototype_memory_{t}.pt'))

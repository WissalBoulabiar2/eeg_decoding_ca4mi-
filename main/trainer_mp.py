import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset, DataLoader
from methods.ca4mi_mp import CA4MI_MP, enabled
from networks import base_ca4mi_mp as network
from trainer import EEGTrainer
from diagnostics_mp import DiagnosticsMP
import utils


class EEGTrainerMP(EEGTrainer):
    """EEGTrainer for CA4MI-MP. The original EEGTrainer / CA4MI pipeline is left untouched."""

    def initialize_dataloader(self):
        if enabled(self.args, 'replay_fix', 'yes'):
            from dataloaders import dataloader_ca4mi_mp as factory
            return factory.IncrementalDataStreamingMP(self.args)
        from dataloaders import dataloader_ca4mi as factory
        return factory.IncrementalDataStreaming(self.args)

    # ------------------------------------------------------------------ replay (replay_fix='yes')
    def with_replay(self, train_loader):
        """Aligned training trials of the current subject + replay buffer (training split only)."""
        if self.replay is None:
            return train_loader
        tensors = [torch.cat([a, b]) for a, b in zip(train_loader.dataset.tensors, self.replay)]
        return DataLoader(TensorDataset(*tensors), batch_size=self.args.batch_size, shuffle=True)

    def add_to_replay(self, train_loader, t, ca4mi):
        """Store n_samples aligned training trials of subject t, class-balanced, with their true labels.
        They keep the IEA alignment of their own subject. Their logits under the model selected for subject t
        are stored too (used by replay_distill)."""
        x, y, sub, dis = train_loader.dataset.tensors
        g = torch.Generator().manual_seed(int(self.args.seed) * 1000 + t)
        per_class = self.args.n_samples // self.args.n_classes
        keep = []
        for c in range(self.args.n_classes):
            idx = (y == c).nonzero(as_tuple=True)[0].cpu()
            keep.append(idx[torch.randperm(len(idx), generator=g)[:per_class]])
        keep = torch.cat(keep).to(x.device)
        new = [v[keep] for v in (x, y, sub, dis)]
        ca4mi.model.eval()
        with torch.no_grad():
            xk = new[0][:, None].float()
            logits = ca4mi.model(xk, xk, new[2], t)
        self.replay = new if self.replay is None else [torch.cat([a, b]) for a, b in zip(self.replay, new)]
        self.replay_logits = logits if self.replay_logits is None else torch.cat([self.replay_logits, logits])
        if len(self.replay[0]) > self.args.global_reservoir_size:
            sel = torch.randperm(len(self.replay[0]), generator=g)[:self.args.global_reservoir_size].to(x.device)
            self.replay = [v[sel] for v in self.replay]
            self.replay_logits = self.replay_logits[sel]
        labels = {c: int((self.replay[1] == c).sum()) for c in range(self.args.n_classes)}
        print(f"  Replay buffer: {len(self.replay[0])} trials | per class {labels}")

    def run_single_experiment(self, run_index):
        self.args.seed = run_index
        self.set_seed(self.args.seed)
        return self.run_ca4mi_mp(run_index)

    def run_ca4mi_mp(self, run_index):
        dataloader = self.initialize_dataloader()
        net = network.Cls_HEAD(self.args).to(self.args.device)
        net.print_model_size()
        ca4mi = CA4MI_MP(net, self.args, network=network)
        # --set diagnostics=yes is parsed by OmegaConf as the boolean True
        diag = DiagnosticsMP(self.args) if self.args.get('diagnostics', 'no') in ('yes', True) else None
        ca4mi.diagnostics = diag
        self.diagnostics = diag
        replay_fix = enabled(self.args, 'replay_fix', 'yes')
        self.replay = None
        self.replay_logits = None

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

            # IEA alignment, identical to CA4MI. With replay_fix the training split holds only the current
            # subject's trials, so the reference is estimated from them alone.
            dataset[t]['train'], updated_refEA = self.align_data(
                dataset[t]['train'], ca4mi, updated_refEA, self.args.device,
                update_ref=True, gamma=0.1, batch_size=self.args.batch_size)
            dataset[t]['valid'], _ = self.align_data(
                dataset[t]['valid'], ca4mi, updated_refEA, self.args.device,
                update_ref=False, batch_size=self.args.batch_size)
            dataset[t]['test'], _ = self.align_data(
                dataset[t]['test'], ca4mi, updated_refEA, self.args.device,
                update_ref=False, batch_size=self.args.batch_size)
            if replay_fix:
                own_train = dataset[t]['train']
                dataset[t]['train'] = self.with_replay(own_train)
                ca4mi.replay_bank = (None if self.replay is None
                                     else (self.replay[0], self.replay[2], self.replay_logits))
            if enabled(self.args, 'replay_private', 'yes'):
                ca4mi.model.freeze_past_subjects(t)

            wandb.init(
                project="CA4MI-MP-EEG",
                name=f"ca4mi-mp-phase{self.args.get('phase', 'custom')}-sub{t + 1}",
                config=self.filter_wandb_config(),
                mode=self.args.get('wandb_mode', 'online')
            )

            if diag is not None:
                diag.begin_subject(t, dataset, dataloader, ca4mi, replay=self.replay)

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
            if replay_fix:
                self.add_to_replay(own_train, t, ca4mi)

            if diag is not None:
                diag.end_subject(t)
                diag.log_feature_shift(ca4mi, t)

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
            if self.diagnostics is not None:
                self.diagnostics.log_test(model, dataset[u]['test'], t, u)

    def save_memory(self, ca4mi, t):
        """Save memory content after each subject (to analyse / visualise the modes)."""
        m = ca4mi.memory
        torch.save({'mu': m.mu.cpu(), 'var': m.var.cpu(), 'count': m.count.cpu(),
                    'labels': m.labels.cpu(), 'subjects': m.subjects.cpu()},
                   os.path.join(self.args.checkpoint, f'prototype_memory_{t}.pt'))

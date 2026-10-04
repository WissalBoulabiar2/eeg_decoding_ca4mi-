import random
import tracemalloc
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from methods.ca4mi import CA4MI
from networks import base_ca4mi as network
import utils
import torch
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
import wandb


class EEGTrainer:
    """Main trainer class for EEG continual learning experiments"""

    def __init__(self, args):
        self.args = args

    def get_approach_and_network(self):
        """
        Dynamically import the appropriate approach and network based on args.approach
        Returns the approach class and network module
        """
        if self.args.approach == 'ewc':
            from methods.ewc import ElasticWeightConsolidation
            from networks import base_ewc as network
            return ElasticWeightConsolidation, network

        elif self.args.approach == 'der':
            from methods.der import DER
            from networks import base_der as network
            return DER, network

        elif self.args.approach == 'finetuning':
            from methods.finetuning import Finetuning
            from networks import base_finetuning as network
            return Finetuning, network

        elif self.args.approach == 'er':
            from methods.er import ExperienceReplay
            from networks import base_er as network
            return ExperienceReplay, network

        elif self.args.approach == 'mudvi':
            from methods.mudvi import mudvi
            from networks import base_mudvi as network
            return mudvi, network

        elif self.args.approach == 'cger':
            from methods.cger import CGER
            from networks import base_cger as network
            return CGER, network

        else:
            raise NotImplementedError(f"Approach '{self.args.approach}' is not implemented")

    def initialize_dataloader(self):
        """Initialize the appropriate dataloader based on experiment type and approach"""
        if self.args.experiment == 'ewc' or self.args.experiment == 'finetuning':
            from dataloaders import dataloader_baseline as factory
            dataloader = factory.IncrementalDataStreaming(self.args)
        elif self.args.experiment == 'ca4mi':
            from dataloaders import dataloader_ca4mi as factory
            dataloader = factory.IncrementalDataStreaming(self.args)
        elif self.args.experiment == 'der' or self.args.experiment == 'er' or self.args.experiment == 'mudvi' or self.args.experiment == 'cger':
            from dataloaders import dataloader_replay as factory
            dataloader = factory.IncrementalDataStreaming(self.args)
        else:
            raise NotImplementedError("Experiment type not recognized")
        print('dataloader>>>>>>>>>>>>>:', dataloader)
        return dataloader

    # --------------------------------------------------- CA4MI Training Functions -----------------------------------------
    def align_data(self, data_loader, ca4mi, refEA, device, update_ref=True, gamma=0.1, batch_size=64):
        """
        Args:
            data_loader: Original DataLoader (unaligned)
            ca4mi: Object containing IEA functions
            refEA: Previous reference matrix
            update_ref: Whether to update refEA (True for training phase; False for val/test phase)
            gamma: Decay coefficient, only effective when update_ref=True
            batch_size: Batch size for the output aligned DataLoader

        Returns:
            aligned_loader: Aligned DataLoader
            updated_refEA: Updated reference matrix (same as input if update_ref=False)
        """
        all_x, all_y, all_sub_module, all_dis_label = [], [], [], []

        for data, target, sub_module_label, dis_label in data_loader:
            all_x.append(data.numpy())
            all_y.append(target)
            all_sub_module.append(sub_module_label)
            all_dis_label.append(dis_label)

        all_x = np.concatenate(all_x, axis=0)
        all_y = torch.cat(all_y)
        all_sub_module = torch.cat(all_sub_module)
        all_dis_label = torch.cat(all_dis_label)

        if update_ref:
            aligned_x, updated_refEA = ca4mi.IEA(all_x, refEA, gamma=gamma)
        else:
            aligned_x, updated_refEA = ca4mi.IEA(all_x, refEA, gamma=0.1)  # No update, alignment only

        dataset = TensorDataset(
            torch.tensor(aligned_x, dtype=torch.float32).to(device),
            all_y.to(device),
            all_sub_module.to(device),
            all_dis_label.to(device)
        )
        aligned_loader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=update_ref)  # Shuffle only if updating refEA

        return aligned_loader, updated_refEA

    def run_ca4mi(self, run_index):
        """CA4MI specific training and evaluation"""
        dataloader = self.initialize_dataloader()
        net = network.Cls_HEAD(self.args).to(self.args.device)
        net.print_model_size()
        ca4mi = CA4MI(net, self.args, network=network)

        num_subjects = self.args.n_subjects
        acc = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        lss = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        f1_mat = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        inf_time_mat = np.zeros((num_subjects, num_subjects), dtype=np.float32)

        max_prototypes = self.args.max_prototypes
        source_center = []
        updated_refEA = None

        for t in range(num_subjects):
            print("-" * 50)
            dataset = dataloader.load_data(t)
            print(f"{' '} Dataset {t + 1:2d} ({dataset[t]['name']})")
            print("-" * 50)

            # ==================================================
            # IEA (Iterative Entity Alignment) Data Processing
            # ==================================================
            # Training Phase: Update reference matrix
            dataset[t]['train'], updated_refEA = self.align_data(
                data_loader=dataset[t]['train'],
                ca4mi=ca4mi,
                refEA=updated_refEA,
                device=self.args.device,
                update_ref=True,
                gamma=0.1,
                batch_size=self.args.batch_size
            )
            # Validation Phase: Use the same refEA, align but do not update
            dataset[t]['valid'], _ = self.align_data(
                data_loader=dataset[t]['valid'],
                ca4mi=ca4mi,
                refEA=updated_refEA,
                device=self.args.device,
                update_ref=False,
                batch_size=self.args.batch_size
            )
            # Test Phase: Use the same refEA, align but do not update
            dataset[t]['test'], _ = self.align_data(
                data_loader=dataset[t]['test'],
                ca4mi=ca4mi,
                refEA=updated_refEA,
                device=self.args.device,
                update_ref=False,
                batch_size=self.args.batch_size
            )
            # =================================================

            # --- train subject t ---
            wandb.init(
                project="CA4MI-EEG",
                name=f"ca4mi-run",
                config=self.filter_wandb_config()
            )

            if self.args.use_prototypes == 'yes':
                ca4mi.adversarial_training(t, dataset[t], prototypes=source_center)
                current_model = utils.load_current_models(
                    network_class=ca4mi.network.Cls_HEAD,
                    args=ca4mi.args,
                    checkpoint_dir=ca4mi.checkpoint,
                    sub_id=t
                )

                prototypes, _= ca4mi.get_prototype_samples(
                    loader=dataset[t]['train'], net=current_model, sub_id=t
                )
                ca4mi.update_prototypes_ema(prototypes, momentum=0.01)

                source_center.append(prototypes)

                total_prototypes = sum([p.size(0) for p in source_center])
                if total_prototypes > max_prototypes:
                    print(f"Total prototypes {total_prototypes} > {max_prototypes}, do reservoir sampling")
                    source_center = ca4mi.reservoir_sampling(source_center, max_prototypes)
            else:
                ca4mi.adversarial_training(sub_id=t, dataset=dataset[t], prototypes=None)

            self.evaluate_all_previous_subjects(ca4mi, dataset, t, acc, lss, f1_mat, inf_time_mat)

        output_path = os.path.join(self.args.checkpoint,
                                   f"{self.args.experiment}_{self.args.n_subjects}_subjects_seed_{self.args.seed}.txt")
        print(f"\nSaved accuracies at {output_path}")
        np.savetxt(output_path, acc, '%.6f')

        avg_acc, gem_bwt, avg_f1, inf_time_mat = utils.print_log_acc_bwt_f1_inference_time(acc, lss, f1_mat,
                                                                                           inf_time_mat,
                                                                                           output_path=output_path,
                                                                                           run_id=run_index)

        self.save_matrices_ca4mi(acc, f1_mat)

        return {
            'avg_acc': avg_acc,
            'bwt': gem_bwt,
            'avg_f1': avg_f1,
            'inf_time_mat': inf_time_mat
        }

    def evaluate_all_previous_subjects(self, ca4mi, dataset, t, acc, lss, f1_mat, inf_time_mat):
        """Evaluate the model on all previously seen subjects and fill matrices."""
        for u in range(t + 1):
            model = utils.load_model_with_shared_update(
                network_class=ca4mi.network.Cls_HEAD,
                args=ca4mi.args,
                current_model=ca4mi.load_current_models(u),
                checkpoint_dir=ca4mi.checkpoint,
                sub_id=u
            )

            res = ca4mi.test_previous_subjects(
                dataset[u]['test'],
                u,
                model=model
            )

            print(
                f">>> Test on {dataset[u]['name']:15s}: "
                f"loss={res['loss_cls']:.3f}, "
                f"acc={res['acc_cls']:5.2f}% "
                f"f1_macro={res['f1_macro']:.3f}, "
                f"avg_inf_time={res['avg_inf_time'] * 1000:.2f} s <<<"
            )

            acc[t, u] = res['acc_cls']
            lss[t, u] = res['loss_cls']
            f1_mat[t, u] = res['f1_macro']
            inf_time_mat[t, u] = res['avg_inf_time']

    def save_matrices_ca4mi(self, acc, f1_mat):
        """Save result matrices for CA4MI"""
        base_dir = os.path.join(
            self.args.checkpoint,
            f"{self.args.experiment}_{self.args.n_subjects}_subjects_seed_{self.args.seed}"
        )
        os.makedirs(base_dir, exist_ok=True)

        output_path = os.path.join(base_dir, "accuracy_matrix.txt")
        np.savetxt(output_path, acc, '%.6f')
        print(f"Saved accuracies at {output_path}")

        f1_path = os.path.join(base_dir, "f1_macro_matrix.txt")
        np.savetxt(f1_path, f1_mat, '%.6f')
        print(f"Saved macro-F1 matrix at {f1_path}")

    def filter_wandb_config(self):
        config = {}
        for k, v in vars(self.args).items():
            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                config[k] = v
        return config

    # --------------------------------------------------- Baseline Training Functions --------------------------------------

    def run_baseline(self, run_index, approach_baseline, network):
        """Generic baseline training and evaluation"""
        # Initialize data loader and model
        dataloader = self.initialize_dataloader()
        self.args.task_cls = dataloader.sub_cls

        model = network.Net(self.args).to(self.args.device)
        model.print_model_size()

        approach = approach_baseline(model, self.args, network)

        num_subjects = self.args.n_subjects
        acc_matrix = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        loss_matrix = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        f1_matrix = np.zeros((num_subjects, num_subjects), dtype=np.float32)
        inf_time_mat = np.zeros((num_subjects, num_subjects), dtype=np.float32)

        # Loop through subjects
        proto_ = []
        for t in range(num_subjects):
            print("-" * 50)
            dataset = dataloader.load_data(t)
            print(f"{' ' * 10} Dataset {t + 1:2d} ({dataset[t]['name']})")
            print("-" * 50)

            print(f'Training on subject {t + 1}...')
            if self.args.approach == 'ewc':
                approach.train(dataset[t]['train'], t, dataset[t], use_ewc=True)
            elif self.args.approach in ['finetuning', 'er', 'mudvi', 'der', 'cger']:
                if self.args.approach == 'cger':
                    prototypes = approach.train(dataset[t], t, prototypes=proto_)
                    proto_.append(prototypes)
                elif self.args.approach == 'finetuning':
                    approach.train(dataset[t], t)
                else:
                    approach.train(dataset[t], t)
            else:
                raise NotImplementedError(f"Training method not found for approach {self.args.approach}")

            print(f'Training completed for subject {t + 1}')

            if hasattr(approach, 'update_mean_params'):
                approach.update_mean_params()
            elif hasattr(approach, 'post_train_process'):
                approach.post_train_process(t)

            # Evaluate the model on all seen subjects so far
            for eval_subject_idx in range(t + 1):
                # Get test data for the evaluation subject
                eval_dataset = dataset
                test_loader = eval_dataset[eval_subject_idx]['test']
                if self.args.approach == 'finetuning':
                    test_result = approach.evaluate(test_loader, eval_subject_idx)
                else:
                    test_result = approach.evaluate(test_loader, eval_subject_idx,
                                                    model=approach.model)

                # Store all metrics
                acc_matrix[t, eval_subject_idx] = test_result['accuracy']
                loss_matrix[t, eval_subject_idx] = test_result['loss']
                f1_matrix[t, eval_subject_idx] = test_result['f1_score']
                inf_time_mat[t, eval_subject_idx] = test_result['inf_time_mat']

                print(
                      f'Test on {dataset[eval_subject_idx]["name"]:15s}: '
                      f'Accuracy = {test_result["accuracy"]:.2f}%, '
                      f'Loss = {test_result["loss"]:.3f}, '
                      f'F1-score = {test_result["f1_score"]:.2f}%, '
                      f'Avg Inference Time = {test_result["inf_time_mat"] * 1000:.2f} ms'
                      )

            # Save all matrices to files after every subject
            self.save_matrices_baseline(run_index, t, acc_matrix, f1_matrix, loss_matrix)

            # Print running metrics
            utils.print_running_acc_bwt_f1(acc_matrix, f1_matrix, t)

        # Calculate final metrics using enhanced function
        results = utils.print_log_acc_bwt_f1(
            task_class=self.args.task_cls,
            acc=acc_matrix,
            lss=loss_matrix,
            f1=f1_matrix,
            inf_time_mat=inf_time_mat,
            output_path=self.args.checkpoint,
            run_id=run_index
        )

        print(f'Final Average Accuracy: {results["avg_acc"]:.4f}%')
        print(f'Final Average F1-score: {results["avg_f1"]:.4f}%')
        print(f'Backward Transfer (GEM BWT): {results["bwt"]:.4f}')

        return results

    def save_matrices_baseline(self, run_index, subject_idx, acc_matrix, f1_matrix, loss_matrix):
        """Save result matrices for baseline approaches"""
        np.savetxt(f'{self.args.checkpoint}/acc_run_{run_index}_subject_{subject_idx + 1}.txt', acc_matrix, fmt='%.4f')
        np.savetxt(f'{self.args.checkpoint}/f1_run_{run_index}_subject_{subject_idx + 1}.txt', f1_matrix, fmt='%.4f')
        np.savetxt(f'{self.args.checkpoint}/loss_run_{run_index}_subject_{subject_idx + 1}.txt', loss_matrix,
                   fmt='%.4f')

    # --------------------------------------------------- Execute Training Function ----------------------------------------
    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run_single_experiment(self, run_index):
        """
        Main run function that delegates to the appropriate approach
        """
        self.args.seed = run_index
        self.set_seed(self.args.seed)

        if self.args.approach == 'ca4mi':
            return self.run_ca4mi(run_index)
        else:
            # Get the appropriate baseline approach and network
            approach_baseline, network = self.get_approach_and_network()
            return self.run_baseline(run_index, approach_baseline, network)

    def run_full_experiment(self):
        """Main function to run the complete experiment with multiple runs"""
        utils.create_checkpoint_structure(self.args)
        utils.copy_project_files_to_destination(self.args)

        print('-' * 50)
        print('Start running:')
        print('-' * 50)

        # Initialize memory tracking
        tracemalloc.start()
        ACC, BWT, F1, AVG_Inference_Time = [], [], [], []

        for run_index in range(self.args.n_runs):
            print(f"Run {run_index + 1}/{self.args.n_runs}")
            # Run the experiment
            results = self.run_single_experiment(run_index)

            # Store metrics
            ACC.append(results['avg_acc'])
            BWT.append(results['bwt'])
            F1.append(results['avg_f1'])
            AVG_Inference_Time.append(results['inf_time_mat'])

            # Check memory usage after each run
            current, peak = tracemalloc.get_traced_memory()
            print(f'Memory usage after run {run_index + 1}: Current = {current / 1e6:.2f}MB; Peak = {peak / 1e6:.2f}MB')
            tracemalloc.reset_peak()

        # Display final results
        self.display_final_results(ACC, BWT, F1, AVG_Inference_Time)

        # Stop memory tracking and print final usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f'\nFinal Memory usage: Current = {current / 1e6:.2f}MB; Peak = {peak / 1e6:.2f}MB')

        return ACC, BWT, F1, AVG_Inference_Time

    def display_final_results(self, ACC, BWT, F1s, AVG_Inference_Time):
        """Display final results summary"""
        mean_acc, std_acc = np.mean(ACC), np.std(ACC)
        mean_bwt, std_bwt = np.mean(BWT), np.std(BWT)
        mean_f1, std_f1 = np.mean(F1s), np.std(F1s)
        mean_Inference_Time, std_AVG_Inference_Time = np.mean(AVG_Inference_Time), np.std(AVG_Inference_Time)

        print('-' * 80)
        print(f"Average over {self.args.n_runs} runs using {self.args.approach.upper()}")
        print('-' * 50)
        print(f'FINAL RESULTS SUMMARY - {self.args.approach.upper()}')
        print('-' * 50)
        print(f'ACC.AVG:     {mean_acc:.4f}% ± {std_acc:.4f}')
        print(f'F1.AVG:      {mean_f1:.4f}% ± {std_f1:.4f}')
        print(f'BWT.AVG:     {mean_bwt:.4f}% ± {std_bwt:.4f}')
        print(f'AVG_INFERENCE_TIME:      {mean_Inference_Time * 1000:.2f} ± {std_AVG_Inference_Time * 1000:.2f} ms')
        print('-' * 50)
        print("Done!!")



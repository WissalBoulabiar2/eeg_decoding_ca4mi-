import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
import os
import time


class ElasticWeightConsolidation:
    def __init__(self, model, args, network):
        """
        Initialize the Elastic Weight Consolidation (EWC) class.

        Args:
            model: Neural network model
            criterion: Loss criterion (e.g., nn.CrossEntropyLoss())
            args: Configuration object containing hyperparameters
        """
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.network = network
        self.checkpoint = args.checkpoint

        self.lr = getattr(args, 'lr', 0.001)
        self.lr_patience=args.lr_patience
        self.ewc_weight = getattr(args, 'ewc_weight', 20000)  # λ parameter
        self.ewc_epochs = getattr(args, 'n_epochs')
        self.ewc_fisher_batches = getattr(args, 'ewc_fisher_batches', 20)
        self.device = getattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=getattr(args, 'wd', 0.01)
        )

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        self._precision_matrices = {}
        self._means = {}

    def print_config(self):
        """Print EWC-specific configuration"""
        print(f"  - EWC weight (λ): {self.ewc_weight}")
        print(f"  - Fisher batches: {self.ewc_fisher_batches}")
        print(f"  - Weight decay: {getattr(self.args, 'wd', 0.01)}")

    def _compute_fisher_matrix(self, dataloader, num_batches):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            precision_matrices[n] = torch.zeros_like(p.data).to(self.device)

        self.model.eval()

        sample_count = 0
        batch_count = 0

        for inputs, targets in dataloader:
            if batch_count >= num_batches:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if len(inputs.shape) == 3:  # Add channel dimension
                inputs = inputs[:, np.newaxis, :, :]

            for i in range(inputs.size(0)):
                if sample_count >= num_batches * dataloader.batch_size:
                    break

                self.model.zero_grad()

                single_input = inputs[i:i + 1]
                single_target = targets[i:i + 1]
                output = self.model(single_input)

                log_prob = F.log_softmax(output, dim=1)
                loss = F.nll_loss(log_prob, single_target)

                loss.backward()

                for n, p in self.model.named_parameters():
                    if p.grad is not None and n in precision_matrices:
                        precision_matrices[n] += p.grad.data ** 2

                sample_count += 1

            batch_count += 1
            if sample_count >= num_batches * dataloader.batch_size:
                break

        for n in precision_matrices:
            precision_matrices[n] /= sample_count

        # self.model.train()

        return precision_matrices

    def _compute_fisher_information(self, dataloader, task_idx):
        self.update_mean_params(task_idx)

        fisher_dict = self._compute_fisher_matrix(dataloader, self.ewc_fisher_batches)
        self._precision_matrices[task_idx] = fisher_dict

    def update_mean_params(self, task_idx=None):
        if task_idx is None:
            if self._means:
                task_idx = max(self._means.keys()) + 1
            else:
                task_idx = 0

        if task_idx not in self._means:
            self._means[task_idx] = {}

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._means[task_idx][n] = p.data.clone()

    def _compute_ewc_penalty(self):
        if not self._precision_matrices or not self._means:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for task_idx in self._precision_matrices:
            fisher_dict = self._precision_matrices[task_idx]
            means_dict = self._means.get(task_idx, {})

            if not means_dict:
                continue

            for n, p in self.model.named_parameters():
                if n.startswith('classifiers') or n.startswith('classifier'):
                    continue

                if n in fisher_dict and n in means_dict:
                    param_diff = (p - means_dict[n]) ** 2
                    _penalty = fisher_dict[n] * param_diff
                    penalty += _penalty.sum()

        return penalty

    def train(self, train_loader, task_idx, dataset=None, use_ewc=True):
        """
        Train the model on the current task with validation-based early stopping.

        Args:
            train_loader: DataLoader for training data
            task_idx: Current task index
            dataset: Dictionary containing 'valid' key for validation data
            use_ewc: Whether to use EWC penalty (should be True for tasks > 0)
        """
        # For first task, don't use EWC
        if task_idx == 0:
            use_ewc = False

        dataset = dataset

        self.model.train()

        # Early stopping
        best_loss = float('inf')
        best_model = None
        patience = getattr(self, 'lr_patience', 5)
        initial_patience = patience
        learning_rate = self.optimizer.param_groups[0]['lr']
        lr_factor = getattr(self, 'lr_factor', 0.5)
        lr_min = getattr(self, 'lr_min', 1.6e-2)

        for epoch in range(self.ewc_epochs):
            # === Training Phase ===
            self.model.train()
            running_loss = 0.0
            num_batches = 0

            for batch_data in train_loader:
                try:
                    if len(batch_data) == 2:
                        inputs, targets = batch_data
                    elif len(batch_data) == 3:
                        inputs, targets, _ = batch_data
                    elif len(batch_data) == 4:
                        inputs, targets, _, _ = batch_data
                    else:
                        print(f"Warning: Unexpected batch format with {len(batch_data)} elements")
                        inputs, targets = batch_data[0], batch_data[1]

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    if len(inputs.shape) == 3:
                        inputs = inputs[:, np.newaxis, :, :]

                    outputs = self.model(inputs, task_id=task_idx)

                    task_loss = self.criterion(outputs, targets)

                    ewc_penalty = 0.0
                    if use_ewc and self._precision_matrices and self._means:
                        ewc_penalty = self._compute_ewc_penalty()

                    total_loss = task_loss + (self.ewc_weight / 2) * ewc_penalty

                    # === Gradient Descent ===
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    running_loss += total_loss.item()
                    num_batches += 1

                except Exception as e:
                    print(f"Error processing batch in task {task_idx}, epoch {epoch}: {e}")
                    continue

            avg_train_loss = running_loss / max(num_batches, 1)
            print(f"Task {task_idx + 1}, Epoch {epoch + 1}/{self.ewc_epochs}, Train Loss: {avg_train_loss:.4f}", end='')

            # === Validation Phase ===
            if dataset and 'valid' in dataset:
                valid_res = self.eval_(dataset['valid'], task_idx)
                print(f", Valid Loss: {valid_res['loss_tot']:.4f}, Valid Acc: {valid_res['acc_t']:.1f}%", end='')

                if valid_res['loss_tot'] < best_loss:
                    best_loss = valid_res['loss_tot']
                    best_model = self.model.state_dict().copy()
                    patience = initial_patience
                    print(' *', end='')
                else:
                    patience -= 1
                    if patience <= 0:
                        learning_rate /= lr_factor

                        if learning_rate < lr_min:
                            print(f"\nEarly stopping: Learning rate {learning_rate:.1e} < minimum {lr_min:.1e}")
                            break

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = learning_rate

                        patience = initial_patience
                        print(f' lr={learning_rate:.1e}', end='')

            print()
        # === Load Best Model ===
        if best_model is not None:
            print(f"Loading best model with validation loss: {best_loss:.4f}")
            self.model.load_state_dict(best_model)

        # === Compute Fisher Information Matrix ===
        if hasattr(self, '_compute_fisher_information'):
            print("Computing Fisher Information Matrix for EWC...")
            self._compute_fisher_information(train_loader, task_idx)

            print("Fisher Information Matrix computed and parameters saved.")

        self.save_model(task_idx)

    def eval_(self, data_loader, task_id):
        """
        Evaluate model on validation/test data (no gradient updates).
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        num_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch_data in data_loader:
                try:
                    if len(batch_data) == 2:
                        inputs, targets = batch_data
                    elif len(batch_data) >= 3:
                        inputs, targets = batch_data[0], batch_data[1]

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    if len(inputs.shape) == 3:
                        inputs = inputs[:, np.newaxis, :, :]

                    outputs = self.model(inputs, task_id=task_id)

                    loss = self.criterion(outputs, targets)
                    total_loss += loss.item()

                    _, pred = outputs.max(1)
                    correct += pred.eq(targets).sum().item()

                    num_samples += inputs.size(0)
                    num_batches += 1

                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    continue

        res = {
            'loss_tot': total_loss / max(num_batches, 1),
            'acc_t': 100.0 * correct / max(num_samples, 1)
        }

        return res

    def evaluate(self, test_loader, task_id, model=None):
        """
        Evaluate the model on test data with proper error handling
        """
        from sklearn.metrics import f1_score, cohen_kappa_score
        import numpy as np

        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        batch_count = 0
        total_time = 0.0
        num_samples = 0

        if len(test_loader) == 0:
            print(f"Warning: Test loader for task {task_id} is empty!")
            return {
                'accuracy': 0.0,
                'loss': 0.0,
                'f1_score': 0.0,
                'inf_time_mat': 0.0
            }

        with torch.no_grad():
            for batch_data in test_loader:
                try:
                    if len(batch_data) == 2:
                        inputs, labels = batch_data
                    elif len(batch_data) == 3:
                        inputs, labels, _ = batch_data
                    elif len(batch_data) == 4:
                        inputs, labels, _, _ = batch_data
                    else:
                        print(f"Warning: Unexpected batch format with {len(batch_data)} elements for task {task_id}")
                        print(f"Batch structure: {[type(x).__name__ for x in batch_data]}")
                        if hasattr(batch_data[0], 'shape'):
                            print(
                                f"Element shapes: {[x.shape if hasattr(x, 'shape') else 'No shape' for x in batch_data]}")
                        inputs, labels = batch_data[0], batch_data[1]

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    start_time = time.time()

                    if len(inputs.shape) == 3:
                        inputs = inputs[:, np.newaxis, :, :]

                    outputs = model(inputs, task_id=task_id)
                    loss = self.criterion(outputs, labels)

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    end_time = time.time()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    batch_count += 1
                    total_time += (end_time - start_time)
                    num_samples += inputs.shape[0]
                    inf_time_mat = total_time / num_samples

                except Exception as e:
                    print(f"Error processing batch in task {task_id}: {e}")
                    print(f"Batch data length: {len(batch_data)}")
                    print(f"Batch data types: {[type(x).__name__ for x in batch_data]}")
                    if len(batch_data) > 0 and hasattr(batch_data[0], 'shape'):
                        print(f"First element shape: {batch_data[0].shape}")
                    if len(batch_data) > 1 and hasattr(batch_data[1], 'shape'):
                        print(f"Second element shape: {batch_data[1].shape}")
                    continue

        if total == 0 or batch_count == 0:
            print(f"Warning: No data processed for task {task_id}!")
            return {
                'accuracy': 0.0,
                'loss': 0.0,
                'f1_score': 0.0,
                'inf_time_mat': 0.0
            }

        accuracy = 100 * correct / total
        avg_loss = total_loss / batch_count

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        try:
            if len(np.unique(all_labels)) > 1:  # Check if we have more than one class
                f1 = f1_score(all_labels, all_predictions, average='macro') * 100
                kappa = cohen_kappa_score(all_labels, all_predictions) * 100
            else:
                print(f"Warning: Only one class found in task {task_id}, setting F1 and Kappa to 0")
                f1 = 0.0
                kappa = 0.0
        except Exception as e:
            print(f"Warning: Error calculating F1/Kappa for task {task_id}: {e}")
            f1 = 0.0
            kappa = 0.0

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'f1_score': f1,
            'inf_time_mat': inf_time_mat
        }

    def save_model(self, task_id):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fisher_matrices': self._precision_matrices,
            'parameter_means': self._means
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))

    def load_model(self, task_id=None):
        """Load model state"""
        model = self.network.Net(self.args)
        checkpoint = torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)),
                                map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        current_feature_extractor = deepcopy(self.model.FeatureExtractor.state_dict())
        model.FeatureExtractor.load_state_dict(current_feature_extractor)

        model = model.to(self.args.device)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._precision_matrices = checkpoint.get('fisher_matrices', {})
        self._means = checkpoint.get('parameter_means', {})

        return model

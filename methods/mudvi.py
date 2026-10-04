import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
import torch
import torch.nn.functional as F
import random
from copy import deepcopy
import os
import torch.nn as nn
import time


class mudvi:
    """
    MUDVI: Online continual decoding of streaming EEG signal without memory buffer
    Modified implementation without replay mechanism
    """

    def __init__(self, model, args, network):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.network = network
        self.checkpoint = args.checkpoint

        # Extract parameters
        self.lr = getattr(args, 'lr', 0.001)
        self.epochs = getattr(args, 'epochs', 200)
        self.device = getattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=getattr(args, 'wd', 0.01)
        )

        self.current_subject = 0
        self.previous_subject = -1

    def _validate_targets(self, targets):
        """Validate targets"""
        targets = torch.clamp(targets, 0, 1)
        if targets.dtype != torch.long:
            targets = targets.long()
        return targets

    def train(self, dataset, subject_idx):
        """
        Train the model on current subject without memory replay
        + validation-based early stopping

        Args:
            dataset: Dictionary containing 'train' and optionally 'valid' keys, or just train_loader
            subject_idx: Current subject index
        """

        if isinstance(dataset, dict):
            train_loader = dataset['train']
            valid_loader = dataset['valid']
        else:
            train_loader = dataset
            valid_loader = None

        if not hasattr(self, 'previous_subject'):
            self.previous_subject = -1

        if subject_idx != self.previous_subject:
            self.previous_subject = subject_idx

        self.current_subject = subject_idx
        self.model.train()

        if len(train_loader) == 0:
            print(f"ERROR: train_loader for subject {subject_idx + 1} is empty!")
            return

        print(f"Training subject {subject_idx + 1} with {len(train_loader)} batches")

        best_loss = float('inf')
        best_model = None
        patience = getattr(self, 'lr_patience', 10)
        initial_patience = patience
        learning_rate = self.optimizer.param_groups[0]['lr']
        lr_factor = getattr(self, 'lr_factor', 0.5)
        lr_min = getattr(self, 'lr_min', 1e-6)

        for epoch in range(self.epochs):
            # === Training Phase ===
            self.model.train()
            running_loss = 0.0
            num_batches = 0
            batch_count = 0

            for batch_data in train_loader:
                batch_count += 1

                if len(batch_data) == 2:
                    # (inputs, targets)
                    inputs, targets = batch_data
                elif len(batch_data) == 4:
                    inputs, targets, _, _ = batch_data
                else:
                    print(f"ERROR: Unexpected batch data format with {len(batch_data)} elements")
                    continue

                if inputs is None or targets is None:
                    print(f"ERROR: Batch {batch_count} contains None data")
                    continue

                if inputs.size(0) == 0:
                    print(f"ERROR: Batch {batch_count} is empty")
                    continue

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                try:
                    targets = self._validate_targets(targets)
                except Exception as e:
                    print(f"ERROR: Target validation failed for batch {batch_count}: {e}")
                    continue

                original_shape = inputs.shape
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)  # [batch, 1, channels, time]
                elif inputs.dim() == 2:
                    print(f"WARNING: 2D input detected with shape {inputs.shape}")
                    continue

                try:
                    outputs = self.model(inputs, task_id=subject_idx)

                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs

                    if logits.size(0) != targets.size(0):
                        print(f"ERROR: Logits batch size {logits.size(0)} != targets batch size {targets.size(0)}")
                        continue

                    # Only compute current task loss
                    loss = self.criterion(logits, targets)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
                    num_batches += 1

                except Exception as e:
                    print(f"ERROR: Batch {batch_count} processing failed: {e}")
                    continue

            if num_batches > 0:
                avg_train_loss = running_loss / num_batches
                print(
                    f"Subject {subject_idx + 1}, Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}",
                    end='')
            else:
                print(f"ERROR: No valid batches processed in epoch {epoch + 1}")
                continue

            if valid_loader is not None:
                valid_res = self._evaluate_on_validation(valid_loader, subject_idx)

                print(f", Valid Loss: {valid_res['loss_tot']:.4f}, Valid Acc: {valid_res['acc_t']:.1f}%", end='')

                if valid_res['loss_tot'] < best_loss:
                    best_loss = valid_res['loss_tot']
                    best_model = self.model.state_dict().copy()
                    patience = initial_patience
                    print(' *', end='')
                else:
                    patience -= 1

                    if patience <= 0:
                        break
                        learning_rate /= lr_factor

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = learning_rate

                        patience = initial_patience
                        print(f' lr={learning_rate:.1e}', end='')

            print()

        if best_model is not None:
            print(f"Loading best model with validation loss: {best_loss:.4f}")
            self.model.load_state_dict(best_model)

        self.save_model(subject_idx)
        print("-" * 50)

    def _evaluate_on_validation(self, valid_loader, subject_idx):
        """
        Evaluate model on validation data (no gradient updates).
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        num_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(valid_loader):
                try:
                    if len(batch_data) == 2:
                        inputs, targets = batch_data
                    elif len(batch_data) == 4:
                        inputs, targets, _, _ = batch_data
                    else:
                        print(f"ERROR: Unexpected validation batch format with {len(batch_data)} elements")
                        continue

                    if inputs is None or targets is None or inputs.size(0) == 0:
                        continue

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    targets = self._validate_targets(targets)

                    if inputs.dim() == 3:
                        inputs = inputs.unsqueeze(1)
                    elif inputs.dim() == 2:
                        continue

                    outputs = self.model(inputs, task_id=subject_idx)

                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs

                    if logits.size(0) != targets.size(0):
                        continue

                    loss = self.criterion(logits, targets)
                    total_loss += loss.item()

                    _, pred = logits.max(1)
                    correct += pred.eq(targets).sum().item()

                    num_samples += inputs.size(0)
                    num_batches += 1

                except Exception as e:
                    print(f"ERROR: Validation batch {batch_idx} processing failed: {e}")
                    continue

        res = {
            'loss_tot': total_loss / max(num_batches, 1),
            'acc_t': 100.0 * correct / max(num_samples, 1)
        }

        return res

    def evaluate(self, test_loader, subject_idx, model=None):
        """Evaluate model performance"""
        self.model.eval()

        correct = 0
        total = 0
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        total_time = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                try:
                    if len(batch_data) == 2:
                        inputs, targets = batch_data
                    elif len(batch_data) == 4:
                        inputs, targets, _, _ = batch_data
                    else:
                        print(f"ERROR: Unexpected validation batch format with {len(batch_data)} elements")
                        continue

                    if inputs is None or targets is None or inputs.size(0) == 0:
                        continue

                    inputs = inputs.to(self.device)
                    labels = targets.to(self.device)
                    labels = self._validate_targets(labels)
                    start_time = time.time()

                    if inputs.dim() == 3:
                        inputs = inputs.unsqueeze(1)
                    elif inputs.dim() == 2:
                        continue

                    outputs = model(inputs, task_id=subject_idx)

                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs

                    loss = self.criterion(logits, labels)

                    total_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    num_samples += inputs.shape[0]
                    inf_time_mat = total_time / num_samples

                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    continue

            accuracy = 100 * correct / total
            avg_loss = total_loss / len(test_loader)

            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)

            f1 = f1_score(all_labels, all_predictions, average='macro') * 100

            return {
                'accuracy': accuracy,
                'loss': avg_loss,
                'f1_score': f1,
                'inf_time_mat': inf_time_mat
            }

    def save_model(self, task_id):
        """Save model state for MuDvi"""
        torch.save({
            'model_state_dict': self.model.state_dict()
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))

    def load_model(self, task_id=None):
        """Load model state with proper structure initialization for MuDvi"""
        model = self.network.Net(self.args)

        checkpoint = torch.load(os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)),
                                map_location=self.device)

        saved_state = checkpoint['model_state_dict']
        model_state = model.state_dict()
        for key in saved_state:
            if key in model_state and saved_state[key].shape == model_state[key].shape:
                model_state[key] = saved_state[key]

        model.load_state_dict(model_state)
        current_feature_extractor = deepcopy(self.model.FeatureExtractor.state_dict())
        model.FeatureExtractor.load_state_dict(current_feature_extractor)

        model = model.to(self.args.device)

        return model
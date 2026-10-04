import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
from copy import deepcopy
import os
import time


class Finetuning:
    def __init__(self, model, args, network):
        """
        Initialize the Finetuning class (baseline without any forgetting prevention).

        Args:
            model: Neural network model
            args: Configuration object containing hyperparameters
            network: Network class for creating new model instances
        """
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.network = network
        self.checkpoint = args.checkpoint

        self.lr = getattr(args, 'lr', 0.001)
        self.lr_patience = args.lr_patience
        self.epochs = getattr(args, 'n_epochs')
        self.device = getattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=getattr(args, 'wd', 0.01)
        )

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def print_config(self):
        """Print Finetuning-specific configuration"""
        print(f"  - Learning rate: {self.lr}")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Weight decay: {getattr(self.args, 'wd', 0.01)}")

    def train(self, dataset, task_idx):
        """
        Train the model on the current task with validation-based early stopping.

        Args:
            dataset: Dictionary containing 'train' and 'valid' keys or just training data
            task_idx: Current task index
        """

        if isinstance(dataset, dict):
            train_loader = dataset['train']
            valid_loader = dataset['valid']
        else:
            train_loader = dataset
            valid_loader = None

        self.model.train()

        best_loss = float('inf')
        best_model = None
        patience = getattr(self, 'lr_patience', 5)
        initial_patience = patience
        learning_rate = self.optimizer.param_groups[0]['lr']
        lr_factor = getattr(self, 'lr_factor', 0.5)
        lr_min = getattr(self, 'lr_min', 1.6e-2)

        for epoch in range(self.epochs):
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

                    total_loss = task_loss

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
            print(f"Task {task_idx + 1}, Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_train_loss:.4f}", end='')

            # === Validation Phase ===
            if valid_loader is not None:
                valid_res = self.eval_(valid_loader, task_idx)
                print(f", Valid Loss: {valid_res['loss_tot']:.4f}, Valid Acc: {valid_res['acc_t']:.1f}%", end='')

                if valid_res['loss_tot'] < best_loss:
                    best_loss = valid_res['loss_tot']
                    # Store the best model parameters
                    best_model = deepcopy(self.model.state_dict())
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
            else:
                # If no validation data, just save the current state
                best_model = deepcopy(self.model.state_dict())
                best_loss = avg_train_loss

            print()

        # === Load Best Model ===
        if best_model is not None:
            print(f"Loading best model with validation loss: {best_loss:.4f}")
            self.model.load_state_dict(best_model)

        # Save the final model state for this task
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
        # Use the provided model or the current model
        eval_model = model if model is not None else self.model
        eval_model.eval()

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
                'inf_time_mat':  0.0
            }

        with torch.no_grad():
            for batch_data in test_loader:
                try:
                    # Handle different data loader formats
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

                    outputs = eval_model(inputs, task_id=task_id)
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
            if len(np.unique(all_labels)) > 1:
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
        os.makedirs(self.checkpoint, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'task_id': task_id
        }, os.path.join(self.checkpoint, f'model_{task_id}.pth.tar'))

        # Also save as the latest model for loading in the next task
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'task_id': task_id
        }, os.path.join(self.checkpoint, 'latest_model.pth.tar'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
from copy import deepcopy
import os
import random
import time

class ExperienceBuffer:
    """
    Experience buffer for storing samples from previous tasks
    """

    def __init__(self, buffer_size=5000, device='cuda'):
        self.buffer_size = buffer_size
        self.device = device
        self.buffer = []
        self.task_buffers = {}

    def add_task(self, task_id):
        if task_id not in self.task_buffers:
            self.task_buffers[task_id] = []

    def add_data(self, inputs, targets, task_id):
        if task_id not in self.task_buffers:
            self.add_task(task_id)

        if inputs.dim() == 5:
            inputs = inputs.squeeze(1)
        elif inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)

        data_to_store = (inputs.cpu(), targets.cpu(), task_id)
        self.task_buffers[task_id].append(data_to_store)

        max_per_task = self.buffer_size // max(1, len(self.task_buffers))
        if len(self.task_buffers[task_id]) > max_per_task:
            self.task_buffers[task_id].pop(random.randint(0, len(self.task_buffers[task_id]) - 1))

    def sample(self, batch_size, exclude_task=None):
        all_samples = []
        for task_id, samples in self.task_buffers.items():
            if exclude_task is not None and task_id == exclude_task:
                continue
            all_samples.extend(samples)

        if len(all_samples) == 0:
            return None

        sample_size = min(batch_size, len(all_samples))
        sampled = random.sample(all_samples, sample_size)

        inputs = torch.stack([s[0] for s in sampled]).to(self.device)
        targets_list = [s[1] for s in sampled]

        targets = torch.cat([t.view(-1) for t in targets_list]).to(self.device)

        task_ids = [s[2] for s in sampled]

        return inputs, targets, task_ids

    def get_size(self):
        return sum(len(samples) for samples in self.task_buffers.values())


class ExperienceReplay:
    def __init__(self, model, args, network):
        """
        Initialize the Experience Replay (ER) class.

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

        self.buffer_size = getattr(args, 'er_buffer_size', 5000)
        self.replay_batch_size = getattr(args, 'er_replay_batch_size', 8)
        self.alpha = getattr(args, 'er_alpha', 0.5)  ### replay loss weight
        self.num_samples_per_task = getattr(args, 'num_samples', 20)  ##0.1
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=getattr(args, 'wd', 0.01)
        )

        self.experience_buffer = ExperienceBuffer(
            buffer_size=self.buffer_size,
            device=self.device
        )

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    def print_config(self):
        """Print ER-specific configuration"""
        print(f"  - Learning rate: {self.lr}")
        print(f"  - Epochs: {self.epochs}")
        print(f"  - Weight decay: {getattr(self.args, 'wd', 0.01)}")
        print(f"  - Buffer size: {self.buffer_size}")
        print(f"  - Replay batch size: {self.replay_batch_size}")
        print(f"  - Alpha (replay weight): {self.alpha}")
        print(f"  - Samples per task: {self.num_samples_per_task}")

    def store_experience(self, train_loader, task_idx, num_samples=None):
        """
        Store samples from current task to experience buffer

        Args:
            train_loader: DataLoader for current task
            task_idx: Current task index
            num_samples: Number of samples to store (if None, use self.num_samples_per_task)
        """
        print(f"Storing experiences for task {task_idx}...")

        max_samples = num_samples if num_samples is not None else self.num_samples_per_task

        stored_count = 0

        all_inputs = []
        all_targets = []

        for batch_data in train_loader:
            if len(batch_data) == 2:
                inputs, targets = batch_data
            elif len(batch_data) == 3:
                inputs, targets, _ = batch_data
            elif len(batch_data) == 4:
                inputs, targets, _, _ = batch_data
            else:
                inputs, targets = batch_data[0], batch_data[1]

            if len(inputs.shape) == 3:
                inputs = inputs[:, np.newaxis, :, :]

            all_inputs.append(inputs)
            all_targets.append(targets)

        all_inputs = torch.cat(all_inputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        total_samples = all_inputs.size(0)
        num_to_store = min(max_samples, total_samples)

        indices = torch.randperm(total_samples)[:num_to_store]

        for idx in indices:
            target_i = all_targets[idx].item() if all_targets[idx].numel() == 1 else all_targets[idx]
            self.experience_buffer.add_data(
                all_inputs[idx].unsqueeze(0),
                torch.tensor([target_i], dtype=torch.long),
                task_idx
            )
            stored_count += 1

        print(f"Stored {stored_count} samples from {total_samples} available. "
              f"(Configured to store {max_samples} samples per task). "
              f"Total buffer size: {self.experience_buffer.get_size()}")

    def train(self, dataset, task_idx):
        """
        Train the model on the current task with experience replay.

        Args:
            dataset: Dictionary containing 'train' and 'valid' DataLoaders
            task_idx: Current task index
        """
        if isinstance(dataset, dict):
            train_loader = dataset['train']
            valid_loader = dataset['valid']
        else:
            train_loader = dataset
            valid_loader = None

        self.model.train()

        # Early stopping
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
            running_task_loss = 0.0
            running_replay_loss = 0.0
            num_batches = 0

            for batch_idx, batch_data in enumerate(train_loader):
                try:
                    if len(batch_data) == 2:
                        inputs, targets = batch_data
                    elif len(batch_data) == 3:
                        inputs, targets, _ = batch_data
                    elif len(batch_data) == 4:
                        inputs, targets, _, _ = batch_data
                    else:
                        inputs, targets = batch_data[0], batch_data[1]

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    if len(inputs.shape) == 3:
                        inputs = inputs[:, np.newaxis, :, :]

                    # === Current task loss ===
                    outputs = self.model(inputs, task_id=task_idx)
                    task_loss = self.criterion(outputs, targets)

                    # === Experience replay loss ===
                    replay_loss = torch.tensor(0.0).to(self.device)
                    if task_idx > 0 and self.experience_buffer.get_size() > 0:
                        replay_data = self.experience_buffer.sample(
                            self.replay_batch_size,
                            exclude_task=None
                        )

                        if replay_data is not None:
                            replay_inputs, replay_targets, replay_task_ids = replay_data

                            if replay_inputs.dim() == 5:
                                replay_inputs = replay_inputs.squeeze(1)

                            if replay_targets.dim() > 1:
                                replay_targets = replay_targets.squeeze(-1)

                            replay_losses = []
                            for i in range(replay_inputs.size(0)):
                                replay_output = self.model(
                                    replay_inputs[i:i + 1],
                                    task_id=replay_task_ids[i]
                                )
                                target_i = replay_targets[i:i + 1]
                                if target_i.dim() > 1:
                                    target_i = target_i.squeeze(-1)
                                replay_loss_i = self.criterion(replay_output, target_i)
                                replay_losses.append(replay_loss_i)

                            if replay_losses:
                                replay_loss = torch.stack(replay_losses).mean()

                    # === Total loss ===
                    if task_idx > 0 and replay_loss > 0:
                        total_loss = (1 - self.alpha) * task_loss + self.alpha * replay_loss
                    else:
                        total_loss = task_loss

                    # === Gradient Descent ===
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    running_loss += total_loss.item()
                    running_task_loss += task_loss.item()
                    running_replay_loss += replay_loss.item() if replay_loss > 0 else 0
                    num_batches += 1

                except Exception as e:
                    print(f"Error processing batch in task {task_idx}, epoch {epoch}: {e}")
                    continue

            avg_train_loss = running_loss / max(num_batches, 1)
            avg_task_loss = running_task_loss / max(num_batches, 1)
            avg_replay_loss = running_replay_loss / max(num_batches, 1)

            print(f"Task {task_idx + 1}, Epoch {epoch + 1}/{self.epochs}, "
                  f"Total Loss: {avg_train_loss:.4f} "
                  f"(Task: {avg_task_loss:.4f}, Replay: {avg_replay_loss:.4f})", end='')

            # === Validation Phase ===
            if valid_loader is not None:
                valid_res = self.eval_(valid_loader, task_idx)
                print(f", Valid Loss: {valid_res['loss_tot']:.4f}, Valid Acc: {valid_res['acc_t']:.1f}%", end='')

                # Early stopping based on validation loss
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

        # === Store experiences from current task ===
        self.store_experience(train_loader, task_idx)

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

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    batch_count += 1
                    end_time = time.time()
                    total_time += (end_time - start_time)
                    num_samples += inputs.shape[0]
                    inf_time_mat = total_time / num_samples


                except Exception as e:
                    print(f"Error processing batch in task {task_id}: {e}")
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
        """Save model state and buffer information"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'buffer_size': self.experience_buffer.get_size(),
            'task_buffers_sizes': {k: len(v) for k, v in self.experience_buffer.task_buffers.items()}
        }, os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)))

    def load_model(self, task_id=None):
        """Load model state"""
        model = self.network.Net(self.args)
        checkpoint = torch.load(
            os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id)),
            map_location=self.device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        current_feature_extractor = deepcopy(self.model.FeatureExtractor.state_dict())
        model.FeatureExtractor.load_state_dict(current_feature_extractor)
        model = model.to(self.args.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'buffer_size' in checkpoint:
            print(f"Buffer size when saved: {checkpoint['buffer_size']}")
        if 'task_buffers_sizes' in checkpoint:
            print(f"Task buffer sizes: {checkpoint['task_buffers_sizes']}")

        return model
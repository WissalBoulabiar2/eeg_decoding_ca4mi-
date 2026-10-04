import torch.nn as nn
import random
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
from copy import deepcopy
import os
import time


class DER:
    """
    DER implementation with Mask-based pruning - Without replay loss
    """

    def __init__(self, model, args, network):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.network = network
        self.checkpoint = args.checkpoint

        # Extract parameters
        self.lr = getattr(args, 'lr', 0.0001)
        self.der_epochs = getattr(args, 'der_epochs')
        self.lambda_aux = getattr(args, 'lambda_aux')
        self.device = getattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Mask-based pruning parameters
        self.use_pruning = getattr(args, 'der_use_pruning')  # True
        self.s_max = getattr(args, 'der_s_max', 5.0)
        self.pruning_threshold = getattr(args, 'der_pruning_threshold', 0.1)
        self.lambda_sparsity = getattr(args, 'lambda_sparsity')  #

        self.pruning_warmup_epochs = getattr(args, 'pruning_warmup_epochs', 20)
        self.pruning_schedule = getattr(args, 'pruning_schedule', 'cosine')
        self.target_sparsity = getattr(args, 'target_sparsity', 0.3)

        # Initialize mask module
        if self.use_pruning:
            self.mask_module = MaskModule(
                s_max=self.s_max,
                pruning_threshold=self.pruning_threshold
            )
            self.mask_module.initialize_masks(self.model)

        # Initialize optimizer
        self.optimizer = None
        self._initialize_optimizer()

        self.current_task = 0

        if self.use_pruning:
            self.mask_module = MaskModule(
                s_max=self.s_max,
                pruning_threshold=self.pruning_threshold
            )
            print("MaskModule created successfully")

            self.mask_module.initialize_masks(self.model)
            print(f"Masks initialized. Number of mask parameters: {len(self.mask_module.mask_params)}")

            print("Mask parameters:")
            for name in self.mask_module.mask_params:
                print(f"  - {name}")

    def get_current_pruning_strength(self, epoch, total_epochs):
        """Adjust pruning intensity according to training progress"""
        if epoch < self.pruning_warmup_epochs:
            return 0.0  # Do not prune during the preheating period

        progress = (epoch - self.pruning_warmup_epochs) / max(1, total_epochs - self.pruning_warmup_epochs)

        if self.pruning_schedule == 'linear':
            return progress * self.target_sparsity
        elif self.pruning_schedule == 'cosine':
            return self.target_sparsity * (1 - np.cos(progress * np.pi)) / 2
        else:
            return self.target_sparsity

    def _initialize_optimizer(self):
        """Initialize optimizer including mask parameters"""
        # Collect all parameters that need to be optimized
        all_params = list(self.model.parameters())

        # If pruning is enabled, add the mask parameter
        if self.use_pruning:
            for param_name in dir(self.model):
                if param_name.startswith('mask_'):
                    mask_param = getattr(self.model, param_name)
                    if isinstance(mask_param, nn.Parameter):
                        all_params.append(mask_param)

        self.optimizer = optim.Adam(
            all_params,
            lr=self.lr,
            weight_decay=getattr(self.args, 'wd', 0.01)
        )

    def _setup_robust_optimizer(self, task_idx):
        """Use different learning rate strategies"""
        param_groups = []

        base_params = list(self.model.FeatureExtractor.base_model.parameters())
        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': 5e-5,
                'name': 'base_extractor'
            })

        # Task-specific layers
        if hasattr(self.model.FeatureExtractor, 'task_specific_layers'):
            task_params = list(self.model.FeatureExtractor.task_specific_layers.parameters())
            if task_params:
                param_groups.append({
                    'params': task_params,
                    'lr': 1e-4,
                    'name': 'task_layers'
                })

        # classification head
        if hasattr(self.model, 'heads') and len(self.model.heads) > task_idx:
            head_params = list(self.model.heads[task_idx].parameters())
            if head_params:
                param_groups.append({
                    'params': head_params,
                    'lr': 5e-4,
                    'name': 'classifier'
                })

        # Mask parameters
        if self.use_pruning:
            mask_params = []
            for param_name in dir(self.model):
                if param_name.startswith('mask_'):
                    mask_param = getattr(self.model, param_name)
                    if isinstance(mask_param, nn.Parameter):
                        mask_params.append(mask_param)

            if mask_params:
                param_groups.append({
                    'params': mask_params,
                    'lr': 1e-4,
                    'name': 'mask_params'
                })

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=5e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )

    def _adjust_learning_rate(self, epoch, config):
        # Use cosine annealing
        if hasattr(self, 'scheduler'):
            self.scheduler.step()
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=1e-6
            )

    def train(self, dataset, task_idx):
        train_loader = dataset['train']
        valid_loader = dataset['valid']

        # Add new task
        if task_idx >= self.model.num_tasks:
            self.model.add_task()

            if self.use_pruning:
                self.mask_module.initialize_masks(self.model)

        self.current_task = task_idx

        self._diagnose_and_fix_model(task_idx)

        # Reset Optimizer
        self._setup_robust_optimizer(task_idx)

        config = self._get_training_config()
        best_loss = float('inf')
        best_model = None
        patience_counter = 0

        # Add verification loss tracking
        val_loss_history = []

        for epoch in range(config['epochs']):
            # Calculate the current pruning intensity
            current_pruning_strength = 0.0
            if self.use_pruning:
                current_pruning_strength = self.get_current_pruning_strength(epoch, config['epochs'])
                self.mask_module.apply_masks(self.model, pruning_strength=current_pruning_strength)

                if epoch % 10 == 0:
                    pruned, total, sparsity = self.mask_module.get_sparsity_statistics(self.model)
                    print(f"\nEpoch {epoch}: Pruning strength={current_pruning_strength:.2f}, "
                          f"Actual sparsity={sparsity:.2f}%")

            self._adjust_learning_rate(epoch, config)

            # train one epoch
            train_metrics = self._train_one_epoch(train_loader, task_idx, epoch, config)

            # validate the model
            valid_metrics = self._validate_epoch(valid_loader, task_idx, epoch)
            val_loss_history.append(valid_metrics['loss'])

            # print epoch progress
            self._print_epoch_progress(epoch, config['epochs'], train_metrics, valid_metrics)

            # Early stopping
            is_improved, should_stop = self._check_early_stopping(
                valid_metrics['loss'], best_loss, patience_counter, config
            )

            if is_improved:
                best_loss = valid_metrics['loss']
                best_model = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if should_stop:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        if best_model is not None:
            self.model.load_state_dict(best_model)

        self.save_model(task_idx)

    def _train_one_epoch(self, train_loader, task_idx, epoch, config):
        """Train one epoch without replay loss"""
        self.model.train()

        metrics = {
            'total_loss': 0.0,
            'main_loss': 0.0,
            'aux_loss': 0.0,
            'mask_reg_loss': 0.0,
            'processed_batches': 0,
            'successful_batches': 0,
            'predictions': [],
            'targets': [],
            'avg_grad_norm': 0.0
        }

        for batch_idx, batch_data in enumerate(train_loader):
            try:
                inputs, targets, tt = self._prepare_batch_data(batch_data, batch_idx)
                if inputs is None:
                    continue

                outputs = self._safe_forward(inputs, targets, task_idx, tt)
                if outputs is None:
                    continue

                losses = self._compute_losses(outputs, targets, inputs, task_idx, epoch)
                if losses is None:
                    continue

                # apply mask regularization if enabled
                if self.use_pruning:
                    self.mask_module.apply_masks(self.model)

                success = self._safe_backward(losses['total'], config)
                if not success:
                    continue

                self._update_batch_metrics(metrics, losses, outputs, targets)

                # update mask regularization loss
                if 'mask_reg' in losses:
                    metrics['mask_reg_loss'] += losses['mask_reg'].item()

            except Exception as e:
                print(f"  Error in batch {batch_idx}: {e}")
                continue

        # Calculate average metrics
        if metrics['successful_batches'] > 0:
            for key in ['total_loss', 'main_loss', 'aux_loss', 'mask_reg_loss', 'avg_grad_norm']:
                metrics[key] /= metrics['successful_batches']

        return metrics

    def _diagnose_and_fix_model(self, task_idx):
        """Diagnose and fix model issues"""
        # Check classifier
        if hasattr(self.model, 'heads') and len(self.model.heads) > task_idx:
            classifier = self.model.heads[task_idx]
            needs_reset = False

            for layer in classifier:
                if isinstance(layer, torch.nn.Linear):
                    weight_std = layer.weight.std().item()
                    weight_norm = layer.weight.norm().item()

                    if weight_std < 1e-3 or weight_norm < 1e-2:
                        needs_reset = True
                        break

            if needs_reset:
                self._reset_classifier_robust(task_idx)

    def _reset_classifier_robust(self, task_idx):
        """Robust classifier reset"""
        if hasattr(self.model, 'heads') and len(self.model.heads) > task_idx:
            classifier = self.model.heads[task_idx]

            with torch.no_grad():
                for layer in classifier:
                    if isinstance(layer, torch.nn.Linear):
                        # Use He initialization, suitable for ELU activation
                        fan_in = layer.weight.shape[1]
                        std = (2.0 / fan_in) ** 0.5
                        layer.weight.normal_(0, std)

                        if layer.bias is not None:
                            layer.bias.uniform_(-0.01, 0.01)

    def _get_training_config(self):
        """Get training configuration"""
        return {
            'epochs': min(getattr(self, 'der_epochs'), 200),
            'lr': 1e-3,
            'patience': 15,
            'grad_clip': 2.0,
            'lr_decay_factor': 0.8,
            'lr_decay_patience': 8
        }

    def _validate_epoch(self, valid_loader, task_idx, epoch):
        """Validate one epoch"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch_data in valid_loader:
                try:
                    inputs, targets, tt = self._prepare_batch_data(batch_data, -1)
                    if inputs is None:
                        continue

                    # Forward pass
                    outputs = self._safe_forward(inputs, targets, task_idx, tt)
                    if outputs is None:
                        continue

                    logits = outputs['logits']

                    # Calculate loss and accuracy
                    loss = self.criterion(logits, targets)
                    total_loss += loss.item()

                    _, predicted = torch.max(logits, 1)
                    correct += (predicted == targets).sum().item()
                    total_samples += targets.size(0)
                    num_batches += 1

                except Exception as e:
                    continue

        if num_batches == 0:
            return {'loss': float('inf'), 'accuracy': 0.0}

        return {
            'loss': total_loss / num_batches,
            'accuracy': 100.0 * correct / max(total_samples, 1)
        }

    def _print_epoch_progress(self, epoch, total_epochs, train_metrics, valid_metrics):
        """Print training progress"""
        print(f"Epoch {epoch + 1:3d}/{total_epochs}: "
              f"Train Loss={train_metrics['total_loss']:.4f} | "
              f"Valid Loss={valid_metrics['loss']:.4f}, "
              f"Valid Acc={valid_metrics['accuracy']:.1f}% ")

    def _detect_and_fix_training_issues(self, train_metrics, epoch, task_idx):
        """Detect and fix training issues"""
        from collections import Counter

        # Check prediction bias
        if train_metrics['predictions']:
            pred_dist = Counter(train_metrics['predictions'])

            if len(pred_dist) == 1:
                # Reset classifier every 10 epochs
                if epoch % 10 == 0 and epoch > 0:
                    self._reset_classifier_robust(task_idx)

                    # Temporarily increase classifier learning rate
                    if hasattr(self.optimizer, 'param_groups'):
                        for group in self.optimizer.param_groups:
                            if group.get('name') == 'classifier':
                                group['lr'] *= 1.5

    def _check_early_stopping(self, current_loss, best_loss, patience_counter, config):
        """Check early stopping conditions"""
        is_improved = current_loss < best_loss
        should_stop = patience_counter >= config['patience']

        return is_improved, should_stop

    def _compute_light_regularization(self):
        """Compute lightweight regularization"""
        reg_loss = 0.0
        for param in self.model.parameters():
            if param.requires_grad:
                reg_loss += param.pow(2).sum()
        return reg_loss

    def _prepare_batch_data(self, batch_data, batch_idx):
        """Prepare batch data"""
        try:
            if len(batch_data) >= 3:
                inputs, targets, tt = batch_data[0], batch_data[1], batch_data[2]
            else:
                inputs, targets, tt = batch_data[0], batch_data[1], None

            if inputs is None or targets is None or inputs.size(0) == 0:
                return None, None, None

            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()
            if tt is not None:
                tt = tt.to(self.device)

            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)

            return inputs, targets, tt

        except Exception as e:
            return None, None, None

    def _safe_forward(self, inputs, targets, task_idx, tt):
        """Safe forward pass"""
        try:
            if tt is not None:
                outputs = self.model(inputs, task_id=task_idx, tt=tt)
            else:
                outputs = self.model(inputs, task_id=task_idx)

            if isinstance(outputs, dict) and 'logits' in outputs:
                logits = outputs['logits']
            else:
                logits = outputs
                outputs = {'logits': logits}

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return None

            return outputs

        except Exception as e:
            return None

    def _safe_backward(self, total_loss, config):
        """Safe backward pass"""
        try:
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=config['grad_clip']
            )

            # Update parameters
            self.optimizer.step()

            return True

        except Exception as e:
            return False

    def _update_batch_metrics(self, metrics, losses, outputs, targets):
        """Update batch statistics"""
        metrics['total_loss'] += losses['total'].item()
        metrics['main_loss'] += losses['main'].item()
        metrics['aux_loss'] += losses['aux'].item()
        metrics['successful_batches'] += 1

        # Collect predictions
        with torch.no_grad():
            _, predicted = torch.max(outputs['logits'], 1)
            metrics['predictions'].extend(predicted.cpu().numpy().tolist())
            metrics['targets'].extend(targets.cpu().numpy().tolist())

    def _compute_losses(self, outputs, targets, inputs, task_idx, epoch):
        """Compute losses without replay loss"""
        try:
            logits = outputs['logits']
            main_loss = self.criterion(logits, targets)

            if torch.isnan(main_loss) or torch.isinf(main_loss):
                return None

            total_loss = main_loss
            aux_loss = torch.tensor(0.0, device=self.device)

            # Mask regularization
            mask_reg_loss = torch.tensor(0.0, device=self.device)
            if self.use_pruning and epoch >= self.pruning_warmup_epochs:
                mask_reg_loss = self.mask_module.get_mask_regularization_loss(self.model)
                #  Increase mask loss weight as training progresses
                progress = epoch / max(1, self.der_epochs)
                dynamic_lambda = self.lambda_sparsity * (1 + progress)
                total_loss = total_loss + dynamic_lambda * mask_reg_loss

            # Weight regularization
            reg_loss = self._compute_light_regularization()
            total_loss = total_loss + 1e-5 * reg_loss

            return {
                'total': total_loss,
                'main': main_loss,
                'aux': aux_loss,
                'reg': reg_loss,
                'mask_reg': mask_reg_loss
            }

        except Exception as e:
            return None

    def _print_training_summary(self, task_idx):
        """Print training summary including sparsity statistics"""
        print(f"\n{'=' * 60}")
        print(f"TRAINING SUMMARY FOR TASK {task_idx}")
        print(f"{'=' * 60}")

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")

        # purning configuration
        print(f"\nPruning Configuration:")
        print(f"  - Enabled: {self.use_pruning}")

        if self.use_pruning:
            try:
                # print pruning module details
                print(f"  - Mask module exists: {hasattr(self, 'mask_module')}")
                print(f"  - Number of mask params: {len(self.mask_module.mask_params)}")

                # calculate and print pruning statistics
                pruned, total, sparsity = self.mask_module.get_sparsity_statistics(self.model)
                print(f"\nPruning Statistics:")
                print(f"  - Total mask parameters: {total:,}")
                print(f"  - Pruned parameters: {pruned:,}")
                print(f"  - Sparsity ratio: {sparsity:.2f}%")

                # check if masks are available
                print(f"\nSample mask values:")
                for i, (name, mask) in enumerate(self.mask_module.masks.items()):
                    if i < 3:
                        mask_vals = mask.flatten()[:10].cpu().numpy()
                        print(f"  - {name}: min={mask.min():.4f}, max={mask.max():.4f}, mean={mask.mean():.4f}")

            except Exception as e:
                print(f"  - ERROR computing pruning stats: {e}")
                import traceback
                traceback.print_exc()

        print(f"{'=' * 60}\n")

    def save_model(self, task_id):
        """Save model state with extensive debug info"""
        try:
            import os

            os.makedirs(self.checkpoint, exist_ok=True)
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'task_id': task_id,
                'num_tasks': self.model.num_tasks,
                'use_pruning': self.use_pruning
            }

            # save pruning masks if applicable
            if self.use_pruning:
                mask_count = 0
                for param_name in dir(self.model):
                    if param_name.startswith('mask_'):
                        mask_count += 1

            # save the model state
            save_path = os.path.join(self.checkpoint, f'model_{task_id}.pth.tar')
            print(f"Save path: {save_path}")

            torch.save(save_dict, save_path)

        except Exception as e:
            print(f"\n✗ ERROR in save_model: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    def evaluate(self, test_loader, task_idx, model=None):
        self.model.eval()

        if hasattr(self.model, 'reset_classifier_for_task'):
            current_classifier = self.model.get_classifier_for_task(task_idx)
            if current_classifier is not None:
                with torch.no_grad():
                    for layer in current_classifier:
                        if isinstance(layer, torch.nn.Linear):
                            weight_norm = torch.norm(layer.weight).item()
                            if weight_norm > 100 or weight_norm < 0.01:
                                self.model.reset_classifier_for_task(task_idx)
                                break

                            if layer.bias is not None:
                                bias_range = layer.bias.max() - layer.bias.min()
                                if bias_range > 10:
                                    self.model.reset_classifier_for_task(task_idx)
                                    break

        # Apply masks during evaluation
        if self.use_pruning:
            self.mask_module.apply_masks(self.model)

        correct = 0
        total = 0
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        batch_count = 0
        total_time = 0.0
        num_samples = 0

        with torch.no_grad():
            for inputs, labels, tt, _ in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                tt = tt.to(self.device)
                start_time = time.time()

                if len(inputs.shape) == 3:
                    inputs = inputs[:, np.newaxis, :, :]
                try:
                    outputs = model(inputs, task_id=task_idx, tt=tt)

                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    elif isinstance(outputs, dict):
                        possible_keys = ['output', 'prediction', 'result']
                        logits = None
                        for key in possible_keys:
                            if key in outputs:
                                logits = outputs[key]
                                break
                        if logits is None:
                            logits = list(outputs.values())[0]
                    else:
                        logits = outputs

                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)

                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logits = torch.zeros_like(logits)

                except Exception as e:
                    batch_size = inputs.size(0)
                    logits = torch.zeros(batch_size, 2).to(inputs.device)

                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                end_time = time.time()

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                batch_count += 1

                total_time += (end_time - start_time)
                num_samples += inputs.shape[0]
                inf_time_mat = total_time / num_samples

        accuracy = 100 * correct / total
        avg_loss = total_loss / batch_count

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        f1 = f1_score(all_labels, all_predictions, average='macro') * 100.0

        self.model.train()

        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'f1_score': f1,
            'inf_time_mat': inf_time_mat
        }

    def load_model(self, task_id=None):
        """Load model state with error handling"""
        checkpoint_path = os.path.join(self.checkpoint, 'model_{}.pth.tar'.format(task_id))

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print(f"Available files in {self.checkpoint}:")
            if os.path.exists(self.checkpoint):
                files = os.listdir(self.checkpoint)
                for f in files:
                    print(f"  - {f}")
            else:
                print(f"  Directory {self.checkpoint} does not exist!")

            print("Creating new model instead...")
            model = self.network.Net(self.args)

            for i in range(task_id + 1):
                if i >= model.num_tasks:
                    n_classes = 2
                    if i < len(self.args.task_cls):
                        n_classes = self.args.task_cls[i][1]
                    model.add_task(n_classes=n_classes)

            model = model.to(self.args.device)
            return model

        model = self.network.Net(self.args)

        for i in range(task_id + 1):
            if i >= model.num_tasks:
                n_classes = 2
                if i < len(self.args.task_cls):
                    n_classes = self.args.task_cls[i][1]
                model.add_task(n_classes=n_classes)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f"Successfully loaded checkpoint from: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            model = model.to(self.args.device)
            return model

        model_state = model.state_dict()
        try:
            model.load_state_dict(model_state, strict=False)
        except Exception as e:
            print(f"Warning during load_state_dict: {e}")

        #  Load mask state
        if self.use_pruning and 'mask_state' in checkpoint:
            mask_state = checkpoint['mask_state']
            for param_name, mask_value in mask_state.items():
                if hasattr(model, param_name):
                    setattr(model, param_name, nn.Parameter(mask_value.to(self.device)))

        # Processing base model
        if hasattr(self.model, 'FeatureExtractor') and hasattr(self.model.FeatureExtractor, 'base_model'):
            try:
                current_base_model = deepcopy(self.model.FeatureExtractor.base_model.state_dict())
                model.FeatureExtractor.base_model.load_state_dict(current_base_model)
            except Exception as e:
                print(f"Warning: Could not copy base model state: {e}")

        model = model.to(self.args.device)

        return model


class MaskModule:
    """
    Mask-based pruning module following DER paper
    Uses learnable masks to dynamically prune network connections
    """

    def __init__(self, s_max=10.0, pruning_threshold=0.25):
        self.s_max = s_max  # Maximum scaling factor for mask
        self.pruning_threshold = pruning_threshold
        self.masks = {}  # Store masks for each layer
        self.mask_params = {}  # Learnable mask parameters

    def initialize_masks(self, model):
        """
        Initialize learnable masks for all task-specific layers
        """
        self.masks = {}
        self.mask_params = {}

        if hasattr(model, 'FeatureExtractor') and hasattr(model.FeatureExtractor, 'task_specific_layers'):
            for task_idx, task_layer in enumerate(model.FeatureExtractor.task_specific_layers):
                for name, module in task_layer.named_modules():
                    if isinstance(module, nn.Linear):
                        layer_name = f"task_{task_idx}_{name}"
                        mask_param = nn.Parameter(torch.randn_like(module.weight.data) * 2.0)
                        self.mask_params[layer_name] = mask_param

                        setattr(model, f"mask_{layer_name}", mask_param)

        # Initialize masks for classification heads
        if hasattr(model, 'heads'):
            for head_idx, head in enumerate(model.heads):
                for name, module in head.named_modules():
                    if isinstance(module, nn.Linear):
                        layer_name = f"head_{head_idx}_{name}"
                        mask_param = nn.Parameter(torch.randn_like(module.weight.data) * 2.0)
                        self.mask_params[layer_name] = mask_param

                        setattr(model, f"mask_{layer_name}", mask_param)

    def compute_masks(self, model):
        """
        Compute binary masks from learnable mask parameters using sigmoid
        """
        masks = {}

        # Compute masks for task-specific layers
        if hasattr(model, 'FeatureExtractor') and hasattr(model.FeatureExtractor, 'task_specific_layers'):
            for task_idx, task_layer in enumerate(model.FeatureExtractor.task_specific_layers):
                for name, module in task_layer.named_modules():
                    if isinstance(module, nn.Linear):
                        layer_name = f"task_{task_idx}_{name}"
                        if hasattr(model, f"mask_{layer_name}"):
                            mask_param = getattr(model, f"mask_{layer_name}")
                            # Compute mask using sigmoid with scaling
                            mask = torch.sigmoid(self.s_max * mask_param)
                            masks[layer_name] = mask

        # Compute masks for classification heads
        if hasattr(model, 'heads'):
            for head_idx, head in enumerate(model.heads):
                for name, module in head.named_modules():
                    if isinstance(module, nn.Linear):
                        layer_name = f"head_{head_idx}_{name}"
                        if hasattr(model, f"mask_{layer_name}"):
                            mask_param = getattr(model, f"mask_{layer_name}")
                            # Compute mask using sigmoid with scaling
                            mask = torch.sigmoid(self.s_max * mask_param)
                            masks[layer_name] = mask

        self.masks = masks
        return masks

    def apply_masks(self, model, pruning_strength=1.0):
        """Consider pruning intensity when applying masks"""
        self.compute_masks(model)

        # Apply masks to task-specific layers
        if hasattr(model, 'FeatureExtractor') and hasattr(model.FeatureExtractor, 'task_specific_layers'):
            for task_idx, task_layer in enumerate(model.FeatureExtractor.task_specific_layers):
                for name, module in task_layer.named_modules():
                    if isinstance(module, nn.Linear):
                        layer_name = f"task_{task_idx}_{name}"
                        if layer_name in self.masks:
                            # Save original weights
                            if not hasattr(module, 'original_weight'):
                                module.original_weight = module.weight.data.clone()

                            # Use original weights when applying mask
                            mask = self.masks[layer_name]
                            # Soft pruning: Adjust according to strength
                            effective_mask = 1.0 - pruning_strength * (1.0 - mask)
                            module.weight.data = module.original_weight * effective_mask

    def get_mask_regularization_loss(self, model):
        """
        Compute regularization loss to encourage sparsity in masks
        L_mask = Σ |mask_ij|
        """
        mask_loss = 0.0
        total_masks = 0

        for layer_name in self.mask_params:
            if hasattr(model, f"mask_{layer_name}"):
                mask_param = getattr(model, f"mask_{layer_name}")
                # L1 regularization on mask parameters
                mask_loss += torch.sum(torch.abs(mask_param))
                total_masks += mask_param.numel()

        # Normalize by number of mask parameters
        if total_masks > 0:
            mask_loss = mask_loss / total_masks

        return mask_loss

    def get_sparsity_statistics(self, model):
        """
        Compute sparsity statistics for current masks
        """
        self.compute_masks(model)

        total_params = 0
        pruned_params = 0

        for layer_name, mask in self.masks.items():
            total_params += mask.numel()
            # Count parameters with mask value < threshold as pruned
            pruned_params += (mask < self.pruning_threshold).sum().item()

        sparsity_ratio = pruned_params / max(total_params, 1) * 100
        return pruned_params, total_params, sparsity_ratio
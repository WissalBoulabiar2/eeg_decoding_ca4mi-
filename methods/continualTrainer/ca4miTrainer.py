import time
from copy import deepcopy
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from methods.continualTrainer import utils
from sklearn.metrics import f1_score
import wandb


class ca4mi_Trainer:
    """Trainer class containing training, evaluation and testing methods for CA4MI"""

    def __init__(self, ca4mi_instance):
        """Initialize trainer with reference to CA4MI instance"""
        self.ca4mi = ca4mi_instance

    def train_single_epoch(self, sub_id, dataset, prototypes=None):
        """Single epoch training function"""
        self.ca4mi.model.train()
        self.ca4mi.discriminator.train()
        self._validate_prototypes(prototypes)

        for data, target, sub_module_label, dis_label in dataset['train']:
            # Prepare batch data
            x, y, sub_module_label, t_real_D, t_fake_D = self._prepare_batch_data(
                data, target, sub_module_label, dis_label)

            # Train encoder based on mixup setting
            if self.ca4mi.mixup == 'yes':
                x_sub_module = self._train_encoder_with_mixup(
                    x, y, sub_module_label, sub_id, t_real_D, prototypes)
            else:
                x_sub_module = self._train_encoder_without_mixup(
                    x, y, sub_module_label, sub_id, t_real_D, prototypes)

            # Train discriminator
            self._train_discriminator(x, x_sub_module, sub_id, t_real_D, t_fake_D)

    def adversarial_training(self, sub_id, dataset, prototypes=None):
        """Main training loop function"""
        # Initialization
        self.ca4mi.discriminator = self.ca4mi.initialize_discriminator(sub_id)

        best_loss = np.inf
        best_model = deepcopy(self.ca4mi.model.state_dict())
        best_loss_d = np.inf
        best_model_d = deepcopy(self.ca4mi.discriminator.state_dict())

        dis_lr_update = True
        discriminator_lr = self.ca4mi.discriminator_lr[sub_id]
        patience_d = self.ca4mi.lr_patience
        self.ca4mi.discriminator_optimizer = self.ca4mi.setup_discriminator_optimizer(sub_id, discriminator_lr)

        encoder_lr = self.ca4mi.encoder_lr[sub_id]
        patience = self.ca4mi.lr_patience
        self.ca4mi.encoder_optimizer = self.ca4mi.setup_encoder_optimizer(sub_id, encoder_lr)

        # Training loop
        for e in range(self.ca4mi.nepochs):
            # Call single epoch training function
            self.train_single_epoch(sub_id, dataset, prototypes)

            train_res = self.eval_(dataset['train'], sub_id, prototypes)
            wandb.log({
                "epoch": e,
                "train/cls_acc": train_res['acc_cls'],
                "train/cls_loss": train_res['loss_cls'],
                "train/adv_loss": train_res.get('loss_adv', 0),
                "train/ort_loss": train_res.get('loss_ort', 0),
                "train/total_loss": train_res['loss_tot'],
            })

            random_chance, threshold = 20., 22.
            if train_res['acc_cls'] < threshold:
                discriminator_lr, encoder_lr = discriminator_lr / 10., encoder_lr / 10.
                self.ca4mi.discriminator_optimizer, self.ca4mi.encoder_optimizer = self.ca4mi.setup_discriminator_optimizer(
                    sub_id,
                    discriminator_lr), self.ca4mi.setup_encoder_optimizer(
                    sub_id, encoder_lr)
                self.ca4mi.discriminator = self.ca4mi.initialize_discriminator(sub_id)
                self.ca4mi.model = self.ca4mi.load_current_models(
                    sub_id - 1) if sub_id > 0 else self.ca4mi.network.Cls_HEAD(
                    self.ca4mi.args).to(self.ca4mi.args.device)

            # Validation and model selection
            valid_res = self.eval_(dataset['valid'], sub_id)
            wandb.log({
                "epoch": e,
                "train/cls_acc": valid_res['acc_cls'],
                "train/cls_loss": valid_res['loss_cls'],
                "train/adv_loss": valid_res.get('loss_adv', 0),
                "train/ort_loss": valid_res.get('loss_ort', 0),
                "train/total_loss": valid_res['loss_tot'],
            })

            # Update best encoder model
            if valid_res['loss_tot'] < best_loss:
                best_loss, best_model, patience = valid_res['loss_tot'], deepcopy(
                    self.ca4mi.model.state_dict()), self.ca4mi.lr_patience
            else:
                patience -= 1
                if patience <= 0:
                    encoder_lr /= self.ca4mi.lr_factor
                    if encoder_lr < self.ca4mi.lr_min:
                        print()
                        break
                    patience, self.ca4mi.encoder_optimizer = self.ca4mi.lr_patience, self.ca4mi.setup_encoder_optimizer(
                        sub_id,
                        encoder_lr)
                    wandb.log({
                        "lr/encoder": encoder_lr,
                        "lr/discriminator": discriminator_lr,
                        "epoch": e
                    })

            # Update best discriminator model
            if train_res['loss_adv'] < best_loss_d:
                best_loss_d, best_model_d, patience_d = train_res['loss_adv'], deepcopy(
                    self.ca4mi.discriminator.state_dict()), self.ca4mi.lr_patience
            else:
                patience_d -= 1
                if patience_d <= 0 and dis_lr_update:
                    discriminator_lr /= self.ca4mi.lr_factor
                    if discriminator_lr < self.ca4mi.lr_min:
                        dis_lr_update = False
                    patience_d, self.ca4mi.discriminator_optimizer = self.ca4mi.lr_patience, self.ca4mi.setup_discriminator_optimizer(
                        sub_id, discriminator_lr)

        # Load best models and save
        self.ca4mi.model.load_state_dict(deepcopy(best_model))
        self.ca4mi.discriminator.load_state_dict(deepcopy(best_model_d))
        utils.save_models(self.ca4mi.model, self.ca4mi.discriminator, self.ca4mi.checkpoint, sub_id)
        wandb.finish()

    def eval_(self, data_loader, sub_id, prototypes=None):
        """Evaluate model on given data loader"""
        # Initialize accumulators
        loss_a, loss_c, loss_o, loss_total = 0, 0, 0, 0
        num_correct_domain, num_correct_class = 0, 0
        num = 0
        batch = 0

        # Set models to evaluation mode
        self.ca4mi.model.eval()
        self.ca4mi.discriminator.eval()

        with torch.no_grad():
            for batch, (data, target, sub_module_label, dis_label) in enumerate(data_loader):
                # Evaluate single batch
                batch_results = self._eval_single_batch(
                    data, target, sub_module_label, dis_label, sub_id, prototypes)

                # Accumulate results
                loss_c += batch_results['cls_loss']
                loss_a += batch_results['adv_loss']
                loss_o += batch_results['ort_loss']
                loss_total += batch_results['total_loss']
                num_correct_class += batch_results['num_correct_class']
                num_correct_domain += batch_results['num_correct_domain']
                num += batch_results['batch_size']

        # Compute final metrics
        num_batches = batch + 1
        return {
            'loss_cls': loss_c.item() / num_batches,
            'acc_cls': 100 * num_correct_class / num,
            'loss_adv': loss_a.item() / num_batches,
            'acc_dis': 100 * num_correct_domain / num,
            'loss_ort': loss_o.item() / num_batches,
            'loss_tot': loss_total.item() / num_batches,
            'size': utils.loader_size(data_loader=data_loader)
        }

    def test_previous_subjects(self, data_loader, sub_id, model, prototypes=None):
        """Test model performance on previous subjects"""
        # Set models to evaluation mode
        model.eval()
        self.ca4mi.discriminator.eval()

        # Initialize all metrics
        metrics = self._initialize_test_metrics()

        with torch.no_grad():
            for batch_idx, (data, target, sub_module_label, dis_label) in enumerate(data_loader):
                # Process current batch and update metrics
                self._process_test_batch(
                    data, target, sub_module_label, dis_label,
                    sub_id, model, prototypes, metrics
                )
                metrics['batch_count'] = batch_idx

        # Compute and return final results
        return self._compute_final_test_results(metrics, data_loader)

    # Helper methods for training
    def _validate_prototypes(self, prototypes):
        """Validate prototypes format"""
        if prototypes is not None and not (
                isinstance(prototypes, list) and all(
            isinstance(p, torch.Tensor) and p.dim() == 2 for p in prototypes)
        ):
            raise ValueError("Prototypes must be a list of 2D tensors")

    def _prepare_batch_data(self, data, target, sub_module_label, dis_label):
        """Prepare batch data"""
        data = data[:, np.newaxis, :, :].to(device=self.ca4mi.device, dtype=torch.float32)
        target = target.to(device=self.ca4mi.device, dtype=torch.long)
        sub_module_label = sub_module_label.to(device=self.ca4mi.device)

        x, y = data, target
        t_real_D = dis_label.to(self.ca4mi.device)
        t_fake_D = torch.zeros_like(dis_label).to(self.ca4mi.device)

        return x, y, sub_module_label, t_real_D, t_fake_D

    def _train_encoder_with_mixup(self, x, y, sub_module_label, sub_id, t_real_D, prototypes):
        """Train encoder with mixup"""
        alpha = self.ca4mi.alpha
        data, targets_a, targets_b, lam = self.ca4mi.mixup_data(x, y, alpha)
        x_sub_module = self.ca4mi.assign_subject_specific_mask(data, sub_module_label, sub_id)

        for step in range(self.ca4mi.encoder_step):
            self.ca4mi.encoder_optimizer.zero_grad()
            self.ca4mi.model.zero_grad()

            output = self.ca4mi.model(data, x_sub_module, sub_module_label, sub_id)
            cls_loss = self.ca4mi.mixup_criterion(self.ca4mi.criterion, output, targets_a, targets_b, lam)

            shared_encoded, private_encoded = self.ca4mi.model.get_encoded(data, x_sub_module, sub_id)
            adv_loss = self.ca4mi.adv_loss_fn(
                self.ca4mi.discriminator(shared_encoded, t_real_D, sub_id), t_real_D)

            ort_loss = (self.ca4mi.ort_loss_fn(shared_encoded, private_encoded)
                        if self.ca4mi.orth == 'yes' else torch.tensor(0).to(self.ca4mi.device))

            if self.ca4mi.use_prototypes == 'yes' and (prototypes is not None):
                pro_loss_a = self.ca4mi.compute_prototype_loss(shared_encoded, targets_a, prototypes)
                pro_loss_b = self.ca4mi.compute_prototype_loss(shared_encoded, targets_b, prototypes)
                pro_loss = lam * pro_loss_a + (1 - lam) * pro_loss_b
            else:
                pro_loss = torch.tensor(0., device=self.ca4mi.device)

            total_loss = (cls_loss + self.ca4mi.adv_loss_reg * adv_loss +
                          self.ca4mi.ort_loss_reg * ort_loss +
                          (self.ca4mi.pro_loss_reg * pro_loss if self.ca4mi.use_prototypes == 'yes' else 0))

            total_loss.backward(retain_graph=True)
            self.ca4mi.encoder_optimizer.step()

        return x_sub_module

    def _train_encoder_without_mixup(self, x, y, sub_module_label, sub_id, t_real_D, prototypes):
        """Train encoder without mixup"""
        x_sub_module = self.ca4mi.assign_subject_specific_mask(x, sub_module_label, sub_id)

        for step in range(self.ca4mi.encoder_step):
            self.ca4mi.encoder_optimizer.zero_grad()
            self.ca4mi.model.zero_grad()

            output = self.ca4mi.model(x, x_sub_module, sub_module_label, sub_id)
            cls_loss = self.ca4mi.criterion(output, y)

            shared_encoded, private_encoded = self.ca4mi.model.get_encoded(x, x_sub_module, sub_id)
            adv_loss = self.ca4mi.adv_loss_fn(
                self.ca4mi.discriminator(shared_encoded, t_real_D, sub_id), t_real_D)

            ort_loss = (self.ca4mi.ort_loss_fn(shared_encoded, private_encoded)
                        if self.ca4mi.orth == 'yes' else torch.tensor(0).to(self.ca4mi.device))

            pro_loss = (self.ca4mi.compute_prototype_loss(private_encoded, y, prototypes)
                        if prototypes else torch.tensor(0).to(self.ca4mi.device))

            total_loss = (cls_loss + self.ca4mi.adv_loss_reg * adv_loss +
                          self.ca4mi.ort_loss_reg * ort_loss +
                          (self.ca4mi.pro_loss_reg * pro_loss if self.ca4mi.use_prototypes == 'yes' else 0))

            total_loss.backward(retain_graph=True)
            self.ca4mi.encoder_optimizer.step()

        return x_sub_module


    def _train_discriminator(self, x, x_sub_module, sub_id, t_real_D, t_fake_D):
        """Train the discriminator to distinguish real subject embeddings from fake ones."""
        for step in range(self.ca4mi.discriminator_step):
            # Reset gradients
            self.ca4mi.discriminator_optimizer.zero_grad()
            self.ca4mi.discriminator.zero_grad()

            # ----- Real encoded features -----
            # Get the shared encoder output and detach it from the computational graph
            shared_encoded = self.ca4mi.model.get_encoded(x, x_sub_module, sub_id)[0].detach()

            # Compute discriminator loss for real latent vectors
            dis_real_loss = self.ca4mi.adv_loss_fn(
                self.ca4mi.discriminator(shared_encoded, t_real_D, sub_id), t_real_D
            )
            dis_real_loss.backward(retain_graph=True)

            # ----- Fake latent features -----
            # Generate fake latent vectors from a Gaussian distribution
            z_fake = torch.randn((x.size(0), self.ca4mi.latent_dim), dtype=torch.float32,
                                 device=self.ca4mi.device) * self.ca4mi.sigma + self.ca4mi.mu

            # Compute discriminator loss for fake samples
            dis_fake_loss = self.ca4mi.adv_loss_fn(
                self.ca4mi.discriminator(z_fake, t_real_D, sub_id), t_fake_D
            )
            dis_fake_loss.backward(retain_graph=True)

            # Update discriminator parameters
            self.ca4mi.discriminator_optimizer.step()

    # Helper methods for evaluation
    def _eval_single_batch(self, data, target, sub_module_label, dis_label, sub_id, prototypes=None):
        """Evaluate single batch and return losses and predictions"""
        # Prepare data
        data = data[:, np.newaxis, :, :].to(device=self.ca4mi.device, dtype=torch.float32)
        target = target.to(device=self.ca4mi.device, dtype=torch.long)
        sub_module_label = sub_module_label.to(device=self.ca4mi.device)

        x = data
        y = target
        t_real_D = dis_label.to(self.ca4mi.device)

        # Forward pass through model
        output = self.ca4mi.model(x, x, sub_module_label, sub_id)
        shared_out, private_out = self.ca4mi.model.get_encoded(x, x, sub_id)

        # predictions
        _, pred = output.max(1)
        num_correct_class = pred.eq(y.view_as(pred)).sum().item()

        # Discriminator predictions
        output_d = self.ca4mi.discriminator.forward(shared_out, t_real_D, sub_id)
        _, pred_d = output_d.max(1)
        num_correct_domain = pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

        # Compute losses
        cls_loss = self.ca4mi.cls_loss_fn(output, y)
        adv_loss = self.ca4mi.adv_loss_fn(output_d, t_real_D)

        if self.ca4mi.orth == 'yes':
            ort_loss = self.ca4mi.ort_loss_fn(shared_out, private_out)
        else:
            ort_loss = torch.tensor(0).to(device=self.ca4mi.device, dtype=torch.float32)
            self.ca4mi.ort_loss_reg = 0

        pro_loss = self.ca4mi.compute_prototype_loss(shared_out, y, prototypes)

        total_loss = (cls_loss + self.ca4mi.adv_loss_reg * adv_loss +
                      self.ca4mi.ort_loss_reg * ort_loss +
                      (self.ca4mi.pro_loss_reg * pro_loss if self.ca4mi.use_prototypes == 'yes' else 0))

        batch_size = x.size(0)

        return {
            'cls_loss': cls_loss,
            'adv_loss': adv_loss,
            'ort_loss': ort_loss,
            'total_loss': total_loss,
            'num_correct_class': num_correct_class,
            'num_correct_domain': num_correct_domain,
            'batch_size': batch_size
        }

    # Helper methods for testing
    def _initialize_test_metrics(self):
        """Initialize all metrics and accumulators for testing"""
        return {
            'losses': {'cls': 0, 'adv': 0, 'ort': 0, 'total': 0},
            'correct': {'class': 0, 'domain': 0},
            'predictions': {'targets': [], 'preds': []},
            'timing': {'total_time': 0.0, 'num_samples': 0},
            'batch_count': 0,
            'sample_count': 0
        }

    def _process_test_batch(self, data, target, sub_module_label, dis_label, sub_id, model, prototypes, metrics):
        """Process a single batch during testing and update metrics"""
        # Data preparation
        data = data[:, np.newaxis, :, :].to(device=self.ca4mi.device, dtype=torch.float32)
        target = target.to(device=self.ca4mi.device, dtype=torch.long)
        sub_module_label = sub_module_label.to(device=self.ca4mi.device)

        x, y = data, target
        t_real_D = dis_label.to(self.ca4mi.device)

        # Time the inference
        start_time = time.time()

        # Forward pass through model
        output = model.forward(x, x, sub_module_label, sub_id)
        shared_out, private_out = model.get_encoded(x, x, sub_id)

        # predictions
        _, pred = output.max(1)
        min_size = min(len(pred), len(y))
        pred_truncated = pred[:min_size]
        y_truncated = y[:min_size]

        # Update accuracy
        metrics['correct']['class'] += pred_truncated.eq(y_truncated).sum().item()
        metrics['predictions']['targets'].extend(y_truncated.cpu().tolist())
        metrics['predictions']['preds'].extend(pred_truncated.cpu().tolist())

        # Discriminator predictions
        output_d = self.ca4mi.discriminator.forward(shared_out, sub_module_label, sub_id)
        _, pred_d = output_d.max(1)
        metrics['correct']['domain'] += pred_d.eq(t_real_D.view_as(pred_d)).sum().item()

        # Compute all losses
        cls_loss = self.ca4mi.cls_loss_fn(output, y)
        adv_loss = self.ca4mi.adv_loss_fn(output_d, t_real_D)

        ort_loss = (self.ca4mi.ort_loss_fn(shared_out, private_out) if self.ca4mi.orth == 'yes'
                    else torch.tensor(0).to(device=self.ca4mi.device, dtype=torch.float32))
        if self.ca4mi.orth != 'yes':
            self.ca4mi.ort_loss_reg = 0

        pro_loss = self.ca4mi.compute_prototype_loss(shared_out, y, prototypes)

        total_loss = (cls_loss + self.ca4mi.adv_loss_reg * adv_loss + self.ca4mi.ort_loss_reg * ort_loss +
                      (self.ca4mi.pro_loss_reg * pro_loss if self.ca4mi.use_prototypes == 'yes' else 0))

        # Update loss accumulators
        metrics['losses']['cls'] += cls_loss
        metrics['losses']['adv'] += adv_loss
        metrics['losses']['ort'] += ort_loss
        metrics['losses']['total'] += total_loss

        # Update timing and sample counts
        end_time = time.time()
        metrics['timing']['total_time'] += (end_time - start_time)
        metrics['timing']['num_samples'] += x.shape[0]
        metrics['sample_count'] += x.size(0)


    def _compute_final_test_results(self, metrics, data_loader):
        """Compute final test results from accumulated metrics"""
        num_batches = metrics['batch_count'] + 1
        num_samples = metrics['sample_count']

        return {
            'loss_cls': metrics['losses']['cls'].item() / num_batches,
            'acc_cls': 100 * metrics['correct']['class'] / num_samples,
            'loss_adv': metrics['losses']['adv'].item() / num_batches,
            'acc_dis': 100 * metrics['correct']['domain'] / num_samples,
            'loss_ort': metrics['losses']['ort'].item() / num_batches,
            'loss_tot': metrics['losses']['total'].item() / num_batches,
            'size': utils.loader_size(data_loader=data_loader),
            'f1_macro': f1_score(metrics['predictions']['targets'],
                                 metrics['predictions']['preds'], average='macro') * 100.0,
            'avg_inf_time': metrics['timing']['total_time'] / metrics['timing']['num_samples']
        }

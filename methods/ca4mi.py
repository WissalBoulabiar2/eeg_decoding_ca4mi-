import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import fractional_matrix_power
from methods.continualTrainer.discriminator import Discriminator, OrthLoss
from methods.continualTrainer.ca4miTrainer import ca4mi_Trainer



sys.path.append('../')


class CA4MI:
    def __init__(self, model, args, network):
        self.model = model
        self.args = args
        self.network = network
        self.checkpoint = args.checkpoint
        self.device = args.device
        self.nepochs = args.n_epochs

        # Core settings
        self.use_prototypes = args.use_prototypes
        self.alpha = args.alpha
        self.n_samples = args.n_samples
        self.mixup = args.mixup
        self.align = args.align
        self.orth = args.orth
        self.latent_dim = args.latent_dim
        self.mu = 0.0
        self.sigma = 1.0

        # Optimization
        self.encoder_lr = [args.encoder_lr] * args.n_subjects
        self.discriminator_lr = [args.discriminator_lr] * args.n_subjects
        self.encoder_step = args.shared_step
        self.discriminator_step = args.discriminator_step
        self.lr_patience = args.lr_patience
        self.lr_factor = args.lr_factor
        self.lr_min = args.lr_min

        # Losses
        self.cls_loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.adv_loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.ort_loss_fn = OrthLoss().to(self.device)
        self.pro_loss_reg = args.pro_loss_reg
        self.adv_loss_reg = args.adv
        self.ort_loss_reg = args.orth_reg
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # Memory
        self.subject_memory = {i: {'x': [], 'y': [], 'sub_module_label': [], 'dis_label': []} for i in range(args.n_subjects)}

        # Trainer wrapper
        self.trainer = ca4mi_Trainer(self)
        self.proto_mu = None
        self.feature_mappers = nn.ModuleDict()

        # ------------------ Prototypes ------------------ #
    def get_prototype_samples(self, loader, net, sub_id):
        net.eval()
        all_fea = []
        all_output = []
        all_label = []

        with torch.no_grad():
            for batch, (data, target, sub_module_label, dis_label) in enumerate(loader):
                data = data[:, np.newaxis, :, :]
                x = data.to(self.device)
                y = target.to(self.device, dtype=torch.long)
                sub_module_label = sub_module_label.to(self.device)

                outputs = net(x, x, sub_module_label, sub_id)
                shared_out, private_out = net.get_encoded(x, x, sub_id)

                all_fea.append(shared_out)
                all_output.append(outputs)
                all_label.append(y)

        all_fea = torch.cat(all_fea).to(self.device)
        all_output = torch.cat(all_output).to(self.device)
        all_label = torch.cat(all_label).to(self.device)

        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(predict == all_label).item() / float(all_label.size(0))

        class_prototypes = []
        for i in range(all_output.size(1)):
            idx = (all_label == i).nonzero(as_tuple=True)[0]
            class_fea = all_fea[idx]
            class_proto = class_fea.mean(dim=0)
            class_prototypes.append(class_proto)

        class_prototypes = torch.stack(class_prototypes).to(self.device)
        print('Memory updated by adding {} prototype n_samples'.format(len(class_prototypes)))

        return class_prototypes, accuracy


    def reservoir_sampling(self, proto_list, max_proto):
        all_proto = torch.cat(proto_list, dim=0)
        if all_proto.size(0) <= max_proto:
            return [all_proto]

        reservoir = all_proto[:max_proto].clone()
        for i in range(max_proto, all_proto.size(0)):
            j = random.randint(0, i)
            if j < max_proto:
                reservoir[j] = all_proto[i]
        return [reservoir]
        
    def get_or_create_mapper(self, current_dim, proto_dim):
        key = f"{current_dim}_{proto_dim}"
        if key not in self.feature_mappers:
            self.feature_mappers[key] = nn.Linear(current_dim, proto_dim).to(self.device)
        return self.feature_mappers[key]

    def update_prototypes_ema(self, mu_hat: torch.Tensor, momentum: float = 0.10) -> torch.Tensor:

        if mu_hat is None:
            return self.proto_mu if hasattr(self, 'proto_mu') else None

        device = mu_hat.device
        C, D = mu_hat.shape

        if not hasattr(self, 'proto_mu') or self.proto_mu is None:
            self.proto_mu = mu_hat.clone().to(device)
            self.proto_init = True
            print(f"[EMA] Initialized proto_mu with shape {self.proto_mu.shape}")
            return self.proto_mu

        if self.proto_mu.shape != mu_hat.shape:
            print(f"[EMA] Dimension mismatch: {self.proto_mu.shape} vs {mu_hat.shape}")

            if self.proto_mu.shape[1] != D:
                print("[EMA] Feature dimension mismatch - reinitializing")
                self.proto_mu = mu_hat.clone().to(device)
                return self.proto_mu

            if self.proto_mu.shape[0] != C:
                old_C = self.proto_mu.shape[0]
                new_proto = torch.zeros(C, D, device=device)

                if C > old_C:
                    new_proto[:old_C] = self.proto_mu
                    new_proto[old_C:] = mu_hat[old_C:]
                    print(f"[EMA] Expanded from {old_C} to {C} classes")
                else:
                    new_proto = self.proto_mu[:C]
                    print(f"[EMA] Reduced from {old_C} to {C} classes")

                self.proto_mu = new_proto

        eps = 1e-8
        nz = (mu_hat.norm(dim=1) > eps)

        if nz.any():
            self.proto_mu[nz] = (1.0 - momentum) * self.proto_mu[nz] + momentum * mu_hat[nz]

            if hasattr(self, 'normalize_prototypes') and self.normalize_prototypes:
                norms = self.proto_mu[nz].norm(dim=1, keepdim=True)
                self.proto_mu[nz] = self.proto_mu[nz] / (norms + 1e-12)

        return self.proto_mu

    def compute_prototype_loss(self, features, targets, prototypes):
        pro_loss = torch.tensor(0).to(self.device, dtype=torch.float32)
        if prototypes is not None:
            for i in range(len(targets)):
                target = targets[i].item()
                if target < len(prototypes):
                    pro_loss += torch.norm(features[i] - prototypes[target])
            pro_loss /= len(targets)
        return pro_loss


    # Main training entry
    def adversarial_training(self, sub_id, dataset, prototypes=None):
        return self.trainer.adversarial_training(sub_id, dataset, prototypes)

    def eval_(self, data_loader, sub_id, prototypes=None):
        return self.trainer.eval_(data_loader, sub_id, prototypes)

    def test_previous_subjects(self, data_loader, sub_id, model, prototypes=None):
        return self.trainer.test_previous_subjects(data_loader, sub_id, model, prototypes)

    # ------------------ Utils ------------------ #
    def IEA(self, x, prev_EA=None, gamma=0.9):
        covs = np.array([np.cov(xi) for xi in x])
        curr_EA = np.mean(covs, axis=0)
        ref_EA = gamma * prev_EA + (1 - gamma) * curr_EA if prev_EA is not None else curr_EA
        sqrt_ref = fractional_matrix_power(ref_EA, -0.5)
        aligned = np.array([sqrt_ref @ xi for xi in x])
        return aligned, ref_EA

    def mixup_data(self, x, y, alpha):
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        index = torch.randperm(x.size(0)).to(x.device)
        index = torch.randperm(x.size(0)).to(x.device)
        return lam * x + (1 - lam) * x[index], y, y[index], lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def assign_subject_specific_mask(self, data, sub_labels, sub_id):
        mask = (sub_id * torch.ones_like(sub_labels) == sub_labels).to(self.device)
        result = data.clone()
        for i in range(data.size(0)):
            if not mask[i]:
                result[i] = result[i].detach()
        return result.to(self.device)

    def setup_encoder_optimizer(self, sub_id, lr=None):
        lr = lr or self.encoder_lrs[sub_id]
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.args.encoder_wd)

    def setup_discriminator_optimizer(self, sub_id, lr=None):
        lr = lr or self.discriminator_lr[sub_id]
        return torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=self.args.discriminator_wd)

    def initialize_discriminator(self, sub_id):
        return Discriminator(self.args, sub_id).to(self.device)

    def load_current_models(self, sub_id):
        return self.network.Cls_HEAD(self.args).to(self.device)

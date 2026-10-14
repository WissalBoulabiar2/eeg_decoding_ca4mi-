"""
CA4MI-MP: CA4MI with a multi-prototype memory.

Everything (IEA, shared/private encoders, adversarial alignment, orthogonality) is inherited
from CA4MI unchanged. Only the prototype part is replaced:
  Phase 1  single prototype per class   -> n_modes anchors per class (clustered, not averaged)
  Phase 2  random reservoir             -> class-balanced, diversity-aware memory (Ward merging)
  Phase 3  fixed prototype loss         -> similarity-adaptive prototype loss
  Phase 4  Euclidean distance to mu     -> Mahalanobis distance with (mu, diag Sigma)
  Option   pull only                    -> pull to own class + push from other classes (contrastive)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from methods.ca4mi import CA4MI
from methods.prototype_memory import PrototypeMemory, PrototypeAnchors
from methods.continualTrainer.ca4miTrainer import ca4mi_Trainer


def enabled(args, key, default='no'):
    """yes/no switch; `--set key=yes` is parsed by OmegaConf as the boolean True."""
    return args.get(key, default) in ('yes', True)


class CA4MI_MP(CA4MI):
    def __init__(self, model, args, network):
        super().__init__(model, args, network)

        # Phase 1 - multi-prototype
        self.n_modes = getattr(args, 'n_modes', 3)
        self.modes_per_subject = getattr(args, 'modes_per_subject', 1)
        # Phase 3 - adaptive consistency
        self.adaptive_proto = getattr(args, 'adaptive_proto', 'no') == 'yes'
        self.adaptive_tau = getattr(args, 'adaptive_tau', 1.0)
        self.adaptive_min_weight = getattr(args, 'adaptive_min_weight', 0.1)
        # Phase 4 - uncertainty
        self.proto_distance = getattr(args, 'proto_distance', 'euclidean')
        self.var_floor_ratio = getattr(args, 'var_floor_ratio', 0.1)
        # Option - contrastive
        self.proto_contrastive_reg = getattr(args, 'proto_contrastive_reg', 0.0)
        self.proto_temperature = getattr(args, 'proto_temperature', 1.0)

        # Phase 2 - memory
        self.memory = PrototypeMemory(
            n_classes=args.n_classes,
            max_prototypes=args.max_prototypes,
            selection=getattr(args, 'memory_selection', 'diversity'),
            class_balanced=getattr(args, 'class_balanced_memory', 'yes') == 'yes',
            seed=args.seed,
        )
        self.trainer = ca4mi_mp_Trainer(self)
        self.diagnostics = None  # set by EEGTrainerMP when diagnostics='yes'
        # Replay logit consistency (Dark Experience Replay, Buzzega et al. 2020)
        self.replay_distill = enabled(args, 'replay_distill', 'no')
        self.replay_distill_reg = float(args.get('replay_distill_reg', 0.5))
        self.replay_bank = None  # (x, sub_module_label, logits) set by EEGTrainerMP before each subject

    # ------------------------------------------------------------ memory
    def extract_shared_features(self, loader, net, sub_id):
        """Shared features of the current subject only (replayed samples of past subjects are skipped)."""
        net.eval()
        feats, labels = [], []
        with torch.no_grad():
            for data, target, sub_module_label, _ in loader:
                x = data[:, np.newaxis, :, :].to(self.device, dtype=torch.float32)
                keep = sub_module_label.to(self.device) == sub_id
                shared_out, _ = net.get_encoded(x, x, sub_id)
                feats.append(shared_out[keep])
                labels.append(target.to(self.device, dtype=torch.long)[keep])
        return torch.cat(feats), torch.cat(labels)

    def update_memory(self, loader, net, sub_id):
        feats, labels = self.extract_shared_features(loader, net, sub_id)
        self.memory.add_subject(feats, labels, sub_id, modes_per_subject=self.modes_per_subject)
        print(f"[Memory] {self.memory.summary()}")

    def get_anchors(self):
        anchors = self.memory.anchors(self.n_modes)
        if anchors is not None:
            print(f"[Anchors] {len(anchors)} anchors | per class {anchors.summary()}")
        return anchors

    # -------------------------------------------------------------- loss
    def _distances(self, features, anchors):
        """Distance of every feature to every anchor: [B, M]."""
        diff = features.unsqueeze(1) - anchors.mu.unsqueeze(0)
        if self.proto_distance == 'mahalanobis':
            var = anchors.var.clamp_min(self.var_floor_ratio * anchors.var.mean().clamp_min(1e-8))
            d2 = (diff.pow(2) / var.unsqueeze(0)).sum(-1)
        elif self.proto_distance == 'euclidean':
            d2 = diff.pow(2).sum(-1)
        else:
            raise ValueError(f"Unknown proto_distance: {self.proto_distance}")
        return (d2 + 1e-12).sqrt()

    def _adaptive_weights(self, d_pos, nearest, anchors):
        """Strong constraint close to a known mode, weaker constraint for a new/different subject."""
        if self.proto_distance == 'mahalanobis':
            spread = torch.full_like(d_pos, float(anchors.mu.size(1)) ** 0.5)
        else:
            spread = anchors.var.sum(-1).clamp_min(1e-8).sqrt()[nearest]
        tau = self.adaptive_tau * spread
        w = torch.exp(-d_pos.pow(2) / (2 * tau.pow(2)))
        return (self.adaptive_min_weight + (1 - self.adaptive_min_weight) * w).detach()

    def compute_prototype_loss(self, features, targets, prototypes):
        """
        Returns  L_pull + (proto_contrastive_reg / pro_loss_reg) * L_con,
        so that  pro_loss_reg * returned = pro_loss_reg * L_pull + proto_contrastive_reg * L_con
        with the loss composition of the original trainer.
        """
        zero = torch.tensor(0., device=self.device)
        if prototypes is None or len(prototypes) == 0:
            return zero

        d = self._distances(features, prototypes)                      # [B, M]
        pos = targets.unsqueeze(1) == prototypes.labels.unsqueeze(0)   # [B, M]
        valid = pos.any(1)
        if not valid.any():
            return zero

        # Pull towards the nearest mode of the correct class
        d_pos_all = d.masked_fill(~pos, float('inf'))
        d_pos, nearest = d_pos_all.min(1)
        d_pos, nearest = d_pos[valid], nearest[valid]
        w = self._adaptive_weights(d_pos, nearest, prototypes) if self.adaptive_proto else torch.ones_like(d_pos)
        pull = (w * d_pos).mean()

        if self.proto_contrastive_reg <= 0:
            return pull

        # Push away from the modes of the other classes (multi-positive prototype InfoNCE)
        has_neg = (~pos).any(1) & valid
        if not has_neg.any():
            return pull
        logits = -d[has_neg] / self.proto_temperature
        p = pos[has_neg]
        con = -(torch.logsumexp(logits.masked_fill(~p, float('-inf')), 1) - torch.logsumexp(logits, 1)).mean()
        scale = self.proto_contrastive_reg / self.pro_loss_reg if self.pro_loss_reg > 0 else 1.0
        return pull + scale * con


class ca4mi_mp_Trainer(ca4mi_Trainer):
    """Same training loop as CA4MI; prototypes are PrototypeAnchors and always use shared features."""

    def _validate_prototypes(self, prototypes):
        if prototypes is not None and not isinstance(prototypes, PrototypeAnchors):
            raise ValueError("Prototypes must be a PrototypeAnchors instance")

    def adversarial_training(self, sub_id, dataset, prototypes=None):
        self._epochs_done, self._valid_loader, self._valid_hist, self._in_diag = 0, dataset['valid'], [], False
        super().adversarial_training(sub_id, dataset, prototypes)
        if self._valid_hist:
            best = int(np.argmin(self._valid_hist))
            print(f"  [Selection] {len(self._valid_hist)} epochs run, selected epoch {best + 1} "
                  f"(valid loss {self._valid_hist[best]:.3f}, min_select_epoch="
                  f"{self.ca4mi.args.get('min_select_epoch', 0)})")

    def eval_(self, data_loader, sub_id, prototypes=None):
        res = super().eval_(data_loader, sub_id, prototypes)
        # model_selection='cls': the trainer keeps the epoch with the lowest valid loss_tot; loss_tot includes
        # +adv_loss (discriminator CE), so lower means *more* subject-separable shared features. Select on the
        # classification loss instead. loss_tot is only used for selection and logging.
        if self.ca4mi.args.get('model_selection', 'cls') == 'cls':
            res['loss_tot'] = res['loss_cls']
        if data_loader is getattr(self, '_valid_loader', None):
            # min_select_epoch=N: no checkpoint before N training epochs. With 22 validation trials the valid loss
            # is noisy and its minimum can fall on epoch 1-2, keeping an almost untrained model for the subject.
            # The forced value decreases every epoch, so the trainer's patience / LR schedule is unaffected.
            forced = self._epochs_done < int(self.ca4mi.args.get('min_select_epoch', 0))
            if not self._in_diag:
                self._valid_hist.append(float('inf') if forced else res['loss_tot'])
            if forced:
                res['loss_tot'] = 1e9 - self._epochs_done
        return res

    def train_single_epoch(self, sub_id, dataset, prototypes=None):
        super().train_single_epoch(sub_id, dataset, prototypes)
        if self.ca4mi.replay_distill and self.ca4mi.replay_bank is not None:
            self._replay_distill_epoch(sub_id)
        self._epochs_done += 1
        # Diagnostics (main/diagnostics_mp.py), only with diagnostics='yes'
        if self.ca4mi.diagnostics is not None:
            self._in_diag = True
            try:
                self.ca4mi.diagnostics.on_epoch_end(self.ca4mi, sub_id, dataset, prototypes)
            finally:
                self._in_diag = False

    def _replay_distill_epoch(self, sub_id):
        """One pass over the replay buffer: keep the logits of replayed trials close to the logits stored when they
        entered the buffer (MSE). With replay_private, private_u and head_u are frozen, so this anchors the shared
        features of past subjects where their own (frozen) head expects them."""
        x, sub, logits = self.ca4mi.replay_bank
        model = self.ca4mi.model
        model.train()
        bs = self.ca4mi.args.batch_size
        order = torch.randperm(len(x), device=x.device)
        for i in range(0, len(x), bs):
            idx = order[i:i + bs]
            xb = x[idx][:, None].float()
            self.ca4mi.encoder_optimizer.zero_grad()
            out = model(xb, xb, sub[idx], sub_id)
            loss = self.ca4mi.replay_distill_reg * torch.nn.functional.mse_loss(out, logits[idx])
            loss.backward()
            self.ca4mi.encoder_optimizer.step()

    def _train_encoder_without_mixup(self, x, y, sub_module_label, sub_id, t_real_D, prototypes):
        """Train encoder without mixup (prototype loss on the shared space, as with mixup)"""
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

            pro_loss = self.ca4mi.compute_prototype_loss(shared_encoded, y, prototypes)

            total_loss = (cls_loss + self.ca4mi.adv_loss_reg * adv_loss +
                          self.ca4mi.ort_loss_reg * ort_loss +
                          (self.ca4mi.pro_loss_reg * pro_loss if self.ca4mi.use_prototypes == 'yes' else 0))

            total_loss.backward(retain_graph=True)
            self.ca4mi.encoder_optimizer.step()

        return x_sub_module

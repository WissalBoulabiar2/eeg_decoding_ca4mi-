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

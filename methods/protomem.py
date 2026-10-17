"""
ProtoMem: subject-incremental MI-EEG decoding with a Gaussian prototype memory used as a feature-level generator.

Per subject t:
  L = CE(f(x), y) on current + replayed trials
    + distill_reg * MSE(logits of replayed trials, logits stored when they entered the buffer)   (DER, Buzzega 2020)
    + gen_reg     * CE(classifier(z~N(mu_m, Sigma_m)), class_m)  on pseudo-features from the memory modes
                                                                  (prototype augmentation, cf. PASS, Zhu 2021)
After training on subject t:
  1. drift compensation: the modes of every past (subject, class) are shifted by the mean change of the embeddings
     of that subject's stored exemplars (semantic drift compensation, Yu et al. 2020);
  2. the embeddings of subject t's training trials are summarised into Gaussian modes and added to the memory
     (class-balanced, Ward consolidation once max_prototypes is reached: methods/prototype_memory.py);
  3. n_samples class-balanced exemplars of subject t are stored (trial, label, subject, logits, embedding).
Subject t is evaluated with its own (frozen) BatchNorm and the shared encoder / classifier.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from methods.prototype_memory import PrototypeMemory
from networks.base_protomem import ProtoMemNet


def enabled(args, key, default='no'):
    """yes/no switch; `--set key=yes` is parsed by OmegaConf as the boolean True."""
    return args.get(key, default) in ('yes', True)


class ProtoMem:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.net = ProtoMemNet(args).to(self.device)
        self.memory = PrototypeMemory(
            n_classes=args.n_classes, max_prototypes=args.max_prototypes,
            selection=args.get('memory_selection', 'diversity'),
            class_balanced=enabled(args, 'class_balanced_memory', 'yes'), seed=args.seed)
        self.replay = None  # dict: x, y, sub, logits, feat
        self.distill_reg = float(args.distill_reg)
        self.gen_reg = float(args.gen_reg)

    # ------------------------------------------------------------------ helpers
    def _optimizer(self, lr):
        params = [p for p in self.net.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=lr, weight_decay=float(self.args.weight_decay))

    @torch.no_grad()
    def _embed_eval(self, x, sub, bs=64, model=None):
        model = model if model is not None else self.net
        model.eval()
        return torch.cat([model.embed(x[i:i + bs], sub[i:i + bs]) for i in range(0, len(x), bs)])

    @torch.no_grad()
    def predict(self, x, sub, model=None):
        model = model if model is not None else self.net
        return model.logits(self._embed_eval(x, sub, model=model))

    def _augment(self, x):
        """augment='yes': random time shift (one per batch), per-trial amplitude scaling and Gaussian noise
        relative to each trial's standard deviation. Applied to training batches only."""
        a = self.args
        if not enabled(a, 'augment', 'no'):
            return x
        shift = int(torch.randint(-int(a.aug_shift), int(a.aug_shift) + 1, (1,)).item())
        x = torch.roll(x, shift, dims=-1)
        scale = 1 + float(a.aug_scale) * (2 * torch.rand(len(x), 1, 1, device=x.device) - 1)
        noise = float(a.aug_noise) * torch.randn_like(x) * x.std(dim=(1, 2), keepdim=True)
        return x * scale + noise

    def _sample_memory(self, n):
        m = self.memory
        idx = torch.randint(len(m), (n,), device=m.mu.device)
        z = m.mu[idx] + m.var[idx].clamp_min(0).sqrt() * torch.randn_like(m.mu[idx])
        return F.normalize(z, dim=1), m.labels[idx]

    # ------------------------------------------------------------------ training
    def train_subject(self, t, x_tr, y_tr, x_va, y_va):
        a = self.args
        self.net.start_subject(t, freeze_past=enabled(a, 'freeze_past_bn', 'yes'))
        sub_tr = torch.full_like(y_tr, t)
        if self.replay is not None:
            r = self.replay
            x = torch.cat([x_tr, r['x']]); y = torch.cat([y_tr, r['y']]); sub = torch.cat([sub_tr, r['sub']])
            is_rep = torch.cat([torch.zeros_like(y_tr, dtype=torch.bool), torch.ones_like(r['y'], dtype=torch.bool)])
            old_logits = torch.cat([torch.zeros(len(y_tr), a.n_classes, device=self.device), r['logits']])
        else:
            x, y, sub = x_tr, y_tr, sub_tr
            is_rep = torch.zeros_like(y_tr, dtype=torch.bool)
            old_logits = torch.zeros(len(y_tr), a.n_classes, device=self.device)
        sub_va = torch.full_like(y_va, t)

        lr, patience = float(a.lr), int(a.patience)
        opt = self._optimizer(lr)
        best_loss, best_state, best_epoch = float('inf'), copy.deepcopy(self.net.state_dict()), 0
        use_gen = self.gen_reg > 0 and len(self.memory) > 0
        # ema_decay > 0: exponential moving average of the weights (Polyak averaging). The averaged model is the one
        # validated, selected and kept; it reduces the variance of picking one checkpoint on 22 validation trials.
        decay = float(a.get('ema_decay', 0.0))
        ema = copy.deepcopy(self.net) if decay > 0 else None
        selected = ema if ema is not None else self.net
        for epoch in range(1, int(a.n_epochs) + 1):
            self.net.train()
            perm = torch.randperm(len(x), device=self.device)
            for i in range(0, len(x), int(a.batch_size)):
                idx = perm[i:i + int(a.batch_size)]
                logits = self.net(self._augment(x[idx]), sub[idx])
                loss = F.cross_entropy(logits, y[idx])
                rep = is_rep[idx]
                if self.distill_reg > 0 and rep.any():
                    loss = loss + self.distill_reg * F.mse_loss(logits[rep], old_logits[idx][rep])
                if use_gen:
                    zp, yp = self._sample_memory(len(idx))
                    loss = loss + self.gen_reg * F.cross_entropy(self.net.logits(zp), yp)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if ema is not None:
                    with torch.no_grad():
                        for pe, p in zip(ema.parameters(), self.net.parameters()):
                            pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
                        for be, b in zip(ema.buffers(), self.net.buffers()):
                            be.copy_(b)

            v_loss = F.cross_entropy(self.predict(x_va, sub_va, model=selected), y_va).item()
            if epoch < int(a.min_select_epoch):
                continue
            if v_loss < best_loss:
                best_loss, best_state, best_epoch, patience = v_loss, copy.deepcopy(selected.state_dict()), epoch, int(a.patience)
            else:
                patience -= 1
                if patience <= 0:
                    lr /= float(a.lr_factor)
                    if lr < float(a.lr_min):
                        break
                    opt, patience = self._optimizer(lr), int(a.patience)
        self.net.load_state_dict(best_state)
        print(f"  [Selection] {epoch} epochs run, selected epoch {best_epoch} (valid CE {best_loss:.3f})")

    # ------------------------------------------------------------------ balanced consolidation
    def consolidate(self, t, x_tr, y_tr):
        """consolidate_epochs > 0: after the selected checkpoint of subject t, a few epochs of class- and
        subject-balanced fine-tuning on the exemplars of the past subjects plus an equal-size class-balanced
        sample of subject t, at lr * consolidate_lr_factor (balanced fine-tuning, Castro et al. ECCV 2018).
        Lets past subjects improve (positive backward transfer) instead of only being preserved."""
        a = self.args
        n_epochs = int(a.get('consolidate_epochs', 0))
        if n_epochs <= 0 or self.replay is None:
            return
        per_class = int(a.n_samples) // a.n_classes
        g = torch.Generator().manual_seed(int(a.seed) * 1000 + 500 + t)
        keep = []
        for c in range(a.n_classes):
            idx = (y_tr == c).nonzero(as_tuple=True)[0].cpu()
            keep.append(idx[torch.randperm(len(idx), generator=g)[:per_class]])
        keep = torch.cat(keep).to(self.device)
        r = self.replay
        x = torch.cat([x_tr[keep], r['x']]); y = torch.cat([y_tr[keep], r['y']])
        sub = torch.cat([torch.full_like(y_tr[keep], t), r['sub']])
        opt = self._optimizer(float(a.lr) * float(a.get('consolidate_lr_factor', 0.1)))
        for _ in range(n_epochs):
            self.net.train()
            perm = torch.randperm(len(x), device=self.device)
            for i in range(0, len(x), int(a.batch_size)):
                idx = perm[i:i + int(a.batch_size)]
                loss = F.cross_entropy(self.net(self._augment(x[idx]), sub[idx]), y[idx])
                if self.gen_reg > 0 and len(self.memory) > 0:
                    zp, yp = self._sample_memory(len(idx))
                    loss = loss + self.gen_reg * F.cross_entropy(self.net.logits(zp), yp)
                opt.zero_grad()
                loss.backward()
                opt.step()
        print(f"  [Consolidation] {n_epochs} balanced epochs on {len(y)} trials ({len(r['y'])} exemplars)")

    # ------------------------------------------------------------------ after a subject
    @torch.no_grad()
    def end_subject(self, t, x_tr, y_tr):
        a = self.args
        # 1. drift compensation of the modes of past subjects, measured on their exemplars
        if self.replay is not None and enabled(a, 'drift_comp', 'yes') and len(self.memory) > 0:
            r = self.replay
            feat_new = self._embed_eval(r['x'], r['sub'])
            m = self.memory
            for u in r['sub'].unique().tolist():
                for c in range(a.n_classes):
                    sel = (r['sub'] == u) & (r['y'] == c)
                    ent = (m.subjects == u) & (m.labels == c)
                    if sel.any() and ent.any():
                        m.mu[ent] = m.mu[ent] + (feat_new[sel] - r['feat'][sel]).mean(0)
            r['feat'] = feat_new
            if int(a.get('consolidate_epochs', 0)) > 0:
                # after consolidation the distillation target is the improved state, not the original one
                r['logits'] = self.net.logits(feat_new)
        # 2. Gaussian modes of subject t
        sub_t = torch.full_like(y_tr, t)
        feats = self._embed_eval(x_tr, sub_t)
        self.memory.add_subject(feats, y_tr, t, modes_per_subject=int(a.modes_per_subject))
        print(f"  [Memory] {self.memory.summary()}")
        # 3. exemplars of subject t (class-balanced), with their logits and embeddings
        if int(a.n_samples) > 0:
            g = torch.Generator().manual_seed(int(a.seed) * 1000 + t)
            keep = []
            for c in range(a.n_classes):
                idx = (y_tr == c).nonzero(as_tuple=True)[0].cpu()
                keep.append(idx[torch.randperm(len(idx), generator=g)[:int(a.n_samples) // a.n_classes]])
            keep = torch.cat(keep).to(self.device)
            new = {'x': x_tr[keep], 'y': y_tr[keep], 'sub': sub_t[keep],
                   'logits': self.net.logits(feats[keep]), 'feat': feats[keep]}
            self.replay = new if self.replay is None else {k: torch.cat([self.replay[k], new[k]]) for k in new}
            print(f"  [Replay] {len(self.replay['y'])} exemplars")

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, x, y, u):
        logits = self.predict(x, torch.full_like(y, u))
        pred = logits.argmax(1)
        acc = 100.0 * (pred == y).float().mean().item()
        f1 = 100.0 * f1_score(y.cpu().numpy(), pred.cpu().numpy(), average='macro')
        return acc, f1

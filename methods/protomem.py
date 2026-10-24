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
        if enabled(self.args, 'gen_noise', 'yes'):
            z = m.mu[idx] + m.var[idx].clamp_min(0).sqrt() * torch.randn_like(m.mu[idx])
        else:                                   # gen_noise='no': mean-only replay (control for the Gaussian noise)
            z = m.mu[idx]
        return F.normalize(z, dim=1), m.labels[idx]

    def _epoch_batches(self, n_own, n_rep):
        """Index batches of one epoch over [current-subject trials | exemplars].
        Default: one shuffled pass over the concatenation, so the share of the current subject shrinks as the
        buffer grows (100 own trials vs 800 exemplars at the 9th subject).
        balanced_replay='yes': one pass over the current subject's trials, each batch paired with an equal-size
        batch drawn from the exemplars (separate replay batch, Chaudhry et al. 2019), so the current subject keeps
        half of every batch whatever the buffer size."""
        bs = int(self.args.batch_size)
        if enabled(self.args, 'balanced_replay', 'no') and n_rep > 0:
            perm = torch.randperm(n_own, device=self.device)
            for i in range(0, n_own, bs):
                own = perm[i:i + bs]
                rep = n_own + torch.randint(n_rep, (len(own),), device=self.device)
                yield torch.cat([own, rep])
        else:
            perm = torch.randperm(n_own + n_rep, device=self.device)
            for i in range(0, n_own + n_rep, bs):
                yield perm[i:i + bs]

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
            for idx in self._epoch_batches(len(y_tr), len(x) - len(y_tr)):
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
        if int(a.get('total_buffer', 0)) > 0:      # same per-subject share as the past subjects' exemplars
            per_class = min(per_class, max(1, int(a.total_buffer) // ((t + 1) * a.n_classes)))
        g = torch.Generator().manual_seed(int(a.seed) * 1000 + 500 + t)
        keep = []
        for c in range(a.n_classes):
            idx = (y_tr == c).nonzero(as_tuple=True)[0].cpu()
            keep.append(idx[torch.randperm(len(idx), generator=g)[:per_class]])
        keep = torch.cat(keep).to(self.device)
        r = self.replay
        if not enabled(a, 'consolidate_balanced', 'yes'):
            # update-matched control: the same number of trials and updates, drawn without balancing from the ordinary
            # training mixture (all training trials of subject t + the exemplars), so only the balancing differs
            px = torch.cat([x_tr, r['x']]); py = torch.cat([y_tr, r['y']])
            ps = torch.cat([torch.full_like(y_tr, t), r['sub']])
            pick = torch.randperm(len(py), generator=g)[:len(keep) + len(r['y'])].to(self.device)
            x, y, sub = px[pick], py[pick], ps[pick]
        elif enabled(a, 'consolidate_current', 'yes'):
            x = torch.cat([x_tr[keep], r['x']]); y = torch.cat([y_tr[keep], r['y']])
            sub = torch.cat([torch.full_like(y_tr[keep], t), r['sub']])
        else:   # control: the same number of extra updates on past exemplars only, without the new subject's
                # trials (the slots of the current subject are filled with randomly re-drawn past exemplars)
            extra = torch.randint(len(r['y']), (len(keep),), generator=g).to(self.device)
            x = torch.cat([r['x'], r['x'][extra]]); y = torch.cat([r['y'], r['y'][extra]])
            sub = torch.cat([r['sub'], r['sub'][extra]])
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
            mu_old = m.mu.clone()
            if a.get('drift_weighting', 'dominant') == 'weighted':
                # a merged mode is shifted by the drifts of all the subjects it summarises, weighted by their
                # number of trials in the mode (m.comp); subjects without exemplars do not contribute
                for c in range(a.n_classes):
                    ent = m.labels == c
                    if not ent.any():
                        continue
                    num = torch.zeros_like(m.mu[ent])
                    den = torch.zeros(int(ent.sum()), 1, device=m.mu.device)
                    for u in r['sub'].unique().tolist():
                        sel = (r['sub'] == u) & (r['y'] == c)
                        if sel.any():
                            w = m.comp[ent][:, u:u + 1]
                            num += w * (feat_new[sel] - r['feat'][sel]).mean(0)
                            den += w
                    m.mu[ent] = m.mu[ent] + num / den.clamp_min(1e-8)
            else:
                # each mode is shifted by the drift of the subject it is attributed to (dominant subject)
                for u in r['sub'].unique().tolist():
                    for c in range(a.n_classes):
                        sel = (r['sub'] == u) & (r['y'] == c)
                        ent = (m.subjects == u) & (m.labels == c)
                        if sel.any() and ent.any():
                            m.mu[ent] = m.mu[ent] + (feat_new[sel] - r['feat'][sel]).mean(0)
            self._drift_diagnostic(t, mu_old, feat_new)
            r['feat'] = feat_new
            if int(a.get('consolidate_epochs', 0)) > 0:
                # after consolidation the distillation target is the improved state, not the original one
                r['logits'] = self.net.logits(feat_new)
        # 2. Gaussian modes of subject t
        sub_t = torch.full_like(y_tr, t)
        feats = self._embed_eval(x_tr, sub_t)
        if enabled(a, 'mode_diag', 'yes'):
            self._multimodality_diagnostic(t, feats, y_tr)
        ids = t * 100000 + torch.arange(len(y_tr), device=y_tr.device)       # unique id of every training trial
        self.memory.add_subject(feats, y_tr, t, modes_per_subject=int(a.modes_per_subject), ids=ids)
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
                   'logits': self.net.logits(feats[keep]), 'feat': feats[keep], 'id': ids[keep]}
            self.replay = new if self.replay is None else {k: torch.cat([self.replay[k], new[k]]) for k in new}
            # total_buffer > 0: fixed total number of stored trials over the whole stream, shared equally by the
            # subjects seen so far and their classes; older subjects keep a random subset (their stored order is random)
            total = int(a.get('total_buffer', 0))
            if total > 0:
                r = self.replay
                cap = max(1, total // (len(r['sub'].unique()) * a.n_classes))
                keep = torch.cat([((r['sub'] == u) & (r['y'] == c)).nonzero(as_tuple=True)[0][:cap]
                                  for u in r['sub'].unique().tolist() for c in range(a.n_classes)])
                keep = keep.sort().values
                self.replay = {k: v[keep] for k, v in r.items()}
            print(f"  [Replay] {len(self.replay['y'])} exemplars")

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def _drift_diagnostic(self, t, mu_old, feat_new):
        """Accuracy of the drift approximation: each corrected mode is compared with the mode recomputed from the
        current embeddings of its own stored trials (exact when all of its trials are stored). Appended to self.diag
        (saved as drift_diag.json by main_protomem.py)."""
        m, r = self.memory, self.replay
        if m.members is None or 'id' not in r:
            return
        e_corr, e_raw, cover, var_ratio, merged = [], [], [], [], 0
        for i, mem in enumerate(m.members):
            mask = torch.isin(r['id'], mem)
            if int(mask.sum()) < 2:
                continue
            rec = feat_new[mask]
            e_corr.append(float((m.mu[i] - rec.mean(0)).norm()))
            e_raw.append(float((mu_old[i] - rec.mean(0)).norm()))
            cover.append(float(mask.sum()) / len(mem))
            var_ratio.append(float(rec.var(0, unbiased=False).mean() / m.var[i].mean().clamp_min(1e-12)))
            merged += int((m.comp[i] > 0).sum() > 1)
        if not hasattr(self, 'diag'):
            self.diag = []
        if e_corr:
            self.diag.append({'stage': t + 1, 'modes': len(e_corr), 'merged_modes': merged,
                              'err_corrected': float(np.mean(e_corr)), 'err_uncorrected': float(np.mean(e_raw)),
                              'coverage': float(np.mean(cover)), 'var_ratio': float(np.mean(var_ratio))})

    @torch.no_grad()
    def _multimodality_diagnostic(self, t, feats, y, n_null=20):
        """Are the embeddings of one (subject, class) multimodal? Silhouette of a 2-means clustering of the real
        embeddings, against the same statistic on samples of a single diagonal Gaussian with the same mean and
        variance (parametric null). p = share of null draws with a silhouette >= the real one. CPU / numpy with its
        own generator, so the training RNG streams are unchanged. Appended to self.mode_diag (mode_diag.json)."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        rng = np.random.default_rng(int(self.args.seed) * 1000 + 700 + t)
        sil = lambda z: silhouette_score(z, KMeans(2, n_init=10, random_state=0).fit_predict(z))
        if not hasattr(self, 'mode_diag'):
            self.mode_diag = []
        for c in range(int(self.args.n_classes)):
            z = feats[y == c].double().cpu().numpy()
            if len(z) < 6:
                continue
            real = float(sil(z))
            mu, sd = z.mean(0), z.std(0)
            null = [float(sil(mu + sd * rng.standard_normal(z.shape))) for _ in range(n_null)]
            # stricter null: a single Gaussian with the full empirical covariance (correlations kept), sampled as
            # mu + sum_i (z_i - mu) g_i / sqrt(n), g ~ N(0, I)
            zc = z - mu
            full = [float(sil(mu + rng.standard_normal((len(z), len(z))) @ zc / np.sqrt(len(z))))
                    for _ in range(n_null)]
            self.mode_diag.append({'subject': t + 1, 'class': c, 'n': len(z), 'silhouette': real,
                                   'null_mean': float(np.mean(null)), 'null_sd': float(np.std(null)),
                                   'p': float(np.mean(np.array(null) >= real)),
                                   'null_full_mean': float(np.mean(full)), 'null_full_sd': float(np.std(full)),
                                   'p_full': float(np.mean(np.array(full) >= real))})

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, x, y, u):
        logits = self.predict(x, torch.full_like(y, u))
        pred = logits.argmax(1)
        acc = 100.0 * (pred == y).float().mean().item()
        f1 = 100.0 * f1_score(y.cpu().numpy(), pred.cpu().numpy(), average='macro')
        return acc, f1

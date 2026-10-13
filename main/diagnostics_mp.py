"""
Diagnostics for CA4MI-MP (experiments A-D). Only active with `diagnostics: 'yes'`; normal runs are unchanged.

  A  training vs validation curves      -> A_curves.jsonl, A_summary.jsonl
  B  per-subject / per-class test        -> B_confusion.jsonl
  C  feature-domain shift                -> C_shift.jsonl
  D  gradients on the shared encoder     -> D_gradients.jsonl
     forgetting during training          -> D_forgetting.jsonl
     memory content                      -> D_memory.jsonl

Files are written to <checkpoint>/diagnostics/seed_<seed>/. Every diagnostic runs with the RNG state saved and
restored, without parameter updates and with the model in eval mode (BatchNorm statistics are not touched), so the
training trajectory is the same as without diagnostics.

C uses validation sets of all seen subjects and D_forgetting uses validation sets of previous subjects: this is offline
analysis only, nothing computed here is fed back into training, alignment, memory or model selection.
"Own" trials are the evaluated subject's own trials, "replay" trials are replayed trials of other subjects that the
shared dataloader puts into the same split.

Offline per-subject summary of saved accuracy matrices (experiment B, no training needed):
    python main/diagnostics_mp.py <checkpoint>/**/ca4mi_mp_9_subjects_seed_*.txt
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import random
from collections import Counter
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
import utils


@contextmanager
def preserved_rng():
    """Run a block without changing python / numpy / torch / cuda RNG states."""
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _jsonable(o):
    if hasattr(o, 'item') and getattr(o, 'ndim', 0) == 0:
        return o.item()
    if hasattr(o, 'tolist'):
        return o.tolist()
    return str(o)


def loader_tensors(loader):
    """(x, y, sub_module_label, dis_label) of an aligned loader (TensorDataset built by EEGTrainer.align_data)."""
    return loader.dataset.tensors


@torch.no_grad()
def predict(model, x, sub, private_id, batch_size=64):
    model.eval()
    logits = []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size][:, None].float()
        logits.append(model(xb, xb, sub[i:i + batch_size], private_id))
    return torch.cat(logits)


@torch.no_grad()
def encode(model, x, private_id, batch_size=64):
    model.eval()
    shared, private = [], []
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size][:, None].float()
        s, p = model.get_encoded(xb, xb, private_id)
        shared.append(s)
        private.append(p)
    return torch.cat(shared), torch.cat(private)


def class_metrics(logits, y, n_classes):
    if len(y) == 0:
        return None
    pred = logits.argmax(1)
    conf = torch.bincount(y * n_classes + pred, minlength=n_classes * n_classes).view(n_classes, n_classes)
    support = conf.sum(1)
    recall = [100. * conf[c, c].item() / support[c].item() if support[c] > 0 else None for c in range(n_classes)]
    return {'n': len(y), 'acc': 100. * (pred == y).float().mean().item(),
            'ce': F.cross_entropy(logits, y).item(), 'confusion': conf.tolist(), 'recall': recall}


def split_metrics(logits, y, sub, own_id, n_classes):
    """Metrics on all trials, on the subject's own trials and on replayed trials of other subjects."""
    own = sub == own_id
    return {'all': class_metrics(logits, y, n_classes),
            'own': class_metrics(logits[own], y[own], n_classes),
            'replay': class_metrics(logits[~own], y[~own], n_classes)}


# ----------------------------------------------------------------------------- SPD / feature statistics
def _spd_log(c):
    w, v = np.linalg.eigh(c)
    return (v * np.log(np.clip(w, 1e-10, None))) @ v.T


def mean_trial_covariance(x):
    """Mean channel covariance of trials x [N, C, T] (float64)."""
    x = x.double()
    xc = x - x.mean(-1, keepdim=True)
    return (xc @ xc.transpose(1, 2) / (x.size(-1) - 1)).mean(0).cpu().numpy()


def log_euclid_to_identity(x):
    """||log(mean cov)||_F: 0 when the trials are perfectly whitened (Euclidean alignment target)."""
    if len(x) == 0:
        return None
    return float(np.linalg.norm(_spd_log(mean_trial_covariance(x))))


def linear_probe(features, labels, max_folds=5):
    """Cross-validated logistic-regression accuracy (%), None if there are too few samples per label."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    counts = np.bincount(labels)
    counts = counts[counts > 0]
    if len(counts) < 2 or counts.min() < 2:
        return None
    cv = StratifiedKFold(n_splits=min(max_folds, int(counts.min())), shuffle=True, random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return float(100. * cross_val_score(clf, features, labels, cv=cv).mean())


def fisher_ratio(f, y):
    """||mu_1 - mu_0||^2 / (tr S_0 + tr S_1) for 2 classes (mean over class pairs otherwise)."""
    classes = np.unique(y)
    vals = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a, b = f[y == classes[i]], f[y == classes[j]]
            if len(a) > 1 and len(b) > 1:
                vals.append(float(np.sum((a.mean(0) - b.mean(0)) ** 2) / (a.var(0).sum() + b.var(0).sum() + 1e-12)))
    return float(np.mean(vals)) if vals else None


def subject_covariance_distances(features, subjects, k=16):
    """Pairwise log-Euclidean distances between subjects' feature covariances (PCA to k dims, Ledoit-Wolf)."""
    from sklearn.covariance import LedoitWolf
    ids = np.unique(subjects)
    if len(ids) < 2:
        return None
    centered = features - features.mean(0)
    k = min(k, centered.shape[0] - 1, centered.shape[1])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    z = centered @ vt[:k].T
    logs = {int(s): _spd_log(LedoitWolf().fit(z[subjects == s]).covariance_) for s in ids}
    dist = [[float(np.linalg.norm(logs[int(a)] - logs[int(b)])) for b in ids] for a in ids]
    off = [dist[i][j] for i in range(len(ids)) for j in range(len(ids)) if i != j]
    return {'subjects': ids.tolist(), 'matrix': dist, 'mean_pairwise': float(np.mean(off))}


class DiagnosticsMP:
    def __init__(self, args):
        self.args = args
        self.every = int(args.get('diag_every', 10))
        self.n_grad = int(args.get('diag_grad_samples', 64))
        self.n_classes = args.n_classes
        self.out_dir = os.path.join(args.checkpoint, 'diagnostics', f'seed_{args.seed}')
        os.makedirs(self.out_dir, exist_ok=True)
        self.t = None
        self.dataset = None
        self.epoch = 0
        self.curves = []
        self.grad_idx = None

    def write(self, name, record):
        with open(os.path.join(self.out_dir, f'{name}.jsonl'), 'a') as f:
            f.write(json.dumps(record, default=_jsonable) + '\n')

    # ------------------------------------------------------------------ subject lifecycle
    def begin_subject(self, t, dataset, dataloader, ca4mi):
        """Call after IEA alignment of subject t, before training."""
        self.t, self.dataset, self.epoch, self.curves = t, dataset, 0, []
        n = len(loader_tensors(dataset[t]['train'])[0])
        g = torch.Generator().manual_seed(int(self.args.seed) * 1000 + t)
        self.grad_idx = torch.randperm(n, generator=g)[:self.n_grad]
        with preserved_rng():
            self.log_memory(t, dataloader, ca4mi)
            self.log_alignment(t, dataset)

    def on_epoch_end(self, ca4mi, sub_id, dataset, prototypes):
        """Called by ca4mi_mp_Trainer.train_single_epoch after every training epoch."""
        with preserved_rng():
            self.log_curves(ca4mi, sub_id, dataset)
            if self.epoch % self.every == 0:
                self.log_gradients(ca4mi, sub_id, dataset, prototypes)
                self.log_forgetting(ca4mi, sub_id)
        self.epoch += 1

    def end_subject(self, t):
        """Call after adversarial_training. Reproduces the epoch chosen by the trainer (min valid loss_tot)."""
        if not self.curves:
            return
        tot = [c['valid_loss_tot'] for c in self.curves]
        ce = [c['valid']['all']['ce'] for c in self.curves]
        e_tot, e_ce = int(np.argmin(tot)), int(np.argmin(ce))
        pick = lambda e: {'epoch': e, 'train_own_acc': self.curves[e]['train']['own']['acc'],
                          'valid_acc': self.curves[e]['valid']['all']['acc'],
                          'valid_own_acc': (self.curves[e]['valid']['own'] or {}).get('acc'),
                          'valid_ce': ce[e], 'valid_loss_tot': tot[e]}
        self.write('A_summary', {'t': t, 'epochs_run': len(self.curves),
                                 'selected_by_loss_tot': pick(e_tot), 'selected_by_valid_ce': pick(e_ce)})

    # ------------------------------------------------------------------ A: curves
    def log_curves(self, ca4mi, t, dataset):
        x, y, sub, _ = loader_tensors(dataset['train'])
        train = split_metrics(predict(ca4mi.model, x, sub, t), y, sub, t, self.n_classes)
        xv, yv, subv, _ = loader_tensors(dataset['valid'])
        valid = split_metrics(predict(ca4mi.model, xv, subv, t), yv, subv, t, self.n_classes)
        valid_tot = ca4mi.eval_(dataset['valid'], t)['loss_tot']  # exactly the trainer's selection criterion
        rec = {'t': t, 'epoch': self.epoch, 'train': train, 'valid': valid, 'valid_loss_tot': valid_tot,
               'lr_encoder': ca4mi.encoder_optimizer.param_groups[0]['lr'],
               'lr_discriminator': ca4mi.discriminator_optimizer.param_groups[0]['lr']}
        self.curves.append(rec)
        self.write('A_curves', rec)

    # ------------------------------------------------------------------ B: test confusion
    def log_test(self, model, loader, t, u):
        x, y, sub, _ = loader_tensors(loader)
        with preserved_rng():
            rec = split_metrics(predict(model, x, sub, u), y, sub, u, self.n_classes)
        self.write('B_confusion', {'t': t, 'u': u, **rec})

    # ------------------------------------------------------------------ D: gradients
    def log_gradients(self, ca4mi, t, dataset, prototypes):
        """Weighted gradient of every loss term w.r.t. the shared encoder, on a fixed training subset (eval mode)."""
        model, disc = ca4mi.model, ca4mi.discriminator
        model.eval()
        disc.eval()
        x, y, sub, dis = [v[self.grad_idx.to(v.device)] for v in loader_tensors(dataset['train'])]
        xb = x[:, None].float()
        params = [p for p in model.shared.parameters() if p.requires_grad]

        shared, private = model.get_encoded(xb, xb, t)
        out = model(xb, xb, sub, t)
        own, n = sub == t, len(y)
        losses = {}
        # Cross-entropy split by origin, scaled so that cls_current + cls_replay = the batch-mean CE used in training
        if own.any():
            losses['cls_current'] = F.cross_entropy(out[own], y[own]) * own.sum() / n
        if (~own).any():
            losses['cls_replay'] = F.cross_entropy(out[~own], y[~own]) * (~own).sum() / n
        losses['adv'] = ca4mi.adv_loss_reg * ca4mi.adv_loss_fn(disc(shared, dis, t), dis)  # through the GRL
        if ca4mi.orth == 'yes':
            losses['ort'] = ca4mi.ort_loss_reg * ca4mi.ort_loss_fn(shared, private)
        if ca4mi.use_prototypes == 'yes' and prototypes is not None:
            pro = ca4mi.pro_loss_reg * ca4mi.compute_prototype_loss(shared, y, prototypes)
            if pro.requires_grad:
                losses['pro'] = pro

        grads = {}
        for name, loss in losses.items():
            g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            grads[name] = torch.cat([(gi if gi is not None else torch.zeros_like(p)).flatten()
                                     for gi, p in zip(g, params)])
        names = list(grads)
        cosine = {f'{a}|{b}': F.cosine_similarity(grads[a], grads[b], dim=0).item()
                  for i, a in enumerate(names) for b in names[i + 1:]}
        self.write('D_gradients', {'t': t, 'epoch': self.epoch, 'n_replay': int((~own).sum()),
                                   'loss': {k: v.item() for k, v in losses.items()},
                                   'weighted_grad_norm': {k: v.norm().item() for k, v in grads.items()},
                                   'cosine': cosine})

    # ------------------------------------------------------------------ D: forgetting during training
    def log_forgetting(self, ca4mi, t):
        """Validation accuracy of every previous subject u < t with the current shared encoder."""
        if t == 0:
            return
        rec = {'t': t, 'epoch': self.epoch, 'valid': {}}
        for u in range(t):
            model = utils.load_model_with_shared_update(
                network_class=ca4mi.network.Cls_HEAD, args=ca4mi.args,
                current_model=ca4mi.model, checkpoint_dir=ca4mi.checkpoint, sub_id=u)
            x, y, sub, _ = loader_tensors(self.dataset[u]['valid'])
            rec['valid'][u] = split_metrics(predict(model, x, sub, u), y, sub, u, self.n_classes)
        self.write('D_forgetting', rec)

    # ------------------------------------------------------------------ D: memory content
    def log_memory(self, t, dataloader, ca4mi):
        mem = getattr(dataloader, 'global_memory', None) or {}
        rec = {'t': t, 'replay_size': len(mem.get('y', [])),
               'replay_label_count': dict(Counter(int(v) for v in mem.get('y', []))),
               'replay_subject_count': dict(Counter(int(v) for v in mem.get('sub_module_label', [])))}
        m = ca4mi.memory
        if len(m) > 0:
            rec['prototype_memory'] = {
                'entries': len(m),
                'per_class': dict(Counter(int(v) for v in m.labels.cpu())),
                'per_subject': dict(Counter(int(v) for v in m.subjects.cpu())),
                'consolidation_active': len(m) >= m.max_prototypes}
        self.write('D_memory', rec)

    # ------------------------------------------------------------------ C: alignment + feature shift
    def log_alignment(self, t, dataset):
        """Input space after IEA: distance of the mean trial covariance to the identity, own vs replayed trials."""
        rec = {'t': t}
        for split in ['train', 'valid', 'test']:
            x, _, sub, _ = loader_tensors(dataset[t][split])
            own = sub == t
            rec[split] = {'own': log_euclid_to_identity(x[own]), 'replay': log_euclid_to_identity(x[~own]),
                          'n_own': int(own.sum()), 'n_replay': int((~own).sum())}
        self.write('C_alignment', rec)

    def log_feature_shift(self, ca4mi, t):
        """After subject t (and the memory update): shift, separability and prototype quality of the features."""
        with preserved_rng(), torch.no_grad():
            model = ca4mi.model
            feats, privs, ys, subs = [], [], [], []
            per_subject = {}
            anchors = ca4mi.memory.anchors(ca4mi.n_modes) if len(ca4mi.memory) > 0 else None
            for u in range(t + 1):
                x, y, sub, _ = loader_tensors(self.dataset[u]['valid'])
                own = sub == u
                x, y = x[own], y[own]
                s, p = encode(model, x, u)
                feats.append(s.cpu().numpy())
                privs.append(p.cpu().numpy())
                ys.append(y.cpu().numpy())
                subs.append(np.full(len(y), u))
                info = {'n': len(y), 'fisher_shared': fisher_ratio(feats[-1], ys[-1])}
                if anchors is not None:
                    d = ca4mi._distances(s, anchors)
                    same = anchors.labels.unsqueeze(0) == y.unsqueeze(1)
                    d_same = d.masked_fill(~same, float('inf')).min(1).values
                    d_other = d.masked_fill(same, float('inf')).min(1).values
                    info['anchor_dist_same'] = d_same.mean().item()
                    info['anchor_dist_other'] = d_other.mean().item()
                    info['nearest_anchor_acc'] = 100. * (d_same < d_other).float().mean().item()
                info['prototype_drift'] = self._prototype_drift(ca4mi, u)
                per_subject[u] = info

            f, fp = np.concatenate(feats), np.concatenate(privs)
            y, s = np.concatenate(ys), np.concatenate(subs)
            rec = {'t': t, 'per_subject': per_subject,
                   'probe_shared_subject': linear_probe(f, s), 'probe_shared_class': linear_probe(f, y),
                   'probe_private_subject': linear_probe(fp, s), 'probe_private_class': linear_probe(fp, y),
                   'shared_cov_distance': subject_covariance_distances(f, s)}
        self.write('C_shift', rec)

    def _prototype_drift(self, ca4mi, u):
        """Stored memory modes of subject u vs class means of u's own training features under the current encoder.
        Uses u's past training split: offline analysis only."""
        m = ca4mi.memory
        if len(m) == 0 or not (m.subjects == u).any():
            return None
        x, y, sub, _ = loader_tensors(self.dataset[u]['train'])
        own = sub == u
        s, _ = encode(ca4mi.model, x[own], u)
        y = y[own]
        drift = {}
        for c in range(self.n_classes):
            sel = (m.subjects == u) & (m.labels == c)
            if sel.any() and (y == c).any():
                now = s[y == c].mean(0)
                stored = (m.mu[sel] * m.count[sel].unsqueeze(1)).sum(0) / m.count[sel].sum()
                drift[c] = ((stored - now).norm() / now.norm().clamp_min(1e-8)).item()
        return drift


# ----------------------------------------------------------------------------- offline B summary
def summarize_matrix(acc, n_classes=2):
    """Per-subject diagonal, final, forgetting and chance-level cells of one accuracy matrix (rows = stages)."""
    n = acc.shape[0]
    chance = 100. / n_classes
    rows = []
    for u in range(n):
        seen = acc[u:, u]
        forgetting = float(np.max(acc[u:n - 1, u]) - acc[n - 1, u]) if u < n - 1 else None
        rows.append({'subject': u, 'diag': float(acc[u, u]), 'final': float(acc[n - 1, u]),
                     'forgetting': forgetting, 'chance_cells': int(np.isclose(seen, chance, atol=0.01).sum())})
    return {'ACC': float(acc[-1].mean()),
            'BWT': float(np.sum(acc[-1] - np.diag(acc)) / (n - 1)),
            'forgetting': float(np.mean([r['forgetting'] for r in rows if r['forgetting'] is not None])),
            'subjects': rows}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Per-subject summary of CA4MI-MP accuracy matrices (experiment B)')
    parser.add_argument('files', nargs='+', help='accuracy matrix .txt files (np.savetxt), one per seed')
    parser.add_argument('--n_classes', type=int, default=2)
    flags = parser.parse_args()

    summaries = [summarize_matrix(np.loadtxt(p), flags.n_classes) for p in flags.files]
    for path, s in zip(flags.files, summaries):
        print(f"\n{path}\n  ACC={s['ACC']:.2f}  BWT={s['BWT']:.2f}  forgetting={s['forgetting']:.2f}")
        print(f"  {'subj':>4} {'diag':>7} {'final':>7} {'forget':>7} {'chance':>6}")
        for r in s['subjects']:
            fg = f"{r['forgetting']:7.2f}" if r['forgetting'] is not None else '      -'
            print(f"  {r['subject']:>4} {r['diag']:7.2f} {r['final']:7.2f} {fg} {r['chance_cells']:>6}")
    if len(summaries) > 1:
        for key in ['ACC', 'BWT', 'forgetting']:
            v = [s[key] for s in summaries]
            print(f"{key}: {np.mean(v):.2f} +- {np.std(v):.2f} over {len(v)} seeds")


if __name__ == '__main__':
    main()

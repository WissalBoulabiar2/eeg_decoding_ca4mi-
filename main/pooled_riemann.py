"""
Pooled, aligned Riemannian decoder that replays covariance matrices instead of raw EEG (review 16), on exactly the
trials the deep models use (same loader and splits as classical_baselines.py; cross_session=yes tests on the whole
second session).
Each subject's covariances (8-30 Hz, OAS) are re-centred to the identity with the Riemannian mean of that subject's
training covariances only (re-centring step of Riemannian Procrustes analysis; validation and test trials use the
same reference, so no statistic of the evaluation trials is used) and mapped to the tangent space at the identity.
At stage t, one logistic regression shared by all subjects is refitted on the current subject's training vectors and
the stored vectors of the past subjects, then evaluated on the test trials of subjects 1..t:
- rct_all: every training covariance is kept (no raw EEG, 253 values per trial);
- rct_buf: the same fixed total buffer as ProtoMem, floor(B / (k C)) covariances per subject and class, a seeded
  random subset (the first trials of a random order drawn when the subject is stored), B = total_buffer.
Matrices are written like classical_baselines.py, so the analysis scripts read them unchanged:

    python main/pooled_riemann.py --config ./configs/protomem.yml --n_runs 10 --checkpoint OUT \
        --set data_dir=/path/ seed_offset=28 total_buffer=180 [cross_session=yes] [n_subjects=14]
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import numpy as np
import scipy.signal
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.tangentspace import tangent_space
from dataloaders.dataloader_ca4mi_mp import IncrementalDataStreamingMP

FS = 250


def to_arrays(loader):
    """Same as main_protomem.to_arrays (not imported: `main` is shadowed by main/main.py on the molab PYTHONPATH)."""
    xs, ys = [], []
    for batch in loader:
        xs.append(np.asarray(batch[0], dtype=np.float32))
        ys.append(np.asarray(batch[1]))
    return np.concatenate(xs), np.concatenate(ys).astype(np.int64)


def covs(x):
    sos = scipy.signal.butter(4, [8, 30], btype='bandpass', fs=FS, output='sos')
    return Covariances('oas').fit_transform(scipy.signal.sosfiltfilt(sos, x.astype(np.float64), axis=-1))


def recentred_vectors(c, ref_isqrt):
    c = ref_isqrt @ c @ ref_isqrt
    return tangent_space(c, np.eye(c.shape[-1]))


def run(args, seed):
    args.seed = seed
    args.ea = 'no'                  # raw trials; the alignment is the Riemannian re-centring below
    loader = IncrementalDataStreamingMP(args)
    n, n_cls = args.n_subjects, args.n_classes
    budget = int(args.get('total_buffer', 180))
    variants = ('rct_all', 'rct_buf')
    res = {k: {m: np.zeros((n, n)) for m in ('acc', 'f1', 'val')} for k in variants}
    tests, valids, stored = [], [], []          # stored: per subject (vectors, labels, order) of training trials
    for t in range(n):
        d = loader.load_data(t)[t]
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = (to_arrays(d[s]) for s in ('train', 'valid', 'test'))
        c_tr = covs(x_tr)
        r = invsqrtm(mean_riemann(c_tr))
        v_tr = recentred_vectors(c_tr, r)
        valids.append((recentred_vectors(covs(x_va), r), y_va))
        tests.append((recentred_vectors(covs(x_te), r), y_te))
        rng = np.random.RandomState(seed * 1000 + t)
        order = {c: rng.permutation(np.flatnonzero(y_tr == c)) for c in range(n_cls)}
        quota = budget // ((t + 1) * n_cls)
        for k in variants:
            xs, ys = [v_tr], [y_tr]
            for v, y, o in stored:
                idx = np.arange(len(y)) if k == 'rct_all' else np.concatenate([o[c][:quota] for c in range(n_cls)])
                xs.append(v[idx]); ys.append(y[idx])
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack(xs), np.concatenate(ys))
            for u in range(t + 1):
                p = lr.predict(tests[u][0])
                res[k]['acc'][t, u] = 100 * np.mean(p == tests[u][1])
                res[k]['f1'][t, u] = 100 * f1_score(tests[u][1], p, average='macro')
                res[k]['val'][t, u] = 100 * np.mean(lr.predict(valids[u][0]) == valids[u][1])
        stored.append((v_tr, y_tr, order))
        print(f"  seed {seed} stage {t + 1}: " + "  ".join(
            f"{k} {res[k]['acc'][t, :t + 1].mean():.1f}" for k in variants), flush=True)
    for k in variants:
        out = os.path.join(args.checkpoint, k, f"protomem_{n}_subjects_seed_{seed}")
        os.makedirs(out, exist_ok=True)
        for name, m in (('accuracy_matrix', 'acc'), ('f1_macro_matrix', 'f1'), ('val_accuracy_matrix', 'val')):
            np.savetxt(os.path.join(out, f'{name}.txt'), res[k][m], '%.6f')
        a = res[k]['acc']
        print(f"seed {seed} [{k}]: ACC={a[-1].mean():.2f} BWT={np.mean(a[-1, :-1] - np.diag(a)[:-1]):.2f} -> {out}")


def main():
    parser = argparse.ArgumentParser(description='Pooled re-centred tangent-space decoder with covariance replay')
    parser.add_argument('--config', type=str, default='./configs/protomem.yml')
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    flags = parser.parse_args()
    args = OmegaConf.merge(OmegaConf.load(flags.config), OmegaConf.from_dotlist(flags.set))
    args.checkpoint = flags.checkpoint
    for r in range(flags.n_runs):
        run(args, r + int(args.get('seed_offset', 0)))


if __name__ == '__main__':
    main()

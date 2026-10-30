"""
Classical per-subject calibration decoders (review 14, M7), on exactly the trials the deep models use: the subject
splits come from the same loader (IncrementalDataStreamingMP), so train / valid / test are identical for every seed,
and the cross-session protocol (cross_session=yes) tests on the whole second session as for the other methods.
Each subject gets its own decoder trained on its training trials only (they never forget and store no raw EEG):
- fbcsp: filter bank of nine 4-Hz bands from 4 to 40 Hz, 4 CSP filters per band (log-variance), shrinkage LDA;
- tslr:  8-30 Hz band-pass, OAS covariances, Riemannian tangent space, logistic regression.
Matrices are written like the reference modes of main_protomem.py (row t = subjects 0..t, no forgetting), so the
analysis scripts read them unchanged:

    python main/classical_baselines.py --config ./configs/protomem.yml --n_runs 10 --checkpoint OUT \
        --set data_dir=/path/ seed_offset=28 [cross_session=yes]
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import numpy as np
import scipy.signal
from omegaconf import OmegaConf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from pyriemann.tangentspace import TangentSpace
from dataloaders.dataloader_ca4mi_mp import IncrementalDataStreamingMP
from main.main_protomem import to_arrays

FS = 250
BANDS = [(lo, lo + 4) for lo in range(4, 40, 4)]


def bandpass(x, lo, hi):
    sos = scipy.signal.butter(4, [lo, hi], btype='bandpass', fs=FS, output='sos')
    return scipy.signal.sosfiltfilt(sos, x, axis=-1)


class FBCSP:
    def fit(self, x, y):
        self.csp = []
        feats = []
        for lo, hi in BANDS:
            c = Covariances('oas').fit_transform(bandpass(x, lo, hi))
            csp = CSP(nfilter=4, log=True).fit(c, y)
            self.csp.append(csp)
            feats.append(csp.transform(c))
        self.lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(np.hstack(feats), y)
        return self

    def predict(self, x):
        feats = [csp.transform(Covariances('oas').fit_transform(bandpass(x, lo, hi)))
                 for csp, (lo, hi) in zip(self.csp, BANDS)]
        return self.lda.predict(np.hstack(feats))


class TSLR:
    def fit(self, x, y):
        self.ts = TangentSpace(metric='riemann').fit(Covariances('oas').fit_transform(bandpass(x, 8, 30)), y)
        self.lr = LogisticRegression(C=1.0, max_iter=1000).fit(self._feat(x), y)
        return self

    def _feat(self, x):
        return self.ts.transform(Covariances('oas').fit_transform(bandpass(x, 8, 30)))

    def predict(self, x):
        return self.lr.predict(self._feat(x))


DECODERS = {'fbcsp': FBCSP, 'tslr': TSLR}


def run(args, seed):
    args.seed = seed
    loader = IncrementalDataStreamingMP(args)
    n = args.n_subjects
    res = {k: np.zeros((3, n)) for k in DECODERS}
    for t in range(n):
        d = loader.load_data(t)[t]
        x_tr, y_tr = to_arrays(d['train'])
        x_va, y_va = to_arrays(d['valid'])
        x_te, y_te = to_arrays(d['test'])
        x_tr, x_va, x_te = (z.astype(np.float64) for z in (x_tr, x_va, x_te))
        for k, cls in DECODERS.items():
            dec = cls().fit(x_tr, y_tr)
            p_te, p_va = dec.predict(x_te), dec.predict(x_va)
            res[k][:, t] = (100 * np.mean(p_te == y_te), 100 * f1_score(y_te, p_te, average='macro'),
                            100 * np.mean(p_va == y_va))
        print(f"  seed {seed} subject {t + 1}: " + "  ".join(f"{k} {res[k][0, t]:.1f}" for k in DECODERS), flush=True)
    for k, (acc_u, f1_u, val_u) in res.items():
        out = os.path.join(args.checkpoint, k, f"protomem_{n}_subjects_seed_{seed}")
        os.makedirs(out, exist_ok=True)
        for name, v in (('accuracy_matrix', acc_u), ('f1_macro_matrix', f1_u), ('val_accuracy_matrix', val_u)):
            np.savetxt(os.path.join(out, f'{name}.txt'), np.tril(np.tile(v, (n, 1))), '%.6f')
        print(f"seed {seed} [{k}]: ACC={acc_u.mean():.2f} -> {out}")


def main():
    parser = argparse.ArgumentParser(description='Per-subject FBCSP+LDA and tangent-space+LR decoders')
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

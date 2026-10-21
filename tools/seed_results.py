"""
Seed-level results and paired statistics, recomputed from the saved accuracy / F1 matrices.

For every (dataset, method, seed): ACC, BWT, forgetting, F1. Then, for ProtoMem against each other method, the
paired difference over seeds with its 95% confidence interval (t distribution, n-1 dof) and the paired t-test p-value.

    python tools/seed_results.py --set 2a final=final_bal/final a0=ablation/A0 \
                                 --set HGD final=hgd/final protomem=hgd_tune/final/gen1.0 --seeds 8 9 10 11 12
Writes seed_results.csv and paired_ci.csv (and prints both).
"""
import argparse
import csv
import glob
import os
import numpy as np
from scipy import stats

BASE = {'finetuning': 'Fine-tuning', 'ewc': 'EWC', 'er': 'ER', 'der': 'DER', 'mudvi': 'MUDVI', 'cger': 'CGER'}
METRICS = ['ACC', 'BWT', 'Forgetting', 'F1']


def metrics(acc_file):
    a = np.loadtxt(acc_file)
    f1_file = (acc_file.replace('acc_run_', 'f1_run_') if 'acc_run_' in acc_file
               else acc_file.replace('accuracy_matrix', 'f1_macro_matrix'))
    n = len(a)
    return [a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (n - 1),
            np.mean([a[u:n - 1, u].max() - a[-1, u] for u in range(n - 1)]), np.loadtxt(f1_file)[-1].mean()]


def seed_of_dir(f):
    return int(os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1])


def load(spec, n):
    """spec: {'final': dir with bench/ and protomem/, optional 'protomem': override dir, optional 'a0': A0 dir}.
    Duplicated runs of one seed (restarted sessions) are identical; the first one is kept."""
    out = {}
    final = spec['final']
    for key, name in BASE.items():
        runs = {}
        for f in sorted(glob.glob(f"{final}/bench/{key}/**/acc_run_*_subject_{n}.txt", recursive=True)):
            runs.setdefault(int(os.path.basename(f).split('_')[2]), metrics(f))
        out[name] = runs
    out['CA4MI'] = {}
    for f in sorted(glob.glob(f"{final}/bench/ca4mi/**/ca4mi_{n}_subjects_seed_*/accuracy_matrix.txt", recursive=True)):
        out['CA4MI'].setdefault(seed_of_dir(f), metrics(f))
    for name, d in [('ProtoMem', spec.get('protomem', f"{final}/protomem")), ('A0', spec.get('a0'))]:
        if d:
            out[name] = {}
            for f in sorted(glob.glob(f"{d}/**/protomem_{n}_subjects_seed_*/accuracy_matrix.txt", recursive=True)):
                out[name].setdefault(seed_of_dir(f), metrics(f))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', nargs='+', action='append', required=True,
                   help='DATASET key=dir ... (keys: final, protomem, a0)')
    p.add_argument('--seeds', type=int, nargs='*', default=[8, 9, 10, 11, 12])
    p.add_argument('--n_subjects', type=int, default=9)
    flags = p.parse_args()

    rows, ci_rows = [], []
    for s in flags.set:
        ds, spec = s[0], dict(kv.split('=', 1) for kv in s[1:])
        data = load(spec, flags.n_subjects)
        for name, runs in data.items():
            for seed in flags.seeds:
                if seed in runs:
                    rows.append([ds, name, seed] + [f"{v:.2f}" for v in runs[seed]])
        proto = data['ProtoMem']
        for name, runs in data.items():
            if name == 'ProtoMem':
                continue
            seeds = [x for x in flags.seeds if x in proto and x in runs]
            if len(seeds) < 2:
                continue
            for k, met in enumerate(METRICS):
                d = np.array([proto[x][k] - runs[x][k] for x in seeds])
                half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                pval = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else float('nan')
                ci_rows.append([ds, f"ProtoMem - {name}", met, len(d), f"{d.mean():.2f}",
                                f"{d.mean() - half:.2f}", f"{d.mean() + half:.2f}", f"{pval:.3f}"])

    with open('seed_results.csv', 'w', newline='') as f:
        csv.writer(f).writerows([['dataset', 'method', 'seed'] + METRICS] + rows)
    with open('paired_ci.csv', 'w', newline='') as f:
        csv.writer(f).writerows([['dataset', 'comparison', 'metric', 'n', 'mean_diff', 'ci95_low', 'ci95_high', 'p']]
                                + ci_rows)
    print(f"{'dataset':8s} {'comparison':24s} {'metric':11s} {'diff':>7s}  {'95% CI':>17s}  p")
    for r in ci_rows:
        print(f"{r[0]:8s} {r[1]:24s} {r[2]:11s} {r[4]:>7s}  [{r[5]:>6s}, {r[6]:>6s}]  {r[7]}")
    print("\nwrote seed_results.csv and paired_ci.csv")


if __name__ == '__main__':
    main()

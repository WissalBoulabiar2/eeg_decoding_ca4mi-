"""
Tables and paired statistics of the review-7 experiments (fixed total buffer of 180 trials, BCI-IV 2a), recomputed
from the saved matrices: K=1 factorial of logit consistency (LC) and balanced consolidation (CO) with update-matched
and past-only controls, shared vs subject-specific BN, independent confirmation on fresh seeds 18-27, subject-level
analysis and the multimodality diagnostic.

    python tools/review7_tables.py <unzipped review7 folder containing results7/> [<review6 folder with results6/>]
"""
import glob
import json
import os
import sys
import numpy as np
from scipy import stats

METRICS = ['ACC', 'BWT', 'F', 'F+', 'Fmax', 'F1', 'LA']
OLD, NEW = [8, 9, 10, 11, 12], list(range(18, 28))


def matrices(f):
    a = np.loadtxt(f)
    f1f = f.replace('acc_run_', 'f1_run_') if 'acc_run_' in f else f.replace('accuracy_matrix', 'f1_macro_matrix')
    return a, np.loadtxt(f1f)


def metrics(a, f1):
    n = len(a)
    d = np.array([a[u:n - 1, u].max() - a[-1, u] for u in range(n - 1)])
    return [a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (n - 1), d.mean(), np.maximum(d, 0).mean(), d.max(),
            f1[-1].mean(), np.diag(a).mean()]


def runs(root, block, name):
    """{seed: (metrics, final per-subject accuracies)} of one configuration."""
    b = os.path.join(root, block)
    pm = glob.glob(f"{b}/protomem/{name}/protomem_9_subjects_seed_*/accuracy_matrix.txt")
    if pm:
        fs = {int(os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1]): f for f in pm}
    elif name == 'ca4mi':
        fs = {int(os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1]): f
              for f in glob.glob(f"{b}/bench*/ca4mi/**/ca4mi_9_subjects_seed_*/accuracy_matrix.txt", recursive=True)}
    else:
        sub = 'bench_der30/der' if name == 'der30' else f'bench/{name}'
        fs = {int(os.path.basename(f).split('_')[2]): f
              for f in glob.glob(f"{b}/{sub}/**/acc_run_*_subject_9.txt", recursive=True)}
    out = {}
    for s, f in fs.items():
        a, f1 = matrices(f)
        out[s] = (metrics(a, f1), a[-1])
    return out


def sub(r, seeds):
    return {s: r[s] for s in seeds if s in r}


def row(label, r):
    v = np.array([r[s][0] for s in sorted(r)])
    print(f"  {label:24s} n={len(v):2d} " + " | ".join(f"{m} {v[:, k].mean():6.2f}±{v[:, k].std():5.2f}"
                                                   for k, m in enumerate(METRICS)))


def diff(a, b, k=0):
    s = sorted(set(a) & set(b))
    return np.array([a[i][0][k] - b[i][0][k] for i in s])


def ci(d):
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else 1.0
    return d.mean(), d.mean() - h, d.mean() + h, p


def paired(label, a, b, ks=(0, 1, 3)):
    out = []
    for k in ks:
        m, lo, hi, p = ci(diff(a, b, k))
        out.append(f"{METRICS[k]} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}] p={p:.3f}")
    print(f"  {label:34s} n={len(diff(a, b)):2d} " + " | ".join(out))
    return ci(diff(a, b, 0))[3]


def holm(ps):
    names, p = list(ps), np.array(list(ps.values()))
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return dict(zip(names, adj))


def subject_level(label, a, b):
    """Per-subject final-accuracy gain averaged over the shared seeds; Wilcoxon over the 9 subjects."""
    s = sorted(set(a) & set(b))
    g = np.mean([a[i][1] - b[i][1] for i in s], axis=0)
    w = stats.wilcoxon(g).pvalue if np.any(g != 0) else 1.0
    print(f"  {label:34s} seeds={len(s):2d} gains per subject: " + " ".join(f"{x:+5.1f}" for x in g)
          + f" | improved {int((g > 0).sum())}/9 | Wilcoxon p={w:.3f}")


R = os.path.join(sys.argv[1] if len(sys.argv) > 1 else '.', 'results7')
FAC = ['A0', 'A0+LC', 'A0+CO', 'A0+LC+CO', 'A0+LC+CO_unbal', 'A0+LC+CO_past', 'sel']
f = {n: runs(R, 'A_factorial', n) for n in FAC}
bn = {n: runs(R, 'B_bn', n) for n in ['sel_sharedBN', 'A0+LC+CO_sharedBN']}

for title, seeds in [("seeds 8-12", OLD), ("fresh seeds 18-27", NEW), ("all 15 seeds", OLD + NEW)]:
    print(f"\n=== A. K=1 factorial, fixed total buffer 180, {title} ===")
    for n in FAC:
        row(n, sub(f[n], seeds))
    F = {n: sub(f[n], seeds) for n in FAC}
    for x, y in [('A0+LC', 'A0'), ('A0+CO', 'A0'), ('A0+LC+CO', 'A0'), ('sel', 'A0'), ('A0+LC+CO', 'A0+LC'),
                 ('A0+LC+CO', 'A0+CO'), ('sel', 'A0+LC+CO'), ('A0+LC+CO', 'A0+LC+CO_unbal'),
                 ('A0+LC+CO', 'A0+LC+CO_past')]:
        paired(f"{x} - {y}", F[x], F[y])
    s = sorted(set.intersection(*[set(F[n]) for n in ['A0', 'A0+LC', 'A0+CO', 'A0+LC+CO']]))
    acc = {n: np.array([F[n][i][0][0] for i in s]) for n in ['A0', 'A0+LC', 'A0+CO', 'A0+LC+CO']}
    for lab, d in [("main effect LC", ((acc['A0+LC'] - acc['A0']) + (acc['A0+LC+CO'] - acc['A0+CO'])) / 2),
                   ("main effect CO", ((acc['A0+CO'] - acc['A0']) + (acc['A0+LC+CO'] - acc['A0+LC'])) / 2),
                   ("interaction LC x CO", (acc['A0+LC+CO'] - acc['A0+CO']) - (acc['A0+LC'] - acc['A0']))]:
        m, lo, hi, p = ci(d)
        print(f"  {lab:34s} ACC {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}] p={p:.3f}")

print("\n=== B. shared BN vs subject-specific BN, all 15 seeds ===")
for n, r in bn.items():
    row(n, r)
paired("sel - sel_sharedBN", f['sel'], bn['sel_sharedBN'])
paired("A0+LC+CO - A0+LC+CO_sharedBN", f['A0+LC+CO'], bn['A0+LC+CO_sharedBN'])

print("\n=== C. independent confirmation, fresh seeds 18-27 (never used for any decision) ===")
BASE = ['finetuning', 'ewc', 'er', 'der', 'der30', 'mudvi', 'cger', 'ca4mi']
c = {n: runs(R, 'C_confirm', n) for n in BASE + ['sel_1mode']}
c['A0'], c['A0+LC+CO'], c['sel'] = sub(f['A0'], NEW), sub(f['A0+LC+CO'], NEW), sub(f['sel'], NEW)
for n in BASE + ['A0', 'A0+LC+CO', 'sel_1mode', 'sel']:
    row(n, c[n])
ps = {n: paired(f"sel - {n}", c['sel'], c[n]) for n in ['A0', 'cger', 'mudvi', 'ca4mi', 'er']}
for n in ['A0+LC+CO', 'sel_1mode', 'der30']:
    paired(f"sel - {n}", c['sel'], c[n])
print("  Holm-adjusted ACC p (primary family): " + ", ".join(f"{n} {p:.3f}" for n, p in holm(ps).items()))

print("\n=== D. subject-level analysis (gain per subject, averaged over seeds) ===")
subject_level("sel - A0 (15 seeds)", f['sel'], f['A0'])
subject_level("A0+LC+CO - A0 (15 seeds)", f['A0+LC+CO'], f['A0'])
for n in ['cger', 'mudvi', 'ca4mi', 'er']:
    subject_level(f"sel - {n} (seeds 18-27)", c['sel'], c[n])

print("\n=== E. multimodality of the (subject, class) embeddings (sel, all seeds) ===")
recs = [x for g in glob.glob(f"{R}/A_factorial/protomem/sel/*/mode_diag.json") for x in json.load(open(g))]
if recs:
    sil = np.array([x['silhouette'] for x in recs]); nul = np.array([x['null_mean'] for x in recs])
    p = np.array([x['p'] for x in recs])
    print(f"  {len(recs)} (seed, subject, class) sets | silhouette real {sil.mean():.4f} vs single-Gaussian null "
          f"{nul.mean():.4f} | real > null in {np.mean(sil > nul) * 100:.0f}% | p<0.05 in {np.mean(p < 0.05) * 100:.0f}%")

if len(sys.argv) > 2:      # determinism: seeds 8-12 of sel re-run here vs the review-6 runs
    old = runs(os.path.join(sys.argv[2], 'results6'), 'A_main', 'sel')
    d = [abs(f['sel'][s][0][0] - old[s][0][0]) for s in OLD if s in old and s in f['sel']]
    print(f"\n  determinism check sel seeds 8-12 vs review6: max |dACC| = {max(d):.3f}")

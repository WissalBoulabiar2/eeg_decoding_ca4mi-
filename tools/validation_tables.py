"""
Independent validation of the simplified linear-head configuration on fresh seeds 38-47 (review 12), from the saved
accuracy matrices. Pre-specified primary tests (written to provenance12.txt before the runs): ProtoMem-linear minus
aligned CGER and ProtoMem-linear minus ProtoMem-cosine, Holm over these two. Everything else is secondary.

    python tools/validation_tables.py <dir with results12/>
"""
import glob
import os
import sys
import numpy as np
from scipy import stats

R = os.path.join(sys.argv[1], 'results12')
COLS = {'ACC': 0, 'BWT': 1, 'F+': 3, 'Fmax': 4, 'F1': 5, 'LA': 6}


def load(f):
    a = np.loadtxt(f)
    f1 = np.loadtxt(f.replace('acc_run_', 'f1_run_') if 'acc_run_' in f else f.replace('accuracy_matrix', 'f1_macro_matrix'))
    n = len(a)
    d = np.array([a[u:n - 1, u].max() - a[-1, u] for u in range(n - 1)])
    return [a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (n - 1), d.mean(), np.maximum(d, 0).mean(), d.max(),
            f1[-1].mean(), np.diag(a).mean()], a[-1]


def pm(name):
    return {int(os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1]): load(f)
            for f in glob.glob(f"{R}/{name}/protomem_9_subjects_seed_*/accuracy_matrix.txt")}


def bench(m):
    return {int(os.path.basename(f).split('_')[2]): load(f)
            for f in glob.glob(f"{R}/bench_ea/{m}/**/acc_run_*_subject_9.txt", recursive=True)}


def paired(a, b, k=0):
    s = sorted(set(a) & set(b))
    d = np.array([a[i][0][k] - b[i][0][k] for i in s])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = stats.ttest_1samp(d, 0).pvalue
    f = lambda x: f"{x:+.2f}".replace("-", "$-$")
    return f"{f(d.mean())} [{f(d.mean() - h)}, {f(d.mean() + h)}], " + ("$p<0.001$" if p < 0.001 else f"$p={p:.3f}$"), p


def holm(ps):
    names, p = list(ps), np.array(list(ps.values()))
    adj, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return dict(zip(names, adj))


def subjects(a, b):
    s = sorted(set(a) & set(b))
    g = np.mean([a[i][1] - b[i][1] for i in s], axis=0)
    return f"subjects improved {(g > 0).sum()}/9, Wilcoxon p={stats.wilcoxon(g).pvalue:.3f}"


runs = {'ProtoMem, linear head': pm('protomem_linear'), 'ProtoMem, cosine head': pm('protomem_cosine'),
        'CGER + alignment': bench('cger'), 'MUDVI + alignment': bench('mudvi')}
print("% ===== fresh seeds 38-47: mean +- SD (ACC BWT F+ Fmax F1 LA)")
for lab, r in runs.items():
    v = np.array([r[s][0] for s in sorted(r)])
    print(f"{lab:22s} & " + " & ".join(f"{v[:, k].mean():.2f} $\\pm$ {v[:, k].std():.2f}" for k in COLS.values())
          + f" \\\\ % n={len(v)}")

L, C = runs['ProtoMem, linear head'], runs['ProtoMem, cosine head']
prim = {'linear - CGER+EA': (L, runs['CGER + alignment']), 'linear - cosine': (L, C)}
adj = holm({k: paired(a, b)[1] for k, (a, b) in prim.items()})
print("\n% ===== PRIMARY (pre-specified), Holm over 2")
for k, (a, b) in prim.items():
    print(f"% {k}: ACC {paired(a, b)[0]} | Holm {adj[k]:.3f} | BWT {paired(a, b, 1)[0]} | "
          f"F+ {paired(a, b, 3)[0]} | {subjects(a, b)}")
print("\n% ===== secondary (exploratory)")
for k, (a, b) in {'linear - MUDVI+EA': (L, runs['MUDVI + alignment']),
                  'cosine - CGER+EA': (C, runs['CGER + alignment']),
                  'cosine - MUDVI+EA': (C, runs['MUDVI + alignment'])}.items():
    print(f"% {k}: ACC {paired(a, b)[0]} | BWT {paired(a, b, 1)[0]} | F+ {paired(a, b, 3)[0]} | {subjects(a, b)}")
for lab, r in runs.items():
    mins = [r[s][1].min() for s in sorted(r)]
    print(f"% {lab}: worst-served user mean {np.mean(mins):.2f} (min {min(mins):.1f}); "
          f"Fmax mean {np.mean([r[s][0][4] for s in r]):.2f} (max {max(r[s][0][4] for s in r):.1f})")

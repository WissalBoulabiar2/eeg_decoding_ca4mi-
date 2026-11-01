"""
Review-15 analyses: High-Gamma (14 subjects, seeds 28-37) with the checkpoint-fixed aligned ER / MUDVI and the
matched-regime aligned ER / MUDVI / CGER, and the fixed aligned MUDVI on the validation seeds 38-47.

    python tools/review15_tables.py <dir results15> <dir results13> <dir results12>
"""
import glob
import os
import re
import sys
import numpy as np
from scipy import stats

R15, R13, R12 = (os.path.join(d, f'results{k}') for d, k in zip(sys.argv[1:4], (15, 13, 12)))
NAMES = ['ACC', 'BWT', 'F+', 'Fmax', 'LA']


def metrics(f):
    a = np.loadtxt(f)
    n = len(a)
    d = np.array([a[u:n - 1, u].max() - a[-1, u] for u in range(n - 1)])
    return np.array([a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (n - 1), np.maximum(d, 0).mean(), d.max(),
                     np.diag(a).mean()]), a[-1]


def load(path, seeds, nsub):
    out = {}
    for f in glob.glob(f"{path}/**/*.txt", recursive=True):
        m = (re.search(rf"_{nsub}_subjects_seed_(\d+)[/\\]accuracy_matrix\.txt$", f)
             or re.search(rf"acc_run_(\d+)_subject_{nsub}\.txt$", f))
        if m and int(m.group(1)) in seeds:
            out[int(m.group(1))] = metrics(f)
    return out


def row(lab, r):
    v = np.array([r[s][0] for s in sorted(r)])
    print(f"  {lab:30s} n={len(v):2d} " + " | ".join(f"{k} {v[:, i].mean():6.2f}±{v[:, i].std():5.2f}"
                                                   for i, k in enumerate(NAMES)))


def diff(a, b, k=0):
    s = sorted(set(a) & set(b))
    d = np.array([a[i][0][k] - b[i][0][k] for i in s])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d, d.mean(), d.mean() - h, d.mean() + h, stats.ttest_1samp(d, 0).pvalue


def paired(lab, a, b, ks=(0, 1, 2)):
    out = []
    for k in ks:
        _, m, lo, hi, p = diff(a, b, k)
        out.append(f"{NAMES[k]} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}] p={p:.3f}")
    s = sorted(set(a) & set(b))
    g = np.mean([a[i][1] - b[i][1] for i in s], axis=0)
    print(f"  {lab:38s} n={len(s):2d} " + " | ".join(out) + f" | subjects {(g > 0).sum()}/{len(g)} "
          f"W p={stats.wilcoxon(g).pvalue:.3f}")
    return diff(a, b)[4]


def holm(ps):
    names, p = list(ps), np.array(list(ps.values()))
    adj, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return ", ".join(f"{n} {x:.3f}" for n, x in zip(names, adj))


def tost(a, b, margin):
    d = diff(a, b)[0]
    return max(stats.ttest_1samp(d + margin, 0, alternative='greater').pvalue,
               stats.ttest_1samp(d - margin, 0, alternative='less').pvalue)


F = range(28, 38)
print("=== High-Gamma, 14 subjects, budget 280, seeds 28-37 ===")
H = {'ProtoMem': load(f"{R13}/P2_hgd/protomem/protomem", F, 14),
     'Replay (shared BN)': load(f"{R13}/P2_hgd/protomem/replay", F, 14),
     'CGER + EA': load(f"{R13}/P2_hgd/bench_ea/cger", F, 14),
     'MUDVI + EA (fixed)': load(f"{R15}/R1_hgd/bench_fix_ea/mudvi", F, 14),
     'ER + EA (fixed)': load(f"{R15}/R1_hgd/bench_fix_ea/er", F, 14),
     'CGER + EA, matched': load(f"{R15}/R1_hgd/bench_matched_ea/cger", F, 14),
     'MUDVI + EA, matched': load(f"{R15}/R1_hgd/bench_matched_ea/mudvi", F, 14),
     'ER + EA, matched': load(f"{R15}/R1_hgd/bench_matched_ea/er", F, 14),
     'MUDVI + EA (bug)': load(f"{R13}/P2_hgd/bench_ea/mudvi", F, 14),
     'ER + EA (bug)': load(f"{R13}/P2_hgd/bench_ea/er", F, 14)}
for lab, r in H.items():
    row(lab, r)
P = H['ProtoMem']
print("\n  checkpoint fix (fixed - bug):")
for m in ['MUDVI', 'ER']:
    paired(f"{m} + EA fixed - bug", H[f"{m} + EA (fixed)"], H[f"{m} + EA (bug)"])
print("\n  family as on 2a cross-session (Holm over 7):")
fam = ['CGER + EA', 'MUDVI + EA (fixed)', 'ER + EA (fixed)', 'CGER + EA, matched', 'MUDVI + EA, matched',
       'ER + EA, matched', 'Replay (shared BN)']
print("  Holm: " + holm({m: paired(f"ProtoMem - {m}", P, H[m]) for m in fam}))
for m in ['CGER', 'MUDVI', 'ER']:
    ref = 'CGER + EA' if m == 'CGER' else f"{m} + EA (fixed)"
    paired(f"{m} + EA: matched - reference", H[f"{m} + EA, matched"], H[ref])
for mg in (2, 3):
    print(f"  TOST ProtoMem vs CGER + EA, matched, +-{mg}: p={tost(P, H['CGER + EA, matched'], mg):.3f}")

V = range(38, 48)
print("\n=== validation seeds 38-47: aligned MUDVI with the fix ===")
Vd = {'ProtoMem, cosine head': load(f"{R12}/protomem_cosine", V, 9),
      'ProtoMem, linear head': load(f"{R12}/protomem_linear", V, 9),
      'MUDVI + EA (fixed)': load(f"{R15}/R2_valid/bench_fix_ea/mudvi", V, 9),
      'MUDVI + EA (bug)': load(f"{R12}/bench_ea/mudvi", V, 9)}
for lab, r in Vd.items():
    row(lab, r)
paired("MUDVI + EA fixed - bug", Vd['MUDVI + EA (fixed)'], Vd['MUDVI + EA (bug)'])
paired("cosine - MUDVI + EA (fixed)", Vd['ProtoMem, cosine head'], Vd['MUDVI + EA (fixed)'])
paired("linear - MUDVI + EA (fixed)", Vd['ProtoMem, linear head'], Vd['MUDVI + EA (fixed)'])

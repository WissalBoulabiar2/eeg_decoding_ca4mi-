"""
Review-13 analyses from the saved accuracy matrices: isolated / joint references and the waterfall from Replay (shared BN)
towards ER settings on 2a (seeds 28-37), High-Gamma with aligned baselines (seeds 28-37), small buffer budgets, and
aligned baselines on the cross-session and four-class protocols (seeds 8-12).

    python tools/review13_tables.py <dir results13> <dir results8> <dir results10>
"""
import glob
import os
import re
import sys
import numpy as np
from scipy import stats

R13, R8, R10 = (os.path.join(d, f'results{k}') for d, k in zip(sys.argv[1:4], (13, 8, 10)))
F, OLD = list(range(28, 38)), [8, 9, 10, 11, 12]
NAMES = ['ACC', 'BWT', 'F+', 'Fmax', 'LA']


def metrics(f):
    a = np.loadtxt(f)
    n = len(a)
    d = np.array([a[u:n - 1, u].max() - a[-1, u] for u in range(n - 1)])
    return np.array([a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (n - 1), np.maximum(d, 0).mean(), d.max(),
                     np.diag(a).mean()]), a[-1]


def pm(path, nsub=9, seeds=None):
    out = {}
    for f in glob.glob(f"{path}/protomem_{nsub}_subjects_seed_*/accuracy_matrix.txt"):
        s = int(re.search(r"seed_(\d+)", f).group(1))
        if seeds is None or s in seeds:
            out[s] = metrics(f)
    return out


def bench(path, nsub=9, seeds=None):
    out = {}
    for f in glob.glob(f"{path}/**/acc_run_*_subject_{nsub}.txt", recursive=True):
        s = int(re.search(r"acc_run_(\d+)_", f).group(1))
        if seeds is None or s in seeds:
            out[s] = metrics(f)
    return out


def row(lab, r):
    v = np.array([r[s][0] for s in sorted(r)])
    print(f"  {lab:28s} n={len(v):2d} " + " | ".join(f"{k} {v[:, i].mean():6.2f}±{v[:, i].std():5.2f}"
                                                   for i, k in enumerate(NAMES)))


def diff(a, b, k=0):
    s = sorted(set(a) & set(b))
    d = np.array([a[i][0][k] - b[i][0][k] for i in s])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else 1.0
    return d, d.mean(), d.mean() - h, d.mean() + h, p


def paired(lab, a, b, ks=(0, 1, 2)):
    out = []
    for k in ks:
        _, m, lo, hi, p = diff(a, b, k)
        out.append(f"{NAMES[k]} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}] p={p:.3f}")
    s = sorted(set(a) & set(b))
    g = np.mean([a[i][1] - b[i][1] for i in s], axis=0)
    w = stats.wilcoxon(g).pvalue if np.any(g != 0) else 1.0
    print(f"  {lab:40s} " + " | ".join(out) + f" | subjects {(g > 0).sum()}/{len(g)} W p={w:.3f}")
    return diff(a, b)[4]


def holm(ps):
    names, p = list(ps), np.array(list(ps.values()))
    adj, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return ", ".join(f"{n} {x:.3f}" for n, x in zip(names, adj))


def tost(a, b, margin):
    d = diff(a, b)[0]
    p_lo = stats.ttest_1samp(d + margin, 0, alternative='greater').pvalue
    p_hi = stats.ttest_1samp(d - margin, 0, alternative='less').pvalue
    return max(p_lo, p_hi)


proto = pm(f"{R8}/H_confirm/protomem/chosen", seeds=F)
replay = pm(f"{R8}/H_confirm/protomem/A0", seeds=F)
er_ea = bench(f"{R10}/bench_ea/er", seeds=F)
cger_ea = bench(f"{R10}/bench_ea/cger", seeds=F)

print("\n=== P1. 2a references (seeds 28-37) ===")
iso, joint = pm(f"{R13}/P1_2a/protomem/isolated", seeds=F), pm(f"{R13}/P1_2a/protomem/joint", seeds=F)
for lab, r in [('ProtoMem', proto), ('Replay (shared BN)', replay), ('isolated per-subject models', iso),
               ('joint training (upper bound)', joint), ('CGER + alignment', cger_ea)]:
    row(lab, r)
paired("ProtoMem - isolated", proto, iso, ks=(0,))
paired("ProtoMem - joint", proto, joint, ks=(0,))
paired("isolated - joint", iso, joint, ks=(0,))
paired("ProtoMem LA - isolated LA", proto, iso, ks=(4,))

print("\n=== P1. waterfall from Replay (shared BN) towards ER settings (seeds 28-37, cumulative) ===")
steps = [('Replay (shared BN)', replay)] + [(n, pm(f"{R13}/P1_2a/protomem/{n}", seeds=F)) for n in
                                            ['W1_linear', 'W2_er_optim', 'W3_er_stop', 'W4_reservoir', 'W5_separate']]
steps.append(('ER + alignment (actual)', er_ea))
for lab, r in steps:
    row(lab, r)
for (la, a), (lb, b) in zip(steps[:-1], steps[1:]):
    paired(f"{lb} - {la}", b, a, ks=(0, 1))

print("\n=== P2. High-Gamma, 14 subjects, budget 280 (seeds 28-37) ===")
H = {n: pm(f"{R13}/P2_hgd/protomem/{n}", 14, F) for n in ['protomem', 'replay', 'isolated', 'joint']}
H.update({f"{m}+EA": bench(f"{R13}/P2_hgd/bench_ea/{m}", 14, F) for m in ['cger', 'mudvi', 'er']})
for n, r in H.items():
    row(n, r)
ps = {n: paired(f"ProtoMem - {n}", H['protomem'], H[n]) for n in ['cger+EA', 'mudvi+EA', 'er+EA', 'replay']}
print("  Holm (4 comparisons): " + holm(ps))
for mgn in (2, 3):
    print(f"  TOST ProtoMem vs CGER+EA, margin +-{mgn}: p={tost(H['protomem'], H['cger+EA'], mgn):.3f}")
paired("ProtoMem - isolated", H['protomem'], H['isolated'], ks=(0,))
paired("ProtoMem - joint", H['protomem'], H['joint'], ks=(0,))

print("\n=== P3. buffer budgets on 2a (seeds 28-37): ProtoMem vs CGER + alignment ===")
for B in (18, 36, 90):
    p, c = pm(f"{R13}/P3_budget{B}/protomem/protomem", seeds=F), bench(f"{R13}/P3_budget{B}/bench_ea/cger", seeds=F)
    row(f"ProtoMem, {B} trials", p); row(f"CGER + alignment, {B} trials", c)
    paired(f"ProtoMem - CGER+EA ({B})", p, c)
row("ProtoMem, 180 trials", proto); row("CGER + alignment, 180 trials", cger_ea)

print("\n=== P4. secondary protocols, aligned baselines (seeds 8-12) ===")
xs = pm(f"{R8}/D_xsess/protomem/chosen", seeds=OLD)
row("ProtoMem cross-session", xs)
for m in ['cger', 'mudvi', 'er']:
    r = bench(f"{R13}/P4_xsess/bench_ea/{m}", seeds=OLD)
    row(f"{m}+EA cross-session", r)
    paired(f"ProtoMem - {m}+EA (cross-session)", xs, r)
fc = pm(f"{R8}/F_4class/protomem/chosen", seeds=OLD)
c4 = bench(f"{R13}/P4_4class/bench_ea/cger", seeds=OLD)
row("ProtoMem four-class", fc); row("cger+EA four-class", c4)
paired("ProtoMem - cger+EA (four-class)", fc, c4)

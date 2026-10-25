"""
Tables and paired statistics of the review-8 experiments (shared BN, K=1, fixed total buffer), recomputed from the
saved matrices, with the baselines of review 6 (seeds 8-12, all protocols) and review 7 (fresh seeds 18-27).

    python tools/review8_tables.py <dir with results8/> <dir with results7/> <dir with results6/>
"""
import glob
import json
import os
import sys
import numpy as np
from scipy import stats

METRICS = ['ACC', 'BWT', 'F', 'F+', 'Fmax', 'F1', 'LA']
OLD, NEW, FRESH = [8, 9, 10, 11, 12], list(range(18, 28)), list(range(28, 38))
R8, R7, R6 = (os.path.join(d, f'results{k}') for d, k in zip(sys.argv[1:4], (8, 7, 6)))


def load(f):
    a = np.loadtxt(f)
    f1 = np.loadtxt(f.replace('acc_run_', 'f1_run_') if 'acc_run_' in f else f.replace('accuracy_matrix', 'f1_macro_matrix'))
    n = len(a)
    d = np.array([a[u:n - 1, u].max() - a[-1, u] for u in range(n - 1)])
    m = [a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (n - 1), d.mean(), np.maximum(d, 0).mean(), d.max(),
         f1[-1].mean(), np.diag(a).mean()]
    return m, a[-1]


def runs(root, block, name, nsub=9, bench='bench'):
    """{seed: (metrics, final per-subject accuracies)}; ProtoMem variants, CA4MI or a benchmark baseline."""
    b = os.path.join(root, block)
    pm = glob.glob(f"{b}/protomem/{name}/protomem_{nsub}_subjects_seed_*/accuracy_matrix.txt")
    if pm:
        fs = {int(os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1]): f for f in pm}
    elif name == 'ca4mi':
        fs = {int(os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1]): f for f in
              glob.glob(f"{b}/{bench}/ca4mi/**/ca4mi_{nsub}_subjects_seed_*/accuracy_matrix.txt", recursive=True)}
    else:
        fs = {int(os.path.basename(f).split('_')[2]): f for f in
              glob.glob(f"{b}/{bench}/{name}/**/acc_run_*_subject_{nsub}.txt", recursive=True)}
    return {s: load(f) for s, f in fs.items()}


def sub(r, seeds):
    return {s: r[s] for s in seeds if s in r}


def row(label, r):
    if not r:
        print(f"  {label:24s} (no runs)"); return
    v = np.array([r[s][0] for s in sorted(r)])
    print(f"  {label:24s} n={len(v):2d} " + " | ".join(f"{m} {v[:, k].mean():6.2f}±{v[:, k].std():5.2f}"
                                                   for k, m in enumerate(METRICS)))


def ci(d):
    if len(d) < 2:
        return float('nan'), float('nan'), float('nan'), float('nan')
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    p = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else 1.0
    return d.mean(), d.mean() - h, d.mean() + h, p


def paired(label, a, b, ks=(0, 1, 3)):
    s = sorted(set(a) & set(b))
    out = []
    for k in ks:
        m, lo, hi, p = ci(np.array([a[i][0][k] - b[i][0][k] for i in s]))
        out.append(f"{METRICS[k]} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}] p={p:.3f}")
    print(f"  {label:34s} n={len(s):2d} " + " | ".join(out))
    return ci(np.array([a[i][0][0] - b[i][0][0] for i in s]))[3]


def holm(ps):
    names, p = list(ps), np.array(list(ps.values()))
    adj, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return ", ".join(f"{n} {x:.3f}" for n, x in zip(names, adj))


def subject_level(label, a, b):
    s = sorted(set(a) & set(b))
    g = np.mean([a[i][1] - b[i][1] for i in s], axis=0)
    w = stats.wilcoxon(g).pvalue if np.any(g != 0) else 1.0
    print(f"  {label:34s} seeds={len(s):2d} per subject: " + " ".join(f"{x:+5.1f}" for x in g)
          + f" | improved {int((g > 0).sum())}/{len(g)} | Wilcoxon p={w:.3f}")


chosen = open(os.path.join(sys.argv[1], 'chosen8.txt')).read().split()[0]
print(f"configuration chosen on validation (seeds 8-12): {chosen}")
print(open(os.path.join(sys.argv[1], 'chosen8.txt')).read())

CAND = ['A0', 'A0+LC', 'A0+CO', 'A0+LC+CO', 'sel', 'sel-gauss', 'sel-drift', 'sel_1mode', 'sel_meanonly', 'sel_ward']
P = {n: runs(R8, 'P1_shared', n) for n in CAND}
for title, seeds in [("seeds 8-12", OLD), ("seeds 18-27", NEW), ("all 15 seeds", OLD + NEW)]:
    print(f"\n=== A. shared BN, K=1, budget 180: factorial and ablations, {title} ===")
    for n in CAND:
        row(n, sub(P[n], seeds))
    F = {n: sub(P[n], seeds) for n in CAND}
    for n in CAND[1:]:
        paired(f"{n} - A0", F[n], F['A0'])
    for n in ['sel', 'sel-gauss', 'sel-drift', 'sel_1mode', 'sel_meanonly', 'sel_ward']:
        paired(f"sel - {n}" if n != 'sel' else "sel - A0+LC+CO", F['sel'], F[n] if n != 'sel' else F['A0+LC+CO'])
    s = sorted(set.intersection(*[set(F[n]) for n in ['A0', 'A0+LC', 'A0+CO', 'A0+LC+CO']]))
    acc = {n: np.array([F[n][i][0][0] for i in s]) for n in ['A0', 'A0+LC', 'A0+CO', 'A0+LC+CO']}
    for lab, d in [("main effect LC", ((acc['A0+LC'] - acc['A0']) + (acc['A0+LC+CO'] - acc['A0+CO'])) / 2),
                   ("main effect CO", ((acc['A0+CO'] - acc['A0']) + (acc['A0+LC+CO'] - acc['A0+LC'])) / 2),
                   ("interaction LC x CO", (acc['A0+LC+CO'] - acc['A0+CO']) - (acc['A0+LC'] - acc['A0']))]:
        m, lo, hi, p = ci(d)
        print(f"  {lab:34s} ACC {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}] p={p:.3f}")

BASE = ['finetuning', 'ewc', 'er', 'der', 'mudvi', 'cger', 'ca4mi']
CH, A0 = P[chosen], P['A0']
print("\n=== B. chosen vs baselines, 2a within-session, budget 180 ===")
for title, seeds, bl in [("seeds 8-12 (baselines: review 6)", OLD, {m: runs(R6, 'A_main', m) for m in BASE}),
                         ("seeds 18-27 (baselines: review 7)", NEW, {m: runs(R7, 'C_confirm', m) for m in BASE}),
                         ("FRESH seeds 28-37 (phase 3)", FRESH, {m: runs(R8, 'H_confirm', m) for m in BASE})]:
    print(f" --- {title}")
    c = sub(runs(R8, 'H_confirm', 'chosen') if seeds == FRESH else CH, seeds)
    a = sub(runs(R8, 'H_confirm', 'A0') if seeds == FRESH else A0, seeds)
    row(f"chosen ({chosen})", c); row("A0 (shared BN)", a)
    for m in BASE:
        row(m, sub(bl[m], seeds))
    ps = {'A0': paired("chosen - A0", c, a)}
    for m in BASE:
        if bl[m]:
            ps[m] = paired(f"chosen - {m}", c, sub(bl[m], seeds))
    prim = {k: v for k, v in ps.items() if k in ('A0', 'cger', 'mudvi', 'ca4mi', 'er') and not np.isnan(v)}
    print("  Holm-adjusted ACC p (primary family): " + holm(prim))
    for m in ['ca4mi', 'cger', 'mudvi']:
        if bl[m]:
            subject_level(f"chosen - {m}", c, sub(bl[m], seeds))
    subject_level("chosen - A0", c, a)

print("\n=== C. other protocols, seeds 8-12 (baselines from review 6) ===")
for title, blk, nsub, bases in [("cross-session (session E test)", 'D_xsess', 9, BASE + ['sel', 'A0']),
                                ("High-Gamma, 14 subjects, budget 280", 'E_hgd14', 14, BASE + ['sel', 'A0']),
                                ("4-class, budget 180", 'F_4class', 9, ['cger', 'ca4mi', 'sel', 'A0']),
                                ("budget 300", 'C_fix300', 9, ['cger', 'sel', 'A0']),
                                ("budget 150", 'C_fix150', 9, ['cger', 'sel', 'A0'])]:
    print(f" --- {title}")
    c, a = runs(R8, blk, 'chosen', nsub), runs(R8, blk, 'A0', nsub)
    row(f"chosen ({chosen})", c); row("A0 (shared BN)", a)
    paired("chosen - A0 (shared BN)", c, a)
    for m in bases:
        r = runs(R6, blk, m, nsub, bench='bench*')
        lab = {'sel': 'ProtoMem subject BN (rev6)', 'A0': 'A0 subject BN, K=3 (rev6)'}.get(m, m)
        row(lab, r)
        paired(f"chosen - {lab}", c, r)

print("\n=== D. subject orders (fixed splits), budget 180, seeds 8-12 ===")
row("chosen sequential", sub(CH, OLD)); row("A0 sequential", sub(A0, OLD))
paired("chosen - A0 (sequential)", sub(CH, OLD), sub(A0, OLD))
for o in ('rev', 'p1', 'p2'):
    c, a = runs(R8, 'G_order', f'chosen_{o}'), runs(R8, 'G_order', f'A0_{o}')
    row(f"chosen {o}", c); row(f"A0 {o}", a)
    paired(f"chosen - A0 ({o})", c, a)

if len(sys.argv) > 4:      # review 9: Euclidean-alignment control of the chosen configuration, seeds 18-37
    R9 = os.path.join(sys.argv[4], 'results9')
    noea = {}
    for f in glob.glob(f"{R9}/chosen_noEA/protomem_9_subjects_seed_*/accuracy_matrix.txt"):
        noea[int(os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1])] = load(f)
    ch20 = {**sub(CH, NEW), **runs(R8, 'H_confirm', 'chosen')}
    print("\n=== F. Euclidean-alignment control, pooled seeds 18-37 ===")
    row(f"chosen ({chosen})", ch20); row("chosen without EA", noea)
    paired("chosen - chosen without EA", ch20, noea)
    subject_level("chosen - chosen without EA", ch20, noea)
    for m in ['er', 'mudvi', 'cger', 'ca4mi']:
        bl = {**runs(R7, 'C_confirm', m), **runs(R8, 'H_confirm', m)}
        paired(f"chosen without EA - {m}", noea, bl)

print("\n=== E. multimodality of the (subject, class) embeddings, full-covariance null ===")
for n in [chosen, 'sel']:
    recs = [x for g in glob.glob(f"{R8}/P1_shared/protomem/{n}/*/mode_diag.json") for x in json.load(open(g))]
    if recs and 'p_full' in recs[0]:
        sil = np.array([x['silhouette'] for x in recs])
        fn = np.array([x['null_full_mean'] for x in recs]); pf = np.array([x['p_full'] for x in recs])
        print(f"  {n:10s} {len(recs)} sets | silhouette real {sil.mean():.3f} vs full-cov null {fn.mean():.3f} | "
              f"real > null in {np.mean(sil > fn) * 100:.0f}% | p_full<0.05 in {np.mean(pf < 0.05) * 100:.0f}% "
              f"(expected by chance: 5%)")

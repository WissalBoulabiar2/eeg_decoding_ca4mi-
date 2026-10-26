"""
LaTeX table bodies of the paper, generated from the saved accuracy matrices (no number is typed by hand).

    python tools/paper_tables.py <dir results8> <dir results7> <dir results6> <dir results9> > tables.tex
Primary comparison: pooled seeds 18-37 (not used for the configuration choice, which used validation of seeds 8-12).
"""
import contextlib
import glob
import io
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    import review8_tables as T          # noqa: E402  (helpers; its own printout is discarded)

R8, R7, R6 = T.R8, T.R7, T.R6
R9 = os.path.join(sys.argv[4], 'results9')
M = {k: i for i, k in enumerate(T.METRICS)}
POOL = T.NEW + T.FRESH
NAMES = {'finetuning': 'Fine-tuning', 'ewc': 'EWC', 'er': 'ER', 'der': 'DER', 'mudvi': 'MUDVI', 'cger': 'CGER',
         'ca4mi': 'CA4MI', 'A0': 'Replay (shared BN)', 'chosen': 'ProtoMem (ours)'}
BASE = ['finetuning', 'ewc', 'er', 'der', 'mudvi', 'cger', 'ca4mi']


def ms(r, k):
    v = np.array([r[s][0][M[k]] for s in sorted(r)])
    return v.mean(), v.std()


def cell(r, k, best=None):
    m, s = ms(r, k)
    txt = f"{m:.2f} $\\pm$ {s:.2f}".replace("-", "$-$", 1) if m < 0 else f"{m:.2f} $\\pm$ {s:.2f}"
    return f"\\textbf{{{txt}}}" if best is not None and abs(m - best) < 1e-9 else txt


def diff(a, b, k):
    s = sorted(set(a) & set(b))
    return T.ci(np.array([a[i][0][M[k]] - b[i][0][M[k]] for i in s])), len(s)


def dtxt(a, b, k):
    (m, lo, hi, p), _ = diff(a, b, k)
    f = lambda x: f"{x:+.2f}".replace("-", "$-$")
    return f"{f(m)} [{f(lo)}, {f(hi)}], $p={p:.3f}$" if p >= 0.001 else f"{f(m)} [{f(lo)}, {f(hi)}], $p<0.001$"


def holm(ps):
    names, p = list(ps), np.array(list(ps.values()))
    adj, run = np.empty(len(p)), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j])); adj[j] = run
    return dict(zip(names, adj))


def bold_rows(rows, cols, higher):
    best = {}
    for k in cols:
        vals = [ms(r, k)[0] for r in rows.values() if r]
        best[k] = (max(vals) if higher[k] else min(vals))
    return best


# ---------------------------------------------------------------- data
main = {m: {**T.runs(R7, 'C_confirm', m), **T.runs(R8, 'H_confirm', m)} for m in BASE}
main['A0'] = {**T.sub(T.P['A0'], T.NEW), **T.runs(R8, 'H_confirm', 'A0')}
main['chosen'] = {**T.sub(T.P['A0+CO'], T.NEW), **T.runs(R8, 'H_confirm', 'chosen')}
hgd = {m: T.runs(R6, 'E_hgd14', m, 14, bench='bench*') for m in BASE}
hgd['A0'], hgd['chosen'] = T.runs(R8, 'E_hgd14', 'A0', 14), T.runs(R8, 'E_hgd14', 'chosen', 14)

print("% ===== Table main: 2a pooled seeds 18-37 | HGD seeds 8-12")
c2a, chg = ['ACC', 'BWT', 'F+', 'F1', 'LA'], ['ACC', 'BWT', 'F+', 'F1']
hi = {'ACC': True, 'BWT': True, 'F+': False, 'F1': True, 'LA': True}
b2a = bold_rows({m: main[m] for m in main if m != 'der'}, c2a, hi)
bhg = bold_rows({m: hgd[m] for m in hgd if m != 'der'}, chg, hi)
for m in BASE + ['A0', 'chosen']:
    print(f"{NAMES[m]:20s} & " + " & ".join(cell(main[m], k, b2a[k] if m != 'der' else None) for k in c2a)
          + f" % n={len(main[m])}\n{'':20s} & " + " & ".join(cell(hgd[m], k, bhg[k] if m != 'der' else None) for k in chg)
          + f" \\\\ % n={len(hgd[m])}")

print("\n% ===== Table retention: 2a pooled seeds 18-37: F, F+, Fmax")
for m in BASE + ['A0', 'chosen']:
    print(f"{NAMES[m]:20s} & " + " & ".join(f"${ms(main[m], k)[0]:.2f} \\pm {ms(main[m], k)[1]:.2f}$"
                                            for k in ['F', 'F+', 'Fmax']) + " \\\\")

print("\n% ===== Table paired: ProtoMem minus method, pooled seeds 18-37, Holm on ACC")
fam = ['A0', 'cger', 'mudvi', 'ca4mi', 'er']
adj = holm({m: diff(main['chosen'], main[m], 'ACC')[0][3] for m in fam})
for m in fam:
    print(f"ProtoMem $-$ {NAMES[m]:18s} & " + " & ".join(dtxt(main['chosen'], main[m], k) for k in ['ACC', 'BWT', 'F+'])
          + f" & {adj[m]:.3f} \\\\ % n={diff(main['chosen'], main[m], 'ACC')[1]}")
for m in ['ca4mi', 'cger', 'mudvi', 'er', 'A0']:
    s = sorted(set(main['chosen']) & set(main[m]))
    g = np.mean([main['chosen'][i][1] - main[m][i][1] for i in s], axis=0)
    from scipy import stats
    print(f"% subject level vs {m}: improved {(g > 0).sum()}/9, Wilcoxon p={stats.wilcoxon(g).pvalue:.3f}, "
          f"gains " + " ".join(f"{x:+.1f}" for x in g))

print("\n% ===== Table ablation: factorial, 15 seeds (8-12, 18-27); shared-BN panel with validation ACC (seeds 8-12)")
val = dict(l.split() for l in open(os.path.join(sys.argv[1], 'chosen8.txt')).read().splitlines()[1:] if l.strip())
SUBJ = {n: T.runs(R7, 'A_factorial', n) for n in ['A0', 'A0+LC', 'A0+CO', 'A0+LC+CO', 'A0+LC+CO_unbal',
                                                    'A0+LC+CO_past', 'sel']}
LAB = {'A0': 'Replay', 'A0+LC': '+ logit consistency (LC)', 'A0+CO': '+ balanced consolidation (CO)',
       'A0+LC+CO': '+ LC + CO', 'A0+LC+CO_unbal': '+ LC + CO, update-matched (unbalanced)',
       'A0+LC+CO_past': '+ LC + CO, past exemplars only', 'sel': '+ LC + CO + Gaussian memory + drift',
       'sel-gauss': '\\quad without Gaussian replay', 'sel-drift': '\\quad without drift correction',
       'sel_1mode': '\\quad one mode per class', 'sel_meanonly': '\\quad mean-only replay',
       'sel_ward': '\\quad Ward merging'}
for panel, D, has_val in [("subject-specific BN", SUBJ, False), ("shared BN", T.P, True)]:
    print(f"% --- {panel}")
    for n in (SUBJ if not has_val else ['A0', 'A0+LC', 'A0+CO', 'A0+LC+CO', 'sel', 'sel-gauss', 'sel-drift',
                                         'sel_1mode', 'sel_meanonly', 'sel_ward']):
        r = D[n]
        ref = 'A0' if not n.startswith('sel') or n == 'sel' else 'sel'
        dd = "--" if n == 'A0' else dtxt(r, D[ref], 'ACC') + (" (vs.\\ full)" if ref == 'sel' and n != 'sel' else "")
        v = f"{float(val[n]):.2f} & " if has_val else ""
        print(f"{LAB[n]:42s} & {v}" + " & ".join(f"${ms(r, k)[0]:.2f} \\pm {ms(r, k)[1]:.2f}$" for k in
                                                 ['ACC', 'BWT', 'F+', 'LA']) + f" & {dd} \\\\ % n={len(r)}")
    s = sorted(set.intersection(*[set(D[n]) for n in ['A0', 'A0+LC', 'A0+CO', 'A0+LC+CO']]))
    a = {n: np.array([D[n][i][0][0] for i in s]) for n in ['A0', 'A0+LC', 'A0+CO', 'A0+LC+CO']}
    for lab, d in [("LC", ((a['A0+LC'] - a['A0']) + (a['A0+LC+CO'] - a['A0+CO'])) / 2),
                   ("CO", ((a['A0+CO'] - a['A0']) + (a['A0+LC+CO'] - a['A0+LC'])) / 2),
                   ("LCxCO", (a['A0+LC+CO'] - a['A0+CO']) - (a['A0+LC'] - a['A0']))]:
        m, lo, hi_, p = T.ci(d)
        print(f"% main effect {lab}: {m:+.2f} [{lo:+.2f}, {hi_:+.2f}] p={p:.3f}")
    if not has_val:
        print(f"% CO balanced vs update-matched: ACC {dtxt(D['A0+LC+CO'], D['A0+LC+CO_unbal'], 'ACC')} | "
              f"BWT {dtxt(D['A0+LC+CO'], D['A0+LC+CO_unbal'], 'BWT')}")

print("\n% ===== Table controls")
sh15 = T.P['A0+CO']
print(f"shared vs subject BN (Replay + CO, 15 seeds): ACC {dtxt(sh15, SUBJ['A0+CO'], 'ACC')} | "
      f"BWT {dtxt(sh15, SUBJ['A0+CO'], 'BWT')} | F+ {dtxt(sh15, SUBJ['A0+CO'], 'F+')}")
print(f"  subject BN: {cell(SUBJ['A0+CO'], 'ACC')}, shared: {cell(sh15, 'ACC')}")
noea = {}
for f in glob.glob(f"{R9}/chosen_noEA/protomem_9_subjects_seed_*/accuracy_matrix.txt"):
    noea[int(os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1])] = T.load(f)
print(f"with vs without EA (seeds 18-37): ACC {dtxt(main['chosen'], noea, 'ACC')} | BWT {dtxt(main['chosen'], noea, 'BWT')}"
      f" | F+ {dtxt(main['chosen'], noea, 'F+')}")
print(f"  without EA: ACC {cell(noea, 'ACC')}, BWT {cell(noea, 'BWT')}, F+ {cell(noea, 'F+')}, LA {cell(noea, 'LA')}")
for m in ['er', 'mudvi', 'cger', 'ca4mi']:
    print(f"  without EA - {m}: ACC {dtxt(noea, main[m], 'ACC')} | BWT {dtxt(noea, main[m], 'BWT')}")

if len(sys.argv) > 5:      # review 10: alignment-matched baselines (ea=yes), seeds 18-37
    R10 = os.path.join(sys.argv[5], 'results10')
    from scipy import stats
    ea = {m: T.runs(R10, '.', m, bench='bench_ea') for m in ['er', 'cger', 'mudvi']}
    print("\n% ===== Table alignment-matched baselines (seeds 18-37)")
    for m in ['er', 'cger', 'mudvi']:
        print(f"{NAMES[m]} + EA & " + " & ".join(cell(ea[m], k) for k in ['ACC', 'BWT', 'F+', 'F1', 'LA'])
              + f" \\\\ % n={len(ea[m])}")
    adj = holm({m: diff(main['chosen'], ea[m], 'ACC')[0][3] for m in ea})
    for m in ['er', 'cger', 'mudvi']:
        print(f"% {m}+EA - {m}: ACC {dtxt(ea[m], main[m], 'ACC')} | BWT {dtxt(ea[m], main[m], 'BWT')}")
        print(f"ProtoMem $-$ {NAMES[m]} + EA & " + " & ".join(dtxt(main['chosen'], ea[m], k)
                                                         for k in ['ACC', 'BWT', 'F+']) + f" & {adj[m]:.3f} \\\\")
        s = sorted(set(main['chosen']) & set(ea[m]))
        g = np.mean([main['chosen'][i][1] - ea[m][i][1] for i in s], axis=0)
        print(f"%   subject level: improved {(g > 0).sum()}/9, Wilcoxon p={stats.wilcoxon(g).pvalue:.3f}, "
              + " ".join(f"{x:+.1f}" for x in g))
        for lab, seeds in [("28-37", T.FRESH)]:
            print(f"%   seeds {lab}: ProtoMem - {m}+EA ACC {dtxt(T.sub(main['chosen'], seeds), T.sub(ea[m], seeds), 'ACC')}")

print("\n% ===== Table protocols (seeds 8-12): ACC mean +- SD (BWT), and ProtoMem - method ACC")
for title, blk, nsub, bases in [("Cross-session", 'D_xsess', 9, ['er', 'mudvi', 'cger', 'ca4mi']),
                                ("Four-class", 'F_4class', 9, ['cger', 'ca4mi']),
                                ("Budget 150", 'C_fix150', 9, ['cger']), ("Budget 300", 'C_fix300', 9, ['cger'])]:
    c, a = T.runs(R8, blk, 'chosen', nsub), T.runs(R8, blk, 'A0', nsub)
    print(f"% --- {title}")
    print(f"%   ProtoMem {cell(c, 'ACC')} (BWT {ms(c, 'BWT')[0]:.2f}, F+ {ms(c, 'F+')[0]:.2f}) | "
          f"Replay {cell(a, 'ACC')} (BWT {ms(a, 'BWT')[0]:.2f}) | diff {dtxt(c, a, 'ACC')}")
    for m in bases:
        r = T.runs(R6, blk, m, nsub, bench='bench*')
        print(f"%   {m}: {cell(r, 'ACC')} (BWT {ms(r, 'BWT')[0]:.2f}, F+ {ms(r, 'F+')[0]:.2f}) | ProtoMem - {m} {dtxt(c, r, 'ACC')}")
print("% --- High-Gamma (seeds 8-12)")
for m in ['A0'] + BASE:
    print(f"%   ProtoMem - {m}: ACC {dtxt(hgd['chosen'], hgd[m], 'ACC')} | F+ {dtxt(hgd['chosen'], hgd[m], 'F+')}")

print("\n% ===== orders (seeds 8-12)")
seq = T.sub(T.P['A0+CO'], T.OLD), T.sub(T.P['A0'], T.OLD)
for o, (c, a) in [('seq', seq)] + [(o, (T.runs(R8, 'G_order', f'chosen_{o}'), T.runs(R8, 'G_order', f'A0_{o}')))
                                  for o in ('rev', 'p1', 'p2')]:
    print(f"% {o}: ProtoMem {cell(c, 'ACC')} BWT {ms(c, 'BWT')[0]:.2f} | Replay {cell(a, 'ACC')} | diff {dtxt(c, a, 'ACC')}")

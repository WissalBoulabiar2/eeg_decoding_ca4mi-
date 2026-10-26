"""
Primary analysis of the paper on the untouched seeds 28-37 (seeds 18-27 and the pooled 18-37 are supplementary),
recomputed from the saved matrices with the helpers of tools/paper_tables.py.

    python tools/paper_tables_primary.py <dir results8> <dir results7> <dir results6> <dir results9> <dir results10>
Primary family (Holm): ProtoMem vs aligned CGER, aligned MUDVI, aligned ER, CA4MI and Replay (shared BN).
"""
import contextlib
import io
import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    import paper_tables as P          # noqa: E402  (loads all results; its printout is discarded)
T = P.T
F = T.FRESH
ea = {m: T.runs(os.path.join(sys.argv[5], 'results10'), '.', m, bench='bench_ea') for m in ['er', 'cger', 'mudvi']}
noea = P.noea
R = {**{m: T.sub(r, F) for m, r in P.main.items()}, **{f'{m}+EA': T.sub(r, F) for m, r in ea.items()}}
POOL = {**P.main, **{f'{m}+EA': r for m, r in ea.items()}}
NAMES = {**P.NAMES, 'er+EA': 'ER + alignment', 'cger+EA': 'CGER + alignment', 'mudvi+EA': 'MUDVI + alignment'}
ROWS = ['finetuning', 'ewc', 'der', 'er', 'mudvi', 'cger', 'ca4mi', 'er+EA', 'mudvi+EA', 'cger+EA', 'A0', 'chosen']
ptxt = lambda p: "$<0.001$" if p < 0.001 else f"{p:.3f}"

print("% ===== Table main (2a): seeds 28-37 ACC BWT F+ F1 LA | pooled 18-37 ACC")
c = ['ACC', 'BWT', 'F+', 'F1', 'LA']
hi = {'ACC': True, 'BWT': True, 'F+': False, 'F1': True, 'LA': True}
best = P.bold_rows({m: R[m] for m in ROWS if m != 'der'}, c, hi)
for m in ROWS:
    print(f"{NAMES[m]:22s} & " + " & ".join(P.cell(R[m], k, best[k] if m != 'der' else None) for k in c)
          + f" & {P.cell(POOL[m], 'ACC')} \\\\ % n={len(R[m])}/{len(POOL[m])}")

print("\n% ===== Table paired (primary family, seeds 28-37, Holm) | pooled 18-37 ACC")
fam = ['cger+EA', 'mudvi+EA', 'er+EA', 'ca4mi', 'A0']
adj = P.holm({m: P.diff(R['chosen'], R[m], 'ACC')[0][3] for m in fam})
adjp = P.holm({m: P.diff(POOL['chosen'], POOL[m], 'ACC')[0][3] for m in fam})
for m in fam:
    s = sorted(set(R['chosen']) & set(R[m]))
    g = np.mean([R['chosen'][i][1] - R[m][i][1] for i in s], axis=0)
    print(f"ProtoMem $-$ {NAMES[m]:20s} & " + " & ".join(P.dtxt(R['chosen'], R[m], k) for k in ['ACC', 'BWT', 'F+'])
          + f" & {ptxt(adj[m])} & {P.dtxt(POOL['chosen'], POOL[m], 'ACC')} (Holm {ptxt(adjp[m])}) \\\\ "
          f"% subjects improved {(g > 0).sum()}/9, Wilcoxon p={stats.wilcoxon(g).pvalue:.3f}")

print("\n% ===== secondary: reference implementations, seeds 28-37")
for m in ['cger', 'mudvi', 'er']:
    print(f"% ProtoMem - {m} (reference): {P.dtxt(R['chosen'], R[m], 'ACC')}; {m}+EA - {m}: "
          f"{P.dtxt(R[m + '+EA'], R[m], 'ACC')} (pooled {P.dtxt(POOL[m + '+EA'], POOL[m], 'ACC')})")

print("\n% ===== Table retention, seeds 28-37: F, F+, Fmax")
for m in ROWS:
    print(f"{NAMES[m]:22s} & " + " & ".join(f"${P.ms(R[m], k)[0]:.2f} \\pm {P.ms(R[m], k)[1]:.2f}$"
                                            for k in ['F', 'F+', 'Fmax']) + " \\\\")
fm = [R['chosen'][s][0][T.METRICS.index('Fmax')] for s in sorted(R['chosen'])]
print(f"% ProtoMem Fmax per seed: min {min(fm):.1f} max {max(fm):.1f}; SD of ACC: " +
      ", ".join(f"{m} {P.ms(R[m], 'ACC')[1]:.2f}" for m in ROWS))

if len(sys.argv) > 6:      # review 11: classifier head (cosine proxy vs standard linear head), seeds 18-37
    lin = {}
    for f in __import__('glob').glob(os.path.join(sys.argv[6], 'results11', 'chosen_linear',
                                                  'protomem_9_subjects_seed_*', 'accuracy_matrix.txt')):
        lin[int(os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1])] = T.load(f)
    lF = T.sub(lin, F)
    print("\n% ===== classifier head, seeds 28-37 (pooled 18-37)")
    print(f"% linear head: " + " | ".join(f"{k} {P.cell(lF, k)}" for k in ['ACC', 'BWT', 'F+', 'Fmax', 'LA'])
          + f" | pooled ACC {P.cell(lin, 'ACC')}")
    for k in ['ACC', 'BWT', 'F+', 'LA']:
        print(f"%   cosine - linear {k}: {P.dtxt(R['chosen'], lF, k)} (pooled {P.dtxt(POOL['chosen'], lin, k)})")
    s = sorted(set(R['chosen']) & set(lF))
    g = np.mean([R['chosen'][i][1] - lF[i][1] for i in s], axis=0)
    print(f"%   subject level 28-37: improved {(g > 0).sum()}/9, Wilcoxon p={stats.wilcoxon(g).pvalue:.3f}, "
          + " ".join(f"{x:+.1f}" for x in g))
    for m in ['cger+EA', 'mudvi+EA', 'er+EA', 'ca4mi']:
        print(f"%   linear-head ProtoMem - {m}: {P.dtxt(lF, R[m], 'ACC')} (pooled {P.dtxt(lin, POOL[m], 'ACC')})")

print("\n% ===== alignment control, seeds 28-37 (pooled 18-37)")
nF = T.sub(noea, F)
print(f"% without EA: {P.cell(nF, 'ACC')} | ProtoMem - noEA {P.dtxt(R['chosen'], nF, 'ACC')} "
      f"(pooled {P.dtxt(POOL['chosen'], noea, 'ACC')}) | BWT {P.dtxt(R['chosen'], nF, 'BWT')}")
for m in ['er', 'mudvi', 'cger', 'ca4mi']:
    print(f"%   noEA - {m} (reference): {P.dtxt(nF, R[m], 'ACC')} (pooled {P.dtxt(noea, POOL[m], 'ACC')})")

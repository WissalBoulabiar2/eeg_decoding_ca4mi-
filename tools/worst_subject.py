"""
Worst-subject analysis on the primary seeds 28-37 from the saved accuracy matrices: which earlier subject declines most
(F^max) in each run, how often each subject is the worst, and the final accuracy of the worst-served user.

    python tools/worst_subject.py <dir results8> <dir results7> <dir results6> <dir results9> <dir results10>
"""
import collections
import contextlib
import glob
import io
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    import paper_tables as P          # noqa: E402  (helpers and loaded results; its printout is discarded)
T = P.T
F = T.FRESH


def matrices(root, block, name, bench=None):
    if bench is None:
        fs = glob.glob(f"{root}/{block}/protomem/{name}/protomem_9_subjects_seed_*/accuracy_matrix.txt")
        seed = lambda f: int(os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1])
    else:
        fs = glob.glob(f"{root}/{block}/{bench}/{name}/**/acc_run_*_subject_9.txt", recursive=True)
        seed = lambda f: int(os.path.basename(f).split('_')[2])
    return {seed(f): np.loadtxt(f) for f in fs if seed(f) in F}


R10 = os.path.join(sys.argv[5], 'results10')
runs = {'ProtoMem': {**matrices(P.R8, 'H_confirm', 'chosen')},
        'CGER + alignment': matrices(R10, '.', 'cger', bench='bench_ea'),
        'ER + alignment': matrices(R10, '.', 'er', bench='bench_ea'),
        'MUDVI + alignment': matrices(R10, '.', 'mudvi', bench='bench_ea')}
mins = {}
for lab, ms in runs.items():
    worst, fmax, final_min, drop_end = [], [], [], []
    for s, a in sorted(ms.items()):
        n = len(a)
        d = np.array([a[u:n - 1, u].max() - a[-1, u] for u in range(n - 1)])
        worst.append(int(d.argmax()) + 1)
        fmax.append(d.max())
        final_min.append(a[-1].min())
    c = collections.Counter(worst)
    print(f"{lab:18s} n={len(ms)} | F^max mean {np.mean(fmax):.2f} (range {min(fmax):.1f}-{max(fmax):.1f}) | "
          f"most-declining subject per seed: {worst} -> " + ", ".join(f"S{k}: {v}" for k, v in c.most_common())
          + f" | final accuracy of the worst-served user: mean {np.mean(final_min):.2f} (min {min(final_min):.1f})")
    mins[lab] = dict(zip(sorted(ms), final_min))
from scipy import stats   # noqa: E402
for lab in list(runs)[1:]:
    s = sorted(set(mins['ProtoMem']) & set(mins[lab]))
    d = np.array([mins['ProtoMem'][i] - mins[lab][i] for i in s])
    print(f"worst-served user, ProtoMem - {lab}: {d.mean():+.2f}, paired t p={stats.ttest_1samp(d, 0).pvalue:.3f}")

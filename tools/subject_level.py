"""
Subject-level statistics for the primary comparisons (review: the population of interest is the 9 subjects, not
the seeds). For each comparison, from the saved accuracy matrices:
- per-subject accuracy averaged over seeds, ProtoMem minus method, Wilcoxon signed-rank and sign tests over the
  9 subjects (smallest attainable two-sided p: 0.0039);
- the corrected resampled t-test of Nadeau & Bengio (2003) over seeds: variance inflated by (1/J + n_test/n_train).

    python tools/subject_level.py <dir results14> <dir results8> <dir results10> <dir results13>
"""
import contextlib
import io
import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    import review14_tables as T          # noqa: E402  (loads every run; its printout is discarded)


def nadeau_bengio(d, ratio):
    J = len(d)
    t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1))
    return 2 * stats.t.sf(abs(t), J - 1)


def report(lab, a, b, ratio):
    s = sorted(set(a) & set(b))
    d = np.array([a[i][0][0] - b[i][0][0] for i in s])
    g = np.mean([a[i][1] - b[i][1] for i in s], axis=0)          # per-subject, averaged over seeds
    w = stats.wilcoxon(g).pvalue if np.any(g != 0) else 1.0
    pos, neg = int((g > 0).sum()), int((g < 0).sum())
    sign = stats.binomtest(pos, pos + neg).pvalue if pos + neg else 1.0
    print(f"  {lab:38s} mean {d.mean():+6.2f} | subjects +{pos}/-{neg}, median {np.median(g):+6.2f}, "
          f"Wilcoxon p={w:.3f}, sign p={sign:.3f} | corrected t p={nadeau_bengio(d, ratio):.3f} "
          f"(naive t p={stats.ttest_1samp(d, 0).pvalue:.3f})")


A, B = T.A, T.B
print("=== within session, seeds 28-37 (n_test/n_train = 22/100)")
for m in ['CGER + EA', 'MUDVI + EA (fixed)', 'ER + EA (fixed)', 'CA4MI', 'Replay (shared BN)', 'CGER + EA, matched',
          'MUDVI + EA, matched', 'ER + EA, matched', 'joint training', 'FBCSP + LDA (per subject)',
          'TS + LR (per subject)', 'isolated per-subject models']:
    report(f"ProtoMem - {m}", A['ProtoMem'], A[m], 22 / 100)
print("\n=== cross-session, seeds 48-57 (n_test/n_train = 144/122)")
for m in ['CGER + EA', 'MUDVI + EA', 'ER + EA', 'Replay (shared BN)', 'CGER + EA, matched', 'MUDVI + EA, matched',
          'ER + EA, matched', 'joint training', 'FBCSP + LDA (per subject)', 'TS + LR (per subject)']:
    report(f"ProtoMem - {m}", B['ProtoMem'], B[m], 144 / 122)

"""
Numbers of the review-14 version of the paper that tools/review14_tables.py does not print: the within-session family
with the same seven comparisons as the cross-session one, the last waterfall step with the fixed ER, and the
seeds 28-47 equivalence test against aligned CGER (reference settings; CGER is unaffected by the checkpoint fix).

    python tools/review14_paper.py <dir results14> <dir results8> <dir results10> <dir results13> <dir results12>
"""
import contextlib
import io
import os
import sys

sys.argv, R12 = sys.argv[:5], os.path.join(sys.argv[5], 'results12')
sys.path.insert(0, os.path.dirname(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    import review14_tables as T       # noqa: E402  (loads every run; its printout is discarded)

A, P = T.A, T.A['ProtoMem']
print("=== within session, seeds 28-37: same 7-comparison family as cross-session (Holm over 7)")
fam = ['CGER + EA', 'MUDVI + EA (fixed)', 'ER + EA (fixed)', 'CGER + EA, matched', 'MUDVI + EA, matched',
       'ER + EA, matched', 'Replay (shared BN)']
print("  Holm: " + T.holm({m: T.paired(f"ProtoMem - {m}", P, A[m]) for m in fam}))

print("\n=== waterfall: last step with the fixed ER")
W5 = T.load(f"{T.R13}/P1_2a/protomem/W5_separate", T.F)
T.paired("ER + EA (fixed) - W5_separate", A['ER + EA (fixed)'], W5, ks=(0, 1))
T.paired("Replay - ER + EA (fixed)", A['Replay (shared BN)'], A['ER + EA (fixed)'], ks=(0,))

print("\n=== seeds 28-47: ProtoMem vs aligned CGER (reference settings), equivalence")
S20 = list(range(28, 48))
p20 = {**T.load(f"{T.R8}/H_confirm/protomem/chosen", S20), **T.load(f"{R12}/protomem_cosine", S20)}
c20 = {**T.load(f"{T.R10}/bench_ea/cger", S20), **T.load(f"{R12}/bench_ea/cger", S20)}
T.paired("ProtoMem - CGER + EA (seeds 28-47)", p20, c20, ks=(0,))
for mg in (2, 3):
    print(f"  TOST +-{mg}: p={T.tost(p20, c20, mg):.3f}")

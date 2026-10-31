"""
Review-14 analyses from the saved accuracy matrices:
A. BCI IV 2a within session, seeds 28-37: baselines with the checkpoint fix, baselines with ProtoMem's training regime
   (matched), per-subject classical decoders (FBCSP+LDA, tangent space+LR), the noise floor of the forgetting metrics
   (cumulative joint training) and the BN-recalibration control.
B. Cross-session (train session 1, test the whole session 2), fresh seeds 48-57: all methods.

    python tools/review14_tables.py <dir results14> <dir results8> <dir results10> <dir results13>
"""
import glob
import os
import re
import sys
import numpy as np
from scipy import stats

R14, R8, R10, R13 = (os.path.join(d, f'results{k}') for d, k in zip(sys.argv[1:5], (14, 8, 10, 13)))
F, N = list(range(28, 38)), list(range(48, 58))
NAMES = ['ACC', 'BWT', 'F+', 'Fmax', 'LA']


def metrics(f):
    a = np.loadtxt(f)
    n = len(a)
    d = np.array([a[u:n - 1, u].max() - a[-1, u] for u in range(n - 1)])
    return np.array([a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (n - 1), np.maximum(d, 0).mean(), d.max(),
                     np.diag(a).mean()]), a[-1]


def load(path, seeds):
    """Every run below path: ProtoMem-style <...>_9_subjects_seed_<s>/accuracy_matrix.txt or acc_run_<s>_subject_9.txt."""
    out = {}
    for f in glob.glob(f"{path}/**/*.txt", recursive=True):
        m = re.search(r"_9_subjects_seed_(\d+)[/\\]accuracy_matrix\.txt$", f) or re.search(r"acc_run_(\d+)_subject_9\.txt$", f)
        if m and int(m.group(1)) in seeds:
            out[int(m.group(1))] = metrics(f)
    return out


def row(lab, r):
    if not r:
        print(f"  {lab:34s} (no runs)"); return
    v = np.array([r[s][0] for s in sorted(r)])
    print(f"  {lab:34s} n={len(v):2d} " + " | ".join(f"{k} {v[:, i].mean():6.2f}±{v[:, i].std():5.2f}"
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
    print(f"  {lab:44s} n={len(s):2d} " + " | ".join(out) + f" | subjects {(g > 0).sum()}/{len(g)} W p={w:.3f}")
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


# ---------------------------------------------------------------- A. 2a within session, seeds 28-37
print("\n=== A. BCI IV 2a within session, seeds 28-37 ===")
Q1 = f"{R14}/Q1_2a"
A = {'ProtoMem': load(f"{R8}/H_confirm/protomem/chosen", F),
     'Replay (shared BN)': load(f"{R8}/H_confirm/protomem/A0", F),
     'Replay + BN recalibration': load(f"{Q1}/protomem/replay_bnrecal", F),
     'CGER + EA': load(f"{R10}/bench_ea/cger", F),
     'MUDVI + EA (fixed)': load(f"{Q1}/bench_fix_ea/mudvi", F),
     'ER + EA (fixed)': load(f"{Q1}/bench_fix_ea/er", F),
     'CGER + EA, matched': load(f"{Q1}/bench_matched_ea/cger", F),
     'MUDVI + EA, matched': load(f"{Q1}/bench_matched_ea/mudvi", F),
     'ER + EA, matched': load(f"{Q1}/bench_matched_ea/er", F),
     'CA4MI': load(f"{R8}/H_confirm/bench/ca4mi", F),
     'fine-tuning': load(f"{R8}/H_confirm/bench/finetuning", F),
     'EWC (fixed)': load(f"{Q1}/bench_fix/ewc", F),
     'DER (fixed)': load(f"{Q1}/bench_fix/der", F),
     'ER (fixed)': load(f"{Q1}/bench_fix/er", F),
     'MUDVI (fixed)': load(f"{Q1}/bench_fix/mudvi", F),
     'CGER': load(f"{R8}/H_confirm/bench/cger", F),
     'isolated per-subject models': load(f"{R13}/P1_2a/protomem/isolated", F),
     'joint training': load(f"{R13}/P1_2a/protomem/joint", F),
     'cumulative joint (noise floor)': load(f"{Q1}/protomem/cumjoint", F),
     'FBCSP + LDA (per subject)': load(f"{Q1}/protomem/fbcsp", F),
     'TS + LR (per subject)': load(f"{Q1}/protomem/tslr", F)}
OLD = {'ER + EA (bug)': load(f"{R10}/bench_ea/er", F), 'MUDVI + EA (bug)': load(f"{R10}/bench_ea/mudvi", F),
       'ER (bug)': load(f"{R8}/H_confirm/bench/er", F), 'MUDVI (bug)': load(f"{R8}/H_confirm/bench/mudvi", F),
       'EWC (bug)': load(f"{R8}/H_confirm/bench/ewc", F), 'DER (bug)': load(f"{R8}/H_confirm/bench/der", F)}
for lab, r in {**A, **OLD}.items():
    row(lab, r)
P = A['ProtoMem']
print("\n  effect of the checkpoint fix (fixed - bug):")
for new, old in [('ER + EA (fixed)', 'ER + EA (bug)'), ('MUDVI + EA (fixed)', 'MUDVI + EA (bug)'),
                 ('ER (fixed)', 'ER (bug)'), ('MUDVI (fixed)', 'MUDVI (bug)'), ('EWC (fixed)', 'EWC (bug)'),
                 ('DER (fixed)', 'DER (bug)')]:
    paired(f"{new} - {old}", A[new], OLD[old])
print("\n  primary family, reference settings (Holm over 5):")
fam = ['CGER + EA', 'MUDVI + EA (fixed)', 'ER + EA (fixed)', 'CA4MI', 'Replay (shared BN)']
print("  Holm: " + holm({m: paired(f"ProtoMem - {m}", P, A[m]) for m in fam}))
print("\n  matched training regime (Holm over 3):")
fam = ['CGER + EA, matched', 'MUDVI + EA, matched', 'ER + EA, matched']
print("  Holm: " + holm({m: paired(f"ProtoMem - {m}", P, A[m]) for m in fam}))
for m in ['CGER', 'MUDVI', 'ER']:
    ref = 'CGER + EA' if m == 'CGER' else f"{m} + EA (fixed)"
    paired(f"{m} + EA: matched - reference settings", A[f"{m} + EA, matched"], A[ref])
for mg in (2, 3):
    print(f"  TOST ProtoMem vs CGER + EA, matched, +-{mg}: p={tost(P, A['CGER + EA, matched'], mg):.3f}")
print("\n  classical per-subject decoders:")
for m in ['FBCSP + LDA (per subject)', 'TS + LR (per subject)']:
    paired(f"ProtoMem - {m}", P, A[m], ks=(0,))
    paired(f"isolated DeepConvNet - {m}", A['isolated per-subject models'], A[m], ks=(0,))
print("\n  noise floor of the forgetting metrics (cumulative joint training forgets nothing by construction):")
for m in ['ProtoMem', 'CGER + EA', 'Replay (shared BN)']:
    paired(f"{m} - cumulative joint", A[m], A['cumulative joint (noise floor)'], ks=(0, 2, 3))
print("\n  BN recalibration control:")
paired("Replay + BN recal - Replay", A['Replay + BN recalibration'], A['Replay (shared BN)'])
paired("ProtoMem - (Replay + BN recal)", P, A['Replay + BN recalibration'])

# ---------------------------------------------------------------- B. cross-session, seeds 48-57
print("\n=== B. Cross-session (train session 1, test all of session 2), seeds 48-57 ===")
Q2 = f"{R14}/Q2_xsess"
B = {'ProtoMem': load(f"{Q2}/protomem/protomem", N),
     'Replay (shared BN)': load(f"{Q2}/protomem/replay", N),
     'CGER + EA': load(f"{Q2}/bench_ea/cger", N),
     'MUDVI + EA': load(f"{Q2}/bench_ea/mudvi", N),
     'ER + EA': load(f"{Q2}/bench_ea/er", N),
     'CGER + EA, matched': load(f"{Q2}/bench_matched_ea/cger", N),
     'MUDVI + EA, matched': load(f"{Q2}/bench_matched_ea/mudvi", N),
     'ER + EA, matched': load(f"{Q2}/bench_matched_ea/er", N),
     'CA4MI': load(f"{Q2}/bench/ca4mi", N),
     'isolated per-subject models': load(f"{Q2}/protomem/isolated", N),
     'joint training': load(f"{Q2}/protomem/joint", N),
     'cumulative joint (noise floor)': load(f"{Q2}/protomem/cumjoint", N),
     'FBCSP + LDA (per subject)': load(f"{Q2}/protomem/fbcsp", N),
     'TS + LR (per subject)': load(f"{Q2}/protomem/tslr", N)}
for lab, r in B.items():
    row(lab, r)
P = B['ProtoMem']
print("\n  primary family (Holm over 7):")
fam = ['CGER + EA', 'MUDVI + EA', 'ER + EA', 'CGER + EA, matched', 'MUDVI + EA, matched', 'ER + EA, matched',
       'Replay (shared BN)']
print("  Holm: " + holm({m: paired(f"ProtoMem - {m}", P, B[m]) for m in fam}))
for m in ['CGER', 'MUDVI', 'ER']:
    paired(f"{m} + EA: matched - reference settings", B[f"{m} + EA, matched"], B[f"{m} + EA"])
for ref in ['CGER + EA', 'CGER + EA, matched']:
    for mg in (2, 3):
        print(f"  TOST ProtoMem vs {ref}, +-{mg}: p={tost(P, B[ref], mg):.3f}")
print("\n  CA4MI (only the seeds that finished):")
paired("ProtoMem - CA4MI", P, B['CA4MI'])
print("\n  references and classical decoders:")
for m in ['isolated per-subject models', 'joint training', 'FBCSP + LDA (per subject)', 'TS + LR (per subject)']:
    paired(f"ProtoMem - {m}", P, B[m], ks=(0,))
paired("ProtoMem LA - isolated LA", P, B['isolated per-subject models'], ks=(4,))
print("\n  noise floor of the forgetting metrics:")
for m in ['ProtoMem', 'CGER + EA', 'CGER + EA, matched', 'Replay (shared BN)']:
    paired(f"{m} - cumulative joint", B[m], B['cumulative joint (noise floor)'], ks=(0, 2, 3))

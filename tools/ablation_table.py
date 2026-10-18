"""
Cumulative ablation table of ProtoMem from the result folders (mean ± std over seeds).

    python tools/ablation_table.py ablation final_bal
A0-A4 are read from <ablation>/<row>/, A5 (final ProtoMem) from <final_bal>/final/protomem/.
"""
import argparse
import glob
import os
import numpy as np

ROWS = [
    ("A0", "single encoder, subject BN, proxy classifier, exemplar replay"),
    ("A1", "+ logit consistency on exemplars"),
    ("A2", "+ pseudo-feature replay from the Gaussian memory"),
    ("A3", "+ multi-mode memory with Ward consolidation ($M=2$)"),
    ("A4", "+ drift compensation"),
    ("A5", "+ balanced consolidation (full ProtoMem)"),
]


def metrics(folder):
    vals = []
    for f in sorted(glob.glob(os.path.join(folder, "protomem_9_subjects_seed_*", "accuracy_matrix.txt"))):
        a, f1 = np.loadtxt(f), np.loadtxt(f.replace("accuracy_matrix", "f1_macro_matrix"))
        n = len(a)
        vals.append((a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (n - 1),
                     np.mean([a[u:n - 1, u].max() - a[-1, u] for u in range(n - 1)]), f1[-1].mean()))
    return np.array(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ablation_dir")
    p.add_argument("final_dir")
    flags = p.parse_args()
    res = {}
    for row, _ in ROWS:
        folder = os.path.join(flags.final_dir, "final", "protomem") if row == "A5" else os.path.join(flags.ablation_dir, row)
        v = metrics(folder)
        res[row] = v if len(v) else None
    best = {k: max((r for r in res if res[r] is not None), key=lambda r: (1 if k != 2 else -1) * res[r][:, k].mean())
            for k in range(4)}

    print(f"\n{'Row':4s} {'Added component':58s} {'ACC':>13s} {'BWT':>13s} {'Forgetting':>13s} {'F1':>13s} seeds")
    for row, name in ROWS:
        v = res[row]
        if v is None:
            print(f"{row:4s} {name:58s} (no results)")
            continue
        print(f"{row:4s} {name:58s} " + " ".join(f"{v[:, k].mean():6.2f} ± {v[:, k].std():4.2f}" for k in range(4))
              + f"  {len(v)}")

    print("\n% ---- LaTeX ----")
    print("\\begin{table}[t]\n\\centering\n\\caption{Cumulative ablation of ProtoMem on BCI Competition IV 2a "
          "(mean $\\pm$ std over five seeds).}\n\\label{tab:ablation}\n\\begin{tabular}{llcccc}\n\\toprule")
    print("Row & Added component & ACC (\\%) & BWT (\\%) & Forgetting (\\%) $\\downarrow$ & F1 (\\%) \\\\\n\\midrule")
    for row, name in ROWS:
        v = res[row]
        if v is None:
            continue
        cells = []
        for k in range(4):
            c = f"{v[:, k].mean():.2f} $\\pm$ {v[:, k].std():.2f}"
            cells.append(f"\\textbf{{{c}}}" if best[k] == row else c)
        print(f"{row} & {name} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")


if __name__ == "__main__":
    main()

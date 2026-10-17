"""
Subject-wise accuracy evolution (same layout as stages_plot) from the log of main/run_final.py.

Reads the accuracy matrices of the final-evaluation runs ([B final] jobs) directly from the saved log:
  - baselines, CA4MI, CA4MI-MP: the "Accuracies =" block printed at the end of each run
  - ProtoMem: the "test acc:" line printed after each subject
and plots, for every method, the mean matrix over the seeds found in the log (or one seed with --seed).

    python tools/plot_stages.py results.txt                      # mean over all final seeds
    python tools/plot_stages.py results.txt --seed 3             # a single seed
    python tools/plot_stages.py results.txt --with_mp            # also draw CA4MI-MP
"""
import argparse
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HEADER = re.compile(r"\[job \d+/\d+\] \[B final\] (.+?) seed (\d+)")
VALUE = re.compile(r"(-?\d+(?:\.\d+)?)%")

# method name in the log -> (label, color, marker, linestyle, linewidth, alpha, zorder)
STYLE = {
    "ProtoMem":   ("Ours",       "blue",   "o", "-",  1.8, 1.0, 10),
    "CA4MI":      ("CA4MI",      "c",      "o", "-",  1.0, 0.7, 4),
    "CA4MI-MP":   ("CA4MI-MP",   "navy",   "s", "--", 1.0, 0.6, 4),
    "ER":         ("ER",         "green",  "*", "-",  1.0, 0.5, 3),
    "Finetuning": ("Finetuning", "red",    "^", "--", 1.0, 0.5, 3),
    "EWC":        ("EWC",        "purple", "s", "-",  1.0, 0.5, 3),
    "CGER":       ("CGER",       "orange", "D", "-",  1.0, 0.5, 3),
    "DER":        ("DER",        "brown",  "v", "-",  1.0, 0.5, 3),
    "MUDVI":      ("MUDVI",      "pink",   "o", "-",  1.0, 0.7, 3),
}


def parse_log(path, n):
    """Return {method: {seed: n x n accuracy matrix}} for the [B final] runs of the log."""
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    out = defaultdict(dict)
    method, seed, proto_rows, i = None, None, [], 0
    while i < len(lines):
        line = lines[i]
        h = HEADER.search(line)
        if h:
            method, seed, proto_rows = h.group(1).strip(), int(h.group(2)), []
        elif "[A selection]" in line:
            method = None
        elif method == "ProtoMem" and line.strip().startswith("test acc:"):
            proto_rows.append([float(v) for v in line.split(":", 1)[1].split()])
            if len(proto_rows) == n:
                a = np.zeros((n, n))
                for t, row in enumerate(proto_rows):
                    a[t, :len(row)] = row
                out[method][seed] = a
        elif method is not None and method != "ProtoMem" and line.strip() == "Accuracies =":
            rows = [[float(v) for v in VALUE.findall(lines[i + 1 + r])] for r in range(n)]
            if all(len(r) == n for r in rows):
                out[method][seed] = np.array(rows)
            i += n
        i += 1
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--n_subjects", type=int, default=9)
    parser.add_argument("--seed", type=int, default=None, help="plot one seed instead of the mean over seeds")
    parser.add_argument("--with_mp", action="store_true", help="also plot CA4MI-MP")
    parser.add_argument("--out", default="stages_protomem.pdf")
    parser.add_argument("--ymin", type=float, default=20)
    flags = parser.parse_args()

    n = flags.n_subjects
    runs = parse_log(flags.log, n)
    data = {}
    for m, by_seed in runs.items():
        if m not in STYLE or (m == "CA4MI-MP" and not flags.with_mp):
            continue
        if flags.seed is not None:
            if flags.seed in by_seed:
                data[m] = by_seed[flags.seed]
        else:
            data[m] = np.mean(list(by_seed.values()), axis=0)
        print(f"{m:10s} seeds {sorted(by_seed)} -> final ACC {data[m][-1].mean():.2f}" if m in data else "")
    if "ProtoMem" not in data:
        raise SystemExit("No ProtoMem final run found in the log.")

    subjects = [f"S{s:02d}" for s in range(1, n + 1)]
    fig, ax = plt.subplots(figsize=(16, 4.2))
    ax.set_xlim(0, n)
    ax.set_ylim(flags.ymin, 100)
    for s in range(n):
        ax.axvline(s, color="black", lw=0.8)
        ax.text(s + 0.5, 102, subjects[s], ha="center", va="bottom", fontsize=13)
        ax.axvline(s + 0.02, color="lightgray", ls="--", lw=1.2, zorder=0)
        n_pts = n - s  # subject s is learned at stage s and tested at stages s .. n-1
        xs = [s + 0.02 + (i / n_pts) * 0.9 for i in range(n_pts)]
        for key, (lab, c, mk, ls, lw, al, z) in STYLE.items():
            if key in data:
                ax.plot(xs, [data[key][t][s] for t in range(s, n)],
                        color=c, marker=mk, ms=3.5, ls=ls, lw=lw, alpha=al, zorder=z)
    ax.axvline(n, color="black", lw=0.8)
    ax.set_xticks([s + 0.02 for s in range(n)])
    ax.set_xticklabels([f"T{s + 1}" for s in range(n)], color="gray", fontsize=12)
    ax.set_yticks([30, 50, 70, 90, 100])
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_xlabel("Training Stage", fontsize=14)
    ax.tick_params(length=0)

    keys = [k for k in STYLE if k in data]
    handles = [Line2D([0], [0], color=STYLE[k][1], marker=STYLE[k][2], ls=STYLE[k][3], lw=STYLE[k][4], ms=4)
               for k in keys]
    fig.legend(handles, [STYLE[k][0] for k in keys], loc="upper center", ncol=len(keys), frameon=True,
               fontsize=12, bbox_to_anchor=(0.5, 1.02))
    plt.subplots_adjust(top=0.82)
    plt.savefig(flags.out, bbox_inches="tight")
    plt.savefig(flags.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"saved {flags.out} and {flags.out.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()

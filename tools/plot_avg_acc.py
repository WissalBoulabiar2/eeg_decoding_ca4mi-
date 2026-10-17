"""
Average accuracy on the subjects seen so far, after each training stage, for every method (one continuous line each).

    acc_t = mean_{u <= t} A[t, u]      (A = accuracy matrix, rows = stages)

Reads the final-evaluation runs from the run_final.py log (same parser as tools/plot_stages.py) and averages over the
seeds found in the log (or one seed with --seed). CA4MI-MP is not drawn.

    python tools/plot_avg_acc.py results.txt
    python tools/plot_avg_acc.py results.txt --band          # shaded ± std over seeds
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_stages import parse_log

# log name -> (label, color, marker, linestyle, linewidth, markersize, zorder)
STYLE = {
    "Finetuning": ("Finetuning (no CL)", "#6b6b6b", "v", "-", 2.0, 6, 3),
    "EWC":        ("EWC",                "#e69f00", "s", "-", 2.0, 6, 3),
    "ER":         ("ER",                 "#009e73", "^", "-", 2.0, 6, 3),
    "DER":        ("DER",                "#d55e00", "D", "-", 2.0, 6, 3),
    "MUDVI":      ("MUDVI",              "#7b2fbe", "P", "-", 2.0, 7, 3),
    "CGER":       ("CGER",               "#f0e442", "X", "-", 2.0, 7, 3),
    "CA4MI":      ("CA4MI",              "#56b4e9", "o", "-", 2.0, 7, 4),
    "ProtoMem":   ("Ours (ProtoMem)",    "#0047ab", "*", "-", 3.0, 12, 10),
}


def seen_accuracy(a):
    """Mean accuracy over the subjects learned so far, after every stage."""
    return np.array([a[t, :t + 1].mean() for t in range(len(a))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--n_subjects", type=int, default=9)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--band", action="store_true", help="shade ± std over seeds")
    parser.add_argument("--title", default="Subject-incremental learning on BCI-IV 2a")
    parser.add_argument("--out", default="avg_acc_protomem.pdf")
    flags = parser.parse_args()

    runs = parse_log(flags.log, flags.n_subjects)
    t = np.arange(1, flags.n_subjects + 1)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for key, (lab, c, mk, ls, lw, ms, z) in STYLE.items():
        by_seed = runs.get(key, {})
        if flags.seed is not None:
            by_seed = {s: a for s, a in by_seed.items() if s == flags.seed}
        if not by_seed:
            print(f"{key}: no run found, skipped")
            continue
        curves = np.array([seen_accuracy(a) for a in by_seed.values()])
        mean = curves.mean(0)
        ax.plot(t, mean, color=c, marker=mk, ls=ls, lw=lw, ms=ms, label=lab, zorder=z,
                markeredgecolor="white" if key != "ProtoMem" else c, markeredgewidth=0.6)
        if flags.band and len(curves) > 1:
            ax.fill_between(t, mean - curves.std(0), mean + curves.std(0), color=c, alpha=0.12, zorder=1)
        print(f"{lab:20s} seeds {sorted(by_seed)}  final {mean[-1]:.2f}")

    ax.set_xticks(t)
    ax.set_xticklabels([f"T{i}" for i in t])
    ax.set_xlabel("Training stage (subjects learned so far)", fontsize=12)
    ax.set_ylabel("Avg. accuracy on seen subjects (%)", fontsize=12)
    ax.set_title(flags.title, loc="left", fontsize=14, fontweight="bold")
    ax.grid(axis="y", color="#dddddd", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
    plt.tight_layout()
    plt.savefig(flags.out, bbox_inches="tight")
    plt.savefig(flags.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"saved {flags.out} and {flags.out.replace('.pdf', '.png')}")


if __name__ == "__main__":
    main()

"""
Effect of buffer ratio on ACC and BWT, in the layout of the earlier buffer figure (two stacked panels).

ProtoMem at ratios 0 / 0.5 / 1.0 = 0 / 50 / 100 exemplars per subject (studies/buffer_0, studies/buffer_50, and the
final runs of results.txt for 1.0). Baselines were not re-run at lower ratios: they are drawn at ratio 1.0 only
(default configuration, from results.txt).

    python tools/plot_buffer_ratio.py studies results.txt
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_stages import load_runs
from plot_buffer_order import read_studies_dir, stats

STYLE = {  # name -> (label, color, marker)
    "ER": ("ER", "green", "o"), "CGER": ("CGER", "orange", "^"), "DER": ("DER", "red", "v"),
    "MUDVI": ("MUDVI", "purple", "D"), "CA4MI": ("CA4MI", "teal", "o"), "ProtoMem": ("Ours", "blue", "s"),
}
plt.rcParams.update({"font.family": "serif"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("studies")
    p.add_argument("final")
    p.add_argument("--out", default="buffer_ratio_protomem.pdf")
    flags = p.parse_args()
    studies, final = read_studies_dir(flags.studies), load_runs(flags.final, 9)

    ratios, ours = [0.0, 0.5, 1.0], []
    for mats in [studies.get("buffer 0", {}), studies.get("buffer 50", {}), final.get("ProtoMem", {})]:
        ours.append(stats(mats))
    fig, axes = plt.subplots(2, 1, figsize=(8, 8.5))
    for k, (ax, title, ylab) in enumerate(zip(
            axes, ["Effect of buffer size on average classification accuracy (ACC)",
                   "Effect of buffer size on backward transfer (BWT)"], ["ACC (%)", "BWT (%)"])):
        for name, (lab, c, mk) in STYLE.items():
            if name == "ProtoMem":
                ax.plot(ratios, [o[0][k] for o in ours], color=c, marker=mk, lw=1.8, ms=6, label=lab, zorder=5)
            elif name in final:
                ax.plot([1.0], [stats(final[name])[0][k]], color=c, marker=mk, ms=7, ls="none", label=lab)
        if k == 1:
            ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(ratios)
        ax.set_xlabel("Buffer ratio")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(flags.out, bbox_inches="tight")
    plt.savefig(flags.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    for r, o in zip(ratios, ours):
        print(f"ProtoMem ratio {r}: ACC {o[0][0]:.2f}  BWT {o[0][1]:.2f}")
    print(f"saved {flags.out}")


if __name__ == "__main__":
    main()

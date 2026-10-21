"""
Effect of the buffer ratio on ACC and BWT for ProtoMem and the baselines (two stacked panels).

Buffer ratio 0 / 0.5 / 1.0 = 0 / 50 / 100 exemplars per subject (a subject has 100 training trials on BCI-IV 2a).
  - ProtoMem: studies/buffer_0, studies/buffer_50, and the final runs (results.txt) for 1.0
  - baselines: <buffer_bl>/buffer_<n>/<method>/ (run_benchmark_clean.py --set n_samples=<n>); the default buffer of
    each baseline comes from the final runs (results.txt). ER and DER with an empty buffer reduce to fine-tuning,
    so their ratio-0 point is the Finetuning run.
All points use the same seeds (--seeds, default 3 4 5).

    python tools/plot_buffer_ratio.py studies results.txt buffer_bl
"""
import argparse
import glob
import os
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_stages import load_runs
from plot_buffer_order import read_studies_dir, acc_bwt

STYLE = {  # name -> (label, color, marker)
    "ER": ("ER", "green", "o"), "CGER": ("CGER", "orange", "^"), "DER": ("DER", "red", "v"),
    "MUDVI": ("MUDVI", "purple", "D"), "CA4MI": ("CA4MI", "teal", "o"), "ProtoMem": ("Ours", "blue", "s"),
}
DIR_NAME = {"ER": "er", "CGER": "cger", "DER": "der", "MUDVI": "mudvi", "CA4MI": "ca4mi"}
DEFAULT_BUFFER = {"ER": 20, "DER": 100, "CGER": 100, "MUDVI": 100, "CA4MI": 20}  # n_samples in configs/<method>.yml
EMPTY_IS_FINETUNING = {"ER", "DER"}
plt.rcParams.update({"font.family": "serif"})


def read_buffer_runs(root, n=9):
    """{name: {buffer: {seed: matrix}}}. A configuration restarted after an interruption can appear in several run
    folders: runs that did not complete (zero accuracy on a newly learned subject) are dropped, then the first one kept."""
    found = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name, d in DIR_NAME.items():
        if d == "ca4mi":
            files = glob.glob(os.path.join(root, "buffer_*", d, "*", f"ca4mi_{n}_subjects_seed_*", "accuracy_matrix.txt"))
            seed_of = lambda f: int(os.path.basename(os.path.dirname(f)).rsplit("_", 1)[1])
        else:
            files = glob.glob(os.path.join(root, "buffer_*", d, "*", f"acc_run_*_subject_{n}.txt"))
            seed_of = lambda f: int(os.path.basename(f).split("_")[2])
        for f in sorted(files):
            nb = int(os.path.normpath(f).split(os.sep)[-4 if d != "ca4mi" else -5].split("_")[1])
            found[name][nb][seed_of(f)].append(np.loadtxt(f))
    out = defaultdict(dict)
    for name, by_nb in found.items():
        for nb, by_seed in by_nb.items():
            out[name][nb] = {}
            for s, mats in by_seed.items():
                ok = [a for a in mats if np.diag(a).min() > 0]
                if len(ok) < len(mats):
                    print(f"{name} buffer {nb} seed {s}: {len(mats) - len(ok)} incomplete run(s) dropped")
                if ok:
                    out[name][nb][s] = ok[0]
    return out


def mean_point(mats, seeds):
    v = [acc_bwt(mats[s]) for s in seeds if s in mats]
    return (np.mean(v, axis=0), len(v)) if v else (None, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("studies")
    p.add_argument("final")
    p.add_argument("buffer_bl")
    p.add_argument("--final_seeds", type=int, nargs="*", default=[8, 9, 10, 11, 12],
                   help="seeds of the results table, used for the points at the default configuration")
    p.add_argument("--min_seeds", type=int, default=2, help="points with fewer finished seeds are not drawn")
    p.add_argument("--out", default="buffer_ratio_protomem.pdf")
    flags = p.parse_args()
    studies, final, bl = read_studies_dir(flags.studies), load_runs(flags.final, 9), read_buffer_runs(flags.buffer_bl)
    all_seeds = range(100)

    ratios, sizes = [0.0, 0.5, 1.0], [0, 50, 100]
    series = {}
    for name in STYLE:
        pts = []
        for r, nb in zip(ratios, sizes):
            # default configurations = the runs of the results table (final, --final_seeds); other points: all seeds
            seeds = flags.final_seeds
            if name == "ProtoMem":
                mats = final.get("ProtoMem", {}) if nb == 100 else studies.get(f"buffer {nb}", {})
                seeds = flags.final_seeds if nb == 100 else all_seeds
            elif nb == 0 and name in EMPTY_IS_FINETUNING:
                mats = final.get("Finetuning", {})
            elif nb == DEFAULT_BUFFER[name] and nb not in bl.get(name, {}):
                mats = final.get(name, {})
            else:
                mats, seeds = bl.get(name, {}).get(nb, {}), all_seeds
            m, cnt = mean_point(mats, seeds)
            if m is not None and cnt < flags.min_seeds:
                print(f"{name} ratio {r}: only {cnt} seed(s), not drawn")
            elif m is not None:
                pts.append((r, m, cnt))
        series[name] = pts
        print(f"{STYLE[name][0]:6s} " + "  ".join(f"ratio {r}: ACC {m[0]:6.2f} BWT {m[1]:6.2f} ({c} seeds)"
                                                for r, m, c in pts))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8.5))
    for k, (ax, title, ylab) in enumerate(zip(
            axes, ["Effect of buffer size on average classification accuracy (ACC)",
                   "Effect of buffer size on backward transfer (BWT)"], ["ACC (%)", "BWT (%)"])):
        for name, (lab, c, mk) in STYLE.items():
            pts = series[name]
            if pts:
                ax.plot([r for r, _, _ in pts], [m[k] for _, m, _ in pts], color=c, marker=mk, lw=1.8 if
                        name == "ProtoMem" else 1.2, ms=6, label=lab, zorder=5 if name == "ProtoMem" else 3)
        if k == 1:
            ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(ratios)
        ax.set_xlabel("Buffer ratio")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(flags.out, bbox_inches="tight")
    plt.savefig(flags.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"saved {flags.out}")


if __name__ == "__main__":
    main()

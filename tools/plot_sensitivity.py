"""
Stability / plasticity heatmaps of ProtoMem over (gen_reg, tau), mean over seeds.
  stability  = BWT (accuracy change on past subjects; higher = better retention)
  plasticity = mean accuracy on each subject right after learning it (diagonal of the accuracy matrix)

    python tools/plot_sensitivity.py sensitivity final_bal
The centre cell (0.5, 0.1) is read from <final_bal>/final/protomem/ (same seeds as the sensitivity runs).
"""
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

GEN = [0.1, 0.5, 1.0]
TAU = [0.05, 0.1, 0.2]
plt.rcParams.update({"font.family": "serif"})


def load(folder):
    mats = {}
    for f in glob.glob(os.path.join(folder, "protomem_9_subjects_seed_*", "accuracy_matrix.txt")):
        mats[int(os.path.basename(os.path.dirname(f)).rsplit("_", 1)[1])] = np.loadtxt(f)
    return mats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sens_dir")
    p.add_argument("final_dir")
    p.add_argument("--out", default="sensitivity_protomem.pdf")
    flags = p.parse_args()

    grids = {s: {} for s in ("bwt", "plast")}
    seeds_used = None
    for g in GEN:
        for t in TAU:
            mats = load(os.path.join(flags.sens_dir, f"g{g}_t{t}"))
            if g == 0.5 and t == 0.1:
                centre = load(os.path.join(flags.final_dir, "final", "protomem"))
                mats = {s: a for s, a in centre.items() if seeds_used is None or s in seeds_used}
            if not mats:
                print(f"gen={g} tau={t}: no results")
                continue
            seeds_used = seeds_used or set(mats)
            n = 9
            grids["bwt"][(g, t)] = np.mean([np.sum(a[-1] - np.diag(a)) / (n - 1) for a in mats.values()])
            grids["plast"][(g, t)] = np.mean([np.diag(a).mean() for a in mats.values()])
            print(f"gen={g} tau={t}: BWT {grids['bwt'][(g, t)]:6.2f}  plasticity {grids['plast'][(g, t)]:6.2f}  "
                  f"seeds {sorted(mats)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, title, cmap, cbar in [(axes[0], "bwt", "Stability", "RdBu_r", "Accuracy change, BWT (%)"),
                                       (axes[1], "plast", "Plasticity", "viridis", "Accuracy on new subjects (%)")]:
        M = np.full((len(TAU), len(GEN)), np.nan)
        for i, t in enumerate(TAU):
            for j, g in enumerate(GEN):
                M[i, j] = grids[key].get((g, t), np.nan)
        kw = {}
        if key == "bwt":
            lim = np.nanmax(np.abs(M))
            kw = dict(vmin=-lim, vmax=lim)
        im = ax.imshow(M, origin="lower", cmap=cmap, aspect="auto", **kw)
        for i in range(len(TAU)):
            for j in range(len(GEN)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                            color="white" if key == "plast" and M[i, j] < np.nanmean(M) else "black")
        ax.set_xticks(range(len(GEN))); ax.set_xticklabels([str(g) for g in GEN])
        ax.set_yticks(range(len(TAU))); ax.set_yticklabels([str(t) for t in TAU])
        ax.set_xlabel(r"Prototype (pseudo-feature) loss weight $\lambda_{\mathrm{gen}}$")
        ax.set_ylabel(r"Classifier temperature $\tau$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label=cbar)
    plt.tight_layout()
    plt.savefig(flags.out, bbox_inches="tight")
    plt.savefig(flags.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"saved {flags.out}")


if __name__ == "__main__":
    main()

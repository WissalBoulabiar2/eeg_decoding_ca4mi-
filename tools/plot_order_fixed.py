"""
Subject-order figure from the fixed-split runs (same train/valid/test trials of every subject in both orders).

    python tools/plot_order_fixed.py <results/protomem dir> [--seq full_K1 --rnd full_order_random]
(a) average accuracy on the subjects seen so far after each stage, (b) final ACC and BWT, mean +- std over seeds.
"""
import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif"})
RANDOM_ORDER = "S4, S5, S7, S3, S9, S1, S2, S6, S8"


def load(d):
    return [np.loadtxt(f) for f in sorted(glob.glob(os.path.join(d, "protomem_9_subjects_seed_*", "accuracy_matrix.txt")))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("protomem_dir")
    p.add_argument("--seq", default="full")
    p.add_argument("--rnd", default="full_order_random")
    p.add_argument("--out", default="order_fixed_split.pdf")
    f = p.parse_args()
    seq, rnd = load(os.path.join(f.protomem_dir, f.seq)), load(os.path.join(f.protomem_dir, f.rnd))
    curve = lambda ms: np.array([[a[t, :t + 1].mean() for t in range(len(a))] for a in ms])
    final = lambda ms: np.array([[a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (len(a) - 1)] for a in ms])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.9, 1]})
    t = np.arange(1, 10)
    for ms, c, lab in [(seq, "blue", "Sequential order  (S1, S2, S3, ...)"),
                       (rnd, "#e8463c", f"Random order  ({RANDOM_ORDER[:10]}, ...)")]:
        cv = curve(ms)
        a1.plot(t, cv.mean(0), color=c, marker="o", lw=2, label=lab)
        a1.fill_between(t, cv.mean(0) - cv.std(0), cv.mean(0) + cv.std(0), color=c, alpha=0.12)
    a1.set_xticks(t); a1.set_xticklabels([f"T{i}" for i in t])
    a1.set_xlabel("Training stage"); a1.set_ylabel("Avg. accuracy on seen subjects (%)")
    a1.set_title("(a) Accuracy during subject-incremental learning"); a1.grid(alpha=0.3); a1.legend(loc="lower right")

    xb, w = np.arange(2), 0.38
    for off, ms, c, lab in [(-w / 2, seq, "blue", "Sequential order"), (w / 2, rnd, "#e8463c", "Random order")]:
        v = final(ms); m, s = v.mean(0), v.std(0)
        bars = a2.bar(xb + off, m, w, yerr=s, capsize=4, color=c, label=lab)
        for b, val, e in zip(bars, m, s):
            a2.text(b.get_x() + b.get_width() / 2, val + e + 1.5, f"{val:.1f}", ha="center", fontsize=10)
        print(f"{lab}: ACC {m[0]:.2f} +- {s[0]:.2f}  BWT {m[1]:.2f} +- {s[1]:.2f}  ({len(ms)} seeds)")
    a2.set_ylim(top=a2.get_ylim()[1] * 1.12)                 # room for the value labels
    a2.axhline(0, color="gray", lw=0.8)
    a2.set_xticks(xb); a2.set_xticklabels(["ACC (%)", "BWT (%)"])
    a2.set_title("(b) Final performance"); a2.grid(axis="y", alpha=0.3); a2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f.out, bbox_inches="tight")
    plt.savefig(f.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"saved {f.out}")


if __name__ == "__main__":
    main()

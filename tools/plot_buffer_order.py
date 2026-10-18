"""
ProtoMem figures: (1) effect of the buffer size, (2) effect of the subject order. Mean ± std over the final seeds.

Inputs (saved logs):
  - the log of main/run_protomem_studies.py (buffer 0/20/50/200, random order)
  - the log of main/run_final.py (results.txt): ProtoMem with 100 exemplars in the sequential order, and the
    baselines at their default configuration, drawn as horizontal reference lines in the buffer figure

    python tools/plot_buffer_order.py results_studies.txt results.txt
"""
import argparse
import os
import re
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_stages import parse_log

STUDY = re.compile(r"\[job \d+/\d+\] \[(buffer \d+|order random)\] ProtoMem seed (\d+)")
RANDOM_ORDER = "S4, S5, S7, S3, S9, S1, S2, S6, S8"
REFERENCES = {  # baselines drawn as reference lines: name -> (color, linestyle)
    "CGER": ("orange", "--"), "MUDVI": ("purple", "--"), "CA4MI": ("teal", "--"), "ER": ("green", "--"),
}
plt.rcParams.update({"font.family": "serif"})


def parse_studies(path, n):
    """{tag: {seed: matrix}} from the studies log."""
    out, tag, seed, rows = defaultdict(dict), None, None, []
    for line in open(path, encoding="utf-8", errors="replace").read().splitlines():
        h = STUDY.search(line)
        if h:
            tag, seed, rows = h.group(1), int(h.group(2)), []
        elif tag and line.strip().startswith("test acc:"):
            rows.append([float(v) for v in line.split(":", 1)[1].split()])
            if len(rows) == n:
                a = np.zeros((n, n))
                for t, r in enumerate(rows):
                    a[t, :len(r)] = r
                out[tag][seed] = a
    return out


def read_studies_dir(root):
    """{tag: {seed: matrix}} from the result folders of run_protomem_studies.py (buffer_<n>/, order_random/)."""
    import glob
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(root, "**", "protomem_*_subjects_seed_*", "accuracy_matrix.txt"), recursive=True):
        parts = os.path.normpath(f).split(os.sep)
        study = next((p for p in parts if p.startswith("buffer_") or p == "order_random"), None)
        if study is None:
            continue
        tag = f"buffer {study.split('_')[1]}" if study.startswith("buffer_") else "order random"
        out[tag][int(parts[-2].rsplit("_", 1)[1])] = np.loadtxt(f)
    return out


def acc_bwt(a):
    n = len(a)
    return a[-1].mean(), np.sum(a[-1] - np.diag(a)) / (n - 1)


def stats(mats):
    v = np.array([acc_bwt(a) for a in mats.values()])
    return v.mean(0), v.std(0)


def seen_curve(mats):
    return np.mean([[a[t, :t + 1].mean() for t in range(len(a))] for a in mats.values()], axis=0)


def buffer_figure(studies, final, out):
    sizes, means, stds = [], [], []
    # 200 is not drawn: on BCI-IV 2a a subject has 100 training trials, so 200 stores the same trials as 100
    for nb in [0, 20, 50, 100]:
        mats = final.get("ProtoMem", {}) if nb == 100 else studies.get(f"buffer {nb}", {})
        if not mats:
            print(f"buffer {nb}: no run found, skipped")
            continue
        m, s = stats(mats)
        sizes.append(nb); means.append(m); stds.append(s)
        print(f"buffer {nb:3d}: ACC {m[0]:.2f} ± {s[0]:.2f}  BWT {m[1]:.2f} ± {s[1]:.2f}  ({len(mats)} seeds)")
    means, stds, x = np.array(means), np.array(stds), np.arange(len(sizes))

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9))
    for k, (ax, name) in enumerate(zip(axes, ["average classification accuracy (ACC)", "backward transfer (BWT)"])):
        for ref, (c, ls) in REFERENCES.items():
            if ref in final:
                ax.axhline(stats(final[ref])[0][k], color=c, ls=ls, lw=1.2, alpha=0.8,
                           label=f"{ref} (default config.)")
        ax.errorbar(x, means[:, k], yerr=stds[:, k], color="blue", marker="s", ms=7, lw=2, capsize=4,
                    label="Ours (ProtoMem)", zorder=5)
        if k == 1:
            ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n(all trials)" if s == 100 else str(s) for s in sizes])
        ax.set_xlabel("Exemplars stored per subject")
        ax.set_ylabel("ACC (%)" if k == 0 else "BWT (%)")
        ax.set_title(f"Effect of buffer size on {name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"saved {out}")


def order_figure(studies, final, out):
    seq, rnd = final.get("ProtoMem", {}), studies.get("order random", {})
    if not seq or not rnd:
        print("order figure: sequential or random runs missing, skipped")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.9, 1]})
    t = np.arange(1, 10)
    a1.plot(t, seen_curve(seq), color="blue", marker="o", lw=2, label="Sequential order  (S1, S2, S3, ...)")
    a1.plot(t, seen_curve(rnd), color="#e8463c", marker="o", lw=2, label=f"Random order  ({RANDOM_ORDER[:10]}, ...)")
    a1.set_xticks(t); a1.set_xticklabels([f"T{i}" for i in t])
    a1.set_xlabel("Training stage"); a1.set_ylabel("Avg. accuracy on seen subjects (%)")
    a1.set_title("(a) Accuracy during subject-incremental learning"); a1.grid(alpha=0.3); a1.legend(loc="lower right")

    (ms, ss), (mr, sr) = stats(seq), stats(rnd)
    xb, w = np.arange(2), 0.38
    for off, m, s, c, lab in [(-w / 2, ms, ss, "blue", "Sequential order"), (w / 2, mr, sr, "#e8463c", "Random order")]:
        bars = a2.bar(xb + off, m, w, yerr=s, capsize=4, color=c, label=lab)
        for b, v, e in zip(bars, m, s):
            a2.text(b.get_x() + b.get_width() / 2, v + e + 1.5 if v >= 0 else v - e - 5, f"{v:.1f}",
                    ha="center", fontsize=10)
    a2.axhline(0, color="gray", lw=0.8)
    a2.set_xticks(xb); a2.set_xticklabels(["ACC (%)", "BWT (%)"])
    a2.set_title("(b) Final performance"); a2.grid(axis="y", alpha=0.3); a2.legend(loc="upper right")
    fig.suptitle("Ours on BCI-IV 2a: effect of subject order", fontsize=14)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"sequential: ACC {ms[0]:.2f} ± {ss[0]:.2f}  BWT {ms[1]:.2f} ± {ss[1]:.2f} | "
          f"random: ACC {mr[0]:.2f} ± {sr[0]:.2f}  BWT {mr[1]:.2f} ± {sr[1]:.2f}")
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("studies_log")
    parser.add_argument("final_log")
    parser.add_argument("--n_subjects", type=int, default=9)
    flags = parser.parse_args()
    if os.path.isdir(flags.studies_log):
        studies = read_studies_dir(flags.studies_log)
    elif os.path.exists(flags.studies_log):
        studies = parse_studies(flags.studies_log, flags.n_subjects)
    else:
        studies = {}
    print({k: sorted(v) for k, v in studies.items()})
    final = parse_log(flags.final_log, flags.n_subjects)
    buffer_figure(studies, final, "buffer_protomem.pdf")
    order_figure(studies, final, "order_protomem.pdf")


if __name__ == "__main__":
    main()

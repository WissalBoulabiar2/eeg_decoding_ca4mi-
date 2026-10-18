"""
Architecture figure of ProtoMem (same layout idea as docs/architecture_p0-5.png):
(A) end-to-end architecture, (B) continual-learning and prototype-memory mechanisms.

    python tools/draw_architecture.py            -> architecture_protomem.pdf / .png
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Rectangle

BLUE, LBLUE = "#2f6fb5", "#e6f0fa"
GREEN, LGREEN = "#3a9a3a", "#e8f5e6"
PURPLE, LPURPLE = "#6a4fb0", "#efeafa"
ORANGE, LORANGE = "#e07b1f", "#fdf0e2"
GREY, LGREY = "#555555", "#f4f4f4"
RED = "#d6453d"
rng = np.random.default_rng(3)

fig = plt.figure(figsize=(17, 13.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.4, r=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))


def txt(x, y, s, size=10, weight="normal", color="black", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, **kw)


def arrow(x1, y1, x2, y2, color="black", ls="-", lw=1.6, rad=0.0, style="-|>", ms=14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}"))


def eeg(x, y, w, h, color="black", seed=0):
    t = np.linspace(0, 1, 200)
    s = np.random.default_rng(seed).standard_normal(200).cumsum()
    s = (s - s.mean()) / (np.abs(s - s.mean()).max() + 1e-9)
    ax.plot(x + t * w, y + h / 2 + s * h / 2, color=color, lw=0.8)


def heat(x, y, w, h, seed=0, cmap="Blues", eye=False):
    m = np.eye(6) * 0.8 + 0.2 * np.random.default_rng(seed).random((6, 6)) if eye else \
        np.random.default_rng(seed).random((6, 6))
    m = (m + m.T) / 2
    ax.imshow(m, cmap=cmap, extent=(x, x + w, y, y + h), zorder=3, vmin=0, vmax=1.2)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=GREY, lw=0.6, zorder=4))


def vec(x, y, w, h, color, n=6):
    for i in range(n):
        ax.add_patch(Rectangle((x, y + i * h / n), w, h / n * 0.9, fc=color, ec="white", lw=0.5, alpha=0.4 + 0.6 * (i % 3) / 2))


# ======================================================================== (A)
box(0.6, 50.5, 98.8, 48.8, "white", GREY, lw=1.0, r=0.8)
txt(1.5, 97.8, "(A) ProtoMem end-to-end architecture", 15, "bold", ha="left")

# subject stream
box(2, 62, 12, 33.5, LBLUE, BLUE, lw=1.0)
for k, (yy, lab) in enumerate([(88, "S$_1$"), (77, "S$_2$"), (66, "S$_9$")]):
    txt(3.6, yy + 3.5, lab, 13, "bold")
    box(5.5, yy, 7.8, 7.5, "white", "#9bb8d8", lw=0.8, r=0.6)
    txt(9.4, yy + 6.3, "22 ch × 1000", 7.5)
    eeg(6.0, yy + 3.2, 6.8, 2.2, seed=k)
    txt(9.4, yy + 1.2, "MI labels\ntrain/val/test", 6.2, color=GREY)
txt(9.4, 75.6, "⋮", 14)
txt(8, 63.5, "subject stream $\\mathcal{D}_1,\\dots,\\mathcal{D}_N$", 8.5, color=BLUE)
arrow(14.2, 79, 17.3, 79)

# Euclidean alignment
box(17.5, 62, 13.5, 33.5, LBLUE, BLUE, lw=1.0)
txt(24.25, 93.2, "Euclidean alignment", 10.5, "bold")
txt(24.25, 90.5, "per subject, from its\ntraining trials only", 8.5, color=GREY)
heat(21.0, 81.5, 6.5, 6.0, seed=4)
txt(24.25, 79.8, "$R_k=\\frac{1}{n_k}\\sum_i \\mathrm{cov}(X_k^i)$", 9.5)
arrow(24.25, 78.3, 24.25, 76.4, lw=1.2)
heat(21.0, 69.5, 6.5, 6.0, seed=5, eye=True)
txt(24.25, 67.2, "$\\tilde X=R_k^{-1/2}X$", 10)
txt(24.25, 64.0, "valid/test aligned with $R_k$", 7.8, color=GREY)
arrow(31.2, 79, 34.3, 79)

# encoder + subject BN
box(34.5, 62, 22, 33.5, LBLUE, BLUE, lw=1.0)
txt(45.5, 93.2, "Single DeepConvNet encoder $E$", 11.5, "bold", color=BLUE)
txt(45.5, 90.5, "convolutions shared by all subjects", 8.5, color=GREY)
for i, hh in enumerate([10, 8.5, 7, 5.5, 4]):
    ax.add_patch(Rectangle((36.5 + i * 3.6, 84 - hh / 2), 2.2, hh, fc=BLUE, ec="white", alpha=0.75))
box(36, 63.5, 19, 13.8, LGREEN, GREEN, lw=1.0)
txt(45.5, 75.6, "subject-specific BatchNorm bank", 9.5, "bold", color=GREEN)
for i, lab in enumerate(["BN S$_1$", "BN S$_2$", "…", "BN S$_k$"]):
    fc = "#cfe8cb" if i < 3 else GREEN
    box(36.8 + i * 4.6, 68.5, 4.1, 4.5, fc, GREEN, lw=0.8, r=0.5)
    txt(38.85 + i * 4.6, 70.75, lab, 7.6, color="white" if i == 3 else "black")
txt(45.5, 65.4, "past subjects frozen  •  $\\theta^{BN}_k\\leftarrow\\theta^{BN}_{k-1}$", 7.8, color=GREEN)
arrow(56.7, 79, 59.8, 79)

# embedding
vec(60.3, 74, 2.2, 10, BLUE)
txt(61.4, 87.0, "$z\\in\\mathbb{R}^{128}$", 10.5, "bold")
txt(61.4, 71.7, "$\\ell_2$-normalized", 7.8, color=GREY)
arrow(63.0, 79, 66.1, 79)

# proxy classifier
box(66.3, 62, 19.5, 33.5, LPURPLE, PURPLE, lw=1.0)
txt(76.05, 93.2, "Multi-proxy cosine classifier", 11.5, "bold", color=PURPLE)
txt(76.05, 90.5, "one classifier shared by all subjects", 8.5, color=GREY)
cx, cy, rr = 76.05, 79.5, 7.2
ax.add_patch(Circle((cx, cy), rr, fill=False, ec=GREY, lw=0.8, ls="--"))
for ang, c in zip(np.deg2rad([25, 55, 85, 205, 235, 265]), [ORANGE] * 3 + [GREEN] * 3):
    ax.add_patch(Circle((cx + rr * np.cos(ang), cy + rr * np.sin(ang) * 0.95), 0.75, fc=c, ec="black", lw=0.5, zorder=5))
ax.add_patch(Circle((cx + 2.5, cy + 1.2), 0.5, fc=BLUE, ec="black", lw=0.5, zorder=5))
txt(cx + 3.6, cy + 2.3, "$z$", 9)
txt(cx, 70.0, "$K=3$ proxies per class on the unit sphere", 8.2)
txt(cx, 66.3, "$f_c(z)=\\log\\sum_j e^{-\\|z-\\hat p_{c,j}\\|^2/\\tau}$,  $\\tau=0.1$", 9.5)
arrow(86.0, 79, 89.3, 79)

# prediction
txt(93.4, 86.5, "motor-imagery\nclass prediction", 10.5, "bold")
for i, c in enumerate([GREEN, RED]):
    ax.add_patch(Circle((91.0 + i * 4.8, 79), 1.5, fc=c, ec="black", lw=0.6))
txt(93.4, 75.2, "0  /  1", 10)
txt(93.4, 72.6, "left / right hand", 8.2)
txt(93.4, 67.8, "subject index\nknown at test:\nselects its BN only", 7.6, color=GREY)

# loss + config
box(28, 52.3, 44, 6.6, "white", "black", lw=1.0, r=0.8)
txt(50, 55.6, "$\\mathcal{L}_{total}=\\mathcal{L}_{cls}+1.0\\,\\mathcal{L}_{dis}+0.5\\,\\mathcal{L}_{gen}$", 16)
box(2, 52.3, 22.5, 7.5, LGREY, GREY, lw=0.8, ls="--")
txt(13.25, 56.05, "BCI-IV 2a | 9 subjects | 2 classes\n22 ch × 1000 | DCN, $D=128$ | Adam\nno adversarial / orthogonality / mixup", 8.2)
box(75.5, 52.3, 22.5, 7.5, LPURPLE, PURPLE, lw=0.8, ls="--")
txt(86.75, 56.05, "$\\mathcal{L}_{dis}$: logit consistency on exemplars\n$\\mathcal{L}_{gen}$: CE on pseudo-features\nsampled from the memory (panel B)", 8.2)

# ======================================================================== (B)
box(0.6, 7.6, 98.8, 42.0, "white", GREY, lw=1.0, r=0.8)
txt(1.5, 48.0, "(B) Continual-learning and prototype-memory mechanisms (stage $k$)", 15, "bold", ha="left")

# train current subject
box(2, 24, 15, 19, "#fff7d6", "#c9a227", lw=1.0)
txt(9.5, 40.6, "Train subject $S_k$", 11.5, "bold")
txt(9.5, 35.8, "current trials\n+ exemplars (own BN)\n+ pseudo-features", 9)
txt(9.5, 28.6, "model selected on the\nvalidation CE of $S_k$", 8, color=GREY)
arrow(17.2, 33.5, 20.8, 33.5)

# exemplar buffer
box(21, 33.5, 22, 12.3, LORANGE, ORANGE, lw=1.0)
txt(32, 43.6, "Exemplar buffer $\\mathcal{R}$", 11, "bold", color=ORANGE)
txt(32, 39.6, "100 class-balanced trials / subject,\nstored with their logits and embeddings", 8.5)
txt(32, 35.4, "replayed through their frozen BN  →  $\\mathcal{L}_{dis}$", 8.5, color=ORANGE)

# memory
box(21, 9.5, 44, 22.5, LPURPLE, PURPLE, lw=1.4)
txt(43, 30.0, "Gaussian prototype memory $\\mathcal{M}$  (fixed capacity $B=24$)", 11.5, "bold", color=PURPLE)
for i in range(7):
    c = ORANGE if i < 4 else GREEN
    ex, ey = 25 + i * 2.6 + rng.uniform(-0.5, 0.5), 18 + rng.uniform(-3.5, 4.5)
    ax.add_patch(Ellipse((ex, ey), rng.uniform(2.2, 3.6), rng.uniform(1.2, 2.2), angle=rng.uniform(0, 180),
                         fc=c, ec="black", lw=0.5, alpha=0.55))
txt(30.5, 11.6, "modes $(\\mu,\\sigma^2,n,c,s)$\n2 per class and subject", 8.2)
box(41.5, 20.0, 10.5, 7.8, "white", PURPLE, lw=0.8, r=0.6)
txt(46.75, 23.9, "class-balanced\n12 modes / class", 8.5)
box(41.5, 11.0, 10.5, 7.8, "white", PURPLE, lw=0.8, r=0.6)
txt(46.75, 14.9, "Ward merging\n$\\frac{n_in_j}{n_i+n_j}\\|\\mu_i-\\mu_j\\|^2$", 8.5)
box(53.3, 11.0, 10.5, 16.8, "white", PURPLE, lw=0.8, r=0.6)
txt(58.55, 19.4, "pseudo-feature\nreplay\n$\\tilde z\\sim\\mathcal{N}(\\mu_m,\\sigma_m^2)$\n→ $\\mathcal{L}_{gen}$\n(proxies only)", 8.5)

# after the stage
box(68, 9.5, 14.5, 36.3, "#f0f7ff", BLUE, lw=1.0)
txt(75.25, 43.6, "After training $S_k$", 11, "bold", color=BLUE)
steps = ["① balanced consolidation\n10 epochs, lr × 0.1,\nall subjects equally weighted",
         "② refresh stored logits",
         "③ drift compensation\n$\\mu_m \\leftarrow \\mu_m + \\delta_{s_m,c_m}$\n(measured on exemplars)",
         "④ add modes of $S_k$ to $\\mathcal{M}$\n+ exemplars of $S_k$ to $\\mathcal{R}$"]
for i, s in enumerate(steps):
    txt(75.25, 38.0 - i * 7.6, s, 8.4)
arrow(65.3, 20, 67.8, 20, color=PURPLE)

# evaluation
box(85.5, 24, 12.5, 21.8, LGREY, GREY, lw=1.0)
txt(91.75, 43.4, "evaluated after\nevery stage", 10, "bold")
for i, (lab, cm) in enumerate([("ACC", "Blues"), ("F1", "Greens")]):
    heat(86.6 + i * 5.8, 33.0, 4.6, 4.6, seed=10 + i, cmap=cm)
    txt(88.9 + i * 5.8, 31.6, lab, 8.5)
txt(91.75, 27.8, "BWT  •  forgetting", 9)
arrow(82.7, 33.5, 85.3, 33.5)

# feedback to the next stage
ax.plot([75.25, 75.25, 9.5], [9.4, 8.5, 8.5], color=PURPLE, ls="--", lw=1.6)
arrow(9.5, 8.5, 9.5, 23.8, color=PURPLE, ls="--")
txt(43, 8.5, " next stage $S_{k+1}$: memory, exemplars and frozen BN feed training ", 8.5, color=PURPLE,
    bbox=dict(fc="white", ec="none", pad=0.6))

# legend
box(0.6, 0.6, 98.8, 6.0, "white", GREY, lw=0.8, r=0.6)
ax.plot([2, 5.5], [4.6, 4.6], color="black", lw=1.6)
txt(6.2, 4.6, "= forward computation", 9.5, ha="left")
ax.plot([2, 5.5], [2.3, 2.3], color="black", lw=1.6, ls="--")
txt(6.2, 2.3, "= loss / feedback", 9.5, ha="left")
for x0, (c, lab) in zip([23, 38.5, 60.5, 84],[(BLUE, "shared"), (GREEN, "subject-specific (BN)"),
                                            (PURPLE, "prototype memory / classifier"), (ORANGE, "exemplar replay")]):
    ax.add_patch(Rectangle((x0, 2.6), 2.4, 2.0, fc=c, ec="black", lw=0.5))
    txt(x0 + 3.0, 3.6, f"= {lab}", 9.5, ha="left")

plt.savefig("architecture_protomem.pdf", bbox_inches="tight")
plt.savefig("architecture_protomem.png", dpi=220, bbox_inches="tight")
print("saved architecture_protomem.pdf / .png")

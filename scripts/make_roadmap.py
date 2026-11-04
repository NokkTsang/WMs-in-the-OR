"""Draw the project roadmap (encoding -> predicting -> generative); green = tested with preliminary conclusion, red = to do."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

OUT = Path(__file__).resolve().parent.parent / "docs" / "roadmap.png"
GREEN, RED, GREY = ("#d7f0d9", "#2e7d32"), ("#fbd5d5", "#c62828"), ("#eeeeee", "#757575")

fig, ax = plt.subplots(figsize=(17, 8.2))
ax.set_xlim(0, 17)
ax.set_ylim(0, 8.2)
ax.axis("off")


def box(x, y, w, h, color, title, body, title_size=12.5, body_size=10, top=0.95):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.15", fc=color[0], ec=color[1], lw=2))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=color[1])
    ax.text(x + 0.2, y + h - top, body, ha="left", va="top", fontsize=body_size, linespacing=1.45)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=22, lw=2, color="#444444"))


ax.text(
    8.5,
    7.95,
    "OR world-model project: roadmap (status after the tests of 2026-11-04)",
    ha="center",
    fontsize=16,
    fontweight="bold",
)

box(
    0.2,
    2.2,
    2.9,
    4.6,
    GREY,
    "Data",
    "EgoExOR (Legacy)\n41 takes (16 MISS + 25 US)\n\nexo + ego cameras,\nscene graph (every frame),\nphase, gaze, hands, audio\n\nMM-OR: tested by the\nother group (to merge)",
)
box(
    3.9,
    2.2,
    3.5,
    4.6,
    GREEN,
    "1  Encoding",
    "Video clip  ->  latent z\n\nTested (frozen): V-JEPA 2,\nSurgMotion, DINOv2,\nSurgeNet, CAFormer\n\nResult:\n- V-JEPA 2 is best for\n  predicting change\n- video > image encoders\n- surgical pre-training:\n  no robust gain",
)
box(
    8.2,
    4.7,
    4.7,
    2.1,
    GREEN,
    "2a  Predicting: latent space",
    "z(t-k..t) -> z(t+h)   vs  copy-last\n\nResult: ridge best (nMSE 0.80 / 0.60\nat +4 s); MLP, GRU, Transformer\nno gain at this data size",
    body_size=9.5,
    top=0.85,
)
box(
    8.2,
    2.2,
    4.7,
    2.3,
    GREEN,
    "2b  Predicting: structured OR state",
    "scene graph: which relations will\nappear / disappear (onset / offset AUC)\n\nResult: latent beats privileged controls\non offsets (+0.04 to +0.13)",
    title_size=12,
    body_size=9.5,
    top=0.85,
)
box(
    8.2,
    0.2,
    4.7,
    1.75,
    RED,
    "2c  To do",
    "joint MISS + US training, token-level\nfeatures, ego cams / gaze / hands,\nfine-tune V-JEPA 2, seeds + CIs",
    title_size=11.5,
    body_size=9.5,
    top=0.85,
)
box(
    13.7,
    2.2,
    3.2,
    4.6,
    RED,
    "3  Generative",
    "Generate future OR\nvideo / scenes from the\npredicted state\n\nNot started.\n\nCandidate (to be checked):\nNVIDIA Cosmos-H-Surgical\n(domain fit, licence,\nconditioning on OR state)",
)

arrow(3.15, 4.5, 3.85, 4.5)
arrow(7.45, 4.5, 8.15, 5.75)
arrow(7.45, 4.5, 8.15, 3.35)
arrow(12.95, 5.75, 13.65, 5.75)
arrow(12.95, 3.35, 13.65, 3.35)

ax.legend(
    handles=[
        Patch(fc=GREEN[0], ec=GREEN[1], label="tested, preliminary conclusion"),
        Patch(fc=RED[0], ec=RED[1], label="to do"),
        Patch(fc=GREY[0], ec=GREY[1], label="data"),
    ],
    loc="lower left",
    bbox_to_anchor=(0.0, 0.0),
    fontsize=10,
    frameon=False,
)
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("saved", OUT)

# %% [markdown]
# # Synthesis across encoders and cameras
# Reads the tables of notebook 01 (`outputs/probe/<procedure>/summary.csv`) and recomputes the per-take phase accuracy of every encoder × camera.
# The procedure is chosen with `EGOEXOR_PROC` (`MISS` default, or `Ultrasound`); results go to `outputs/synthesis/<procedure>/`.
# Each cell is preceded by **Does** and **Shows**.

# %%
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import egoexor_io as eio
from probe_utils import ENCODERS, full_tags, load_splits, load_tag, loto_phase, split_tag

pd.set_option("display.width", 220)
PROC = os.environ.get("EGOEXOR_PROC", "MISS")
OUT = eio.out_dir(f"synthesis/{PROC}")
summary = pd.read_csv(eio.OUT_DIR / "probe" / PROC / "summary.csv", index_col=0)
summary["encoder"] = [split_tag(t)[0] for t in summary.index]
summary["cam"] = [split_tag(t)[1] for t in summary.index]

# %% [markdown]
# ## 1. Encoder × camera heatmaps
# **Does:** one heatmap per metric of notebook 01 (rows = encoders, columns = cameras). **Shows:** how much the *camera* matters compared with the *encoder*; the spread over cameras of one encoder is the camera effect, the spread over encoders in one column is the encoder effect.

# %%
METRICS = {
    "phase_bal_acc": "phase balanced accuracy (higher better)",
    "take_id_acc": "take-ID accuracy (lower = less recording-specific)",
    "sg_auc_now": "scene-graph AUC now",
    "sg_auc_+4s": "scene-graph AUC at +4 s",
    "nMSE_+1s": "latent nMSE +1 s (< 1 beats copy-last)",
    "nMSE_+4s": "latent nMSE +4 s (< 1 beats copy-last)",
}
cams = sorted(summary.cam.unique())
fig, axes = plt.subplots(2, 3, figsize=(5 * 3, 2.6 * 2 + 1.5))
for ax, (m, title) in zip(axes.ravel(), METRICS.items()):
    piv = summary.pivot(index="encoder", columns="cam", values=m).reindex(index=ENCODERS, columns=cams)
    reverse = m.startswith("nMSE") or m == "take_id_acc"
    ax.imshow(piv.values, cmap="viridis_r" if reverse else "viridis", aspect="auto")
    ax.set_xticks(range(len(cams)), [c.replace("external_", "ext") for c in cams], fontsize=8)
    ax.set_yticks(range(len(ENCODERS)), ENCODERS, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    ax.set_title(title, fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "heatmaps.png", dpi=110)
plt.show()

# %% [markdown]
# ## 2. Camera effect vs encoder effect
# **Does:** for every metric, the spread (max - min) of the encoder-averaged value over cameras, and of the camera-averaged value over encoders. **Shows:** which factor moves the result more. If the camera spread is larger than the encoder spread, choosing the viewpoint matters more than choosing the encoder.

# %%
rows = []
for m in METRICS:
    by_cam = summary.groupby("cam")[m].mean()
    by_enc = summary.groupby("encoder")[m].mean()
    rows.append(
        dict(
            metric=m,
            camera_spread=by_cam.max() - by_cam.min(),
            encoder_spread=by_enc.max() - by_enc.min(),
            best_cam=by_cam.idxmax(),
            worst_cam=by_cam.idxmin(),
        )
    )
spread_res = pd.DataFrame(rows).round(3)
spread_res

# %% [markdown]
# ## 3. Surgical vs general pre-training, pooled over cameras
# **Does:** recomputes the per-take phase accuracy of every tag (leave-one-take-out), then for each pair (same architecture, surgical vs general pre-training) averages the per-take difference over all cameras and bootstraps over takes (10,000 resamples, 95 % CI). Also counts the cameras where the surgical encoder is ahead.
# **Shows:** the net effect of surgical pre-training *after* removing the camera choice as a source of noise. Takes, not clips, are resampled (clips of a take are strongly correlated).

# %%
TAGS = full_tags(PROC)
splits = load_splits()
per_take = {}
for tag in TAGS:
    meta, X = load_tag(tag, PROC, splits)
    per_take[tag] = pd.Series(loto_phase(meta, X) == meta.phase.values).groupby(meta.take_id.values).mean()
per_take = pd.DataFrame(per_take)
per_take.to_csv(OUT / "phase_acc_per_take.csv")

rng = np.random.default_rng(0)
rows = []
for a, b in [("surgenet", "caformer_in1k"), ("surgmotion", "vjepa2")]:
    diffs = {}
    for cam in cams:
        ta, tb = f"{a}_{cam}_L16s4", f"{b}_{cam}_L16s4"
        if ta in per_take and tb in per_take:
            diffs[cam] = per_take[ta] - per_take[tb]
    d = pd.DataFrame(diffs)
    pooled = d.mean(1).values
    boots = rng.choice(pooled, size=(10000, len(pooled))).mean(1)
    rows.append(
        dict(
            pair=f"{a} - {b}",
            n_cams=d.shape[1],
            n_takes=len(pooled),
            pooled_diff=pooled.mean(),
            ci_lo=np.percentile(boots, 2.5),
            ci_hi=np.percentile(boots, 97.5),
            cams_ahead=int((d.mean() > 0.005).sum()),
            cams_behind=int((d.mean() < -0.005).sum()),
            per_cam_diff=", ".join(f"{c.replace('external_', 'ext')}:{v:+.3f}" for c, v in d.mean().items()),
        )
    )
pooled_res = pd.DataFrame(rows).round(3)
pooled_res

# %% [markdown]
# ## 4. Camera ranking
# **Does:** ranks cameras by the encoder-averaged phase accuracy and scene-graph AUC. **Shows:** which viewpoint to use for a world model of this procedure.

# %%
rank_res = (
    summary.groupby("cam")[["phase_bal_acc", "sg_auc_now", "sg_auc_+4s", "nMSE_+4s", "take_id_acc"]]
    .mean()
    .round(3)
    .sort_values("phase_bal_acc", ascending=False)
)
rank_res

# %% [markdown]
# ## Save

# %%
spread_res.to_csv(OUT / "camera_vs_encoder_spread.csv", index=False)
pooled_res.to_csv(OUT / "pooled_pretraining.csv", index=False)
rank_res.to_csv(OUT / "camera_ranking.csv")
print("saved to", OUT)

# %% [markdown]
# # Frozen-feature probes and latent prediction (one procedure, exocentric camera)
# The procedure is chosen with the environment variable `EGOEXOR_PROC` (`MISS` default, or `Ultrasound`); results go to `outputs/probe/<procedure>/`.
# Inputs: `features/<tag>/*.npz` (from `scripts/extract_features.py`) and `data/raw/splits.h5` / `*.h5` annotations.
# Each cell is preceded by **Does** (what it computes) and **Shows** (how to read the result).
# Protocol everywhere: **leave-one-take-out** (train on all other takes, test on the held-out take). A "clip" is 16 frames (step 4, ~4 s) and
# clips start every 15 frames (1 s), so neighbouring clips overlap ~75 %.

# %%
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import egoexor_io as eio
from probe_utils import (
    cos,
    fit_clf,
    full_tags,
    load_splits,
    load_tag,
    loto_phase,
    mean_auc,
    pairs,
    scene_graph_sets,
    stack,
    triplet_matrix,
)

pd.set_option("display.width", 200)
PROC = os.environ.get("EGOEXOR_PROC", "MISS")  # "MISS" or "Ultrasound"; phase labels are only comparable inside one procedure
OUT = eio.out_dir(f"probe/{PROC}")
HORIZONS = [1, 2, 4]  # in clips (1 clip step = 1 s)
TAGS = full_tags(PROC)  # Ultrasound `external_*` exist only in 18 takes and are analysed in notebook 02

# %% [markdown]
# ## 0. Load features and labels
# **Does:** loads every `features/<tag>` (encoder × camera) for `PROC`, with phase, take and official split per clip. **Shows:** the same clips for every tag (encoders and cameras are compared on identical samples).


# %%
splits = load_splits()
data = {t: load_tag(t, PROC, splits) for t in TAGS}
for t, (m, X) in data.items():
    print(f"{t}: {X.shape}, {m.take_id.nunique()} takes, phases {sorted(m.phase.unique())}")

# %% [markdown]
# ## 1. Phase probe
# **Does:** logistic regression on the clip feature predicts the phase of the held-out take. Baselines: majority class and the same pipeline with shuffled training labels.
# **Shows:** whether the frozen feature contains phase information that transfers to unseen recordings (chance = 1 / number of phases: 0.25 MISS, 0.20 Ultrasound; Ultrasound phases 1 and 2 are the same stage repeated). Per-take accuracy shows where it fails.


# %%
res = []
phase_pred = {}
for tag, (meta, X) in data.items():
    y = meta.phase.values
    pred = loto_phase(meta, X)
    phase_pred[tag] = pred
    maj = np.full_like(y, pd.Series(y).mode()[0])
    res.append(
        dict(
            tag=tag,
            n_clips=len(y),
            n_takes=meta.take_id.nunique(),
            macroF1=f1_score(y, pred, average="macro"),
            bal_acc=balanced_accuracy_score(y, pred),
            shuffled=balanced_accuracy_score(y, loto_phase(meta, X, shuffle=True)),
            majority=balanced_accuracy_score(y, maj),
        )
    )
    print(tag, pd.Series(pred == y).groupby(meta.take_id.values).mean().round(2).to_dict())
phase_res = pd.DataFrame(res).round(3)
phase_res

# %% [markdown]
# ## 2. Take-identity probe
# **Does:** predicts *which take* a clip comes from, training on the first 70 % of every take and testing on the last 30 % (gap in between). **Shows:** how much of the feature is recording-specific appearance; high values mean features partly encode the scene instance, not only the activity.

# %%
res = []
for tag, (meta, X) in data.items():
    pos = meta.groupby("take_id").clip.transform(lambda c: c / c.max())
    tr, te = (pos < 0.7).values, (pos >= 0.75).values
    sc, clf = fit_clf(X[tr], meta.take_id.values[tr], C=1.0)
    res.append(
        dict(
            tag=tag,
            take_id_acc=(clf.predict(sc.transform(X[te])) == meta.take_id.values[te]).mean(),
            chance=1 / meta.take_id.nunique(),
        )
    )
take_res = pd.DataFrame(res).round(3)
take_res

# %% [markdown]
# ## 3. UMAP
# **Does:** projects the features of the main camera (MISS `external_3`, Ultrasound `simstation`) to 2-D, coloured by phase (left) and by take (right). **Shows:** a visual check only: clusters by phase = phase information; one cluster per take = recording-specific appearance. Distances between clusters are not meaningful.

# %%
umap_data = {t: v for t, v in data.items() if ("_external_3_" if PROC == "MISS" else "_simstation_") in t}
fig, axes = plt.subplots(len(umap_data), 2, figsize=(11, 4.4 * len(umap_data)), squeeze=False)
for (tag, (meta, X)), (a1, a2) in zip(umap_data.items(), axes):
    Z = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    for ph in sorted(meta.phase.unique()):
        m = (meta.phase == ph).values
        a1.scatter(Z[m, 0], Z[m, 1], s=4, label=f"phase {ph}")
    a1.legend()
    a1.set_title(f"{tag}: phase")
    for tid in meta.take_id.unique():
        m = (meta.take_id == tid).values
        a2.scatter(Z[m, 0], Z[m, 1], s=4, label=tid)
    a2.legend(fontsize=6, ncol=2)
    a2.set_title("take")
plt.tight_layout()
plt.savefig(OUT / "umap.png", dpi=110)
plt.show()

# %% [markdown]
# ## 4. Latent prediction vs copy-last
# **Does:** predicts the feature `h` seconds ahead from the current clip (`history=1`) or the last 3 clips, with a ridge regression on the residual `z[t+h] - z[t]`; compares to **copy-last** (`z[t+h] = z[t]`). Metrics in standardized feature space:
# `cos` = cosine to the true future feature; `nMSE` = ridge MSE / copy-last MSE (**< 1 beats copy-last**); `phase_acc_*` = phase probe applied to true / copied / predicted features.
# **Shows:** whether the feature dynamics are learnable beyond "nothing changes". `phase_acc_*` cannot score prediction quality because the phase label is constant inside a take.


# %%
rows = []
for tag, (meta, X) in data.items():
    meta = meta.reset_index(drop=True)
    folds = {}  # one scaler + phase probe per held-out take
    for tid in meta.take_id.unique():
        train = ~(meta.take_id == tid).values
        Z = StandardScaler().fit(X[train]).transform(X)
        folds[tid] = (
            Z,
            LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(Z[train], meta.phase.values[train]),
        )
    for h in HORIZONS:
        for hist in (1, 3):
            P, T = pairs(meta, h, hist)
            take_of = meta.take_id.values[T]
            acc = {k: [] for k in ["cos_copy", "cos_ridge", "mse_copy", "mse_ridge", "pa_true", "pa_copy", "pa_ridge"]}
            for tid in np.unique(take_of):
                te, tr = take_of == tid, take_of != tid
                Z, clf = folds[tid]
                ridge = Ridge(alpha=1000.0).fit(stack(Z, P, tr), Z[T[tr]] - Z[P[tr][:, -1]])
                last, true = Z[P[te][:, -1]], Z[T[te]]
                pred = last + ridge.predict(stack(Z, P, te))
                y = meta.phase.values[T[te]]
                acc["cos_copy"].append(cos(last, true))
                acc["cos_ridge"].append(cos(pred, true))
                acc["mse_copy"].append(((last - true) ** 2).mean(1))
                acc["mse_ridge"].append(((pred - true) ** 2).mean(1))
                acc["pa_true"].append(clf.predict(true) == y)
                acc["pa_copy"].append(clf.predict(last) == y)
                acc["pa_ridge"].append(clf.predict(pred) == y)
            m = {k: np.concatenate(v).mean() for k, v in acc.items()}
            rows.append(
                dict(
                    tag=tag,
                    horizon_clips=h,
                    history=hist,
                    cos_copy=m["cos_copy"],
                    cos_ridge=m["cos_ridge"],
                    nMSE_ridge=m["mse_ridge"] / m["mse_copy"],
                    phase_acc_true=m["pa_true"],
                    phase_acc_copy=m["pa_copy"],
                    phase_acc_ridge=m["pa_ridge"],
                )
            )
temporal_res = pd.DataFrame(rows).round(3)
temporal_res

# %% [markdown]
# ## 5. Scene-graph probe (current and future)
# **Does:** reads the scene-graph triplets (e.g. `head_surgeon | holding | dressing_material`) at each clip's centre frame; keeps triplets present in 3-97 % of clips and in >= 3 takes; trains a ridge probe `z_t -> y_{t+h}`
# and reports the mean ROC-AUC over triplets (0.5 = chance). `AUC_persist_GT` uses the *ground-truth* triplets at t as the prediction for t+h (privileged labels: a ceiling for "nothing changes", not a fair baseline). `h = 0` reads the current scene graph.
# **Shows:** whether the feature carries semantic scene content, and whether it degrades with the horizon (if it barely does, the probe mostly reads slow context, not dynamics).

# %%
meta0 = next(iter(data.values()))[0].reset_index(drop=True)
assert all((m.t.values == meta0.t.values).all() and (m.take_id.values == meta0.take_id.values).all() for m, _ in data.values())
sets = scene_graph_sets(meta0, PROC)
TRIPLETS, Y = triplet_matrix(sets, meta0.take_id.values)
print(f"{len(sets)} clips, {len(TRIPLETS)} triplets kept; positives per clip: {Y.sum(1).mean():.1f}")

rows = []
for tag, (meta, X) in data.items():
    meta = meta.reset_index(drop=True)
    for h in [0, *HORIZONS]:
        if h == 0:
            src = tgt = np.arange(len(meta))
        else:
            P, tgt = pairs(meta, h)
            src = P[:, -1]
        take_of = meta.take_id.values[tgt]
        S = np.zeros((len(src), Y.shape[1]))
        for tid in np.unique(take_of):
            te = take_of == tid
            sc = StandardScaler().fit(X[src[~te]])
            S[te] = Ridge(alpha=1000.0).fit(sc.transform(X[src[~te]]), Y[tgt[~te]]).predict(sc.transform(X[src[te]]))
        rows.append(
            dict(
                tag=tag,
                horizon_clips=h,
                AUC_probe=mean_auc(Y[tgt], S)[0],
                AUC_persist_GT=mean_auc(Y[tgt], Y[src])[0] if h else np.nan,
            )
        )
sg_res = pd.DataFrame(rows).round(3)
sg_res

# %% [markdown]
# ## 6. Summary
# **Does:** one row per encoder × camera with the headline metric of each probe (phase balanced accuracy, take-identity accuracy, scene-graph AUC now and at the longest horizon, latent-prediction nMSE at the shortest and longest horizon, history = 1).
# **Shows:** the cross-encoder comparison at a glance. `surgenet` (surgical pre-training) and `caformer_in1k` (ImageNet) share the architecture, so their difference isolates the effect of surgical pre-training; `dinov2` is a larger general-domain model.

# %%
h_first, h_last = HORIZONS[0], HORIZONS[-1]
summary = phase_res.set_index("tag")[["bal_acc"]].rename(columns={"bal_acc": "phase_bal_acc"})
summary["take_id_acc"] = take_res.set_index("tag").take_id_acc
sg = sg_res.pivot(index="tag", columns="horizon_clips", values="AUC_probe")
summary["sg_auc_now"], summary[f"sg_auc_+{h_last}s"] = sg[0], sg[h_last]
nmse = temporal_res[temporal_res.history == 1].pivot(index="tag", columns="horizon_clips", values="nMSE_ridge")
summary[f"nMSE_+{h_first}s"], summary[f"nMSE_+{h_last}s"] = nmse[h_first], nmse[h_last]
summary = summary.round(3)
summary

# %% [markdown]
# ## 7. Paired comparison of pre-training (per held-out take)
# **Does:** for each controlled pair (same architecture and camera, different pre-training) computes the per-take phase accuracy difference and a bootstrap 95 % CI over takes (10,000 resamples); `win/loss/tie` count takes where the first encoder is better / worse / equal.
# Pairs: `surgenet` vs `caformer_in1k` (surgical vs ImageNet pre-training, image model), `surgmotion` vs `vjepa2` (surgical vs general video pre-training, same ViT-L).
# **Shows:** whether surgical pre-training helps beyond noise. A CI containing 0 means no evidence of an effect at this sample size (takes, not clips, are the independent units).

# %%
PAIRS = [("surgenet", "caformer_in1k"), ("surgmotion", "vjepa2")]
rng = np.random.default_rng(0)
rows = []
for a, b in PAIRS:
    for tag_a in [t for t in data if t.startswith(a + "_")]:
        tag_b = tag_a.replace(a + "_", b + "_", 1)
        if tag_b not in data:
            continue
        meta = data[tag_a][0]
        per_take = {
            tag: pd.Series(phase_pred[tag] == meta.phase.values).groupby(meta.take_id.values).mean() for tag in (tag_a, tag_b)
        }
        d = (per_take[tag_a] - per_take[tag_b]).values
        boots = rng.choice(d, size=(10000, len(d))).mean(1)
        rows.append(
            dict(
                pair=f"{a} - {b}",
                cam=tag_a.split("_", 1)[1],
                mean_diff=d.mean(),
                ci_lo=np.percentile(boots, 2.5),
                ci_hi=np.percentile(boots, 97.5),
                win=int((d > 0.005).sum()),
                loss=int((d < -0.005).sum()),
                tie=int((abs(d) <= 0.005).sum()),
            )
        )
paired_res = pd.DataFrame(rows).round(3)
paired_res

# %% [markdown]
# ## Save
# **Does:** writes all result tables to `outputs/probe/`.

# %%
tables = {
    "phase_probe": phase_res,
    "take_id_probe": take_res,
    "temporal": temporal_res,
    "scenegraph_probe": sg_res,
    "paired_pretraining": paired_res,
    "summary": summary,
}
for name, df in tables.items():
    df.to_csv(OUT / f"{name}.csv", index=name == "summary")
print("saved to", OUT)

# %% [markdown]
# # Dynamics-aware evaluation, anomaly detection, multi-view fusion and a non-linear predictor
# Companion of notebook 01. The procedure is chosen with the environment variable `EGOEXOR_PROC` (`MISS` default, or `Ultrasound`); results go to `outputs/dynamics/<procedure>/`.
# Notebook 01 showed that frozen features carry phase and scene-graph information, but both are *slow* signals (the phase is constant inside a take, the scene-graph probe barely degrades with the horizon).
# This notebook asks the questions a world model has to answer: **do the latents predict what changes, do they tell normal from abnormal, does fusing views help, and is the ridge predictor the limit?**
# Same protocol as 01 (leave-one-take-out unless stated, 1 clip step = 1 s). Each cell is preceded by **Does** and **Shows**.

# %%
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import egoexor_io as eio
from probe_utils import (
    ENCODERS,
    event_metrics,
    full_tags,
    load_splits,
    load_tag,
    loto_phase,
    mean_auc,
    pairs,
    scene_graph_sets,
    split_tag,
    stack,
    triplet_matrix,
)

pd.set_option("display.width", 220)
PROC = os.environ.get("EGOEXOR_PROC", "MISS")
OUT = eio.out_dir(f"dynamics/{PROC}")
MAIN_CAMS = {"MISS": ["external_3", "external_5"], "Ultrasound": ["simstation"]}[
    PROC
]  # cameras of the (slow) non-linear predictor study
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## 0. Load
# **Does:** loads every encoder × camera that covers all takes of the procedure, plus the scene-graph triplet matrix `Y` (n_clips × n_triplets, triplets present in 3-97 % of the clips and in >= 3 takes; cached in `data/cache`).
# **Shows:** the same clips for every tag.

# %%
TAGS = full_tags(PROC)
TAG_OF = {split_tag(t): t for t in TAGS}
splits = load_splits()
data = {t: load_tag(t, PROC, splits) for t in TAGS}
meta0 = next(iter(data.values()))[0]
assert all((m.t.values == meta0.t.values).all() and (m.take_id.values == meta0.take_id.values).all() for m, _ in data.values())
Z = {t: StandardScaler().fit_transform(X).astype(np.float32) for t, (_, X) in data.items()}  # unsupervised global standardisation
sets = scene_graph_sets(meta0, PROC)
TRIPLETS, Y = triplet_matrix(sets, meta0.take_id.values)
TAKES = meta0.take_id.values
MAIN = [t for t in TAGS if split_tag(t)[1] in MAIN_CAMS]
print(
    f"{PROC}: {len(TAGS)} tags, {len(meta0)} clips, {meta0.take_id.nunique()} takes, {len(TRIPLETS)} triplets, main tags: {len(MAIN)}"
)


def loto_ridge(F_, T_, groups, alpha, col_scale=None):
    """Out-of-take ridge predictions; features are standardised on the training takes (then optionally re-weighted per column)."""
    S = np.zeros(T_.shape)
    for g in np.unique(groups):
        te = groups == g
        sc = StandardScaler().fit(F_[~te])
        a, b = sc.transform(F_[~te]), sc.transform(F_[te])
        if col_scale is not None:
            a, b = a * col_scale, b * col_scale
        S[te] = Ridge(alpha=alpha).fit(a, T_[~te]).predict(b)
    return S


# %% [markdown]
# ## 1. Phase probe with a sensible label set (Ultrasound only)
# **Does:** Ultrasound phases 1 and 2 are the same stage and phase 1 has a single take. Re-scores the phase probe with phases 1+2 merged (4 classes, chance 0.25) and on phases 3-5 only (chance 0.33).
# **Shows:** the real phase-recognition ability once the ill-defined label is removed. MISS has no such problem, so the table is empty there.

# %%
rows = []
if PROC == "Ultrasound":
    y_merged = np.where(meta0.phase.values == 2, 1, meta0.phase.values)
    sub = meta0.phase.values >= 3
    for tag, (meta, X) in data.items():
        merged = balanced_accuracy_score(y_merged, loto_phase(meta, X, y=y_merged))
        m345 = meta[sub].reset_index(drop=True)
        late = balanced_accuracy_score(m345.phase.values, loto_phase(m345, X[sub]))
        rows.append(dict(tag=tag, bal_acc_phase12_merged=merged, bal_acc_phase3to5=late))
merged_res = pd.DataFrame(rows).round(3)
merged_res

# %% [markdown]
# ## 2. Multi-view fusion
# **Does:** concatenates the clip features of several cameras of the same encoder (MISS: the five `external_*` cameras, and + `microscope`; Ultrasound: `simstation` + `or_light`) and runs the same probes as notebook 01 (phase balanced accuracy, scene-graph AUC now / +4 s, nMSE vs copy-last +1 s / +4 s). Single-camera numbers come from `outputs/probe/<procedure>/summary.csv` (notebook 01).
# **Shows:** whether more viewpoints add information over the average / best single camera.

# %%
FUSION = {
    "MISS": {
        "ext1-5": [f"external_{i}" for i in range(1, 6)],
        "ext1-5+microscope": [f"external_{i}" for i in range(1, 6)] + ["microscope"],
    },
    "Ultrasound": {"simstation+or_light": ["simstation", "or_light"]},
}[PROC]


def ridge_nmse(meta, X, h, hist=1):
    """Out-of-take ridge nMSE vs copy-last in standardised feature space (same protocol as notebook 01, section 4)."""
    P, T = pairs(meta, h, hist)
    take_of = meta.take_id.values[T]
    err_r, err_c = [], []
    for tid in np.unique(take_of):
        te, tr = take_of == tid, take_of != tid
        Zf = StandardScaler().fit(X[(meta.take_id != tid).values]).transform(X)
        ridge = Ridge(alpha=1000.0).fit(stack(Zf, P, tr), Zf[T[tr]] - Zf[P[tr][:, -1]])
        last, true = Zf[P[te][:, -1]], Zf[T[te]]
        err_r.append(((last + ridge.predict(stack(Zf, P, te)) - true) ** 2).mean(1))
        err_c.append(((last - true) ** 2).mean(1))
    return np.concatenate(err_r).mean() / np.concatenate(err_c).mean()


def probe_suite(meta, X, Ym):
    """Probes of notebook 01 on any subset of takes: `meta` clip metadata, `X` features, `Ym` scene-graph matrix of the same clips."""
    takes = meta.take_id.values
    out = {"phase_bal_acc": balanced_accuracy_score(meta.phase.values, loto_phase(meta, X))}
    out["sg_auc_now"] = mean_auc(Ym, loto_ridge(X, Ym, takes, 1000.0))[0]
    P, T = pairs(meta, 4)
    out["sg_auc_+4s"] = mean_auc(Ym[T], loto_ridge(X[P[:, -1]], Ym[T], takes[T], 1000.0))[0]
    out["nMSE_+1s"], out["nMSE_+4s"] = ridge_nmse(meta, X, 1), ridge_nmse(meta, X, 4)
    return out


rows = []
for enc in ENCODERS:
    for name, cams in FUSION.items():
        if not all((enc, c) in TAG_OF for c in cams):
            continue
        Xf = np.concatenate([data[TAG_OF[(enc, c)]][1] for c in cams], axis=1)
        rows.append(dict(encoder=enc, views=name, n_views=len(cams), **probe_suite(meta0, Xf, Y)))
fusion_res = pd.DataFrame(rows).round(3)
summary_file = eio.OUT_DIR / "probe" / PROC / "summary.csv"
if len(fusion_res) and summary_file.exists():
    single = pd.read_csv(summary_file, index_col=0)
    single["encoder"] = [split_tag(t)[0] for t in single.index]
    for col in ["phase_bal_acc", "sg_auc_now", "sg_auc_+4s", "nMSE_+4s"]:
        grp = single.groupby("encoder")[col]
        fusion_res[f"{col}_single_mean"] = fusion_res.encoder.map(grp.mean()).round(3)
        fusion_res[f"{col}_single_best"] = fusion_res.encoder.map(grp.min() if col.startswith("nMSE") else grp.max()).round(3)
fusion_res

# %% [markdown]
# ## 2b. Cameras that exist in only some takes (Ultrasound `external_3`)
# **Does:** Ultrasound `external_*` cameras exist only in `ultrasound_3/4/5_*`. For each such camera, compares it with the full-coverage cameras (`simstation`, `or_light`) on exactly the takes where it exists (same probes as above, scene-graph triplets re-selected on that subset). Empty for MISS.
# **Shows:** whether the room camera we used for Ultrasound (`simstation`) is a good choice compared with the Azure Kinect view that MISS uses.

# %%
rows = []
partial = sorted(
    p.name
    for p in eio.FEAT_DIR.iterdir()
    if p.is_dir() and not p.name.startswith("_") and 0 < len(list(p.glob(f"{PROC}_*.npz"))) < len(meta0.take_id.unique())
)
for ptag in partial:
    enc, pcam = split_tag(ptag)
    takes = {tuple(map(int, p.stem.split("_")[1:])) for p in (eio.FEAT_DIR / ptag).glob(f"{PROC}_*.npz")}
    for cam in [pcam] + [c for (e, c) in TAG_OF if e == enc]:
        meta_s, X_s = load_tag(ptag if cam == pcam else TAG_OF[(enc, cam)], PROC, splits, takes=takes)
        _, Y_s = triplet_matrix(scene_graph_sets(meta_s, PROC), meta_s.take_id.values)
        rows.append(
            dict(encoder=enc, cam=cam, n_takes=meta_s.take_id.nunique(), n_clips=len(meta_s), **probe_suite(meta_s, X_s, Y_s))
        )
partial_res = pd.DataFrame(rows).round(3)
partial_res

# %% [markdown]
# ## 3. Event prediction: does the latent know what will *change*?
# **Does:** for every triplet and horizon `h`, splits the clips into those where the triplet is currently **absent** (then the event is an *onset*, label = present at t+h) and **present** (event = *offset*, label = absent at t+h), and scores how well a ridge probe ranks the event clips (mean ROC-AUC over triplets with >= 10 events and >= 10 non-events; 0.5 = chance).
# Probes (all out-of-take): `latent` = z_t, `latent+motion` = [z_t, z_t - z_{t-1}], and **controls that do not use the encoder**: `GT_state` (the current ground-truth scene graph, privileged), `time_phase` (normalised position inside the take and the true phase, privileged), `GT_state+time_phase`, and `latent+GT_state+time_phase`.
# **Shows:** whether the encoder carries information about *upcoming* changes beyond what the current graph and the position in the procedure already tell. If `latent+GT_state+time_phase` does not beat `GT_state+time_phase`, the latent adds nothing to predict events.

# %%
HZ_EV = [1, 4]
POS = meta0.groupby("take_id").clip.transform(lambda c: c / c.max()).values
PHASES = np.eye(int(meta0.phase.max()))[meta0.phase.values - 1]
CTRL = {"GT_state": Y, "time_phase": np.c_[POS, POS**2, POS**3, PHASES]}
CTRL["GT_state+time_phase"] = np.c_[CTRL["GT_state"], CTRL["time_phase"]]
A_LAT, A_CTRL = 1000.0, 10.0  # ridge strength for the 1024-d latent / the few control columns

rows = []
for h in HZ_EV:
    P, T = pairs(meta0, h, hist=2)
    src, prev, take_of = P[:, -1], P[:, 0], TAKES[T]
    Ysrc, Ytgt = Y[src], Y[T]
    for name, C in CTRL.items():
        S = loto_ridge(C[src], Ytgt, take_of, A_CTRL)
        rows.append(dict(tag="(control)", horizon_clips=h, variant=name, **event_metrics(S, Ysrc, Ytgt)))
    for tag in TAGS:
        Zt = Z[tag]
        variants = {
            "latent": (Zt[src], None),
            "latent+motion": (np.c_[Zt[src], Zt[src] - Zt[prev]], None),
            "latent+GT_state+time_phase": (
                np.c_[Zt[src], CTRL["GT_state+time_phase"][src]],
                np.r_[
                    np.ones(Zt.shape[1]), np.full(CTRL["GT_state+time_phase"].shape[1], np.sqrt(A_LAT / A_CTRL))
                ],  # controls keep their own ridge strength
            ),
        }
        for name, (Fm, scale) in variants.items():
            S = loto_ridge(Fm, Ytgt, take_of, A_LAT, col_scale=scale)
            rows.append(dict(tag=tag, horizon_clips=h, variant=name, **event_metrics(S, Ysrc, Ytgt)))
event_res = pd.DataFrame(rows).round(3)
event_res["encoder"] = [split_tag(t)[0] if t != "(control)" else "(control)" for t in event_res.tag]
EV_COLS = ["onset_AUC", "offset_AUC"]
event_summary = event_res.groupby(["horizon_clips", "variant", "encoder"])[EV_COLS].mean().round(3).unstack("encoder")
control_base = event_res[(event_res.tag == "(control)") & (event_res.variant == "GT_state+time_phase")].set_index(
    "horizon_clips"
)[EV_COLS]
lat_full = event_res[event_res.variant == "latent+GT_state+time_phase"].groupby(["horizon_clips", "encoder"])[EV_COLS].mean()
lat_only = event_res[event_res.variant == "latent"].groupby(["horizon_clips", "encoder"])[EV_COLS].mean()
event_gain = pd.concat(
    {
        "latent_only_minus_controls": lat_only.sub(control_base, level="horizon_clips"),
        "latent_plus_controls_minus_controls": lat_full.sub(control_base, level="horizon_clips"),
    },
    axis=1,
).round(3)
print("mean over all cameras; gain of the latent over the privileged controls (GT_state + time_phase)")
event_gain

# %% [markdown]
# **Does:** bar chart of the onset AUC at +4 s (mean over all cameras of each encoder), controls on the right. **Shows:** the encoder variants against the controls at a glance.

# %%
sel = event_res[event_res.horizon_clips == HZ_EV[-1]]
pv = sel.groupby(["variant", "encoder"]).onset_AUC.mean().unstack("encoder")
ax = pv.plot.bar(figsize=(11, 4), rot=20, width=0.85)
ax.axhline(0.5, color="k", lw=0.8, ls="--")
ax.set_ylabel(f"onset AUC at +{HZ_EV[-1]} s")
ax.set_title(f"{PROC}: predicting triplet onsets (0.5 = chance)")
plt.tight_layout()
plt.savefig(OUT / "event_onset_auc.png", dpi=110)
plt.show()

# %% [markdown]
# ## 4. Does latent motion track semantic change?
# **Does:** for consecutive clips (1 s apart) of each take, correlates the latent displacement `||z_{t+1} - z_t||` with the number of scene-graph triplets that appear / disappear in that second (Spearman, per take), and scores how well the displacement detects "any triplet changed" (ROC-AUC, per take). Reports the mean over takes with a bootstrap 95 % CI over takes.
# **Shows:** whether the latent moves when the semantic scene moves (a requirement for using latent change as an event signal). Clips overlap by 75 %, so this is a smooth, low-resolution measure.

# %%
rng = np.random.default_rng(0)
rows = []
for tag in TAGS:
    rho, auc = [], []
    for _, g in meta0.groupby("take_id"):
        idx = g.index.values
        mot = np.linalg.norm(Z[tag][idx[1:]] - Z[tag][idx[:-1]], axis=1)
        chg = np.array([len(sets[a] ^ sets[b]) for a, b in zip(idx[:-1], idx[1:])])
        if chg.std() == 0:
            continue
        rho.append(spearmanr(mot, chg)[0])
        auc.append(roc_auc_score(chg > 0, mot) if 0 < (chg > 0).sum() < len(chg) else np.nan)
    rho, auc = np.array(rho), np.array(auc)
    boots = rng.choice(rho, size=(5000, len(rho))).mean(1)
    rows.append(
        dict(
            tag=tag,
            n_takes=len(rho),
            spearman=rho.mean(),
            ci_lo=np.percentile(boots, 2.5),
            ci_hi=np.percentile(boots, 97.5),
            change_detect_AUC=np.nanmean(auc),
        )
    )
motion_res = pd.DataFrame(rows).round(3)
motion_res["encoder"] = [split_tag(t)[0] for t in motion_res.tag]
motion_res["cam"] = [split_tag(t)[1] for t in motion_res.tag]
motion_res.pivot(index="cam", columns="encoder", values="spearman")

# %% [markdown]
# ## 5. Anomaly detection (take level, exploratory)
# **Does:** uses the scripted anomalies of `TAKE_METADATA.md` (nominal = `none`; the Ultrasound phase 3 "skin marker" takes are an explicit *variation* and are dropped; phases without both nominal and anomalous takes are dropped). Each held-out take gets three scores:
# `novelty` = mean distance of its clips to the 5 nearest clips of the *nominal takes of the same phase* (other takes only); `dyn_mse` = error of a ridge latent predictor (h = 2 s) trained on nominal takes only; `dyn_ratio` = that error divided by the copy-last error.
# Scores are z-scored within each phase and pooled; reported: ROC-AUC (anomalous = positive), permutation p-value (labels shuffled within phase, 1000 draws), and the AUC of the **recording order** (take number) as a confound baseline.
# **Shows:** whether anomalous takes are separable at all. **Caveat:** in the metadata, nominal takes are the first ones of each phase and anomalies come later, so any drift between recordings (people, lighting, layout) can also produce a high AUC; the `take_index` row is what a detector would get from the recording order alone.

# %%
tbl = meta0.groupby("take_id").agg(phase=("phase", "first"), take=("take", "first")).reset_index()
tbl["anomaly"] = [eio.anomaly_of(PROC, p, t) for p, t in zip(tbl["phase"], tbl["take"])]
tbl = tbl[~tbl.anomaly.isin({"(not in metadata)", "Skin marker (variation)"})].copy()
tbl["label"] = (tbl.anomaly != "none").astype(int)
valid = [p for p, g in tbl.groupby("phase") if g.label.nunique() == 2]
tbl = tbl[tbl.phase.isin(valid)].reset_index(drop=True)
NOMINAL = set(tbl[tbl.label == 0].take_id)
print(f"phases used: {valid}; nominal takes {int((tbl.label == 0).sum())}, anomalous takes {int(tbl.label.sum())}")
print(tbl.groupby(["phase", "anomaly"]).size().to_string())


def novelty_scores(Zt, k=5):
    out = {}
    for r in tbl.itertuples():
        ref = tbl[(tbl.phase == r.phase) & (tbl.label == 0) & (tbl.take_id != r.take_id)].take_id
        d = cdist(Zt[TAKES == r.take_id], Zt[np.isin(TAKES, ref)])
        out[r.take_id] = np.sort(d, 1)[:, :k].mean() / np.sqrt(Zt.shape[1])
    return out


def surprise_scores(Zt, h=2):
    P, T = pairs(meta0, h)
    src, take_of = P[:, -1], TAKES[T]
    nominal_rows = np.isin(take_of, [t for t in np.unique(take_of) if eio.anomaly_of(PROC, *map(int, t.split("_"))) == "none"])
    mse, ratio = {}, {}
    for r in tbl.itertuples():
        tr, te = nominal_rows & (take_of != r.take_id), take_of == r.take_id
        ridge = Ridge(alpha=1000.0).fit(Zt[src[tr]], Zt[T[tr]] - Zt[src[tr]])
        err = ((Zt[src[te]] + ridge.predict(Zt[src[te]]) - Zt[T[te]]) ** 2).mean()
        mse[r.take_id], ratio[r.take_id] = err, err / ((Zt[src[te]] - Zt[T[te]]) ** 2).mean()
    return mse, ratio


def pooled_auc(scores, n_perm=1000, seed=0):
    d = tbl.assign(score=tbl.take_id.map(scores))
    d["z"] = d.groupby("phase").score.transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12))
    auc = roc_auc_score(d.label, d.z)
    rng_p = np.random.default_rng(seed)
    groups = [np.flatnonzero(d.phase.values == p) for p in valid]
    perm = []
    for _ in range(n_perm):
        lab = d.label.values.copy()
        for ix in groups:
            lab[ix] = rng_p.permutation(lab[ix])
        perm.append(roc_auc_score(lab, d.z.values))
    return auc, (np.sum(np.array(perm) >= auc) + 1) / (n_perm + 1)


rows, take_scores = [], []
base_auc, base_p = pooled_auc(dict(zip(tbl["take_id"], tbl["take"])))
rows.append(dict(tag="(recording order)", score="take_index", AUC=base_auc, p_value=base_p))
for tag in TAGS:
    nov = novelty_scores(Z[tag])
    mse, ratio = surprise_scores(Z[tag])
    for name, sc in (("novelty", nov), ("dyn_mse", mse), ("dyn_ratio", ratio)):
        auc, p = pooled_auc(sc)
        rows.append(dict(tag=tag, score=name, AUC=auc, p_value=p))
        take_scores += [dict(tag=tag, score=name, take_id=k, value=v) for k, v in sc.items()]
anomaly_res = pd.DataFrame(rows).round(3)
pd.DataFrame(take_scores).to_csv(OUT / "anomaly_take_scores.csv", index=False)
anomaly_res["encoder"] = [split_tag(t)[0] if t in TAG_OF.values() else t for t in anomaly_res.tag]
anomaly_res["cam"] = [split_tag(t)[1] if t in TAG_OF.values() else "" for t in anomaly_res.tag]
anomaly_summary = anomaly_res[anomaly_res.tag.isin(MAIN)].pivot_table(index=["encoder", "cam"], columns="score", values="AUC")
print(f"recording-order baseline: AUC {base_auc:.3f}, p {base_p:.3f}")
anomaly_summary.round(3)

# %% [markdown]
# **Does:** bar chart of the anomaly AUC of the main cameras for the three scores, with the recording-order baseline. **Shows:** how close each score comes to (or stays below) the confound baseline.

# %%
pv = anomaly_res[anomaly_res.tag.isin(MAIN)].pivot_table(index="tag", columns="score", values="AUC")
ax = pv.plot.bar(figsize=(11, 4), rot=30)
ax.axhline(0.5, color="k", lw=0.8, ls="--")
ax.axhline(base_auc, color="r", lw=1.0, ls=":", label="recording order")
ax.set_ylabel("pooled within-phase AUC")
ax.set_title(f"{PROC}: take-level anomaly detection")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "anomaly_auc.png", dpi=110)
plt.show()

# %% [markdown]
# ## 6. Non-linear latent predictor vs ridge vs copy-last
# **Does:** on the main cameras, predicts the residual `z[t+h] - z[t]` from the last 1 or 3 clips with (a) ridge and (b) a 2-hidden-layer MLP (512 units, GELU, dropout 0.3, AdamW, 80 epochs, fixed hyper-parameters, no tuning), with takes split into 5 folds (a fold never shares a take with its training data). nMSE = MSE / copy-last MSE in standardised feature space (< 1 beats copy-last).
# **Shows:** whether a non-linear predictor gets beyond the ridge baseline, i.e. whether the weak predictability of image encoders is a model limit or a feature limit.


# %%
def mlp_predict(Xtr, Ytr, Xte, epochs=80, hidden=512, seed=0):
    torch.manual_seed(seed)
    xt, yt = torch.tensor(Xtr, dtype=torch.float32, device=DEV), torch.tensor(Ytr, dtype=torch.float32, device=DEV)
    net = nn.Sequential(
        nn.Linear(xt.shape[1], hidden),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, yt.shape[1]),
    ).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=DEV)
        for i in range(0, len(xt), 256):
            b = perm[i : i + 256]
            opt.zero_grad()
            F.mse_loss(net(xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte, dtype=torch.float32, device=DEV)).cpu().numpy()


rows = []
for tag in MAIN:
    X = data[tag][1]
    for h in (1, 4):
        for hist in (1, 3):
            P, T = pairs(meta0, h, hist)
            take_of = TAKES[T]
            err = {k: [] for k in ("copy", "ridge", "mlp")}
            for tr_t, te_t in GroupKFold(5).split(P, groups=take_of):
                train_clips = np.isin(TAKES, np.unique(take_of[tr_t]))
                Zf = StandardScaler().fit(X[train_clips]).transform(X).astype(np.float32)
                Xtr, Xte = stack(Zf, P, tr_t), stack(Zf, P, te_t)
                Rtr = Zf[T[tr_t]] - Zf[P[tr_t][:, -1]]
                last, true = Zf[P[te_t][:, -1]], Zf[T[te_t]]
                err["copy"].append(((last - true) ** 2).mean(1))
                err["ridge"].append(((last + Ridge(alpha=1000.0).fit(Xtr, Rtr).predict(Xte) - true) ** 2).mean(1))
                err["mlp"].append(((last + mlp_predict(Xtr, Rtr, Xte) - true) ** 2).mean(1))
            m = {k: np.concatenate(v).mean() for k, v in err.items()}
            rows.append(
                dict(tag=tag, horizon_clips=h, history=hist, nMSE_ridge=m["ridge"] / m["copy"], nMSE_mlp=m["mlp"] / m["copy"])
            )
    print("done", tag, flush=True)
mlp_res = pd.DataFrame(rows).round(3)
mlp_res

# %% [markdown]
# **Does:** bar chart of ridge vs MLP nMSE at +4 s, history 3. **Shows:** the gain (if any) of the non-linear predictor per encoder.

# %%
sel = mlp_res[(mlp_res.horizon_clips == 4) & (mlp_res.history == 3)].set_index("tag")[["nMSE_ridge", "nMSE_mlp"]]
ax = sel.plot.bar(figsize=(11, 4), rot=30)
ax.axhline(1.0, color="k", lw=0.8, ls="--")
ax.set_ylabel("nMSE vs copy-last (+4 s, history 3)")
plt.tight_layout()
plt.savefig(OUT / "mlp_vs_ridge.png", dpi=110)
plt.show()

# %% [markdown]
# ## Save
# **Does:** writes all result tables to `outputs/dynamics/<procedure>/`.

# %%
tables = {
    "merged_phase": merged_res,
    "fusion": fusion_res,
    "partial_cameras": partial_res,
    "event_prediction": event_res,
    "motion_vs_change": motion_res,
    "anomaly": anomaly_res,
    "mlp_predictor": mlp_res,
}
for name, df in tables.items():
    if len(df):
        df.to_csv(OUT / f"{name}.csv", index=False)
event_gain.to_csv(OUT / "event_gain_over_controls.csv")
print("saved to", OUT)

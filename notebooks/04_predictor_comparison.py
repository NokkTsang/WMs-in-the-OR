# %% [markdown]
# # Predictor comparison on a frozen encoder (simple test)
# Fixed encoder (default V-JEPA 2, environment variable `EGOEXOR_ENCODER`), **all cameras of the procedure pooled** (every camera is one training sample, so the model sees every viewpoint), procedure chosen with `EGOEXOR_PROC` (`MISS` or `Ultrasound`).
# Question: which predictor on top of the frozen latent is best for (A) predicting the future latent and (B) predicting which scene-graph relations will appear / disappear?
# Protocol: 5-fold cross-validation **by take** (a take is never in train and test), history 8 s, horizons +1 s and +4 s, one seed, fixed hyper-parameters (no tuning).
# Predictors: `copy_last` (future = now), `ridge` (linear, last clip), `mlp1` (MLP, last clip), `mlp8` (MLP, 8 clips flattened), `gru8` (GRU over 8 clips), `tf8` (small Transformer over 8 clips).
# Task A metric: nMSE = MSE of the predicted latent change / MSE of "no change" (< 1 beats copy-last). Task B metric: onset / offset ROC-AUC (see notebook 02), with the controls that use the privileged current scene graph + position in the take.

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
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import egoexor_io as eio
from probe_utils import event_metrics, full_tags, load_splits, load_tag, scene_graph_sets, split_tag, triplet_matrix

pd.set_option("display.width", 200)
PROC = os.environ.get("EGOEXOR_PROC", "MISS")
ENC = os.environ.get("EGOEXOR_ENCODER", "vjepa2")
OUT = eio.out_dir(f"predictors/{PROC}")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HIST, HORIZONS, EPOCHS = 8, [1, 4], 40

# %% [markdown]
# ## 0. Load
# **Does:** loads the clip features of every camera of the encoder, the scene-graph presence matrix `Y` (triplets present in 3-97 % of clips and >= 3 takes) and the indices of the 8-clip history of every clip (clamped at the start of the take).
# **Shows:** the data size.

# %%
tags = [t for t in full_tags(PROC) if split_tag(t)[0] == ENC]
cams = [split_tag(t)[1] for t in tags]
splits = load_splits()
loaded = [load_tag(t, PROC, splits) for t in tags]
meta0 = loaded[0][0]
assert all((m.t.values == meta0.t.values).all() and (m.take_id.values == meta0.take_id.values).all() for m, _ in loaded)
Zt = torch.tensor(
    np.stack([StandardScaler().fit_transform(X) for _, X in loaded]), dtype=torch.float32, device=DEV
)  # (cameras, clips, D)
C, N, D = Zt.shape
sets = scene_graph_sets(meta0, PROC)
TRIPLETS, Y = triplet_matrix(sets, meta0.take_id.values)
Yt = torch.tensor(Y, dtype=torch.float32, device=DEV)
TAKE, POS_IN_TAKE = meta0.take_id.values, meta0["clip"].values
HIST_IDX = np.maximum(np.arange(N)[:, None] - np.arange(HIST - 1, -1, -1)[None], (np.arange(N) - POS_IN_TAKE)[:, None])
POS = POS_IN_TAKE / meta0.groupby("take_id")["clip"].transform("max").values
CTRL = np.c_[Y, POS, POS**2, POS**3, np.eye(int(meta0.phase.max()))[meta0.phase.values - 1]]  # privileged controls
print(f"{PROC} / {ENC}: cameras {cams}, {N} clips per camera, {meta0.take_id.nunique()} takes, D={D}, {Y.shape[1]} triplets")


# %% [markdown]
# ## 1. Models
# **Does:** defines the three neural predictors (MLP, GRU, 2-layer Transformer encoder; all take a history of clips and output either the latent change (task A) or the triplet logits (task B)). Training: AdamW (lr 1e-3, weight decay 1e-2), batch 256, 40 epochs, MSE (A) or BCE (B).


# %%
class Net(nn.Module):
    def __init__(self, kind, hist, out):
        super().__init__()
        self.kind, self.hist = kind, hist
        if kind == "mlp":
            self.body = nn.Sequential(
                nn.Linear(D * hist, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 512), nn.GELU(), nn.Dropout(0.3)
            )
            width = 512
        elif kind == "gru":
            self.rnn, self.drop, width = nn.GRU(D, 256, batch_first=True), nn.Dropout(0.3), 256
        else:
            layer = nn.TransformerEncoderLayer(128, 4, 256, 0.1, batch_first=True, norm_first=True)
            self.inp, self.pos, self.enc, width = (
                nn.Linear(D, 128),
                nn.Parameter(torch.zeros(1, hist, 128)),
                nn.TransformerEncoder(layer, 2),
                128,
            )
        self.head = nn.Linear(width, out)

    def forward(self, x):
        x = x[:, -self.hist :]
        if self.kind == "mlp":
            z = self.body(x.flatten(1))
        elif self.kind == "gru":
            z = self.drop(self.rnn(x)[1][-1])
        else:
            z = self.enc(self.inp(x) + self.pos)[:, -1]
        return self.head(z)


def fit_predict(kind, hist, xtr, ytr, xte, loss, seed=0):
    torch.manual_seed(seed)
    net = Net(kind, hist, ytr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(xtr), device=DEV)
        for i in range(0, len(xtr), 256):
            b = perm[i : i + 256]
            opt.zero_grad()
            out = net(xtr[b])
            (F.mse_loss(out, ytr[b]) if loss == "mse" else F.binary_cross_entropy_with_logits(out, ytr[b])).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(xte).cpu().numpy()


NETS = {"mlp1": ("mlp", 1), "mlp8": ("mlp", HIST), "gru8": ("gru", HIST), "tf8": ("tf", HIST)}


# %% [markdown]
# ## 2. Cross-validation by take
# **Does:** for each horizon, splits the takes into 5 folds, trains every predictor on the other folds (all cameras pooled) and predicts the held-out takes (all cameras). Task A target: `z[t+h] - z[t]`. Task B target: which triplets are present at `t+h`.
# **Shows:** the predictions of every held-out sample exactly once; metrics are computed on the pooled out-of-fold predictions.


# %%
def gather(ix):
    """History (C*len(ix), HIST, D), camera-major."""
    idx = torch.as_tensor(HIST_IDX[ix], device=DEV)
    return torch.cat([Zt[c][idx] for c in range(C)])


rows_a, rows_b = [], []
for h in HORIZONS:
    ok = np.flatnonzero((np.arange(N) + h < N) & (TAKE[np.minimum(np.arange(N) + h, N - 1)] == TAKE))
    preds_a = {k: [] for k in ["ridge", *NETS]}
    preds_b = {k: [] for k in ["controls", "ridge", *NETS]}
    true_a, ysrc, ytgt = [], [], []
    for tr, te in GroupKFold(5).split(ok, groups=TAKE[ok]):
        itr, ite = ok[tr], ok[te]
        xtr, xte = gather(itr), gather(ite)
        a_tr = torch.cat([Zt[c][itr + h] - Zt[c][itr] for c in range(C)])
        a_te = torch.cat([Zt[c][ite + h] - Zt[c][ite] for c in range(C)])
        b_tr, b_te = Yt[itr + h].repeat(C, 1), Yt[ite + h].repeat(C, 1)
        true_a.append(a_te.cpu().numpy())
        ysrc.append(np.tile(Y[ite], (C, 1)))
        ytgt.append(b_te.cpu().numpy())
        last_tr, last_te = xtr[:, -1].cpu().numpy(), xte[:, -1].cpu().numpy()
        preds_a["ridge"].append(Ridge(alpha=1000.0).fit(last_tr, a_tr.cpu().numpy()).predict(last_te))
        preds_b["ridge"].append(Ridge(alpha=1000.0).fit(last_tr, b_tr.cpu().numpy()).predict(last_te))
        sc = StandardScaler().fit(np.tile(CTRL[itr], (C, 1)))
        preds_b["controls"].append(
            Ridge(alpha=10.0)
            .fit(sc.transform(np.tile(CTRL[itr], (C, 1))), b_tr.cpu().numpy())
            .predict(sc.transform(np.tile(CTRL[ite], (C, 1))))
        )
        for name, (kind, hist) in NETS.items():
            preds_a[name].append(fit_predict(kind, hist, xtr, a_tr, xte, "mse"))
            preds_b[name].append(fit_predict(kind, hist, xtr, b_tr, xte, "bce"))
    true_a, ysrc, ytgt = np.concatenate(true_a), np.concatenate(ysrc), np.concatenate(ytgt)
    for name, p in preds_a.items():
        rows_a.append(dict(predictor=name, horizon_clips=h, nMSE=((np.concatenate(p) - true_a) ** 2).mean() / (true_a**2).mean()))
    rows_a.append(dict(predictor="copy_last", horizon_clips=h, nMSE=1.0))
    for name, p in preds_b.items():
        rows_b.append(dict(predictor=name, horizon_clips=h, **event_metrics(np.concatenate(p), ysrc, ytgt)))
    print("done horizon", h, flush=True)
latent_res = pd.DataFrame(rows_a).round(3)
event_res = pd.DataFrame(rows_b).round(3)

# %% [markdown]
# ## 3. Results
# **Does:** task A: nMSE of the future latent (< 1 beats copy-last). **Shows:** whether a learned predictor predicts the latent change better than "nothing changes", and which one.

# %%
latent_res.pivot(index="predictor", columns="horizon_clips", values="nMSE").reindex(
    ["copy_last", "ridge", "mlp1", "mlp8", "gru8", "tf8"]
)

# %% [markdown]
# **Does:** task B: onset / offset AUC of the predicted scene-graph change. `controls` = ridge on the privileged current scene graph + position in the take + true phase (no encoder). **Shows:** which predictor best forecasts upcoming relation changes, and whether it beats the controls.

# %%
event_pivot = event_res.pivot(index="predictor", columns="horizon_clips", values=["onset_AUC", "offset_AUC"]).reindex(
    ["controls", "ridge", "mlp1", "mlp8", "gru8", "tf8"]
)
event_pivot

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
order = ["copy_last", "ridge", "mlp1", "mlp8", "gru8", "tf8"]
latent_res.pivot(index="predictor", columns="horizon_clips", values="nMSE").reindex(order).plot.bar(
    ax=axes[0], rot=30, title="A: latent nMSE (< 1 good)"
)
axes[0].axhline(1.0, color="k", lw=0.8, ls="--")
for ax, col in zip(axes[1:], ["onset_AUC", "offset_AUC"]):
    event_res.pivot(index="predictor", columns="horizon_clips", values=col).reindex(["controls", *order[1:]]).plot.bar(
        ax=ax, rot=30, title=f"B: {col}"
    )
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
plt.suptitle(f"{PROC}, {ENC}, all cameras pooled, 5-fold by take")
plt.tight_layout()
plt.savefig(OUT / f"{ENC}_predictors.png", dpi=110)
plt.show()

# %% [markdown]
# ## Save

# %%
latent_res.to_csv(OUT / f"{ENC}_latent_prediction.csv", index=False)
event_res.to_csv(OUT / f"{ENC}_event_prediction.csv", index=False)
print("saved to", OUT)

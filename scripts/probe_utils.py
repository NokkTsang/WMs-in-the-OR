"""Shared helpers for the probe notebooks: feature / label loading, leave-one-take-out probes, scene-graph targets."""

import pickle
from collections import Counter

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import egoexor_io as eio

ENCODERS = ["dinov2", "surgenet", "caformer_in1k", "vjepa2", "surgmotion"]
N_TAKES = {"MISS": 16, "Ultrasound": 25}
CACHE_DIR = eio.ROOT / "data" / "cache"


def split_tag(tag):
    """`caformer_in1k_external_3_L16s4` -> (`caformer_in1k`, `external_3`)."""
    enc = next(e for e in ENCODERS if tag.startswith(e + "_"))
    return enc, tag[len(enc) + 1 :].rsplit("_L", 1)[0]


def full_tags(proc):
    """Feature folders that cover every take of the procedure."""
    return sorted(
        p.name
        for p in eio.FEAT_DIR.iterdir()
        if p.is_dir() and not p.name.startswith("_") and len(list(p.glob(f"{proc}_*.npz"))) == N_TAKES[proc]
    )


def load_splits():
    out = {}
    with h5py.File(eio.RAW_DIR / "splits.h5", "r") as f:
        for s in ["train", "validation", "test"]:
            d = f["splits"][s][:]
            for st, ph, tk in set(zip(d["surgery_type"].astype(str), d["procedure_id"], d["take_id"])):
                out[(st, int(ph), int(tk))] = s
    return out


def load_tag(tag, proc, splits, takes=None):
    """Clip metadata (phase, take, t, clip, split, take_id) and clip features of one encoder x camera; optional take filter."""
    rows, feats = [], []
    for p in sorted((eio.FEAT_DIR / tag).glob(f"{proc}_*.npz")):
        _, phase, take = p.stem.split("_")
        if takes is not None and (int(phase), int(take)) not in takes:
            continue
        d = np.load(p)
        rows.append(
            pd.DataFrame(
                dict(
                    phase=int(phase),
                    take=int(take),
                    t=d["t_center"],
                    clip=np.arange(len(d["feat"])),
                    split=splits[(proc, int(phase), int(take))],
                )
            )
        )
        feats.append(d["feat"])
    meta = pd.concat(rows, ignore_index=True)
    meta["take_id"] = meta["phase"].astype(str) + "_" + meta["take"].astype(str)
    return meta, np.concatenate(feats)


def pairs(meta, h, hist=1):
    """Row indices (history stack, target) of clip pairs inside each take."""
    past, tgt = [], []
    for _, g in meta.groupby("take_id"):
        idx = g.index.values
        for i in range(hist - 1, len(idx) - h):
            past.append(idx[i - hist + 1 : i + 1])
            tgt.append(idx[i + h])
    return np.array(past), np.array(tgt)


def cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


def stack(Z, P, mask):
    """Flatten the history stack of the selected pairs (boolean mask or index array): (n_selected, hist * D)."""
    sel = P[mask]
    return Z[sel].reshape(len(sel), -1)


def fit_clf(Xtr, ytr, C=0.1):
    sc = StandardScaler().fit(Xtr)
    return sc, LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)


def loto_phase(meta, X, y=None, shuffle=False, seed=0):
    """Leave-one-take-out phase prediction; `y` overrides the phase labels (e.g. merged phases)."""
    rng = np.random.default_rng(seed)
    y = meta.phase.values if y is None else y
    pred = np.empty_like(y)
    for tid in meta.take_id.unique():
        te = (meta.take_id == tid).values
        ytr = y[~te].copy()
        if shuffle:
            rng.shuffle(ytr)
        sc, clf = fit_clf(X[~te], ytr)
        pred[te] = clf.predict(sc.transform(X[te]))
    return pred


def scene_graph_sets(meta, proc):
    """Scene-graph triplet set (frozenset of 'subject|relation|object') at each clip centre frame; cached on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"scene_graph_{proc}.pkl"
    cache = pickle.loads(cache_file.read_bytes()) if cache_file.exists() else {}
    take_file = {}
    for p in eio.h5_files():
        with h5py.File(p, "r") as f:
            take_file.update({t: p for t in eio.list_takes(f)})
    changed = False
    for (ph, tk), g in meta.groupby(["phase", "take"]):
        keys = [(int(ph), int(tk), int(t)) for t in g.t]
        if all(k in cache for k in keys):
            continue
        tpath = eio.take_path(proc, int(ph), int(tk))
        with h5py.File(take_file[(proc, int(ph), int(tk))], "r") as f:
            for k in keys:
                ds = f.get(f"{tpath}/annotations/frame_{k[2]}/rel_annotations")
                cache[k] = frozenset("|".join(x.decode() for x in r) for r in (ds[:] if ds is not None else []))
        changed = True
    if changed:
        cache_file.write_bytes(pickle.dumps(cache))
    return [cache[(int(p), int(k), int(t))] for p, k, t in zip(meta["phase"], meta["take"], meta["t"])]


def triplet_matrix(sets, take_ids, min_frac=0.03, max_frac=0.97, min_takes=3):
    """Binary presence matrix (n_clips, n_triplets) of triplets present in 3-97 % of the clips and in >= 3 takes."""
    count, takes_with = Counter(), {}
    for s, tid in zip(sets, take_ids):
        for x in s:
            count[x] += 1
            takes_with.setdefault(x, set()).add(tid)
    n = len(sets)
    triplets = [x for x, c in count.items() if min_frac * n <= c <= max_frac * n and len(takes_with[x]) >= min_takes]
    return triplets, np.array([[x in s for x in triplets] for s in sets], dtype=float)


def event_metrics(S, Ysrc, Ytgt):
    """Onset / offset AUC. Onset: triplets absent now, event = present at t+h. Offset: present now, event = absent at t+h.

    `S` scores the presence at t+h; mean ROC-AUC over triplets with >= 10 events and >= 10 non-events.
    """
    out = {}
    for name, cond in (("onset", 0), ("offset", 1)):
        aucs, n_ev = [], 0
        for k in range(Ysrc.shape[1]):
            m = Ysrc[:, k] == cond
            ev = (Ytgt[m, k] == 1) if cond == 0 else (Ytgt[m, k] == 0)
            if ev.sum() >= 10 and (~ev).sum() >= 10:
                aucs.append(roc_auc_score(ev, S[m, k] if cond == 0 else -S[m, k]))
                n_ev += int(ev.sum())
        out[f"{name}_AUC"], out[f"{name}_n_triplets"], out[f"{name}_events"] = (
            (np.mean(aucs) if aucs else np.nan),
            len(aucs),
            n_ev,
        )
    return out


def mean_auc(Yt, S, min_pos=1):
    """Mean ROC-AUC over columns with at least `min_pos` positives and `min_pos` negatives."""
    aucs = [roc_auc_score(Yt[:, k], S[:, k]) for k in range(Yt.shape[1]) if min_pos <= Yt[:, k].sum() <= len(Yt) - min_pos]
    return float(np.mean(aucs)) if aucs else np.nan, len(aucs)

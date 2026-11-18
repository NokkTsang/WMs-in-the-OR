"""Dual-encoder exo/ego fusion for scene-graph prediction, MISS phase 3 only ("Microscope-assisted disc removal").

Tests the idea: encode exocentric cameras with V-JEPA 2 and egocentric cameras with SurgMotion,
then fuse the two pooled vectors for the triplet head, instead of one encoder for every camera.
Same causal protocol as evaluate_triplets.py (now / +1s / +4s, Ridge head, train-only vocabulary,
validation-only thresholds) but take-held-out over the 6 MISS/phase-3 takes only, since that is
the richest, highest-activity phase (microscope positioning, bone cutting, instrument hand-overs;
the same take shown in the OR-Action Fig. 3 qualitative example is MISS/3/take/1).

Horizon meaning ("now" vs "+1s"/"+4s"): the input clip is `x(end_frame - 15*4 .. end_frame)`,
i.e. it always *ends* at frame `end_frame` (never sees anything after it). horizon_s=0 ("now")
decodes the scene graph annotated AT that same `end_frame` -- a *nowcast* / current-state read-out
from the clip that just ended, not a future prediction. horizon_s=1 and 4 decode the graph
`1*15` / `4*15` frames (1s / 4s) *after* `end_frame`, which the model never saw: those are the
actual future-prediction targets.

Four primary variants compared here, all with the identical Ridge head / split / thresholds:
  all_vjepa2     -- V-JEPA 2 for every camera (exo + ego), the single-encoder baseline
  all_surgmotion -- SurgMotion for every camera (exo + ego), the other single-encoder baseline
  dual_fusion    -- V-JEPA 2 on exo cameras + SurgMotion on ego cameras, concatenated (proposed idea:
                    SurgMotion's surgical-video fine-tuning should help most on close-up egocentric
                    hand/instrument motion, while general V-JEPA 2 stays fine for wide room views)
  reverse_fusion -- SurgMotion on exo cameras + V-JEPA 2 on ego cameras (the opposite assignment,
                    a negative control for the dual_fusion hypothesis: if the "SurgMotion helps ego"
                    story is right, this should not beat dual_fusion)
exo_vjepa2 / ego_surgmotion (single-view-only) are kept as weaker reference ablations.
"""

import json
from datetime import UTC, datetime

import h5py
import numpy as np
import pandas as pd
from evaluate_triplets import (
    HORIZONS,
    binary_targets,
    bootstrap_f1,
    choose_threshold,
    decode,
    graph_metrics,
    training_vocabulary,
    valid_clip_ends,
)
from extract_multiview_features import feature_path, inventory, read_features, read_validity
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import egoexor_io as eio

SEED = 2026
PHASE = 3
QUALITATIVE_TAKE = "3_1"  # MISS/3/take/1, the same take as the OR-Action paper's Fig. 3 example
VJEPA2, SURGMOTION = "vjepa2", "surgmotion"


def masked_mean(selected: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """selected: (N, C, D) camera features; weights: (N, C) validity mask -> mean over valid cameras, any-valid mask."""
    denom = weights.sum(axis=1, keepdims=True)
    any_valid = denom[:, 0] > 0
    safe_denom = np.where(denom > 0, denom, 1.0)
    mean = (selected * weights[:, :, None]).sum(axis=1) / safe_denom
    return mean, any_valid


def load_phase3():
    meta = []
    vecs = {"exo_vjepa2": [], "exo_surgmotion": [], "ego_vjepa2": [], "ego_surgmotion": []}
    labels = {h: [] for h in HORIZONS}
    coverage = []
    for take in inventory():
        if take["proc"] != "MISS" or take["phase"] != PHASE:
            continue
        exo_cams = sorted(set(take["cams"]) & (eio.EXO_SOURCES - {"ultrasound"}))
        ego_cams = sorted(set(take["cams"]) & eio.EGO_SOURCES)
        if not exo_cams or not ego_cams:
            raise ValueError(f"{take['path']}: missing exo or ego cameras for fusion")
        starts, ends = valid_clip_ends(take["n_frames"])
        valid_views = read_validity(take)[: len(starts)] == 16
        exo_columns = [take["cams"].index(cam) for cam in exo_cams]
        ego_columns = [take["cams"].index(cam) for cam in ego_cams]
        input_mask = valid_views[:, exo_columns].any(axis=1) & valid_views[:, ego_columns].any(axis=1)
        with h5py.File(take["file"], "r") as f:
            frames = sorted({int(end + h * eio.FPS) for end in ends for h in HORIZONS})
            by_frame = {}
            for frame in frames:
                path = f"{take['path']}/annotations/frame_{frame}/rel_annotations"
                if f"{take['path']}/annotations/frame_{frame}" not in f:
                    continue
                rows = f[path][:]
                by_frame[frame] = frozenset("|".join(x.decode("utf-8") for x in row) for row in rows)
            annotation_mask = np.array([all(int(end + h * eio.FPS) in by_frame for h in HORIZONS) for end in ends])
            mask = annotation_mask & input_mask
            if not mask.any():
                raise ValueError(f"{take['path']}: no usable clips")
            coverage.append(
                dict(
                    take_id=f"{take['phase']}_{take['take']}",
                    candidates=len(ends),
                    kept=int(mask.sum()),
                    excluded_missing_annotations=int((~annotation_mask).sum()),
                    excluded_no_usable_view=int((~input_mask).sum()),
                    exo_cams=",".join(exo_cams),
                    ego_cams=",".join(ego_cams),
                )
            )
            exo_weights = valid_views[mask][:, exo_columns]
            ego_weights = valid_views[mask][:, ego_columns]
            for view, cams, _columns, weights in (
                ("exo", exo_cams, exo_columns, exo_weights),
                ("ego", ego_cams, ego_columns, ego_weights),
            ):
                for model in (VJEPA2, SURGMOTION):
                    feats = np.stack(
                        [read_features(feature_path(model, cam, take), take["n_frames"]) for cam in cams], axis=1
                    )[: len(starts)][mask]
                    mean, _ = masked_mean(feats, weights)
                    vecs[f"{view}_{model}"].append(mean)
            starts_k, ends_k = starts[mask], ends[mask]
            for h in HORIZONS:
                labels[h].extend(by_frame[int(end + h * eio.FPS)] for end in ends_k)
        meta.extend(
            dict(
                phase=take["phase"],
                take=take["take"],
                take_id=f"{take['phase']}_{take['take']}",
                start_frame=int(s),
                end_frame=int(e),
            )
            for s, e in zip(starts_k, ends_k)
        )
    meta = pd.DataFrame(meta)
    vecs = {k: np.concatenate(v) for k, v in vecs.items()}
    features = {
        "exo_vjepa2": vecs["exo_vjepa2"],
        "ego_surgmotion": vecs["ego_surgmotion"],
        "all_vjepa2": np.concatenate([vecs["exo_vjepa2"], vecs["ego_vjepa2"]], axis=1),
        "all_surgmotion": np.concatenate([vecs["exo_surgmotion"], vecs["ego_surgmotion"]], axis=1),
        "dual_fusion": np.concatenate([vecs["exo_vjepa2"], vecs["ego_surgmotion"]], axis=1),
        "reverse_fusion": np.concatenate([vecs["exo_surgmotion"], vecs["ego_vjepa2"]], axis=1),
    }
    return meta, features, labels, coverage


def leave_one_take_out(groups: np.ndarray):
    takes = sorted(set(groups))
    for fold, test_take in enumerate(takes):
        test = np.flatnonzero(groups == test_take)
        development = np.flatnonzero(groups != test_take)
        train_local, val_local = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + fold).split(
                development, groups=groups[development]
            )
        )
        train, val = development[train_local], development[val_local]
        yield fold, train, val, test


def evaluate() -> pd.DataFrame:
    out = eio.out_dir("feasibility/MISS_phase3_fusion")
    meta, features, labels, coverage = load_phase3()
    groups = meta.take_id.to_numpy()
    records, take_rows, qualitative_rows = {}, [], []
    for fold, train, val, test in leave_one_take_out(groups):
        vocabulary = training_vocabulary(labels, train)
        y_train = np.concatenate([binary_targets([labels[h][i] for i in train], vocabulary) for h in HORIZONS], axis=1)
        for name, X in features.items():
            scaler = StandardScaler().fit(X[train])
            head = Ridge(alpha=1000.0).fit(scaler.transform(X[train]), y_train)
            val_scores = head.predict(scaler.transform(X[val])).reshape(len(val), len(HORIZONS), len(vocabulary))
            test_scores = head.predict(scaler.transform(X[test])).reshape(len(test), len(HORIZONS), len(vocabulary))
            for j, h in enumerate(HORIZONS):
                truth = [labels[h][i] for i in test]
                threshold = choose_threshold(val_scores[:, j], [labels[h][i] for i in val], vocabulary)
                prediction = decode(test_scores[:, j], vocabulary, threshold)
                key = (name, h)
                acc = records.setdefault(key, dict(truth=[], prediction=[]))
                acc["truth"].extend(truth)
                acc["prediction"].extend(prediction)
                metrics, _ = graph_metrics(truth, prediction)
                take_rows.append(dict(method=name, horizon_s=h, fold=fold, take_id=groups[test][0], **metrics))
                if groups[test][0] == QUALITATIVE_TAKE:
                    for k, idx in enumerate(test):
                        row = meta.iloc[idx]
                        gt, pred = truth[k], prediction[k]
                        qualitative_rows.append(
                            dict(
                                method=name,
                                horizon_s=h,
                                input_start_s=round(row.start_frame / eio.FPS, 2),
                                input_end_s=round(row.end_frame / eio.FPS, 2),  # = t, last input frame
                                target_time_s=round((row.end_frame + h * eio.FPS) / eio.FPS, 2),
                                gt_triplets="; ".join(sorted(gt)),
                                predicted_triplets="; ".join(sorted(pred)),
                                missing="; ".join(sorted(gt - pred)),
                                extra="; ".join(sorted(pred - gt)),
                                sample_f1=(2 * len(gt & pred) / (len(gt) + len(pred)) if gt or pred else 1.0),
                            )
                        )
        print(f"MISS/phase3 fusion: fold {fold + 1}/{len(set(groups))} (test take {groups[test][0]})", flush=True)

    takes_df = pd.DataFrame(take_rows)
    summary = []
    for (method, h), acc in records.items():
        metrics, _ = graph_metrics(acc["truth"], acc["prediction"])
        selected = takes_df[(takes_df.method == method) & (takes_df.horizon_s == h)]
        low, high = bootstrap_f1(selected)
        metrics.update(micro_f1_ci_low=low, micro_f1_ci_high=high)
        summary.append(dict(method=method, horizon_s=h, **metrics))
    result = pd.DataFrame(summary)
    result.to_csv(out / "summary.csv", index=False)
    takes_df.to_csv(out / "per_take.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    qualitative = pd.DataFrame(qualitative_rows).sort_values(["method", "horizon_s", "input_end_s"])
    qualitative.to_csv(out / f"qualitative_take{QUALITATIVE_TAKE}.csv", index=False)
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                status="complete",
                completed_utc=datetime.now(UTC).isoformat(),
                procedure="MISS",
                phase=PHASE,
                phase_name=eio.MISS_PHASE[PHASE],
                n_samples=len(meta),
                n_takes=len(set(groups)),
                horizons_s=HORIZONS,
                horizon_meaning=(
                    "input clip always ends at frame `end_frame` (=t, never sees later frames); "
                    "horizon_s=0 decodes the graph AT t (nowcast, no future info); "
                    "horizon_s=1/4 decode the graph 1s/4s AFTER t (true future prediction)"
                ),
                split="leave-one-take-out (6 folds), inner GroupShuffleSplit(test_size=0.2) for threshold validation",
                decoder="StandardScaler + Ridge(alpha=1000), train takes only",
                qualitative_take=QUALITATIVE_TAKE,
                variants={
                    "exo_vjepa2": "mean of V-JEPA 2 features over available exo cameras (external_1..5, microscope)",
                    "ego_surgmotion": (
                        "mean of SurgMotion features over available ego cameras "
                        "(head_surgeon, assistant, circulator, anesthetist)"
                    ),
                    "all_vjepa2": "concat(V-JEPA 2 exo mean, V-JEPA 2 ego mean) -- single-encoder baseline",
                    "all_surgmotion": "concat(SurgMotion exo mean, SurgMotion ego mean) -- single-encoder baseline",
                    "dual_fusion": "concat(V-JEPA 2 exo mean, SurgMotion ego mean) -- the proposed dual-encoder design",
                    "reverse_fusion": (
                        "concat(SurgMotion exo mean, V-JEPA 2 ego mean) -- opposite assignment, negative control"
                    ),
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    res = evaluate()
    print(res[["method", "horizon_s", "micro_f1", "micro_f1_ci_low", "micro_f1_ci_high", "macro_f1", "exact_match"]])


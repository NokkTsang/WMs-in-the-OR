"""Causal, take-held-out triplet-set decoding with separate Ego / Exo models.

Frozen, uniformly averaged camera features -> ridge heads for now / +1s / +4s.
Vocabulary, scaling and heads use training takes only; thresholds use validation
takes only. All ground-truth triplets, including unseen combinations, are scored.
"""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime

import h5py
import numpy as np
import pandas as pd
from extract_multiview_features import (
    CLIP_LEN,
    CLIP_SPAN,
    MODELS,
    clip_starts,
    feature_path,
    inventory,
    read_features,
    read_validity,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import egoexor_io as eio
from probe_utils import event_metrics

HORIZONS = (0, 1, 4)
SEED = 2026


def valid_clip_ends(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    starts = clip_starts(n_frames)
    ends = starts + CLIP_SPAN - 1
    valid = ends + max(HORIZONS) * eio.FPS < n_frames
    if not valid.any():
        raise ValueError(f"No clips with a full +{max(HORIZONS)}s target in {n_frames} frames")
    return starts[valid], ends[valid]


def complete_target_mask(ends: np.ndarray, available_frames: set[int]) -> np.ndarray:
    return np.array([all(int(end + h * eio.FPS) in available_frames for h in HORIZONS) for end in ends])


def load_data(proc: str, view: str):
    meta, features = [], {name: [] for name in MODELS}
    labels = {h: [] for h in HORIZONS}
    coverage = []
    for take in inventory():
        if take["proc"] != proc:
            continue
        allowed = eio.EGO_SOURCES if view == "ego" else eio.EXO_SOURCES - {"ultrasound"}
        cameras = sorted(set(take["cams"]) & allowed)
        if not cameras:
            raise ValueError(f"{take['path']}: no {view} cameras")
        starts, ends = valid_clip_ends(take["n_frames"])
        valid_views = read_validity(take)[: len(starts)] == CLIP_LEN
        ego_columns = [i for i, cam in enumerate(take["cams"]) if cam in eio.EGO_SOURCES]
        exo_columns = [i for i, cam in enumerate(take["cams"]) if cam in eio.EXO_SOURCES - {"ultrasound"}]
        input_mask = valid_views[:, ego_columns].any(axis=1) & valid_views[:, exo_columns].any(axis=1)
        columns = [take["cams"].index(cam) for cam in cameras]
        weights = valid_views[:, columns]
        with h5py.File(take["file"], "r") as f:
            frames = sorted({int(end + h * eio.FPS) for end in ends for h in HORIZONS})
            by_frame = {}
            for frame in frames:
                path = f"{take['path']}/annotations/frame_{frame}/rel_annotations"
                if f"{take['path']}/annotations/frame_{frame}" not in f:
                    continue
                if path not in f:
                    raise KeyError(f"Annotation frame exists but has no relations dataset: {path}")
                rows = f[path][:]
                if rows.size and (rows.ndim != 2 or rows.shape[1] != 3):
                    raise ValueError(f"Expected subject/relation/object rows at {path}, got {rows.shape}")
                by_frame[frame] = frozenset("|".join(x.decode("utf-8") for x in row) for row in rows)
            annotation_mask = complete_target_mask(ends, set(by_frame))
            mask = annotation_mask & input_mask
            if not mask.any():
                raise ValueError(f"{take['path']}: no clips with usable Ego/Exo inputs and annotations at all horizons")
            coverage.append(
                dict(
                    take_id=f"{take['phase']}_{take['take']}",
                    candidates=len(ends),
                    kept=int(mask.sum()),
                    excluded_missing_annotations=int((~annotation_mask).sum()),
                    excluded_no_usable_view=int((~input_mask).sum()),
                    excluded_total=int((~mask).sum()),
                    cameras=",".join(cameras),
                    **{f"missing_at_{h}s": sum(int(end + h * eio.FPS) not in by_frame for end in ends) for h in HORIZONS},
                )
            )
            if not mask.all():
                print(
                    f"{proc}/{view} {take['phase']}/{take['take']}: excluding {int((~mask).sum())}/{len(mask)} "
                    f"clips; missing annotations={int((~annotation_mask).sum())}, "
                    f"no usable Ego/Exo input={int((~input_mask).sum())}",
                    flush=True,
                )
            for name in MODELS:
                camera_features = [read_features(feature_path(name, cam, take), take["n_frames"]) for cam in cameras]
                selected = np.stack(camera_features, axis=1)[: len(starts)][mask]
                features[name].append((selected * weights[mask, :, None]).sum(axis=1) / weights[mask].sum(axis=1)[:, None])
            used_cameras = [",".join(cam for cam, usable in zip(cameras, row) if usable) for row in weights[mask]]
            starts, ends = starts[mask], ends[mask]
            for h in HORIZONS:
                labels[h].extend(by_frame[int(end + h * eio.FPS)] for end in ends)
        meta.extend(
            dict(
                phase=take["phase"],
                take=take["take"],
                take_id=f"{take['phase']}_{take['take']}",
                start_frame=int(start),
                end_frame=int(end),
                cameras=used,
            )
            for start, end, used in zip(starts, ends, used_cameras)
        )
    if not meta:
        raise ValueError(f"No samples for {proc} / {view}")
    return pd.DataFrame(meta), {name: np.concatenate(parts) for name, parts in features.items()}, labels, coverage


def binary_targets(sets: list, vocabulary: list[str]) -> np.ndarray:
    return np.array([[triplet in s for triplet in vocabulary] for s in sets], dtype=np.float64)


def training_vocabulary(labels: dict, indices: np.ndarray) -> list[str]:
    return sorted({triplet for h in HORIZONS for i in indices for triplet in labels[h][i]})


def take_splits(groups: np.ndarray):
    for fold, (development, test) in enumerate(GroupKFold(5).split(groups, groups=groups)):
        train_local, val_local = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + fold).split(development, groups=groups[development])
        )
        train, val = development[train_local], development[val_local]
        assert not (set(groups[train]) & set(groups[val]))
        assert not (set(groups[train]) & set(groups[test]))
        assert not (set(groups[val]) & set(groups[test]))
        yield fold, train, val, test


def choose_threshold(scores: np.ndarray, truth: list, vocabulary: list[str]) -> float:
    targets = binary_targets(truth, vocabulary).astype(bool)
    n_true = sum(map(len, truth))
    candidates = []
    for threshold in np.linspace(0.0, 1.0, 21):
        prediction = scores >= threshold
        tp = int(np.count_nonzero(prediction & targets))
        denominator = int(prediction.sum()) + n_true
        candidates.append((2 * tp / denominator if denominator else 1.0, float(threshold)))
    return max(candidates)[1]


def decode(scores: np.ndarray, vocabulary: list[str], threshold: float) -> list[frozenset[str]]:
    return [frozenset(vocabulary[i] for i in np.flatnonzero(row >= threshold)) for row in scores]


def topk(scores: np.ndarray, vocabulary: list[str], k: int) -> list[frozenset[str]]:
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return [frozenset(vocabulary[i] for i in row) for row in indices]


def graph_metrics(truth: list, prediction: list) -> tuple[dict, list[dict]]:
    if len(truth) != len(prediction) or not truth:
        raise ValueError("Need equal, nonempty lists of ground-truth and predicted graphs")
    support, predicted, correct = Counter(), Counter(), Counter()
    exact, sample_f1 = 0, []
    for gt, pred in zip(truth, prediction):
        support.update(gt)
        predicted.update(pred)
        correct.update(gt & pred)
        exact += gt == pred
        sample_f1.append(2 * len(gt & pred) / (len(gt) + len(pred)) if gt or pred else 1.0)
    details = []
    for triplet in sorted(support.keys() | predicted.keys()):
        tp, n_true, n_pred = correct[triplet], support[triplet], predicted[triplet]
        details.append(
            dict(
                triplet=triplet,
                support=n_true,
                tp=tp,
                fp=n_pred - tp,
                fn=n_true - tp,
                precision=tp / n_pred if n_pred else 0.0,
                recall=tp / n_true if n_true else 0.0,
                f1=2 * tp / (n_true + n_pred) if n_true + n_pred else 0.0,
            )
        )
    tp, n_true, n_pred = sum(correct.values()), sum(support.values()), sum(predicted.values())
    metrics = dict(
        n_samples=len(truth),
        tp=tp,
        fp=n_pred - tp,
        fn=n_true - tp,
        precision=tp / n_pred if n_pred else 0.0,
        recall=tp / n_true if n_true else 0.0,
        micro_f1=2 * tp / (n_true + n_pred) if n_true + n_pred else 1.0,
        macro_f1=float(np.mean([d["f1"] for d in details if d["support"]])) if n_true else np.nan,
        sample_f1=float(np.mean(sample_f1)),
        exact_match=exact / len(truth),
    )
    return metrics, details


def known_map(scores: np.ndarray, truth: list, vocabulary: list[str]) -> float:
    targets = binary_targets(truth, vocabulary)
    columns = np.flatnonzero(targets.sum(axis=0))
    if not len(columns):
        return float("nan")
    return float(np.mean([average_precision_score(targets[:, i], scores[:, i]) for i in columns]))


def bootstrap_f1(take_rows: pd.DataFrame) -> tuple[float, float]:
    counts = take_rows[["tp", "fp", "fn"]].to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    sums = counts[rng.integers(len(counts), size=(2000, len(counts)))].sum(axis=1)
    denominator = 2 * sums[:, 0] + sums[:, 1] + sums[:, 2]
    values = np.divide(2 * sums[:, 0], denominator, out=np.ones(len(sums)), where=denominator > 0)
    return tuple(np.quantile(values, [0.025, 0.975]))


def paired_f1_delta(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    a, b = a.set_index("take_id").sort_index(), b.set_index("take_id").sort_index()
    if not a.index.equals(b.index) or not np.array_equal(a.n_samples, b.n_samples):
        raise ValueError("Paired comparisons require the same held-out takes and sample counts")
    draws = np.random.default_rng(SEED).integers(len(a), size=(2000, len(a)))
    scores = []
    for frame in (a, b):
        counts = frame[["tp", "fp", "fn"]].to_numpy(dtype=float)
        summed = np.vstack([counts.sum(axis=0), counts[draws].sum(axis=1)])
        denom = 2 * summed[:, 0] + summed[:, 1] + summed[:, 2]
        scores.append(np.divide(2 * summed[:, 0], denom, out=np.ones(len(summed)), where=denom > 0))
    delta = scores[0] - scores[1]
    low, high = np.quantile(delta[1:], [0.025, 0.975])
    return dict(delta_micro_f1=delta[0], ci_low=low, ci_high=high)


def evaluate(proc: str, view: str) -> pd.DataFrame:
    out = eio.out_dir(f"feasibility/{proc}_{view}")
    (out / "protocol.json").write_text(json.dumps(dict(status="running", procedure=proc, view=view)), encoding="utf-8")
    meta, features, labels, coverage = load_data(proc, view)
    groups = meta.take_id.to_numpy()
    records, fold_rows, take_rows, manifest = {}, [], [], []

    def record(method, h, fold, indices, truth, prediction, scores=None, vocabulary=None, threshold=None):
        key = (method, h)
        acc = records.setdefault(key, dict(truth=[], prediction=[], top5=[], top10=[], known=0, ap=[], events=[]))
        acc["truth"].extend(truth)
        acc["prediction"].extend(prediction)
        metrics, _ = graph_metrics(truth, prediction)
        if scores is not None:
            acc["top5"].extend(topk(scores, vocabulary, 5))
            acc["top10"].extend(topk(scores, vocabulary, 10))
            ap = known_map(scores, truth, vocabulary)
            acc["ap"].append(ap)
            known = sum(len(gt & set(vocabulary)) for gt in truth)
            acc["known"] += known
            metrics.update(mAP_known=ap, known_label_fraction=known / sum(map(len, truth)))
            if h:
                events = event_metrics(
                    scores, binary_targets([labels[0][i] for i in indices], vocabulary), binary_targets(truth, vocabulary)
                )
                acc["events"].append(events)
                metrics.update(events)
        fold_rows.append(dict(method=method, horizon_s=h, fold=fold, threshold=threshold, **metrics))
        held_groups = groups[indices]
        for take_id in np.unique(held_groups):
            mask = np.flatnonzero(held_groups == take_id)
            take_metrics, _ = graph_metrics([truth[i] for i in mask], [prediction[i] for i in mask])
            take_rows.append(dict(method=method, horizon_s=h, fold=fold, take_id=take_id, **take_metrics))

    predictions_path = out / "predictions.tmp.jsonl"
    with predictions_path.open("w", encoding="utf-8") as predictions_file:
        for fold, train, val, test in take_splits(groups):
            vocabulary = training_vocabulary(labels, train)
            y_train = np.concatenate([binary_targets([labels[h][i] for i in train], vocabulary) for h in HORIZONS], axis=1)
            manifest.append(
                dict(
                    fold=fold,
                    train_takes=sorted(set(groups[train])),
                    validation_takes=sorted(set(groups[val])),
                    test_takes=sorted(set(groups[test])),
                    vocabulary=vocabulary,
                )
            )
            for name in (*MODELS, "frequency"):
                if name == "frequency":
                    val_scores = np.broadcast_to(y_train.mean(0), (len(val), y_train.shape[1]))
                    test_scores = np.broadcast_to(y_train.mean(0), (len(test), y_train.shape[1]))
                else:
                    scaler = StandardScaler().fit(features[name][train])
                    head = Ridge(alpha=1000.0).fit(scaler.transform(features[name][train]), y_train)
                    val_scores = head.predict(scaler.transform(features[name][val]))
                    test_scores = head.predict(scaler.transform(features[name][test]))
                val_scores = val_scores.reshape(len(val), len(HORIZONS), len(vocabulary))
                test_scores = test_scores.reshape(len(test), len(HORIZONS), len(vocabulary))
                current_prediction, current_threshold = None, None
                for j, h in enumerate(HORIZONS):
                    truth = [labels[h][i] for i in test]
                    threshold = choose_threshold(val_scores[:, j], [labels[h][i] for i in val], vocabulary)
                    scores = test_scores[:, j]
                    prediction = decode(scores, vocabulary, threshold)
                    record(name, h, fold, test, truth, prediction, scores, vocabulary, threshold)
                    if h == 0:
                        current_prediction, current_threshold = prediction, threshold
                    elif name != "frequency":
                        record(
                            f"{name}_persist",
                            h,
                            fold,
                            test,
                            truth,
                            current_prediction,
                            test_scores[:, 0],
                            vocabulary,
                            current_threshold,
                        )
                    if name != "frequency":
                        for k, index in enumerate(test):
                            row = meta.iloc[index]
                            ranked = np.argsort(-scores[k], kind="stable")[:10]
                            entry = dict(
                                procedure=proc,
                                view=view,
                                model=name,
                                fold=fold,
                                take_id=row.take_id,
                                input_start_frame=int(row.start_frame),
                                input_end_frame=int(row.end_frame),
                                target_frame=int(row.end_frame + h * eio.FPS),
                                horizon_s=h,
                                cameras=row.cameras.split(","),
                                threshold=threshold,
                                truth=sorted(truth[k]),
                                predicted=sorted(prediction[k]),
                                missing=sorted(truth[k] - prediction[k]),
                                extra=sorted(prediction[k] - truth[k]),
                                top10=[dict(triplet=vocabulary[t], score=float(scores[k, t])) for t in ranked],
                            )
                            predictions_file.write(json.dumps(entry) + "\n")
            for h in HORIZONS[1:]:
                record("GT_persist_oracle", h, fold, test, [labels[h][i] for i in test], [labels[0][i] for i in test])
            print(f"{proc}/{view}: fold {fold + 1}/5, {len(vocabulary)} training triplets", flush=True)

    takes_df = pd.DataFrame(take_rows)
    summary, triplet_rows, relation_rows = [], [], []
    for (method, h), acc in records.items():
        metrics, details = graph_metrics(acc["truth"], acc["prediction"])
        selected = takes_df[(takes_df.method == method) & (takes_df.horizon_s == h)]
        low, high = bootstrap_f1(selected)
        metrics.update(micro_f1_ci_low=low, micro_f1_ci_high=high)
        if acc["ap"]:
            metrics["mAP_known_fold_mean"] = float(np.mean(acc["ap"]))
            metrics["known_label_fraction"] = acc["known"] / sum(map(len, acc["truth"]))
            for k in (5, 10):
                ranked_metrics, ranked_details = graph_metrics(acc["truth"], acc[f"top{k}"])
                metrics[f"recall_at_{k}"] = ranked_metrics["recall"]
                metrics[f"mean_recall_at_{k}"] = float(np.mean([d["recall"] for d in ranked_details if d["support"]]))
        if acc["events"]:
            events = pd.DataFrame(acc["events"])
            for event in ("onset", "offset"):
                metrics[f"{event}_auc_fold_mean"] = events[f"{event}_AUC"].mean()
                metrics[f"{event}_eligible_classes_fold_mean"] = events[f"{event}_n_triplets"].mean()
        summary.append(dict(procedure=proc, view=view, method=method, horizon_s=h, **metrics))
        triplet_rows.extend(dict(method=method, horizon_s=h, **d) for d in details)
        relations = sorted({d["triplet"].split("|")[1] for d in details})
        for relation in relations:
            subset = [d for d in details if d["triplet"].split("|")[1] == relation]
            tp, fp, fn = (sum(d[key] for d in subset) for key in ("tp", "fp", "fn"))
            relation_rows.append(
                dict(
                    method=method,
                    horizon_s=h,
                    relation=relation,
                    support=tp + fn,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                )
            )
    result = pd.DataFrame(summary)
    comparisons = []
    for h in HORIZONS:
        primary = takes_df[(takes_df.method == "vjepa2") & (takes_df.horizon_s == h)]
        for baseline in ("dinov2", "frequency", *(("vjepa2_persist",) if h else ())):
            other = takes_df[(takes_df.method == baseline) & (takes_df.horizon_s == h)]
            comparisons.append(dict(horizon_s=h, comparison=f"vjepa2 - {baseline}", **paired_f1_delta(primary, other)))
    pd.DataFrame(comparisons).to_csv(out / "paired_comparisons.csv", index=False)
    result.to_csv(out / "summary.csv", index=False)
    takes_df.to_csv(out / "per_take.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "per_fold.csv", index=False)
    pd.DataFrame(triplet_rows).to_csv(out / "per_triplet.csv", index=False)
    pd.DataFrame(relation_rows).to_csv(out / "per_relation.csv", index=False)
    meta.to_csv(out / "samples.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    predictions_path.replace(out / "predictions.jsonl")
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                status="complete",
                completed_utc=datetime.now(UTC).isoformat(),
                procedure=proc,
                view=view,
                n_samples=len(meta),
                n_takes=len(set(groups)),
                fps=eio.FPS,
                horizons_s=HORIZONS,
                clip_span_frames=CLIP_SPAN,
                fusion="uniform mean of room-camera features whose 16 sampled input frames are all nonzero",
                encoder_training="frozen",
                decoder="StandardScaler + Ridge(alpha=1000), train takes only",
                threshold="per horizon, validation micro-F1 over grid 0..1 in steps of 0.05; no refit",
                vocabulary="union of triplets in training targets only, no frequency filtering",
                evaluation="all annotated target triplets; unseen triplets count as false negatives",
                missing_annotations="exclude clips without annotations at any of the three target times; see coverage.csv",
                input_validity="require >=1 usable camera in each view group; same timestamps for Ego/Exo; see coverage.csv",
                bootstrap="2000 resamples of held-out takes; descriptive, not independent training runs",
                event_auc="fold mean over training-vocabulary triplets with >=10 events and >=10 non-events per fold",
                folds=manifest,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result[["method", "horizon_s", "micro_f1", "precision", "recall"]].to_string(index=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--procedures", nargs="+", choices=["MISS", "Ultrasound"], default=["MISS", "Ultrasound"])
    parser.add_argument("--views", nargs="+", choices=["exo", "ego"], default=["exo", "ego"])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        for procedure in args.procedures:
            for view in args.views:
                evaluate(procedure, view)

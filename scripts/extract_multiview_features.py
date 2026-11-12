"""Complete frozen V-JEPA 2 / DINOv2 features, reading all cameras together.

Uses the existing L16s4 format. Ego and Exo cameras are encoded independently;
the ultrasound screen stream is not a room camera and is excluded.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

import egoexor_io as eio

CLIP_LEN, FRAME_STEP, CLIP_STRIDE = 16, 4, 15
CLIP_SPAN = (CLIP_LEN - 1) * FRAME_STEP + 1
MODELS = ("vjepa2", "dinov2")


def inventory():
    takes = []
    for path in eio.h5_files():
        with h5py.File(path, "r") as f:
            for proc, phase, take in eio.list_takes(f):
                tp = eio.take_path(proc, phase, take)
                sources = eio.get_sources(f, tp)
                unknown = set(sources) - eio.EGO_SOURCES - eio.EXO_SOURCES
                if unknown:
                    raise ValueError(f"{tp}: unknown cameras {unknown}")
                takes.append(
                    dict(
                        file=path,
                        proc=proc,
                        phase=phase,
                        take=take,
                        path=tp,
                        n_frames=eio.n_frames(f, tp),
                        sources=sources,
                        cams=[cam for cam in sources if cam != "ultrasound"],
                    )
                )
    for proc, expected in (("MISS", 16), ("Ultrasound", 25)):
        actual = sum(t["proc"] == proc for t in takes)
        if actual != expected:
            raise ValueError(f"{proc}: expected {expected} complete takes, found {actual}")
    return takes


def feature_path(model: str, cam: str, take: dict) -> Path:
    return eio.FEAT_DIR / f"{model}_{cam}_L16s4" / f"{take['proc']}_{take['phase']}_{take['take']}.npz"


def clip_starts(n_frames: int) -> np.ndarray:
    starts = np.arange(0, n_frames - CLIP_SPAN + 1, CLIP_STRIDE)
    if not len(starts):
        raise ValueError(f"Take has only {n_frames} frames; need {CLIP_SPAN}")
    return starts


def read_features(path: Path, n_frames: int) -> np.ndarray:
    starts = clip_starts(n_frames)
    with np.load(path) as d:
        if (
            int(d["n_frames"]) != n_frames
            or not np.array_equal(d["t_start"], starts)
            or not np.array_equal(d["t_center"], starts + CLIP_SPAN // 2)
            or d["feat"].shape != (len(starts), 1024)
            or d["seq"].shape != (len(starts), CLIP_LEN // 2, 1024)
            or not np.isfinite(d["feat"]).all()
            or not np.isfinite(d["seq"]).all()
        ):
            raise ValueError(f"Incomplete or incompatible L16s4 feature file: {path}")
        return d["feat"].copy()


def validity_path(take: dict) -> Path:
    return eio.RAW_DIR.parent / "cache" / "clip_validity" / f"{take['proc']}_{take['phase']}_{take['take']}_L16s4.npz"


def read_validity(take: dict) -> np.ndarray:
    path = validity_path(take)
    with np.load(path) as data:
        counts = data["nonblack_frames"]
        if (
            not np.array_equal(data["t_start"], clip_starts(take["n_frames"]))
            or data["cameras"].tolist() != take["cams"]
            or counts.shape != (len(data["t_start"]), len(take["cams"]))
            or np.any(counts > CLIP_LEN)
        ):
            raise ValueError(f"Incompatible input-validity file: {path}")
        return counts


def nonblack_counts(raw: np.ndarray, idx: np.ndarray, camera_indices: list[int]) -> np.ndarray:
    frame_valid = raw.max(axis=(2, 3, 4))[:, camera_indices] > 0
    return frame_valid[idx].sum(axis=1).astype(np.uint8)


def extract(batch_size: int, names: list[str]) -> None:
    import torch
    from extract_features import CKPT, build_model, encode

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this feature extraction")
    torch.set_num_threads(4)
    pending = []
    for take in inventory():
        todo = []
        for cam in take["cams"]:
            for name in names:
                path = feature_path(name, cam, take)
                if path.exists():
                    read_features(path, take["n_frames"])
                else:
                    todo.append((cam, name, path))
        if validity_path(take).exists():
            read_validity(take)
        if todo or not validity_path(take).exists():
            pending.append((take, todo))
    print(f"Pending: {len(pending)} takes, {sum(len(todo) for _, todo in pending)} feature files", flush=True)
    if not pending:
        return
    needed = {name for _, todo in pending for _, name, _ in todo}
    models = {
        name: (build_model(name, CLIP_LEN, 256 if name == "vjepa2" else 224, CKPT.get(name)), 256 if name == "vjepa2" else 224)
        for name in names
        if name in needed
    }
    for take, todo in pending:
        begin = time.perf_counter()
        starts = clip_starts(take["n_frames"])
        seqs = {(cam, name): [] for cam, name, _ in todo}
        cameras = sorted({cam for cam, _, _ in todo})
        counts = []
        camera_indices = [take["sources"].index(cam) for cam in take["cams"]]
        with h5py.File(take["file"], "r") as f:
            ds = f[f"{take['path']}/frames/rgb"]
            for b in range(0, len(starts), batch_size):
                batch = starts[b : b + batch_size]
                first, stop = int(batch[0]), int(batch[-1] + CLIP_SPAN)
                raw = ds[first:stop]
                idx = batch[:, None] - first + np.arange(CLIP_LEN)[None] * FRAME_STEP
                counts.append(nonblack_counts(raw, idx, camera_indices))
                for cam in cameras:
                    frames = eio.to_rgb(raw[:, take["sources"].index(cam)], cam)
                    clips = torch.from_numpy(frames[idx]).cuda()
                    for name, (model, res) in models.items():
                        if (cam, name) in seqs:
                            seqs[(cam, name)].append(encode(model, name, clips, res).cpu().numpy())
                    del clips, frames
                del raw
        for cam, name, path in todo:
            seq = np.concatenate(seqs[(cam, name)])
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(
                tmp,
                feat=seq.mean(1),
                seq=seq.astype(np.float16),
                t_start=starts,
                t_center=starts + CLIP_SPAN // 2,
                n_frames=take["n_frames"],
            )
            tmp.replace(path)
            read_features(path, take["n_frames"])
        path = validity_path(take)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        counts = np.concatenate(counts)
        np.savez(tmp, nonblack_frames=counts, cameras=take["cams"], t_start=starts)
        tmp.replace(path)
        read_validity(take)
        print(
            f"{take['proc']} {take['phase']}/{take['take']}: {len(starts)} clips, "
            f"{len(todo)} files, {int((counts < CLIP_LEN).sum())} camera-clips contain all-zero frames, "
            f"{time.perf_counter() - begin:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    extract(args.batch_size, args.models)

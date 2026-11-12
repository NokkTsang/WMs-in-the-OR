"""Measure the actual encoder I/O on one synchronized Ego / Exo clip."""

import json
import time

import h5py
import numpy as np
import torch
from extract_features import CKPT, build_model, encode
from extract_multiview_features import CLIP_LEN, CLIP_SPAN, FRAME_STEP, inventory
from PIL import Image, ImageDraw

import egoexor_io as eio


def main() -> None:
    torch.set_num_threads(4)
    out = eio.out_dir("feasibility/io")
    take = next(take for take in inventory() if (take["proc"], take["phase"], take["take"]) == ("MISS", 3, 1))
    start_frame = ((take["n_frames"] - CLIP_SPAN) // 2) // eio.FPS * eio.FPS
    model = build_model("vjepa2", CLIP_LEN, 256, CKPT["vjepa2"])
    captured = {}

    def capture_input(_module, inputs):
        x = inputs[0]
        captured["encoder_input"] = dict(shape=list(x.shape), dtype=str(x.dtype), finite=bool(x.isfinite().all()))

    def capture_output(_module, _inputs, output):
        captured["encoder_output"] = dict(shape=list(output.shape), dtype=str(output.dtype), finite=bool(output.isfinite().all()))

    before = model.register_forward_pre_hook(capture_input)
    after = model.register_forward_hook(capture_output)
    records = []
    try:
        with h5py.File(take["file"], "r") as f:
            raw = f[f"{take['path']}/frames/rgb"][start_frame : start_frame + CLIP_SPAN]
        for cam in ("external_1", "head_surgeon"):
            frames = eio.to_rgb(raw[:, take["sources"].index(cam)], cam)[::FRAME_STEP]
            if np.any(frames.max(axis=(1, 2, 3)) == 0):
                raise ValueError(f"{cam}: I/O example contains an all-zero frame")
            clips = torch.from_numpy(frames[None]).cuda()
            seq = encode(model, "vjepa2", clips, 256)
            assert captured["encoder_input"]["shape"] == [1, 3, 16, 256, 256]
            assert captured["encoder_output"]["shape"] == [1, 2048, 1024]
            assert captured["encoder_input"]["finite"] and captured["encoder_output"]["finite"]
            assert seq.shape == (1, 8, 1024) and bool(seq.isfinite().all())
            times = []
            torch.cuda.reset_peak_memory_stats()
            for _ in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                seq = encode(model, "vjepa2", clips, 256)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            canvas = Image.new("RGB", (4 * 168, 4 * 190), "white")
            draw = ImageDraw.Draw(canvas)
            for i, frame in enumerate(frames):
                x, y = (i % 4) * 168, (i // 4) * 190
                canvas.paste(Image.fromarray(frame).resize((168, 168)), (x, y))
                frame_id = start_frame + i * FRAME_STEP
                draw.text((x + 3, y + 169), f"frame {frame_id} / {frame_id / eio.FPS:.2f}s", fill="black")
            canvas.save(out / f"{cam}_input.png")
            record = dict(
                camera=cam,
                take=[take["proc"], take["phase"], take["take"]],
                raw_clip_shape=list(clips.shape),
                raw_dtype=str(clips.dtype),
                source_fps=eio.FPS,
                sampled_frame_indices=list(range(start_frame, start_frame + CLIP_SPAN, FRAME_STEP)),
                first_to_last_seconds=(CLIP_SPAN - 1) / eio.FPS,
                tubelet_shape=[2, 16, 16],
                **captured,
                seq_shape=list(seq.shape),
                feat_shape=list(seq.mean(1).shape),
                median_preprocess_and_encode_ms=float(np.median(times)),
                peak_allocated_MiB=torch.cuda.max_memory_allocated() / 1024**2,
                gpu=torch.cuda.get_device_name(),
                torch_version=torch.__version__,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        before.remove()
        after.remove()
    (out / "io.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

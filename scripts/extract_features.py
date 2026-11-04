"""Extract frozen clip-level features from one EgoExOR camera.

Image encoders (frames are encoded independently, then pooled per 2-frame tubelet):
  dinov2         DINOv2 ViT-L/14, general-domain self-supervised
  surgenet       SurgeNetXL CAFormer-S18, DINO self-supervised on ~4.7M surgical frames (public)
  caformer_in1k  CAFormer-S18, ImageNet-1k supervised: same architecture as surgenet, no surgical pre-training (control)
Video encoders:
  vjepa2         V-JEPA 2 ViT-L/16, public general-domain video encoder (initialisation of SurgMotion)
  surgmotion     SurgMotion-L, same architecture with surgical pre-training (gated checkpoint)

Per take, writes features/<tag>/<procedure>_<phase>_<take>.npz with
  feat      (n_clips, D)       mean over time and space
  seq       (n_clips, T/2, D)  spatially pooled, one vector per tubelet (float16)
  t_start, t_center (n_clips,) frame indices inside the take
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import egoexor_io as eio

SURGMOTION_REPO = eio.ROOT / "third_party" / "SurgMotion"
CKPT = {
    "vjepa2": eio.ROOT / "ckpts" / "vjepa2_vitl.pt",
    "surgmotion": eio.ROOT / "ckpts" / "SurgMotion-vitl.pt",
    "surgenet": eio.ROOT / "ckpts" / "surgenetxl_teacher.pth",
    "caformer_in1k": eio.ROOT / "ckpts" / "caformer_s18_in1k.pth",
}
IMAGE_MODELS = {"dinov2", "surgenet", "caformer_in1k"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def build_model(name, clip_len, res, ckpt):
    if name == "dinov2":
        import timm

        model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=res)
        return model.eval().cuda()

    if name in ("surgenet", "caformer_in1k"):
        sys.path.insert(0, str(SURGMOTION_REPO / "foundation_models" / "SurgeNet"))
        from metaformer import caformer_s18

        model = caformer_s18(num_classes=0, pretrained="SurgeNet" if name == "surgenet" else "ImageNet")
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        msg = model.load_state_dict({k: v for k, v in state.items() if not k.startswith("head.")}, strict=False)
        print("load_state_dict:", msg)
        assert not msg.unexpected_keys and all(k.startswith("head.") for k in msg.missing_keys), (
            "checkpoint does not match the model"
        )
        return model.eval().cuda()

    sys.path.insert(0, str(SURGMOTION_REPO))
    import src.models.vision_transformer as vit

    model = vit.vit_large(
        img_size=res, num_frames=clip_len, patch_size=16, tubelet_size=2, use_rope=True, uniform_power=(name == "surgmotion")
    )
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["encoder"]
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print("load_state_dict:", msg)
    missing = [k for k in msg.missing_keys if k != "pos_embed"]  # RoPE models have no absolute pos_embed
    assert not missing, f"missing keys: {missing[:5]}"
    return model.eval().cuda()


def frame_features(model, name, frames):
    """(N,3,res,res) -> (N,D) float32 for the image encoders."""
    if name == "dinov2":
        return model.forward_features(frames)[:, model.num_prefix_tokens :].float().mean(1)  # mean of patch tokens
    return model.forward_features(frames)[0].float()  # CAFormer: LayerNorm(spatial mean of the last stage)


@torch.no_grad()
def encode(model, name, x, res):
    """x: (B,T,H,W,3) uint8 on GPU -> (B, T/2, D) float32, spatially pooled per tubelet (2 frames)."""
    x = x.permute(0, 4, 1, 2, 3).float() / 255.0  # B,3,T,H,W
    B, _, T, H, W = x.shape
    if (H, W) != (res, res):
        frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
        frames = F.interpolate(frames, size=(res, res), mode="bicubic", align_corners=False)
        x = frames.reshape(B, T, 3, res, res).permute(0, 2, 1, 3, 4)
    x = (x - MEAN.to(x.device)) / STD.to(x.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if name in IMAGE_MODELS:
            frames = x.permute(0, 2, 1, 3, 4).reshape(B * T, 3, res, res)
            per_frame = frame_features(model, name, frames).reshape(B, T, -1)
            return per_frame.reshape(B, T // 2, 2, -1).mean(2)  # same temporal resolution as V-JEPA (tubelet 2)
        tokens = model(x).float()  # (B, (T/2)*S, D)
        return tokens.reshape(B, T // 2, tokens.shape[1] // (T // 2), -1).mean(2)


def extract_take(f, take, args, models, todo):
    """Read the camera once and encode it with every model in `todo` ({name: out_path})."""
    tp = eio.take_path(*take)
    span = (args.clip_len - 1) * args.frame_step + 1
    n_total = eio.n_frames(f, tp)
    starts = np.arange(0, n_total - span + 1, args.clip_stride)[: args.max_clips]

    t0 = time.time()
    frames = eio.load_cam(f, tp, args.cam, upto=int(starts[-1]) + span)
    t_read = time.time() - t0

    offsets = np.arange(args.clip_len) * args.frame_step
    for name, out_path in todo.items():
        model, res = models[name]
        seqs = []
        for b in range(0, len(starts), args.batch_size):
            idx = starts[b : b + args.batch_size, None] + offsets[None]
            clips = torch.from_numpy(frames[idx]).cuda(non_blocking=True)  # (B,T,H,W,3)
            seqs.append(encode(model, name, clips, res).cpu())
        seq = torch.cat(seqs).numpy()
        tmp = out_path.with_suffix(".tmp.npz")  # write-then-rename: an interrupted run never leaves a corrupt feature file
        np.savez(tmp, feat=seq.mean(1), seq=seq.astype(np.float16), t_start=starts, t_center=starts + span // 2, n_frames=n_total)
        tmp.replace(out_path)
    print(f"{tp}: {len(starts)} clips, read {t_read:.0f}s, total {time.time() - t0:.0f}s -> {', '.join(todo)}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--model",
        required=True,
        nargs="+",
        choices=["dinov2", "surgenet", "caformer_in1k", "vjepa2", "surgmotion"],
        help="one or more encoders; the camera is read once and shared",
    )
    ap.add_argument("--cam", default="external_3", help="MISS: external_1..5, microscope; Ultrasound: simstation, or_light")
    ap.add_argument("--files", nargs="*", default=None, help="h5 file names in data/raw (default: all complete files)")
    ap.add_argument("--procedure", default=None, choices=["MISS", "Ultrasound"], help="only takes of this procedure")
    ap.add_argument("--clip-len", type=int, default=16, help="frames per clip")
    ap.add_argument("--frame-step", type=int, default=4, help="temporal subsampling inside a clip (source is 15 FPS)")
    ap.add_argument("--clip-stride", type=int, default=15, help="frames between clip starts (takes are only 1-4 min long)")
    ap.add_argument("--res", type=int, default=None, help="input size; default 224 for image encoders, 256 for V-JEPA models")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (single model only); default ckpts/<model>")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-clips", type=int, default=None, help="limit clips per take (smoke tests)")
    ap.add_argument("--tag", default=None, help="output folder name under features/ (single model only)")
    args = ap.parse_args()
    if len(args.model) > 1 and (args.ckpt or args.tag):
        ap.error("--ckpt and --tag need a single --model")
    return args


def main():
    args = parse_args()
    models, out_dirs = {}, {}
    for name in args.model:
        res = args.res or (224 if name in IMAGE_MODELS else 256)
        out_dirs[name] = eio.FEAT_DIR / (args.tag or f"{name}_{args.cam}_L{args.clip_len}s{args.frame_step}")
        out_dirs[name].mkdir(parents=True, exist_ok=True)
        models[name] = (build_model(name, args.clip_len, res, args.ckpt or CKPT.get(name)), res)

    files = [eio.RAW_DIR / n for n in args.files] if args.files else eio.h5_files()
    for h5_path in files:
        with h5py.File(h5_path, "r") as f:
            for take in eio.list_takes(f):
                if args.procedure and take[0] != args.procedure:
                    continue
                if args.cam not in eio.get_sources(f, eio.take_path(*take)):
                    print(f"skip {take}: no camera {args.cam}")
                    continue
                todo = {n: d / "{}_{}_{}.npz".format(*take) for n, d in out_dirs.items()}
                todo = {n: p for n, p in todo.items() if not p.exists()}
                if todo:
                    extract_take(f, take, args, models, todo)


if __name__ == "__main__":
    main()

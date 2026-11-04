"""Shared helpers for reading the EgoExOR Legacy (336x336) HDF5 files.

Layout: /data/{MISS|Ultrasound}/{phase}/take/{take}/frames/rgb -> (T, n_cams, H, W, 3) uint8
Camera names live in the attrs `source_0..source_{n-1}` of <take>/sources.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from pillow_lut import load_cube_file

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FEAT_DIR = ROOT / "features"
OUT_DIR = ROOT / "outputs"
LUT_FILE = ROOT / "third_party" / "EgoExOR" / "data" / "utils" / "egoexor.cube"

FPS = 15
EGO_SOURCES = {"head_surgeon", "assistant", "circulator", "anesthetist"}
# Per the official constants.py, or_light / microscope are Aria devices on the OR light / microscope, i.e. exocentric.
EXO_SOURCES = {
    "or_light",
    "microscope",
    "external_1",
    "external_2",
    "external_3",
    "external_4",
    "external_5",
    "simstation",
    "ultrasound",
}

# File sizes from the HF listing; used to ignore partially downloaded files.
EXPECTED_BYTES = {
    "miss_1.h5": 23096195996,
    "miss_2.h5": 10948046356,
    "miss_3.h5": 19237956096,
    "miss_4.h5": 6659063255,
    "ultrasound_1.h5": 1669545224,
    "ultrasound_2.h5": 15086406795,
    "ultrasound_3.h5": 25419165066,
    "ultrasound_4.h5": 38208302030,
    "ultrasound_5_14.h5": 29290147542,
    "ultrasound_5_58.h5": 25362991532,
}

MISS_PHASE = {
    1: "Patient entry, anesthesia, field prep",
    2: "Incision and initial access",
    3: "Microscope-assisted disc removal",
    4: "Cleanup and dressing",
}
UI_PHASE = {
    1: "Patient entry and skin antisepsis",
    2: "Patient entry and skin antisepsis (repeated)",
    3: "US-based target identification",
    4: "US-guided injection and cleanup (A)",
    5: "US-guided injection and cleanup (B)",
}
PHASE_NAME = {"MISS": MISS_PHASE, "Ultrasound": UI_PHASE}

# Transcribed from third_party/EgoExOR/data/TAKE_METADATA.md: (procedure, phase) -> [(takes, anomaly)]; "none" = nominal workflow
ANOMALY = {
    ("MISS", 1): [((1, 2), "none"), ((3,), "Instrument missing"), ((4,), "Role changes")],
    ("MISS", 2): [((1, 2), "none"), ((3, 4), "Microscope misalignment")],
    ("MISS", 3): [
        ((1, 2), "none"),
        ((3,), "Microscope misalignment"),
        ((4,), "Role changes"),
        ((5, 6), "Scalpel required + role changes"),
    ],
    ("MISS", 4): [((1, 2), "none")],
    ("Ultrasound", 1): [((1,), "none")],
    ("Ultrasound", 2): [((1, 2), "none"), ((3, 4, 5, 6), "Instrument dropped")],
    ("Ultrasound", 3): [((1, 2), "Gloves omitted"), ((3, 4), "Skin marker (variation)")],
    ("Ultrasound", 4): [((1, 2), "none"), ((3, 4), "Syringe jammed"), ((5, 6), "Needle repositioning")],
    ("Ultrasound", 5): [((1, 2), "none"), ((3, 4), "Needle repositioning"), ((5, 6), "Syringe jammed")],
}


def anomaly_of(proc, phase, take):
    for takes, a in ANOMALY.get((proc, phase), []):
        if take in takes:
            return a
    return "(not in metadata)"


def out_dir(name):
    """Result folder `outputs/<name>` (created on demand)."""
    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def h5_files():
    """Completely downloaded take files (splits.h5 excluded); files with leftover `.seg*` progress files are still downloading."""
    return sorted(
        p
        for p in RAW_DIR.glob("*.h5")
        if p.name in EXPECTED_BYTES and p.stat().st_size == EXPECTED_BYTES[p.name] and not list(RAW_DIR.glob(p.name + ".seg*"))
    )


def take_path(procedure, phase, take):
    return f"/data/{procedure}/{phase}/take/{take}"


def list_takes(f):
    """Return sorted [(procedure, phase, take)] for an open legacy file."""
    out = []
    for proc in f["data"].keys():
        for phase in f[f"data/{proc}"].keys():
            grp = f[f"data/{proc}/{phase}"].get("take")
            if grp is None:
                continue
            out.extend((proc, int(phase), int(t)) for t in grp.keys())
    return sorted(out)


def get_sources(f, tp):
    g = f[f"{tp}/sources"]
    names = []
    for i in range(int(g.attrs["source_count"])):
        n = g.attrs[f"source_{i}"]
        names.append(n.decode() if isinstance(n, (bytes, bytearray)) else str(n))
    return names


def cam_index(f, tp, cam):
    return get_sources(f, tp).index(cam)


def n_frames(f, tp):
    return f[f"{tp}/frames/rgb"].shape[0]


@lru_cache(maxsize=1)
def _lut():
    return load_cube_file(str(LUT_FILE))


def to_rgb(frames, cam):
    """Stored frames (N,H,W,3) -> RGB.

    external_* (Azure Kinect) are stored uncorrected (dark, bluish): apply the official colour LUT, no channel flip.
    All other cameras are stored BGR.
    """
    if cam.startswith("external_"):
        return np.stack([np.asarray(Image.fromarray(x).filter(_lut())) for x in frames])
    return np.ascontiguousarray(frames[..., ::-1])


def read_frames(f, tp, cam, idxs):
    """Read frames `idxs` of one camera -> (len(idxs), H, W, 3) uint8 RGB. Slow for scattered indices (see load_cam)."""
    ci = cam_index(f, tp, cam)
    ds = f[f"{tp}/frames/rgb"]
    return to_rgb(np.stack([ds[int(i), ci] for i in idxs]), cam)


def load_cam(f, tp, cam, upto=None):
    """Read the take (first `upto` frames) of one camera into RAM as (T,H,W,3) RGB uint8.

    The file uses tiny gzip chunks spanning all cameras, so random single-frame reads are ~10x slower than
    contiguous blocks.
    """
    ci = cam_index(f, tp, cam)
    ds = f[f"{tp}/frames/rgb"]
    T = ds.shape[0] if upto is None else min(upto, ds.shape[0])
    out = np.empty((T,) + ds.shape[2:], dtype=np.uint8)
    for s in range(0, T, 200):
        out[s : s + 200] = ds[s : min(s + 200, T), ci]
    return to_rgb(out, cam)

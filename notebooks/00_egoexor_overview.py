# %% [markdown]
# # EgoExOR overview (Legacy release, 336×336)
# Each cell is preceded by one line: **Does** (what the cell does) and **Shows** (what to read from the result).
# Dataset-level statistics come from the small `splits.h5` and the official `TAKE_METADATA.md`, so they cover the *whole* dataset even if only some h5 files are downloaded.
# Frame examples run only on downloaded files; missing phases are skipped. Figures are saved to `outputs/overview/`.

# %%
import json
import subprocess
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
OFFICIAL = ROOT / "third_party" / "EgoExOR"
if not OFFICIAL.exists():
    (ROOT / "third_party").mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ardamamur/EgoExOR", str(OFFICIAL)], check=True)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(OFFICIAL / "data" / "utils"))
import egoexor_io as eio
from constants import CAMERA_TYPE_MAPPING, GAZE_FIXATION, GAZE_FIXATION_TO_TAKE

OUT = eio.out_dir("overview")
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 80)
FPS = eio.FPS

# %% [markdown]
# ## 1. Versions and download
# | Version | HF repo | Resolution | Note |
# |---|---|---|---|
# | **Legacy** (used here) | `ardamamur/EgoExOR` | 336×336 | stored Azure Kinect frames need the official LUT at read time (section 4) |
# | HQ | `TUM/EgoExOR` | 1344×1344 | pre-corrected RGB, raw depth, per-device audio; different h5 layout, not supported here |
#
# Every h5 bundles **all cameras and modalities** (ego + exo), so one cannot download exo only.

# %% [markdown]
# **Does:** lists the h5 files on Hugging Face with size and local download status. **Shows:** the full dataset is ~195 GB; only the listed subset is local.

# %%
import requests


def hf_listing(repo):
    r = requests.get(f"https://huggingface.co/api/datasets/{repo}/tree/main", timeout=20)
    r.raise_for_status()
    return pd.DataFrame(
        [{"file": x["path"], "GB": round(x["size"] / 1e9, 2)} for x in r.json() if x["type"] == "file" and x["size"] > 1e6]
    )


try:
    listing = hf_listing("ardamamur/EgoExOR")
except Exception as e:  # offline fallback
    print("HF API unavailable:", e)
    listing = pd.DataFrame([{"file": k, "GB": round(v / 1e9, 2)} for k, v in eio.EXPECTED_BYTES.items()])
done = {p.name for p in eio.h5_files()}
listing["downloaded"] = listing.file.isin(done) | ((listing.file == "splits.h5") & (eio.RAW_DIR / "splits.h5").exists())
print(f"total {listing.GB.sum():.0f} GB, downloaded {listing[listing.downloaded].GB.sum():.0f} GB")
listing

# %% [markdown]
# ## 2. HDF5 structure
# Path: `/data/{MISS|Ultrasound}/{phase}/take/{take}/...`. A **take** is one continuous recording; a **phase** is a surgical stage; each phase has several takes.
#
# **Does:** indexes the downloaded takes and prints the datasets inside one take. **Shows:** every modality and its shape (`T` = number of frames).

# %%
TAKE_FILE = {}
for p in eio.h5_files():
    with h5py.File(p, "r") as f:
        for t in eio.list_takes(f):
            TAKE_FILE[t] = p
print(len(TAKE_FILE), "downloaded takes from", sorted({p.name for p in TAKE_FILE.values()}))


def sources_of(t):
    with h5py.File(TAKE_FILE[t], "r") as f:
        return eio.get_sources(f, eio.take_path(*t))


def frames_of(t, cam, idxs):
    with h5py.File(TAKE_FILE[t], "r") as f:
        return eio.read_frames(f, eio.take_path(*t), cam, idxs)


def n_frames_of(t):
    with h5py.File(TAKE_FILE[t], "r") as f:
        return eio.n_frames(f, eio.take_path(*t))


EX = ("MISS", 4, 1)  # example take for modality demos
with h5py.File(TAKE_FILE[EX], "r") as f:
    tp = eio.take_path(*EX)
    rows = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and not name.startswith("annotations/frame_"):
            rows.append(dict(path=name, shape=obj.shape, dtype=str(obj.dtype)))

    f[tp].visititems(visit)
print(f"{tp}  (+ annotations/frame_i/{{rel_annotations, scene_graph}} per frame)")
pd.DataFrame(rows)

# %% [markdown]
# | Dataset | Meaning |
# |---|---|
# | `frames/rgb (T, n_cam, 336, 336, 3)` | synchronized RGB of all cameras, 15 FPS; camera names in `sources` attrs |
# | `eye_gaze/coordinates (T, 4, 3)` | per ego wearer: `[type id, x, y]` in that ego image; -1 = invalid |
# | `hand_tracking/positions (T, 4, 17)` | per wearer: `[type id, 8 (x, y) points]` (wrist/palm per hand + normal end points); may be NaN |
# | `audio/waveform (N, 2)`, `audio/snippets (T, 48000, 2)` | 48 kHz stereo; snippets = 1 s window per frame |
# | `point_cloud/{coordinates,colors} (T, 2500, 3)` | fused exo RGB-D, downsampled (Legacy has no raw depth) |
# | `annotations/frame_i/rel_annotations (k, 3)` | scene-graph triplets (subject, relation, object) |
# | `metadata/vocabulary` | 37 entities, 23 relations |

# %% [markdown]
# ## 3. Whole dataset: size, phases, anomalies
# Two simulated procedures (actors + phantom, not real surgery): **MISS** (spine surgery, 4 phases) and **Ultrasound** (US-guided injection, 5 phases). Phase and anomaly descriptions are transcribed from the official `data/TAKE_METADATA.md`.
#
# **Does:** builds one row per take of the *entire* dataset from `splits.h5` (one row = one frame). **Shows:** 41 takes, 84,553 frames (94 min), 54–278 s per take, with phase, split and scripted anomaly.

# %%
PHASE_NAME, anomaly_of = eio.PHASE_NAME, eio.anomaly_of

rows = []
with h5py.File(eio.RAW_DIR / "splits.h5", "r") as f:
    for s in ["train", "validation", "test"]:
        d = pd.DataFrame(f["splits"][s][:])
        d["split"] = s
        rows.append(d)
fr = pd.concat(rows)
full = (
    fr.groupby(
        [fr.surgery_type.str.decode("utf-8").rename("procedure"), fr.procedure_id.rename("phase"), fr.take_id.rename("take")]
    )
    .agg(frames=("frame_id", "size"), split=("split", "first"))
    .reset_index()
)
full["seconds"] = (full.frames / FPS).round(1)
full["phase_name"] = [PHASE_NAME[p][ph] for p, ph in zip(full["procedure"], full["phase"])]
full["anomaly"] = [anomaly_of(p, ph, t) for p, ph, t in zip(full["procedure"], full["phase"], full["take"])]
full["downloaded"] = [(p, ph, t) in TAKE_FILE for p, ph, t in zip(full["procedure"], full["phase"], full["take"])]
print(
    f"{len(full)} takes, {full.frames.sum()} frames, {full.seconds.sum() / 60:.1f} min; take length {full.seconds.min():.0f}-{full.seconds.max():.0f} s (mean {full.seconds.mean():.0f} s)"
)
full

# %% [markdown]
# **Does:** plots minutes per phase, split by official train/val/test, and counts takes per split. **Shows:** the splits are take-level; **MISS phase 4 has no test take** and Ultrasound phase 1 is train-only, so phase-held-out evaluation is limited.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, proc in zip(axes, ["MISS", "Ultrasound"]):
    g = full[full.procedure == proc]
    piv = (
        g.pivot_table(index="phase", columns="split", values="seconds", aggfunc="sum")
        .reindex(columns=["train", "validation", "test"])
        .fillna(0)
        / 60
    )
    piv.plot.bar(stacked=True, ax=ax, color=["tab:blue", "tab:orange", "tab:red"])
    ax.set_title(f"{proc}: {len(g)} takes")
    ax.set_ylabel("minutes")
plt.tight_layout()
plt.savefig(OUT / "phases.png", dpi=110)
plt.show()
print(full.anomaly.value_counts().to_dict())
print(
    full.pivot_table(index=["procedure", "phase"], columns="split", values="take", aggfunc="count")
    .fillna(0)
    .astype(int)
    .reindex(columns=["train", "validation", "test"])
)

# %% [markdown]
# ## 4. Cameras
# * **Ego (4):** `head_surgeon, assistant, circulator, anesthetist`: Aria glasses of each role (fisheye).
# * **Exo:** `external_1..5` = fixed Azure Kinect room cameras (**MISS only**); `microscope`, `or_light` = Aria devices on the microscope / OR light; `simstation` = fixed room camera (**Ultrasound only**, not further documented); `ultrasound` = US screen feed.
#
# **Does:** shows every camera of one MISS take and one Ultrasound take at the same time stamp. **Shows:** the camera sets differ per procedure (MISS 10 cameras, Ultrasound 5 here).

# %%
ROLE = {**{c: "ego" for c in eio.EGO_SOURCES}, **{c: "exo" for c in eio.EXO_SOURCES}}


def cam_mosaic(take, title, fname):
    cams = sources_of(take)
    t_frame = n_frames_of(take) // 2
    fig, axes = plt.subplots(1, len(cams), figsize=(2.6 * len(cams), 3.0))
    for ax, c in zip(np.atleast_1d(axes), cams):
        ax.imshow(frames_of(take, c, [t_frame])[0])
        ax.axis("off")
        ax.set_title(f"{c}\n({ROLE.get(c, '?')})", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(OUT / fname, dpi=110)
    plt.show()


first_take = {}
for t in sorted(TAKE_FILE):
    first_take.setdefault(t[0], t)
for proc, t in first_take.items():
    cam_mosaic(t, f"{proc}, take {t[1:]}, same time stamp", f"cameras_{proc}.png")

# %% [markdown]
# **Does:** compares stored `external_*` frames with the same frames after the official LUT. **Shows:** the h5 stores *uncorrected* (dark, bluish) Azure Kinect frames; the official loader applies `egoexor.cube` at read time, and we do the same in `egoexor_io.to_rgb`. (Our observation; the docs only state that a LUT is applied.)

# %%
t = first_take["MISS"]
cams = [c for c in sources_of(t) if c.startswith("external_")]
with h5py.File(TAKE_FILE[t], "r") as f:
    ds = f[eio.take_path(*t) + "/frames/rgb"]
    ci = eio.get_sources(f, eio.take_path(*t))
    raw = {c: ds[n_frames_of(t) // 2, ci.index(c)] for c in cams}
fig, axes = plt.subplots(2, len(cams), figsize=(3 * len(cams), 6))
for j, c in enumerate(cams):
    axes[0, j].imshow(raw[c][..., ::-1])
    axes[0, j].set_title(f"{c}: stored", fontsize=9)
    axes[0, j].axis("off")
    axes[1, j].imshow(eio.to_rgb(raw[c][None], c)[0])
    axes[1, j].set_title("after LUT", fontsize=9)
    axes[1, j].axis("off")
plt.tight_layout()
plt.savefig(OUT / "lut.png", dpi=110)
plt.show()

# %% [markdown]
# **Does:** plots the five Azure Kinect poses from `calibration/exocentric/camera0{1-5}.json` together with a point cloud. **Shows:** cameras hang ~2.3 m high looking down on the room and the cloud lies in the same frame (y is up). Assumption: `camera0k` = `external_k` (not stated in the docs; the overlay is consistent with it).

# %%
from scipy.spatial.transform import Rotation as Rot

cam_pose = {}
for i in range(1, 6):
    d = json.load(open(OFFICIAL / "data" / "calibration" / "exocentric" / f"camera0{i}.json"))["value0"]
    tr, q = d["camera_pose"]["translation"], d["camera_pose"]["rotation"]
    pos = np.array([tr["m00"], tr["m10"], tr["m20"]])
    fwd = Rot.from_quat([q["x"], q["y"], q["z"], q["w"]]).apply([0, 0, 1])
    cam_pose[f"external_{i}"] = (pos, fwd)
with h5py.File(TAKE_FILE[EX], "r") as f:
    pc = f[eio.take_path(*EX) + "/point_cloud/coordinates"][n_frames_of(EX) // 2]

fig = plt.figure(figsize=(7.5, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], s=1, c="lightgray", label="point cloud")
for k, (pos, fwd) in cam_pose.items():
    ax.scatter(pos[0], pos[2], pos[1], s=40)
    ax.text(pos[0], pos[2], pos[1], k.replace("external_", "E"))
    ax.quiver(pos[0], pos[2], pos[1], fwd[0], fwd[2], fwd[1], length=0.8)
ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_zlabel("y (up)")
ax.legend()
ax.view_init(elev=22, azim=-60)
plt.tight_layout()
plt.savefig(OUT / "camera_positions.png", dpi=110)
plt.show()

# %% [markdown]
# ## 5. One example per phase
# **Does:** for each downloaded (procedure, phase), shows the first take at 4 time points from a room-level camera (`external_3` for MISS, `simstation` for Ultrasound), with phase name and anomaly. **Shows:** how the scene looks in each phase and how it changes inside a take (e.g. the team leaves at the end of MISS phase 4; Ultrasound take 1 ends with black frames).

# %%
EXO = {"MISS": "external_3", "Ultrasound": "simstation"}
avail = sorted({(t[0], t[1]) for t in TAKE_FILE})
fig, axes = plt.subplots(len(avail), 4, figsize=(12, 3.1 * len(avail)), squeeze=False)
for row, (proc, ph) in zip(axes, avail):
    t = min(k for k in TAKE_FILE if k[:2] == (proc, ph))
    idx = np.linspace(0, n_frames_of(t) - 1, 4).astype(int)
    for ax, im, i in zip(row, frames_of(t, EXO[proc], idx), idx):
        ax.imshow(im)
        ax.axis("off")
        ax.text(
            0.03,
            0.96,
            f"t={i / FPS:.0f}s",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="white",
            bbox=dict(fc="black", alpha=0.6, pad=1.5),
        )
    row[0].set_title(
        f"{proc} phase {ph} take {t[2]}: {PHASE_NAME[proc][ph]} | anomaly: {anomaly_of(*t)}",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
fig.subplots_adjust(hspace=0.22, wspace=0.02, left=0.01, right=0.99, top=0.96, bottom=0.01)
plt.savefig(OUT / "phases_frames.png", dpi=100, bbox_inches="tight")
plt.show()
print("not downloaded yet (procedure, phase -> #takes):", full[~full.downloaded].groupby(["procedure", "phase"]).size().to_dict())

# %% [markdown]
# ## 6. One example per modality
# Example take: `MISS phase 4 take 1`, frame 700 (t ≈ 47 s).

# %%
T = 700
tp = eio.take_path(*EX)

# %% [markdown]
# ### 6.1 Eye gaze
# **Does:** draws each wearer's gaze point (red) on their own ego image (with the official per-take offset `GAZE_FIXATION` where required). **Shows:** gaze falls on the surgical field (patient, hands, monitor), i.e. where attention is.

# %%
ego = [c for c in sources_of(EX) if c in eio.EGO_SOURCES]
with h5py.File(TAKE_FILE[EX], "r") as f:
    gaze = f[tp + "/eye_gaze/coordinates"][T]
    hands = f[tp + "/hand_tracking/positions"][T]
imgs = {c: frames_of(EX, c, [T])[0] for c in ego}


def gaze_of(role):
    typ = CAMERA_TYPE_MAPPING[role]
    for g in gaze:
        if g[0] == typ and not np.allclose(g[1:], [-1, -1]):
            x, y = g[1], g[2]
            if tp in GAZE_FIXATION_TO_TAKE.get(role, []):
                x, y = x + GAZE_FIXATION["x"], y + GAZE_FIXATION["y"]
            return x, y
    return None


fig, axes = plt.subplots(1, len(ego), figsize=(3.4 * len(ego), 3.6))
for ax, c in zip(np.atleast_1d(axes), ego):
    ax.imshow(imgs[c])
    ax.axis("off")
    g = gaze_of(c)
    if g:
        ax.scatter([g[0]], [g[1]], s=120, c="red", edgecolors="white")
    ax.set_title(c, fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "gaze.png", dpi=110)
plt.show()

# %% [markdown]
# ### 6.2 Hand tracking
# **Does:** draws the 8 hand points per wearer (blue/green dots = left/right wrist and palm, orange/yellow lines = to normal end points). **Shows:** tracking exists only when hands are visible to the wearer (here assistant and head surgeon; the others are NaN).

# %%
col = ["tab:blue", "tab:blue", "tab:green", "tab:green", "orange", "orange", "gold", "gold"]
fig, axes = plt.subplots(1, len(ego), figsize=(3.4 * len(ego), 3.6))
for ax, c in zip(np.atleast_1d(axes), ego):
    ax.imshow(imgs[c])
    ax.axis("off")
    row = next((h for h in hands if h[0] == CAMERA_TYPE_MAPPING[c]), None)
    n_ok = 0
    if row is not None:
        pts = row[1:].reshape(8, 2)
        for i in range(8):
            if not np.isnan(pts[i]).any():
                ax.scatter(*pts[i], s=25, c=col[i])
                n_ok += 1
        for b in range(4):
            if not np.isnan(pts[[b, b + 4]]).any():
                ax.plot(*zip(pts[b], pts[b + 4]), c=col[b + 4], lw=1.5)
    ax.set_title(f"{c}: {n_ok}/8 points", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "hands.png", dpi=110)
plt.show()

# %% [markdown]
# ### 6.3 Audio
# **Does:** plots the full waveform and spectrogram of the take (red line = frame 700). **Shows:** 48 kHz stereo audio aligned to the video; bursts mark speech and activity. Amplitudes are raw (unnormalized) values.

# %%
from scipy.signal import spectrogram

with h5py.File(TAKE_FILE[EX], "r") as f:
    wav = f[tp + "/audio/waveform"][:]
sr = 48000
mono = wav.mean(1)
print(f"waveform {wav.shape}: {wav.shape[0] / sr:.1f} s at {sr} Hz")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 4.5), sharex=True)
a1.plot(np.arange(len(mono)) / sr, mono, lw=0.3)
a1.set_ylabel("amplitude")
f_, t_, S = spectrogram(mono, sr, nperseg=4096, noverlap=2048)
a2.pcolormesh(t_, f_, 10 * np.log10(S + 1e-12), shading="auto")
a2.set_ylim(0, 8000)
a2.set_ylabel("Hz")
a2.set_xlabel("seconds")
for a in (a1, a2):
    a.axvline(T / FPS, c="red")
plt.tight_layout()
plt.savefig(OUT / "audio.png", dpi=110)
plt.show()

# %% [markdown]
# ### 6.4 Point cloud
# **Does:** shows the fused 2,500-point cloud at 3 time points (y up). **Shows:** coarse 3D room structure (table, people); too sparse for fine detail. Legacy has no raw depth.

# %%
fr_ids = (0, T, 1300)
with h5py.File(TAKE_FILE[EX], "r") as f:
    pcs = [f[tp + "/point_cloud/coordinates"][i] for i in fr_ids]
    cols = [f[tp + "/point_cloud/colors"][i] for i in fr_ids]
fig = plt.figure(figsize=(14, 4.5))
for k, (p, c, i) in enumerate(zip(pcs, cols, fr_ids)):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    c = c / 255.0 if c.max() > 1.5 else c
    ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=2, c=np.clip(c, 0, 1))
    ax.set_title(f"frame {i} (t={i / FPS:.0f}s)")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y (up)")
    ax.view_init(elev=22, azim=-60)
plt.tight_layout()
plt.savefig(OUT / "pointcloud.png", dpi=110)
plt.show()

# %% [markdown]
# ### 6.5 Ultrasound screen
# **Does:** lists the cameras present in each downloaded file and shows the `ultrasound` screen stream (first take of each Ultrasound phase, 4 time points). **Shows:** the ultrasound image the operator sees; it exists only in the Ultrasound files.

# %%
src = {
    p.name: sorted({c for t, q in TAKE_FILE.items() if q == p for c in sources_of(t)}) for p in sorted(set(TAKE_FILE.values()))
}
for k, v in src.items():
    print(f"{k:18s} {v}")
us = [(t, c) for t in TAKE_FILE for c in sources_of(t) if c == "ultrasound"]
if us:
    first = {}
    for t, _ in sorted(us):
        first.setdefault(t[:2], t)
    fig, axes = plt.subplots(len(first), 4, figsize=(11, 2.9 * len(first)), squeeze=False)
    for row, t in zip(axes, first.values()):
        idx = np.linspace(0, n_frames_of(t) - 1, 4).astype(int)
        for ax, im, i in zip(row, frames_of(t, "ultrasound", idx), idx):
            ax.imshow(im)
            ax.axis("off")
            ax.set_title(f"t={i / FPS:.0f}s", fontsize=8)
        row[0].set_title(f"{t[0]} phase {t[1]} take {t[2]}\nt={idx[0] / FPS:.0f}s", loc="left", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "ultrasound_screen.png", dpi=100)
    plt.show()
else:
    print("no `ultrasound` stream in the downloaded files")

# %% [markdown]
# ### 6.6 Scene graph
# **Does:** draws the scene graph (nodes = entities, edges = relations) of four frames and prints the vocabulary. **Shows:** a per-frame structured description of the room (who holds / touches / is close to what). It is sparse (3–9 triplets) and changes over time, unlike the phase label.

# %%
import networkx as nx

with h5py.File(TAKE_FILE[EX], "r") as f:
    vocab_e = [x[0].decode() for x in f["metadata/vocabulary/entity"][:]]
    vocab_r = [x[0].decode() for x in f["metadata/vocabulary/relation"][:]]
    trip = {
        i: [tuple(x.decode() for x in r) for r in f[f"{tp}/annotations/frame_{i}/rel_annotations"][:]] for i in (0, 300, T, 1300)
    }
print(f"{len(vocab_e)} entities: {vocab_e}\n\n{len(vocab_r)} relations: {vocab_r}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (i, tr) in zip(axes.ravel(), trip.items()):
    G = nx.MultiDiGraph()
    for s, r, o in tr:
        G.add_edge(s, o, label=r)
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1400, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, connectionstyle="arc3,rad=0.15", node_size=1400)
    elab = {}
    for s, o, d in G.edges(data=True):
        elab.setdefault((s, o), []).append(d["label"])
    nx.draw_networkx_edge_labels(G, pos, {k: ",".join(v) for k, v in elab.items()}, font_size=6, ax=ax)
    ax.set_title(f"frame {i} (t={i / FPS:.0f}s): {len(tr)} triplets")
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "scenegraph.png", dpi=100)
plt.show()

# %% [markdown]
# **Does:** counts relations and entities over all downloaded takes (1 frame/s), then plots which triplets are present over time in the example take. **Shows:** `closeTo`/`holding`-type relations dominate, and triplets switch on/off in blocks as the activity changes, so scene graphs give a time-varying target for "future OR state". (`anaesthetist`/`anesthetist` are both used in the vocabulary.)

# %%
from collections import Counter


def rel_at(f, tpath, i):
    ds = f.get(f"{tpath}/annotations/frame_{i}/rel_annotations")
    return [] if ds is None else ds[:]


rel_c, ent_c, cover = Counter(), Counter(), {}
for t, p in TAKE_FILE.items():
    with h5py.File(p, "r") as f:
        tpath = eio.take_path(*t)
        n = eio.n_frames(f, tpath)
        have = 0
        for i in range(0, n, FPS):
            rr = rel_at(f, tpath, i)
            have += len(rr) > 0
            for s, r, o in rr:
                rel_c[r.decode()] += 1
                ent_c[s.decode()] += 1
                ent_c[o.decode()] += 1
        cover[t] = f"{have}/{len(range(0, n, FPS))}"
print("frames with annotations / sampled frames:", cover)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
pd.Series(rel_c).sort_values().plot.barh(ax=a1, title="relation frequency")
pd.Series(ent_c).sort_values().tail(20).plot.barh(ax=a2, title="top-20 entities")
plt.tight_layout()
plt.savefig(OUT / "sg_stats.png", dpi=110)
plt.show()

with h5py.File(TAKE_FILE[EX], "r") as f:
    n = eio.n_frames(f, tp)
    per_sec = [frozenset(" | ".join(x.decode() for x in r) for r in rel_at(f, tp, i)) for i in range(0, n, FPS)]
allt = sorted(set().union(*per_sec), key=lambda s: -sum(s in x for x in per_sec))[:30]
M = np.array([[s in x for x in per_sec] for s in allt], dtype=float)
plt.figure(figsize=(12, 6.5))
plt.imshow(M, aspect="auto", cmap="Greys", interpolation="nearest")
plt.yticks(range(len(allt)), allt, fontsize=6)
plt.xlabel("seconds")
plt.title(f"{EX}: 30 most frequent triplets over time")
plt.subplots_adjust(left=0.3)
plt.savefig(OUT / "sg_timeline.png", dpi=110, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Storyboard
# **Does:** shows 6 time points of one take from two exo cameras and one ego camera. **Shows:** cameras are time-synchronized and the scene changes substantially within a take (the team leaves and the room empties at the end).

# %%
cams = ["external_3", "external_5", "head_surgeon"]
idx = np.linspace(0, n_frames_of(EX) - 1, 6).astype(int)
fig, axes = plt.subplots(len(cams), 6, figsize=(15, 2.7 * len(cams)))
for r, c in enumerate(cams):
    imgs_ = frames_of(EX, c, idx)
    for k in range(6):
        axes[r, k].imshow(imgs_[k])
        axes[r, k].axis("off")
        if r == 0:
            axes[r, k].set_title(f"t={idx[k] / FPS:.0f}s", fontsize=9)
    axes[r, 0].text(-0.02, 0.5, c, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right")
plt.tight_layout()
plt.savefig(OUT / "storyboard.png", dpi=100)
plt.show()

# %% [markdown]
# ## 8. Summary
# | Content | Spec | Use for the world model |
# |---|---|---|
# | exo RGB | MISS: 5 fixed cams + microscope; Ultrasound: 2; 15 FPS, 336² | input video → latent OR state |
# | ego RGB, gaze, hands | 4 roles | attention / actions (later) |
# | scene graph | 37 entities, 23 relations, every frame | time-varying semantic target |
# | phase + anomaly | 9 (procedure, phase) pairs; scripted anomalies per take | semantic supervision; hard cases |
# | point cloud, audio | 2,500 pts/frame; 48 kHz | optional |
# | splits | take-level train/val/test | generalization tests |
#
# Limits: simulated; 94 min, 84,553 frames, 41 takes (MISS 16, Ultrasound 25); no raw depth in Legacy; sparse scene graph (3–9 triplets/frame).

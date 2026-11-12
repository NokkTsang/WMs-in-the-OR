# %% [markdown]
# # V-JEPA 2 feasibility: inputs, outputs and generated triplet sets
# Run `scripts/check_vjepa_io.py`, `scripts/extract_multiview_features.py`, then
# `scripts/evaluate_triplets.py` first. This notebook reads the completed results,
# rather than silently retraining. `EGOEXOR_PROC` selects MISS or Ultrasound.
#
# Ego and Exo use separate heads. All available room cameras within a group are
# uniformly fused at each timestamp; the ultrasound screen is excluded. The
# forecast clock starts at the LAST input frame, not the clip centre. Missing
# annotations are excluded, not interpreted as empty scene graphs.
# Entirely zero raw frames invalidate a camera clip. Both groups must have at
# least one usable camera, keeping their evaluated timestamps identical.
#
# These are closed-vocabulary triplet-set predictions, not object localization,
# open-vocabulary generation, image synthesis, or a full learned world model.

# %%
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from IPython.display import display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import egoexor_io as eio

PROC = os.environ.get("EGOEXOR_PROC", "MISS")
OUT = eio.out_dir("feasibility")
pd.set_option("display.max_columns", 30)

# %% [markdown]
# ## 1. Measured I/O
# Pixels `[B, 3, 16, 256, 256]` -> tokens `[B, 2048, 1024]`.
# A tubelet is 2 sampled frames x 16 x 16 pixels; it is created INSIDE the encoder.
# The current probes spatially average to `[B, 8, 1024]`, then temporally average
# to `[B, 1024]`. Triplets come from the fitted head, NOT directly from V-JEPA.
# The 16 sampled frames have relative indices 0, 4, ..., 60 at source 15 FPS (4 seconds).
# New clips start every 15 source frames (one clip per second).

# %%
io = json.loads((OUT / "io" / "io.json").read_text(encoding="utf-8"))
display(pd.json_normalize(io))
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
for ax, cam in zip(axes, ("external_1", "head_surgeon")):
    ax.imshow(plt.imread(OUT / "io" / f"{cam}_input.png"))
    ax.set_title(f"Actual sampled input: {cam}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Current and future triplet-set accuracy
# Micro-F1 counts all annotated triplets, including combinations unseen in
# training. Macro-F1 weights each test-supported triplet equally. `mAP_known`
# covers only supported training-vocabulary classes and is not full-graph accuracy.
# Recall@K ranks closed-vocabulary triplets (no bounding-box matching).
#
# `frequency` has no visual input. DINOv2 sees the same sampled frames and
# camera groups, at its existing 224px resolution (V-JEPA uses 256px).
# `*_persist` copies that encoder's predicted current graph into the future.
# `GT_persist_oracle` copies the true current graph: privileged, not deployable.
# Confidence intervals resample held-out takes; they do not capture training-seed
# variation or prove generalization to real operations / new datasets.
# Both groups target the room-level graph; there are no visibility-specific
# labels guaranteeing that an Ego camera can see every scored triplet.

# %%
for view in ("exo", "ego"):
    protocol = json.loads((OUT / f"{PROC}_{view}" / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "complete":
        raise RuntimeError(f"Incomplete evaluation for {PROC}/{view}; run scripts/evaluate_triplets.py first")
tables = {view: pd.read_csv(OUT / f"{PROC}_{view}" / "summary.csv") for view in ("exo", "ego")}
summary = pd.concat(tables.values(), ignore_index=True)
display(summary.round(3))
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, (view, table) in zip(axes, tables.items()):
    for method in ("vjepa2", "dinov2", "frequency", "vjepa2_persist", "GT_persist_oracle"):
        rows = table[table.method == method].sort_values("horizon_s")
        ax.plot(rows.horizon_s, rows.micro_f1, marker="o", label=method)
    ax.set(title=f"{PROC}: {view}", xlabel="Seconds after last observed frame", ylabel="Triplet micro-F1", ylim=(0, 1))
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / f"{PROC}_triplet_f1.png", dpi=150)
plt.show()

# %%
for view in ("exo", "ego"):
    print(f"{PROC} / {view}: paired take-bootstrap differences")
    display(pd.read_csv(OUT / f"{PROC}_{view}" / "paired_comparisons.csv").round(3))
    print("Annotation coverage and actual camera sets")
    display(pd.read_csv(OUT / f"{PROC}_{view}" / "coverage.csv"))

# %% [markdown]
# ## 3. Relation-level failures (dataset relation names, no invented global/local labels)
# Relation F1 below still requires an EXACT subject/relation/object match.
# Inspect support: rare relations can have very uncertain estimates.

# %%
relation_tables = []
for view in ("exo", "ego"):
    frame = pd.read_csv(OUT / f"{PROC}_{view}" / "per_relation.csv")
    frame = frame[(frame.method == "vjepa2") & (frame.horizon_s == 0)].copy()
    frame["view"] = view
    relation_tables.append(frame)
relations = pd.concat(relation_tables)
display(relations.pivot(index="relation", columns="view", values=["support", "f1"]).round(3))

# %% [markdown]
# ## 4. Actual graph outputs: correct, missed and spurious edges
# One preselected example per group: the first phase-3 take, nearest its middle
# evaluated timestamp, at h=0. Selection does NOT use prediction correctness.
# All evaluated timestamps and both encoders are in `predictions.jsonl`.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, view in zip(axes, ("exo", "ego")):
    sample_meta = pd.read_csv(OUT / f"{PROC}_{view}" / "samples.csv")
    take_id = sample_meta.loc[sample_meta.phase == 3, "take_id"].iloc[0]
    frames = sample_meta.loc[sample_meta.take_id == take_id, "end_frame"].sort_values().tolist()
    chosen_frame = frames[len(frames) // 2]
    example = None
    with (OUT / f"{PROC}_{view}" / "predictions.jsonl").open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if (row["model"], row["take_id"], row["input_end_frame"], row["horizon_s"]) == ("vjepa2", take_id, chosen_frame, 0):
                example = row
                break
    if example is None:
        raise ValueError(f"No held-out example at {take_id}/{chosen_frame}")
    gt, pred = set(example["truth"]), set(example["predicted"])
    print(f"{PROC}/{view}, take {take_id}, frame {chosen_frame}")
    display(
        pd.DataFrame(
            [
                dict(triplet=t, status="correct" if t in gt & pred else "missed" if t in gt else "spurious")
                for t in sorted(gt | pred)
            ]
        )
    )
    graph = nx.DiGraph()
    edges = {}
    for t in sorted(gt | pred):
        subject, relation, obj = t.split("|")
        status = "OK" if t in gt & pred else "MISSED" if t in gt else "EXTRA"
        graph.add_edge(subject, obj)
        edges.setdefault((subject, obj), []).append(f"{relation} [{status}]")
    positions = nx.kamada_kawai_layout(graph.to_undirected())
    nx.draw_networkx(graph, positions, ax=ax, node_size=1400, font_size=7, node_color="#dce8f8", arrowsize=15)
    nx.draw_networkx_edge_labels(
        graph, positions, ax=ax, edge_labels={edge: "\n".join(labels) for edge, labels in edges.items()}, font_size=6
    )
    ax.set_title(f"{PROC} {view}: take {take_id}, frame {chosen_frame}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / f"{PROC}_graph_examples.png", dpi=150)
plt.show()

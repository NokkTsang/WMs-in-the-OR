"""Figures for the MISS phase-3 dual-encoder fusion test (2026-11-18).

1. Bar chart: all-V-JEPA2 vs all-SurgMotion vs dual-encoder fusion, 3 horizons (micro-F1).
2. Qualitative timeline for MISS/3/take/1 (the OR-Action Fig. 3 take): per-clip sample-F1
   over time, for the same 3 methods, with one annotated example clip showing the input
   video window and the predicted graph's target time.
3. Metrics chart: micro-F1 / macro-F1 / exact-match side by side, since they don't always
   agree on which variant is best (see README discussion).

Source: `outputs/feasibility/MISS_phase3_fusion/{summary,qualitative_take3_1}.csv`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "outputs" / "feasibility" / "MISS_phase3_fusion"
BAR_OUT = ROOT / "docs" / "phase3_fusion_2026-11-18.png"
TIMELINE_OUT = ROOT / "docs" / "phase3_fusion_timeline_2026-11-18.png"
METRICS_OUT = ROOT / "docs" / "phase3_fusion_metrics_2026-11-18.png"

# The 3 variants the discussion is actually about: single-encoder-for-everything (x2) vs. the
# proposed exo/ego dual-encoder split. exo_vjepa2 / ego_surgmotion (single-view-only) are weaker
# reference ablations, reported in the README table but left out of this headline chart.
LABELS = {
    "all_vjepa2": "all cameras,\nV-JEPA2",
    "all_surgmotion": "all cameras,\nSurgMotion",
    "dual_fusion": "exo=V-JEPA2\n+ ego=SurgMotion\n(proposed fusion)",
}
ORDER = list(LABELS)
COLORS = ["#90a4ae", "#64b5f6", "#2e7d32"]
HORIZONS = [0, 1, 4]
HORIZON_LABELS = {
    0: "now: graph AT t\n(nowcast, t = last input frame)",
    1: "+1s: graph 1s AFTER t\n(future)",
    4: "+4s: graph 4s AFTER t\n(future)",
}


def bar_chart(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=True)
    positions = [0, 0.8, 1.6]
    for ax, h in zip(axes, HORIZONS):
        sub = summary[summary.horizon_s == h].set_index("method").loc[ORDER]
        err = [sub.micro_f1 - sub.micro_f1_ci_low, sub.micro_f1_ci_high - sub.micro_f1]
        bars = ax.bar(positions, sub.micro_f1, width=0.6, yerr=err, capsize=4, color=COLORS, edgecolor="#333", linewidth=0.8)
        for bar, value in zip(bars, sub.micro_f1):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", fontsize=9)
        ax.set_xticks(positions)
        ax.set_xticklabels([LABELS[m] for m in ORDER], fontsize=8)
        ax.set_xlim(-0.55, 2.15)
        ax.set_title(HORIZON_LABELS[h], fontsize=9.5)
        ax.set_ylim(0.70, 0.95)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("scene-graph triplet micro-F1")
    fig.suptitle(
        "MISS phase 3 (\"Microscope-assisted disc removal\"), 6 takes, leave-one-take-out\n"
        "all-V-JEPA2 vs. all-SurgMotion vs. dual-encoder (exo=V-JEPA2 / ego=SurgMotion) fusion -- test run 2026-11-18",
        fontsize=11.5,
    )
    fig.text(
        0.5,
        0.005,
        "Error bars: 95% take-bootstrap CI (2,000 resamples). scripts/evaluate_phase3_fusion.py",
        ha="center",
        fontsize=8,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    BAR_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(BAR_OUT, dpi=130)
    print("saved", BAR_OUT)


def timeline(qual: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    for ax, h in zip(axes, HORIZONS):
        sub = qual[qual.horizon_s == h]
        for method, color in zip(ORDER, COLORS):
            row = sub[sub.method == method].sort_values("target_time_s")
            ax.plot(row.target_time_s, row.sample_f1, label=LABELS[method].replace("\n", " "), color=color, lw=1.6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("sample F1")
        ax.set_title(f"predicted-graph time = {HORIZON_LABELS[h].splitlines()[0]}", fontsize=9.5, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time into MISS/3/take/1 (s) -- x-position = time the predicted scene graph is FOR")
    axes[0].legend(loc="lower left", fontsize=8, ncol=3)

    # Annotate one example clip on the "now" panel to make input-window vs. target-time concrete.
    example = qual[(qual.method == "dual_fusion") & (qual.horizon_s == 0)].sort_values("target_time_s").iloc[10]
    axes[0].annotate(
        f"e.g. input video [{example.input_start_s:.0f}s - {example.input_end_s:.0f}s] (= x_exo, x_ego)\n"
        f"-> predicted graph AT {example.target_time_s:.0f}s (same as input end, nowcast)",
        xy=(example.target_time_s, example.sample_f1),
        xytext=(example.target_time_s + 8, 0.25),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#333"),
    )
    fig.suptitle(
        "MISS/3/take/1 (the OR-Action Fig. 3 take): per-clip scene-graph match quality over time, "
        "held-out fold -- test run 2026-11-18",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    TIMELINE_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(TIMELINE_OUT, dpi=130)
    print("saved", TIMELINE_OUT)


def metrics_chart(summary: pd.DataFrame) -> None:
    """micro-F1 / macro-F1 / exact-match grouped by metric, 3 methods per group, one panel per horizon."""
    metrics = ["micro_f1", "macro_f1", "exact_match"]
    metric_labels = ["micro-F1\n(all triplets)", "macro-F1\n(per-triplet-class avg)", "exact graph\nmatch"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), sharey=True)
    group_w, bar_w = 0.75, 0.22
    for ax, h in zip(axes, HORIZONS):
        sub = summary[summary.horizon_s == h].set_index("method").loc[ORDER]
        for i, method in enumerate(ORDER):
            offsets = (i - 1) * bar_w
            positions = [g * group_w + offsets for g in range(len(metrics))]
            values = sub.loc[method, metrics].to_numpy(dtype=float)
            bars = ax.bar(positions, values, width=bar_w * 0.95, color=COLORS[i], edgecolor="#333", linewidth=0.7)
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", fontsize=6.8, rotation=90)
        ax.set_xticks([g * group_w for g in range(len(metrics))])
        ax.set_xticklabels(metric_labels, fontsize=8)
        ax.set_title(HORIZON_LABELS[h], fontsize=9.5)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(
        handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec="#333") for c in COLORS],
        labels=[LABELS[m].replace("\n", " ") for m in ORDER],
        loc="lower left",
        fontsize=7.5,
    )
    fig.suptitle(
        "MISS phase 3, 6 takes, leave-one-take-out -- micro-F1 does not always agree with macro-F1 / exact-match\n"
        "all-V-JEPA2 vs. all-SurgMotion vs. dual-encoder (exo=V-JEPA2 / ego=SurgMotion) fusion -- test run 2026-11-18",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.86))
    METRICS_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(METRICS_OUT, dpi=130)
    print("saved", METRICS_OUT)


def main() -> None:
    summary = pd.read_csv(RESULT_DIR / "summary.csv")
    bar_chart(summary)
    metrics_chart(summary)
    timeline(pd.read_csv(RESULT_DIR / "qualitative_take3_1.csv"))


if __name__ == "__main__":
    main()

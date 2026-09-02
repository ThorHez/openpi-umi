#!/usr/bin/env python3
"""Generate the publication figure for the 12-frame teacher-memory ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURVE = ROOT / "evaluation/shellgame/teacher_memory_necessity_12f_260826/validation_curve.csv"
DEFAULT_OUTPUT_DIR = ROOT / "docs/figures"

STATE_ONLY = "GT state only"
WITH_TEACHER = "+ teacher memory"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, default=DEFAULT_CURVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="teacher_memory_necessity_ablation_12f_260826")
    return parser.parse_args()


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    steps = np.asarray([int(row["step"]) for row in rows])
    state_only = 100.0 * np.asarray([float(row["state_only_stage_accuracy"]) for row in rows])
    with_teacher = 100.0 * np.asarray([float(row["state_plus_teacher_stage_accuracy"]) for row in rows])
    return steps, state_only, with_teacher


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def plot(curve_path: Path, output_dir: Path, stem: str) -> list[Path]:
    configure_style()
    steps, state_only_curve, teacher_curve = load_curve(curve_path)
    chance = 100.0 / 3.0

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.85), constrained_layout=True)
    ax_curve, ax_stage = axes

    ax_curve.plot(
        steps,
        state_only_curve,
        color=BLUE,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.2,
        label=STATE_ONLY,
    )
    ax_curve.plot(
        steps,
        teacher_curve,
        color=ORANGE,
        marker="s",
        markerfacecolor=ORANGE,
        label=WITH_TEACHER,
    )
    ax_curve.axhline(chance, color=GRAY, linestyle=(0, (3, 2)), linewidth=1.0, zorder=0)
    ax_curve.text(990, chance + 1.2, "chance", color=GRAY, ha="right", va="bottom", fontsize=7.5)
    ax_curve.set_title("(a) Validation learning curve", loc="left", fontweight="bold")
    ax_curve.set_xlabel("Training step")
    ax_curve.set_ylabel("Mean stage accuracy (%)")
    ax_curve.set_xlim(80, 1020)
    ax_curve.set_ylim(25, 75)
    ax_curve.set_xticks([100, 300, 500, 700, 900, 1000])
    ax_curve.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax_curve.spines[["top", "right"]].set_visible(False)

    stage_labels = ["Update 1", "Update 2", "Update 3", "Mean"]
    state_only_final = np.asarray([65.0, 36.25, 33.75, 45.0])
    teacher_final = np.asarray([81.6667, 74.1667, 50.0, 68.6111])
    x = np.arange(len(stage_labels))
    width = 0.34
    bars_a = ax_stage.bar(
        x - width / 2,
        state_only_final,
        width,
        color="white",
        edgecolor=BLUE,
        linewidth=1.3,
        hatch="///",
        label=STATE_ONLY,
        zorder=2,
    )
    bars_b = ax_stage.bar(
        x + width / 2,
        teacher_final,
        width,
        color=ORANGE,
        edgecolor=ORANGE,
        linewidth=1.0,
        label=WITH_TEACHER,
        zorder=2,
    )
    ax_stage.axhline(chance, color=GRAY, linestyle=(0, (3, 2)), linewidth=1.0, zorder=1)
    ax_stage.set_title("(b) Final stage-wise accuracy", loc="left", fontweight="bold")
    ax_stage.set_xlabel("Recurrent update")
    ax_stage.set_ylabel("Accuracy (%)")
    ax_stage.set_xticks(x, stage_labels)
    # Bars start at zero so their visual magnitude is not exaggerated.
    ax_stage.set_ylim(0, 95)
    ax_stage.set_yticks([0, 20, 40, 60, 80])
    ax_stage.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8, zorder=0)
    ax_stage.spines[["top", "right"]].set_visible(False)

    for bar_a, bar_b, improvement in zip(bars_a, bars_b, teacher_final - state_only_final, strict=True):
        for bar in (bar_a, bar_b):
            value = bar.get_height()
            ax_stage.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )
        ax_stage.text(
            bar_b.get_x() + bar_b.get_width() / 2,
            min(bar_b.get_height() + 8.0, 89.0),
            f"+{improvement:.1f} pp",
            color=ORANGE,
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
        )

    handles, labels = ax_curve.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        frameon=False,
        ncol=2,
        handlelength=2.5,
        columnspacing=1.8,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png")]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    outputs = plot(args.curve.resolve(), args.output_dir.resolve(), args.stem)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()

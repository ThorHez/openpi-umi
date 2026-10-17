"""Summarize ablation result JSON files.

The script expects one JSON file per ablation mode in a result directory. It
writes human-readable summaries plus CSV/JSON artifacts back into that directory.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path(
    "/data1/hzl_workspace_for_pi/openpi-umi/ablation_results/"
    "pi0_mem_compress_umi_wbcd_history_light_v1_260605_59999_val_0608"
)
SUMMARY_NAMES = {
    "summary.json",
    "summary.txt",
    "summary.csv",
    "summary.md",
    "per_episode_summary.csv",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_result_json(path: Path) -> bool:
    if path.name in SUMMARY_NAMES:
        return False
    if path.name.startswith("ablation_summary"):
        return False
    return path.suffix == ".json"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def fmt(value: Any, precision: int = 6) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{precision}g}"


def pct_delta(value: Any, reference: Any) -> float | None:
    value_float = safe_float(value)
    reference_float = safe_float(reference)
    if value_float is None or reference_float in (None, 0.0):
        return None
    return (value_float - reference_float) / reference_float * 100.0


def abs_delta(value: Any, reference: Any) -> float | None:
    value_float = safe_float(value)
    reference_float = safe_float(reference)
    if value_float is None or reference_float is None:
        return None
    return value_float - reference_float


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        if not is_result_json(path):
            continue
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        data = dict(data)
        data.setdefault("mode", path.stem)
        data["_source_file"] = path.name
        results.append(data)
    return sorted(results, key=lambda row: (row.get("mode") != "normal", str(row.get("mode"))))


def total_frames(result: dict[str, Any]) -> int | None:
    per_episode = result.get("per_episode")
    if not isinstance(per_episode, list):
        return None
    frames = [episode.get("num_frames") for episode in per_episode if isinstance(episode, dict)]
    if not frames:
        return None
    return int(sum(frame for frame in frames if isinstance(frame, int | float)))


def build_summary_rows(results: list[dict[str, Any]], reference_mode: str) -> list[dict[str, Any]]:
    reference = next((row for row in results if row.get("mode") == reference_mode), None)
    rows = []
    for result in results:
        gate_stats = result.get("gate_stats") if isinstance(result.get("gate_stats"), dict) else {}
        pred_delta = (
            result.get("prediction_delta_vs_normal")
            if isinstance(result.get("prediction_delta_vs_normal"), dict)
            else {}
        )
        row = {
            "mode": result.get("mode"),
            "num_episodes": result.get("num_episodes"),
            "num_frames": total_frames(result),
            "overall_mse": result.get("overall_mse"),
            "overall_rmse": math.sqrt(result["overall_mse"]) if safe_float(result.get("overall_mse")) is not None else None,
            "overall_mae": result.get("overall_mae"),
            "first_step_mse": result.get("first_step_mse"),
            "first_step_mae": result.get("first_step_mae"),
            "gate_sigmoid_mean": gate_stats.get("sigmoid_mean"),
            "pred_delta_rmse": pred_delta.get("pred_delta_rmse"),
            "pred_delta_mae": pred_delta.get("pred_delta_mae"),
            "fraction_improved": pred_delta.get("fraction_improved"),
            "gt_mse_delta_mean": pred_delta.get("gt_mse_delta_mean"),
            "source_file": result.get("_source_file"),
        }
        if reference is not None:
            row["overall_mse_delta_vs_ref"] = abs_delta(result.get("overall_mse"), reference.get("overall_mse"))
            row["overall_mse_pct_vs_ref"] = pct_delta(result.get("overall_mse"), reference.get("overall_mse"))
            row["overall_mae_delta_vs_ref"] = abs_delta(result.get("overall_mae"), reference.get("overall_mae"))
            row["overall_mae_pct_vs_ref"] = pct_delta(result.get("overall_mae"), reference.get("overall_mae"))
            row["first_step_mse_delta_vs_ref"] = abs_delta(result.get("first_step_mse"), reference.get("first_step_mse"))
            row["first_step_mse_pct_vs_ref"] = pct_delta(result.get("first_step_mse"), reference.get("first_step_mse"))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) if key != "mode" else str(row.get(key)) for _, key in columns) + " |")
    return "\n".join(lines)


def text_table(rows: list[dict[str, Any]], columns: list[tuple[str, str, int]]) -> str:
    header = " ".join(title.ljust(width) for title, _, width in columns)
    separator = "-" * len(header)
    body = []
    for row in rows:
        cells = []
        for _, key, width in columns:
            value = str(row.get(key)) if key == "mode" else fmt(row.get(key))
            cells.append(value.ljust(width))
        body.append(" ".join(cells))
    return "\n".join([header, separator, *body])


def build_per_episode_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        episodes = result.get("per_episode")
        if not isinstance(episodes, list):
            continue
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            rows.append(
                {
                    "mode": result.get("mode"),
                    "episode_idx": episode.get("episode_idx"),
                    "num_frames": episode.get("num_frames"),
                    "overall_mse": episode.get("overall_mse"),
                    "overall_mae": episode.get("overall_mae"),
                    "first_step_mse": episode.get("first_step_mse"),
                    "first_step_mae": episode.get("first_step_mae"),
                }
            )
    return rows


def write_reports(results_dir: Path, rows: list[dict[str, Any]], per_episode_rows: list[dict[str, Any]]) -> None:
    summary_columns = [
        ("Mode", "mode", 20),
        ("Episodes", "num_episodes", 9),
        ("Frames", "num_frames", 8),
        ("MSE", "overall_mse", 12),
        ("RMSE", "overall_rmse", 12),
        ("MAE", "overall_mae", 12),
        ("MSE_vs_ref_%", "overall_mse_pct_vs_ref", 13),
        ("FirstMSE", "first_step_mse", 12),
        ("PredDeltaRMSE", "pred_delta_rmse", 14),
        ("FracImproved", "fraction_improved", 13),
        ("GateMean", "gate_sigmoid_mean", 10),
    ]
    text = "\n".join(
        [
            "Ablation Summary",
            "=" * 80,
            text_table(rows, summary_columns),
            "",
            "Notes:",
            "- MSE_vs_ref_% is relative to the reference mode, default normal; lower MSE/MAE is better.",
            "- PredDeltaRMSE and FracImproved come from prediction_delta_vs_normal when present.",
            "",
        ]
    )
    (results_dir / "summary.txt").write_text(text, encoding="utf-8")

    markdown_columns = [
        ("Mode", "mode"),
        ("Episodes", "num_episodes"),
        ("Frames", "num_frames"),
        ("MSE", "overall_mse"),
        ("RMSE", "overall_rmse"),
        ("MAE", "overall_mae"),
        ("MSE vs ref %", "overall_mse_pct_vs_ref"),
        ("First-step MSE", "first_step_mse"),
        ("Pred delta RMSE", "pred_delta_rmse"),
        ("Fraction improved", "fraction_improved"),
        ("Gate mean", "gate_sigmoid_mean"),
    ]
    markdown = "\n".join(
        [
            "# Ablation Summary",
            "",
            markdown_table(rows, markdown_columns),
            "",
            "Lower MSE/MAE is better. `MSE vs ref %` is relative to `normal` by default.",
            "",
        ]
    )
    (results_dir / "summary.md").write_text(markdown, encoding="utf-8")

    write_csv(results_dir / "summary.csv", rows)
    write_csv(results_dir / "per_episode_summary.csv", per_episode_rows)
    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, allow_nan=True)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation experiment JSON results.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing one JSON file per ablation mode. Defaults to {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--reference-mode",
        default="normal",
        help="Mode used as the reference for delta columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results directory does not exist: {results_dir}")

    results = load_results(results_dir)
    if not results:
        raise RuntimeError(f"No ablation result JSON files found in {results_dir}")

    rows = build_summary_rows(results, args.reference_mode)
    per_episode_rows = build_per_episode_rows(results)
    write_reports(results_dir, rows, per_episode_rows)

    print(f"Loaded {len(results)} modes from {results_dir}")
    print(f"Wrote {results_dir / 'summary.txt'}")
    print(f"Wrote {results_dir / 'summary.md'}")
    print(f"Wrote {results_dir / 'summary.csv'}")
    print(f"Wrote {results_dir / 'summary.json'}")
    print(f"Wrote {results_dir / 'per_episode_summary.csv'}")


if __name__ == "__main__":
    main()

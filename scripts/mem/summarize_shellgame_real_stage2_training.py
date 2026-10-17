#!/usr/bin/env python3
"""Summarize Stage-2 loss and trainable-interface gradients from its log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

TRAIN_PATTERN = re.compile(r"^Step (\d+): (.+)$")
EVAL_PATTERN = re.compile(r"^Step (\d+) \[eval\]: (.+)$")
KEY_VALUE_PATTERN = re.compile(r"([A-Za-z0-9_./-]+)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
GRADIENT_KEYS = (
    "grad/action_expert_l2",
    "grad/action_projection_l2",
    "grad/semantic_action_cross_attn_l2",
    "grad/semantic_query_resampler_l2",
)


def parse_metrics(text: str) -> dict[str, float]:
    return {key: float(value) for key, value in KEY_VALUE_PATTERN.findall(text)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows: list[tuple[int, dict[str, float]]] = []
    eval_rows: list[tuple[int, dict[str, float]]] = []
    for raw_line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.lstrip("\r")
        if match := EVAL_PATTERN.match(line):
            eval_rows.append((int(match.group(1)), parse_metrics(match.group(2))))
        elif match := TRAIN_PATTERN.match(line):
            train_rows.append((int(match.group(1)), parse_metrics(match.group(2))))
    if not train_rows or not eval_rows:
        raise RuntimeError(f"Missing train/eval rows in {args.log}")

    val_losses = [(step, metrics["val/action_loss"]) for step, metrics in eval_rows]
    best_val_step, best_val_loss = min(val_losses, key=lambda item: item[1])
    gradient_summary = {}
    for key in GRADIENT_KEYS:
        values = np.asarray([metrics[key] for _, metrics in train_rows if key in metrics])
        gradient_summary[key] = {
            "samples": int(values.size),
            "min": float(values.min()),
            "median": float(np.median(values)),
            "max": float(values.max()),
            "nonzero_fraction": float(np.mean(values > 0)),
        }

    payload = {
        "log": str(args.log.resolve()),
        "train_logged_steps": len(train_rows),
        "first_train_step": train_rows[0][0],
        "last_train_step": train_rows[-1][0],
        "first_action_loss": train_rows[0][1]["action_loss"],
        "last_action_loss": train_rows[-1][1]["action_loss"],
        "eval_logged_steps": len(eval_rows),
        "first_val_step": val_losses[0][0],
        "first_val_action_loss": val_losses[0][1],
        "last_val_step": val_losses[-1][0],
        "last_val_action_loss": val_losses[-1][1],
        "best_val_step": best_val_step,
        "best_val_action_loss": best_val_loss,
        "gradients": gradient_summary,
        "all_required_gradients_nonzero": all(row["nonzero_fraction"] == 1.0 for row in gradient_summary.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

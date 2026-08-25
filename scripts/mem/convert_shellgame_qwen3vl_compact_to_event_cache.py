# ruff: noqa: E402
"""Convert compact Qwen3-VL generations to the recurrent-memory event cache.

The compact Qwen model predicts camera-relative cup names.  This utility
performs the audited camera-to-world conversion and emits the small cache
contract consumed by ``eval_qwenvl_event_recurrent_memory.py``.  It never
copies video frames and never uses expected labels as predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from openpi.tasks.shellgame.qwenvl_event_adapter import normalize_cup_entity


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _world_prediction(row: dict[str, Any]) -> tuple[str, Any] | None:
    if not bool(row.get("valid", False)) or not isinstance(row.get("prediction"), dict):
        return None
    sample_type = str(row["sample_type"])
    prediction = row["prediction"]
    if sample_type == "reveal" and set(prediction) == {"screen_cup"}:
        return "reveal", normalize_cup_entity(str(prediction["screen_cup"]))
    if sample_type == "swap" and set(prediction) == {"screen_pair"}:
        raw_pair = prediction["screen_pair"]
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            return None
        pair = tuple(normalize_cup_entity(str(value)) for value in raw_pair)
        canonical = sorted(pair, key=("left", "middle", "right").index)
        return f"swap_{int(row['event_index'])}", canonical
    return None


def convert_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        result = _world_prediction(row)
        if result is None:
            continue
        query_key, prediction = result
        episode = int(row["episode_index"])
        key = (episode, query_key)
        if key in seen:
            raise ValueError(f"Duplicate compact generation for episode/query {key}")
        seen.add(key)
        converted.append(
            {
                "episode_index": episode,
                "query_key": query_key,
                "prediction": prediction,
                "schema_valid": True,
                "adapter_valid": True,
                "source_sample_type": str(row["sample_type"]),
                "source_frame_indices": row.get("frame_indices"),
            }
        )
    converted.sort(key=lambda row: (int(row["episode_index"]), str(row["query_key"])))
    return converted


def main() -> None:
    args = _parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    converted = convert_records(rows)
    episodes = sorted({int(row["episode_index"]) for row in converted})
    expected = {(episode, query) for episode in episodes for query in ("reveal", "swap_0", "swap_1", "swap_2")}
    actual = {(int(row["episode_index"]), str(row["query_key"])) for row in converted}
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"Incomplete compact event sequences; first missing keys: {missing[:8]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"converted episodes={len(episodes)} events={len(converted)} output={args.output}")


if __name__ == "__main__":
    main()

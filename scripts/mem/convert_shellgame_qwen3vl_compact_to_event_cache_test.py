# ruff: noqa: E402

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mem.convert_shellgame_qwen3vl_compact_to_event_cache import convert_records


def test_compact_screen_coordinates_convert_without_using_expected_labels():
    rows = [
        {
            "episode_index": 7,
            "sample_type": "reveal",
            "event_index": None,
            "valid": True,
            "prediction": {"screen_cup": "screen_right_cup"},
            "expected": {"screen_cup": "screen_left_cup"},
        },
        {
            "episode_index": 7,
            "sample_type": "swap",
            "event_index": 0,
            "valid": True,
            "prediction": {"screen_pair": ["screen_left_cup", "screen_middle_cup"]},
            "expected": {"screen_pair": ["screen_middle_cup", "screen_right_cup"]},
        },
    ]
    converted = convert_records(rows)
    assert converted[0]["prediction"] == "left"
    assert converted[1]["prediction"] == ["middle", "right"]


def test_invalid_and_non_state_rows_are_ignored():
    rows = [
        {
            "episode_index": 1,
            "sample_type": "no_event",
            "valid": True,
            "prediction": {"event": "no_event"},
        },
        {
            "episode_index": 1,
            "sample_type": "swap",
            "event_index": 0,
            "valid": False,
            "prediction": None,
        },
    ]
    assert convert_records(rows) == []

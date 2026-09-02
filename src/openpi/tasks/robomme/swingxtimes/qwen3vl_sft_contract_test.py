import pytest

from openpi.tasks.robomme.swingxtimes.qwen3vl_sft_contract import compact_response
from openpi.tasks.robomme.swingxtimes.qwen3vl_sft_contract import validate_compact_response


def test_progress_contract() -> None:
    right = compact_response(
        "causal_prefix", event="right_arrival", right_count=2, left_count=1, target_round_trips=3
    )
    assert validate_compact_response(right, target_round_trips=3)["completed_round_trips"] == 1
    final = compact_response(
        "causal_prefix", event="left_arrival", right_count=3, left_count=3, target_round_trips=3
    )
    assert validate_compact_response(final, target_round_trips=3)["ready_to_stop"] is True


def test_rejection_contract() -> None:
    assert validate_compact_response(compact_response("no_event")) == {
        "event": "no_completed_arrival"
    }
    assert validate_compact_response(compact_response("local_only")) == {
        "event": "insufficient_history"
    }


def test_rejects_invalid_count_order() -> None:
    with pytest.raises(ValueError, match="must lead"):
        compact_response(
            "causal_prefix", event="right_arrival", right_count=1, left_count=1, target_round_trips=2
        )


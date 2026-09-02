import pytest

from openpi.tasks.robomme.pickxtimes.qwen3vl_sft_contract import compact_response
from openpi.tasks.robomme.pickxtimes.qwen3vl_sft_contract import validate_compact_response


def test_pick_place_press_states() -> None:
    pick = compact_response(
        "causal_prefix", event="pick_complete", completed_count=1, required_count=3
    )
    assert validate_compact_response(pick, required_count=3)["holding"] is True
    place = compact_response(
        "causal_prefix", event="place_complete", completed_count=3, required_count=3
    )
    assert validate_compact_response(place, required_count=3)["ready_to_press"] is True
    press = compact_response(
        "causal_prefix", event="press_complete", completed_count=3, required_count=3
    )
    assert validate_compact_response(press, required_count=3)["done"] is True


def test_rejection_states() -> None:
    assert validate_compact_response(compact_response("no_event")) == {
        "event": "no_completed_event"
    }
    assert validate_compact_response(compact_response("local_only")) == {
        "event": "insufficient_history"
    }


def test_rejects_early_press() -> None:
    with pytest.raises(ValueError, match="only after all placements"):
        compact_response(
            "causal_prefix", event="press_complete", completed_count=1, required_count=2
        )


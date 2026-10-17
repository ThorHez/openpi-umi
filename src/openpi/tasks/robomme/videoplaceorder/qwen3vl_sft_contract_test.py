import pytest

from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import cell_from_xy
from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import compact_response
from openpi.tasks.robomme.videoplaceorder.qwen3vl_sft_contract import validate_compact_response


def test_grounded_contract() -> None:
    text = compact_response(
        "full_demo", target_color="green", ordinal=3, target_cell="r3_c4"
    )
    assert validate_compact_response(text) == {
        "event": "ordinal_target_grounded",
        "target_color": "green",
        "ordinal": 3,
        "target_cell": "r3_c4",
    }


def test_insufficient_contract() -> None:
    assert validate_compact_response(compact_response("truncated_demo")) == {
        "event": "insufficient_evidence"
    }


def test_cell_and_invalid_ordinal() -> None:
    assert cell_from_xy(101, 168) == "r5_c3"
    with pytest.raises(ValueError, match="Invalid grounded target"):
        compact_response("full_demo", target_color="blue", ordinal=5, target_cell="r2_c2")


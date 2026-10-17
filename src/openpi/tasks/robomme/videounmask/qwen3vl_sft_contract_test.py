import pytest

from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import cell_from_yx
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import compact_response
from openpi.tasks.robomme.videounmask.qwen3vl_sft_contract import validate_compact_response


def test_grounded_and_masked_contract() -> None:
    grounded = compact_response("paired_memory", "red", "r2_c5")
    assert validate_compact_response(grounded) == {
        "event": "target_covered",
        "target_color": "red",
        "target_cell": "r2_c5",
    }
    assert validate_compact_response(compact_response("masked_only")) == {
        "event": "insufficient_evidence"
    }


def test_cell_is_clipped_to_image_grid() -> None:
    assert cell_from_yx(95, 165) == "r2_c5"
    assert cell_from_yx(-1, 999) == "r0_c7"


def test_rejects_noncanonical_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported compact response"):
        validate_compact_response('{"target_cell":"r2_c5"}')

import numpy as np
import pytest

from openpi.tasks.shellgame import v10_action_weight_transplant as transplant


def _trees():
    current = {
        "PaliGemma": {
            "llm": {
                "layers": {"attn_1": {"kernel": np.zeros((2, 3))}},
                "prefix": {"kernel": np.zeros((2, 3))},
            }
        },
        "action_in_proj": {"kernel": np.zeros((3, 4))},
        "SemanticMemoryActionConditioner": {"kernel": np.zeros((4, 5))},
    }
    v10 = {
        "PaliGemma": {
            "llm": {
                "layers": {"attn_1": {"kernel": np.ones((2, 3))}},
                "prefix": {"kernel": np.ones((2, 3))},
            }
        },
        "action_in_proj": {"kernel": np.ones((3, 4))},
        "SemanticMemoryActionConditioner": {"kernel": np.ones((4, 5))},
        "OldMemory": {"kernel": np.ones((1,))},
    }
    return current, v10


def test_transplant_replaces_only_action_branch():
    current, v10 = _trees()
    merged, report = transplant.transplant_v10_action_params(current, v10)
    np.testing.assert_array_equal(merged["PaliGemma"]["llm"]["layers"]["attn_1"]["kernel"], 1)
    np.testing.assert_array_equal(merged["action_in_proj"]["kernel"], 1)
    np.testing.assert_array_equal(merged["PaliGemma"]["llm"]["prefix"]["kernel"], 0)
    np.testing.assert_array_equal(merged["SemanticMemoryActionConditioner"]["kernel"], 0)
    assert "OldMemory" not in merged
    assert report.selected_leaves == 2
    assert report.action_expert_leaves == 1
    assert report.projection_leaves == 1


def test_transplant_rejects_shape_mismatch():
    current, v10 = _trees()
    v10["action_in_proj"]["kernel"] = np.ones((3, 5))
    with pytest.raises(ValueError, match="incompatible shapes"):
        transplant.transplant_v10_action_params(current, v10)

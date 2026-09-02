import numpy as np
import pytest

from examples.shellgame import eval_absolute_eef_fixed_history_xy_before_z_isolated as isolated
from examples.shellgame import main as shell_main
from examples.shellgame import main_absolute_eef_fixed_history as fixed_eef
from examples.shellgame import v10_exact_parallel_semantic_adapter as exact


def test_blend_old_memory_condition_has_exact_endpoints():
    raw = np.asarray([[[1.0, 2.0]]], dtype=np.float32)
    conditioned = np.asarray([[[4.0, -2.0]]], dtype=np.float32)
    np.testing.assert_array_equal(
        exact.blend_old_memory_condition(raw, conditioned, 0.0), raw
    )
    np.testing.assert_array_equal(
        exact.blend_old_memory_condition(raw, conditioned, 1.0), conditioned
    )


def test_no_memory_mode_is_exposed_without_enabling_parallel_adapter():
    from examples.shellgame import serve_v10_exact_parallel_semantic_adapter_deterministic as server

    assert "v10_action_no_memory" in server.ADAPTER_MODES


def test_current_only_compat_input_repeats_live_frame_in_every_slot():
    history = [
        {
                "base": np.full((2, 2, 3), index, dtype=np.uint8),
                "wrist": np.full((2, 2, 3), index + 1, dtype=np.uint8),
                "eef_pos": np.zeros(3, dtype=np.float32),
                "eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
                "gripper_width": 0.04,
        }
        for index in range(60)
    ]
    args = shell_main.Args(
        num_frames=61,
        frame_stride=1,
        action_mode="raw7",
        action_dim=7,
        observation_position_frame="absolute",
        osc_input_type="absolute",
    )
    element = fixed_eef._current_only_compat_policy_input(  # noqa: SLF001
        history, np.zeros(3, dtype=np.float32), args=args
    )
    for index in range(61):
        np.testing.assert_array_equal(
            element[f"left_wrist_0_rgb_0_{index}"], history[-1]["wrist"]
        )
        np.testing.assert_array_equal(
            element[f"left_wrist_0_rgb_1_{index}"], history[-1]["base"]
        )


def test_gripper_contacted_cups_ignores_table_contacts():
    contacts = [
        {"body1": "table", "body2": "left_cup_root"},
        {"body1": "middle_cup_root", "body2": "gripper0_right_leftfinger"},
        {"body1": "gripper0_right_rightfinger", "body2": "right_cup_root"},
    ]
    assert shell_main._gripper_contacted_cups(contacts) == {"middle", "right"}  # noqa: SLF001


def test_episode_spec_shards_match_single_full_sequence():
    full = isolated._episode_specs(isolated.Args(num_trials=100, seed=260813))  # noqa: SLF001
    sharded = []
    for start in (0, 25, 50, 75):
        sharded.extend(
            isolated._episode_specs(  # noqa: SLF001
                isolated.Args(
                    num_trials=25,
                    episode_start_index=start,
                    seed=260813,
                )
            )
        )
    assert sharded == full


def test_merge_restores_every_v10_leaf_and_preserves_only_parallel_adapter():
    target = {
        "PaliGemma": {"kernel": np.zeros((2, 3), dtype=np.float32)},
        "ActionMemoryCrossAttention": {"gate": np.zeros((1,), dtype=np.float32)},
        "ParallelSemanticMemoryActionConditioner": {
            "gate_delta": np.zeros((1,), dtype=np.float32),
            "kernel": np.full((2, 2), 7.0, dtype=np.float32),
        },
    }
    v10 = {
        "PaliGemma": {"kernel": np.ones((2, 3), dtype=np.float32)},
        "ActionMemoryCrossAttention": {"gate": np.ones((1,), dtype=np.float32)},
    }

    merged, counts = exact.merge_exact_v10_with_fresh_parallel_adapter(target, v10)

    np.testing.assert_array_equal(merged["PaliGemma"]["kernel"], 1.0)
    np.testing.assert_array_equal(merged["ActionMemoryCrossAttention"]["gate"], 1.0)
    np.testing.assert_array_equal(
        merged["ParallelSemanticMemoryActionConditioner"]["kernel"], 7.0
    )
    assert counts == {"v10": 2, "parallel_adapter": 2}


def test_merge_rejects_any_non_adapter_contract_difference():
    target = {
        "PaliGemma": {"kernel": np.zeros((2, 3), dtype=np.float32)},
        "ParallelSemanticMemoryActionConditioner": {
            "gate_delta": np.zeros((1,), dtype=np.float32)
        },
    }
    with pytest.raises(ValueError, match="Exact V10 restore failed"):
        exact.merge_exact_v10_with_fresh_parallel_adapter(target, {})


def test_merge_normalizes_checkpoint_dtype_to_model_contract():
    target = {
        "PaliGemma": {"kernel": np.zeros((2,), dtype=np.float16)},
        "ParallelSemanticMemoryActionConditioner": {
            "gate_delta": np.zeros((1,), dtype=np.float32)
        },
    }
    source = {"PaliGemma": {"kernel": np.ones((2,), dtype=np.float32)}}
    merged, _ = exact.merge_exact_v10_with_fresh_parallel_adapter(target, source)
    assert merged["PaliGemma"]["kernel"].dtype == np.float16

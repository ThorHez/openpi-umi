import jax
import jax.numpy as jnp
import numpy as np

from openpi.tasks.robomme.pickxtimes import semantic_memory_event


def test_pickxtimes_tracker_teacher_window_shapes_and_padding():
    model = semantic_memory_event.PickXtimesSlidingWindowEventMemoryTracker(
        input_width=16,
        prompt_width=24,
        encoder_width=8,
        encoder_depth=1,
        encoder_heads=2,
        memory_width=8,
        memory_depth=1,
        memory_heads=2,
        num_memory_tokens=6,
        evidence_tokens_per_event=2,
    )
    patches = jax.random.normal(
        jax.random.key(0),
        (2, 7, semantic_memory_event.WINDOW_SIZE, 256, 16),
    )
    prompts = jax.random.normal(jax.random.key(1), (2, 5, 24))
    gripper = jnp.zeros((2, 7, semantic_memory_event.WINDOW_SIZE), dtype=jnp.bool_)
    prompt_mask = jnp.asarray([[True, True, True, False, False], [True, True, False, False, False]])
    sequence_positions = jnp.asarray([[0, 1, 2, 0], [0, 1, 0, 0]])
    sequence_mask = jnp.asarray([[True, True, True, False], [True, True, False, False]])
    variables = model.init(
        jax.random.key(2),
        patches,
        gripper,
        prompts,
        prompt_mask,
        sequence_positions,
        sequence_mask,
        causal_selection=False,
    )
    outputs = model.apply(
        variables,
        patches,
        gripper,
        prompts,
        prompt_mask,
        sequence_positions,
        sequence_mask,
        causal_selection=False,
    )

    assert outputs["event_logits"].shape == (2, 7)
    assert outputs["event_type_logits"].shape == (2, 7, 3)
    assert outputs["stage_memories"].shape == (2, 4, 6, 8)
    assert outputs["completed_count_logits"].shape == (2, 4, 6)
    assert outputs["goal_color_logits"].shape == (2, 3)
    assert outputs["goal_required_count_logits"].shape == (2, 5)
    np.testing.assert_allclose(outputs["stage_memories"][0, 3], outputs["stage_memories"][0, 2], atol=1e-6)
    np.testing.assert_allclose(outputs["stage_memories"][1, 2], outputs["stage_memories"][1, 1], atol=1e-6)


def test_pickxtimes_tracker_causal_hysteresis_selects_rising_edges():
    logits = jnp.asarray([[-2.0, 2.0, 2.0, -2.0, 2.0, -2.0]])
    triggers, active = semantic_memory_event.event_memory.causal_event_triggers(
        logits,
        high_threshold=0.8,
        low_threshold=-0.8,
    )

    np.testing.assert_array_equal(triggers, [[False, True, False, False, True, False]])
    np.testing.assert_array_equal(active, [False])


def test_gripper_fusion_zero_init_preserves_visual_logits():
    fusion = semantic_memory_event.PickXtimesGripperTypeFusion()
    visual = jnp.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    gripper = jnp.asarray(
        [
            [False, False, True, True, True, True, True, True, True, True],
            [True, True, False, False, False, False, False, False, False, False],
        ]
    )
    variables = fusion.init(jax.random.key(3), visual, gripper)
    np.testing.assert_array_equal(fusion.apply(variables, visual, gripper), visual)


def test_gripper_gate_fusion_zero_init_preserves_visual_logits():
    fusion = semantic_memory_event.PickXtimesGripperGateFusion()
    visual = jnp.asarray([1.0, -1.0])
    gripper = jnp.asarray(
        [
            [False, False, True, True, True, True, True, True, True, True],
            [True, True, False, False, False, False, False, False, False, False],
        ]
    )
    variables = fusion.init(jax.random.key(4), visual, gripper)
    np.testing.assert_array_equal(fusion.apply(variables, visual, gripper), visual)


def test_press_gate_fusion_zero_init_preserves_visual_logits():
    fusion = semantic_memory_event.PickXtimesPressGateFusion()
    visual = jnp.asarray([1.0, -1.0])
    semantic = jax.random.normal(jax.random.key(5), (2, 8))
    variables = fusion.init(jax.random.key(6), visual, semantic)
    np.testing.assert_array_equal(fusion.apply(variables, visual, semantic), visual)


def test_transition_grammar_aligns_edges_rejects_failed_grasp_and_adds_press():
    gripper = np.zeros(130, dtype=np.bool_)
    for frame, closed in ((20, True), (40, False), (50, True), (55, False), (65, True), (85, False), (110, True)):
        gripper[frame:] = closed
    event_logits = np.full(130 - semantic_memory_event.WINDOW_SIZE + 1, -2.0, dtype=np.float32)
    for start in (14, 35, 60, 80):
        event_logits[start] = 2.0
    event_logits[105] = 3.0
    event_type_logits = np.zeros((len(event_logits), semantic_memory_event.NUM_EVENT_CLASSES), dtype=np.float32)
    event_type_logits[:, semantic_memory_event.PICK_COMPLETE] = 1.0
    event_type_logits[105] = [-1.0, -1.0, 3.0]

    triggers, event_types = semantic_memory_event.transition_grammar_events(
        gripper,
        event_logits,
        event_type_logits,
        required_count=2,
    )

    assert triggers == [14, 35, 60, 80, 105]
    assert event_types == [
        semantic_memory_event.PICK_COMPLETE,
        semantic_memory_event.PLACE_COMPLETE,
        semantic_memory_event.PICK_COMPLETE,
        semantic_memory_event.PLACE_COMPLETE,
        semantic_memory_event.PRESS_COMPLETE,
    ]


def test_transition_grammar_can_use_goal_gated_press_head_without_type_gate():
    gripper = np.zeros(90, dtype=np.bool_)
    gripper[20:40] = True
    event_logits = np.full(90 - semantic_memory_event.WINDOW_SIZE + 1, -2.0, dtype=np.float32)
    event_logits[14] = 2.0
    event_logits[35] = 2.0
    press_logits = event_logits.copy()
    press_logits[70] = 3.0
    type_logits = np.zeros((len(event_logits), semantic_memory_event.NUM_EVENT_CLASSES), dtype=np.float32)
    type_logits[:, semantic_memory_event.PICK_COMPLETE] = 1.0

    triggers, event_types = semantic_memory_event.transition_grammar_events(
        gripper,
        event_logits,
        type_logits,
        press_logits,
        required_count=1,
        require_press_type=False,
    )

    assert triggers == [14, 35, 70]
    assert event_types == [
        semantic_memory_event.PICK_COMPLETE,
        semantic_memory_event.PLACE_COMPLETE,
        semantic_memory_event.PRESS_COMPLETE,
    ]

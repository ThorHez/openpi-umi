import jax
import jax.numpy as jnp

from openpi.tasks.robomme.anchor_conditioned_transition_memory import AnchorConditionedTransitionMemory


def test_anchor_conditioned_transition_memory_teacher_and_free_rollout_shapes():
    model = AnchorConditionedTransitionMemory(
        max_steps=3,
        frames=12,
        spatial_tokens=16,
        input_width=8,
        width=16,
        hidden_width=24,
        memory_tokens=12,
    )
    inputs = {
        "patch_tokens": jnp.zeros((2, 3, 12, 16, 8), dtype=jnp.float32),
        "sequence_mask": jnp.asarray([[True, True, False], [True, True, True]]),
        "task_ids": jnp.asarray([0, 2]),
        "goal_color_ids": jnp.asarray([[1, 0], [3, 0]]),
        "queried_ordinals": jnp.asarray([0, 2]),
        "num_regions": jnp.asarray([3, 4]),
        "anchor_yx": jnp.zeros((2, 4, 2), dtype=jnp.float32),
        "anchor_mask": jnp.asarray([[True, True, True, False], [True] * 4]),
        "teacher_previous_tables": jnp.zeros((2, 3, 7), dtype=jnp.int32),
    }
    params = model.init(
        jax.random.key(0),
        **inputs,
        teacher_force_mask=jnp.ones((2, 3), dtype=jnp.bool_),
        train=False,
    )
    teacher = model.apply(
        params,
        **inputs,
        teacher_force_mask=jnp.ones((2, 3), dtype=jnp.bool_),
        train=False,
    )
    free = model.apply(
        params,
        **inputs,
        teacher_force_mask=jnp.zeros((2, 3), dtype=jnp.bool_),
        train=False,
    )
    assert teacher["all_tables"].shape == (2, 4, 7, 5)
    assert free["all_tables"].shape == (2, 4, 7, 5)
    assert free["event_type_logits"].shape == (2, 3, 2, 3)
    assert free["committed_event_gates"].shape == (2, 3, 2, 3)
    assert free["write_region_logits"].shape == (2, 3, 2, 4)
    assert free["swap_pair_logits"].shape == (2, 3, 2, 6)
    # Padding must preserve the previous table.
    assert jnp.allclose(free["all_tables"][0, 3], free["all_tables"][0, 2])


def test_hard_event_commit_keeps_table_categorical():
    model = AnchorConditionedTransitionMemory(
        max_steps=2,
        frames=12,
        spatial_tokens=16,
        input_width=8,
        width=16,
        hidden_width=24,
        memory_tokens=12,
        hard_event_commit=True,
    )
    inputs = {
        "patch_tokens": jnp.zeros((1, 2, 12, 16, 8), dtype=jnp.float32),
        "sequence_mask": jnp.ones((1, 2), dtype=jnp.bool_),
        "task_ids": jnp.asarray([1]),
        "goal_color_ids": jnp.asarray([[1, 2]]),
        "queried_ordinals": jnp.asarray([0]),
        "num_regions": jnp.asarray([3]),
        "anchor_yx": jnp.zeros((1, 4, 2), dtype=jnp.float32),
        "anchor_mask": jnp.asarray([[True, True, True, False]]),
        "teacher_previous_tables": jnp.zeros((1, 2, 7), dtype=jnp.int32),
        "teacher_force_mask": jnp.zeros((1, 2), dtype=jnp.bool_),
    }
    variables = model.init(jax.random.key(1), **inputs, train=False)
    output = model.apply(variables, **inputs, train=False)
    tables = output["all_tables"]
    assert jnp.all((tables == 0.0) | (tables == 1.0))
    assert jnp.allclose(tables.sum(axis=-1), 1.0)


def test_decomposed_completion_and_phase_heads_drive_hard_commit():
    model = AnchorConditionedTransitionMemory(
        max_steps=2,
        frames=12,
        spatial_tokens=64,
        input_width=12,
        width=16,
        hidden_width=24,
        memory_tokens=12,
        hard_event_commit=True,
        event_head_mode="completion",
        use_auxiliary_heads=True,
        commit_threshold=0.7,
    )
    inputs = {
        "patch_tokens": jnp.zeros((1, 2, 12, 64, 12), dtype=jnp.float32),
        "sequence_mask": jnp.ones((1, 2), dtype=jnp.bool_),
        "task_ids": jnp.asarray([2]),
        "goal_color_ids": jnp.asarray([[1, 0]]),
        "queried_ordinals": jnp.asarray([1]),
        "num_regions": jnp.asarray([4]),
        "anchor_yx": jnp.zeros((1, 4, 2), dtype=jnp.float32),
        "anchor_mask": jnp.ones((1, 4), dtype=jnp.bool_),
        "teacher_previous_tables": jnp.zeros((1, 2, 7), dtype=jnp.int32),
        "teacher_force_mask": jnp.zeros((1, 2), dtype=jnp.bool_),
    }
    variables = model.init(jax.random.key(2), **inputs, train=False)
    output = model.apply(variables, **inputs, train=False)
    assert output["completion_logits"].shape == (1, 2, 2)
    assert output["event_kind_logits"].shape == (1, 2, 2, 2)
    assert output["phase_logits"].shape == (1, 2, 4)
    gates = output["committed_event_gates"]
    assert jnp.all((gates == 0.0) | (gates == 1.0))
    assert jnp.allclose(gates.sum(axis=-1), 1.0)
    assert jnp.all((output["all_tables"] == 0.0) | (output["all_tables"] == 1.0))

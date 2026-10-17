import jax
import jax.numpy as jnp

from openpi.tasks.robomme import unified_gt_teacher


def _inputs():
    return {
        "task_ids": jnp.asarray([0, 3], dtype=jnp.int32),
        "goal_color_ids": jnp.asarray([[1, 0], [3, 0]], dtype=jnp.int32),
        "required_counts": jnp.asarray([0, 2], dtype=jnp.int32),
        "queried_ordinals": jnp.asarray([0, 0], dtype=jnp.int32),
        "num_regions": jnp.asarray([3, 0], dtype=jnp.int32),
        "event_ids": jnp.asarray([[1, 2, 0], [4, 5, 6]], dtype=jnp.int32),
        "entity_ids": jnp.asarray([[1, 1, 0], [3, 3, 0]], dtype=jnp.int32),
        "region_a_ids": jnp.asarray([[2, 2, 0], [0, 0, 0]], dtype=jnp.int32),
        "region_b_ids": jnp.zeros((2, 3), dtype=jnp.int32),
        "step_mask": jnp.asarray([[True, True, False], [True, True, True]]),
    }


def test_unified_teacher_shapes_and_padding_preserves_memory():
    model = unified_gt_teacher.UnifiedRoboMMEGTTeacher(
        width=16,
        num_memory_tokens=8,
        memory_depth=1,
        memory_heads=4,
        readout_heads=4,
    )
    inputs = _inputs()
    variables = model.init(jax.random.key(0), **inputs)
    outputs = model.apply(variables, **inputs)
    assert outputs["all_memories"].shape == (2, 4, 8, 16)
    assert outputs["state_logits"].shape == (
        2,
        4,
        len(unified_gt_teacher.STATE_FIELDS),
        unified_gt_teacher.MAX_FIELD_CLASSES,
    )
    assert jnp.array_equal(outputs["update_mask"], inputs["step_mask"])
    assert jnp.allclose(outputs["all_memories"][0, 2], outputs["all_memories"][0, 3])


def test_rejection_event_is_an_exact_noop():
    model = unified_gt_teacher.UnifiedRoboMMEGTTeacher(
        width=16,
        num_memory_tokens=8,
        memory_depth=1,
        memory_heads=4,
        readout_heads=4,
    )
    inputs = _inputs()
    inputs["event_ids"] = inputs["event_ids"].at[0, 1].set(
        unified_gt_teacher.EVENTS.index("incomplete_event")
    )
    variables = model.init(jax.random.key(1), **inputs)
    outputs = model.apply(variables, **inputs)
    assert not bool(outputs["update_mask"][0, 1])
    assert jnp.allclose(outputs["all_memories"][0, 1], outputs["all_memories"][0, 2])


def test_invalid_tail_classes_are_masked():
    logits = jnp.zeros((2, len(unified_gt_teacher.STATE_FIELDS), 6))
    masked = unified_gt_teacher.mask_invalid_field_classes(logits)
    covered = unified_gt_teacher.STATE_FIELDS.index("covered")
    assert jnp.all(masked[:, covered, :2] == 0)
    assert jnp.all(masked[:, covered, 2:] < -1e8)


def test_teacher_loss_reports_strict_state_and_final_metrics():
    batch, states = 2, 3
    targets = jnp.zeros((batch, states, len(unified_gt_teacher.STATE_FIELDS)), dtype=jnp.int32)
    mask = jnp.zeros_like(targets, dtype=jnp.bool_)
    task = unified_gt_teacher.STATE_FIELDS.index("task")
    covered = unified_gt_teacher.STATE_FIELDS.index("covered")
    mask = mask.at[:, :, task].set(True)
    mask = mask.at[:, :, covered].set(True)
    logits = jnp.full(
        (batch, states, len(unified_gt_teacher.STATE_FIELDS), 6),
        -10.0,
    )
    logits = logits.at[..., 0].set(10.0)
    outputs = {
        "state_logits": unified_gt_teacher.mask_invalid_field_classes(logits),
        "all_memories": jnp.ones((batch, states, 4, 8)),
    }
    loss, metrics = unified_gt_teacher.compute_teacher_losses(outputs, targets, mask)
    assert float(loss) < 1e-6
    assert float(metrics["state_exact_accuracy"]) == 1.0
    assert float(metrics["sequence_exact_accuracy"]) == 1.0
    assert float(metrics["final_state_exact_accuracy"]) == 1.0


def test_gt_state_encoder_and_shared_readout_branch():
    model = unified_gt_teacher.UnifiedRoboMMEGTTeacher(
        width=16,
        num_memory_tokens=32,
        memory_depth=1,
        memory_heads=4,
        readout_heads=4,
    )
    inputs = _inputs()
    targets = jnp.zeros((2, 4, len(unified_gt_teacher.STATE_FIELDS)), dtype=jnp.int32)
    field_mask = jnp.zeros_like(targets, dtype=jnp.bool_)
    field_mask = field_mask.at[:, :, unified_gt_teacher.STATE_FIELDS.index("task")].set(True)
    variables = model.init(
        jax.random.key(2),
        **inputs,
        teacher_state_targets=targets,
        teacher_state_field_mask=field_mask,
    )
    outputs = model.apply(
        variables,
        **inputs,
        teacher_state_targets=targets,
        teacher_state_field_mask=field_mask,
    )
    assert outputs["gt_state_memories"].shape == (2, 4, 32, 16)
    assert outputs["gt_state_logits"].shape == (2, 4, len(unified_gt_teacher.STATE_FIELDS), 6)
    loss, metrics = unified_gt_teacher.compute_teacher_losses(outputs, targets, field_mask)
    assert jnp.isfinite(loss)
    assert "memory_alignment_loss" in metrics
    assert "canonical_state_exact_accuracy" in metrics

import jax
import jax.numpy as jnp

from openpi.tasks.robomme import unified_gt_teacher as teacher_contract
from openpi.tasks.robomme import unified_visual_student as student_lib


def test_unified_visual_student_shapes_and_masked_noop():
    model = student_lib.UnifiedVisualRecurrentStudent(
        max_steps=3,
        frames=2,
        spatial_tokens=4,
        input_width=16,
        width=16,
        num_memory_tokens=24,
        encoder_width=16,
        encoder_depth=1,
        encoder_heads=4,
        memory_depth=1,
        memory_heads=4,
        dtype_mm="float32",
    )
    patches = jax.random.normal(jax.random.key(1), (2, 3, 2, 4, 16))
    inputs = {
        "patch_tokens": patches,
        "task_ids": jnp.asarray([0, 3], dtype=jnp.int32),
        "goal_color_ids": jnp.asarray([[1, 0], [2, 0]], dtype=jnp.int32),
        "required_counts": jnp.asarray([0, 3], dtype=jnp.int32),
        "queried_ordinals": jnp.asarray([0, 0], dtype=jnp.int32),
        "num_regions": jnp.asarray([3, 4], dtype=jnp.int32),
        "step_mask": jnp.asarray([[True, True, False], [True, False, False]]),
    }
    variables = model.init(jax.random.key(0), **inputs)
    output = model.apply(variables, **inputs)
    assert output["all_memories"].shape == (2, 4, 24, 16)
    assert jnp.array_equal(output["event_memories"][0, 2], output["event_memories"][0, 1])
    assert jnp.array_equal(output["event_memories"][1, 1], output["event_memories"][1, 0])
    assert jnp.array_equal(output["event_memories"][1, 2], output["event_memories"][1, 1])


def test_memory_distillation_loss_prefers_identical_memory():
    teacher = jax.random.normal(jax.random.key(2), (2, 3, 24, 16))
    valid = jnp.asarray([[True, True, False], [True, True, True]])
    same, metrics = student_lib.memory_distillation_loss(teacher, teacher, valid)
    shifted, _ = student_lib.memory_distillation_loss(teacher + 0.5, teacher, valid)
    assert same < 1e-6
    assert metrics["memory_mse_loss"] == 0
    assert shifted > same


def test_semantic_token_weight_changes_loss():
    shape = (1, 1, 32, 8)
    teacher = jnp.zeros(shape)
    student = jnp.zeros(shape).at[:, :, : len(teacher_contract.STATE_FIELDS)].set(1.0)
    valid = jnp.ones((1, 1), dtype=jnp.bool_)
    low, _ = student_lib.memory_distillation_loss(
        student, teacher, valid, semantic_token_weight=1.0
    )
    high, _ = student_lib.memory_distillation_loss(
        student, teacher, valid, semantic_token_weight=8.0
    )
    assert high > low

import jax
import jax.numpy as jnp

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib
from openpi.tasks.robomme import unified_semantic_feedback_student as student_lib


def _run_shapes_and_masked_copy(*, straight_through_hard_feedback: bool):
    batch, steps = 2, 2
    fields = len(teacher_lib.STATE_FIELDS)
    model = student_lib.UnifiedSemanticFeedbackStudent(
        max_steps=steps,
        input_width=8,
        proprio_dim=3,
        width=8,
        encoder_width=8,
        encoder_depth=1,
        encoder_heads=2,
        straight_through_hard_feedback=straight_through_hard_feedback,
    )
    initial = jnp.zeros((batch, fields), dtype=jnp.int32)
    initial = initial.at[:, teacher_lib.STATE_FIELDS.index("task")].set(3)
    previous = jnp.repeat(initial[:, None], steps, axis=1)
    field_mask = jnp.ones((batch, steps, fields), dtype=jnp.bool_)
    sequence_mask = jnp.asarray([[True, False], [True, True]])
    inputs = {
        "patch_tokens": jnp.zeros((batch, steps, 12, 16, 8), dtype=jnp.float32),
        "proprio": jnp.zeros((batch, steps, 12, 3), dtype=jnp.float32),
        "sequence_mask": sequence_mask,
        "initial_state_targets": initial,
        "state_field_mask": field_mask,
        "teacher_previous_targets": previous,
        "teacher_force_mask": jnp.zeros((batch, steps), dtype=jnp.bool_),
    }
    variables = model.init(jax.random.key(0), **inputs, train=False)
    output = model.apply(variables, **inputs, train=False)
    assert output["state_logits"].shape == (
        batch,
        steps + 1,
        fields,
        teacher_lib.MAX_FIELD_CLASSES,
    )
    predictions = jnp.argmax(output["state_logits"], axis=-1)
    assert jnp.array_equal(predictions[:, 0], initial)
    # Zero-initialized semantic delta copies state; padded steps preserve the carry too.
    assert jnp.array_equal(predictions[:, 1], initial)
    assert jnp.array_equal(predictions[0, 2], initial[0])


def test_soft_semantic_feedback_student_shapes_and_masked_copy():
    _run_shapes_and_masked_copy(straight_through_hard_feedback=False)


def test_straight_through_hard_semantic_feedback_student_shapes_and_masked_copy():
    _run_shapes_and_masked_copy(straight_through_hard_feedback=True)

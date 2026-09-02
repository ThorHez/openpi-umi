import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.tasks.robomme.explicit_event_bottleneck_memory import ExplicitEventBottleneckMemory


def _inputs():
    batch, steps, frames, spatial, width = 2, 3, 4, 4, 12
    return {
        "patch_tokens": jnp.ones((batch, steps, frames, spatial, width), jnp.float32),
        "sequence_mask": jnp.asarray([[True, True, True], [True, True, False]]),
        "task_ids": jnp.asarray([0, 1], jnp.int32),
        "goal_color_ids": jnp.asarray([[1, 0], [2, 0]], jnp.int32),
        "queried_ordinals": jnp.asarray([0, 1], jnp.int32),
        "num_regions": jnp.asarray([3, 4], jnp.int32),
        "anchor_yx": jnp.asarray(
            [
                [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.0], [0.0, 0.0]],
                [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]],
            ],
            jnp.float32,
        ),
        "anchor_mask": jnp.asarray(
            [[True, True, True, False], [True, True, True, True]]
        ),
        "teacher_previous_tables": jnp.zeros((batch, steps, 7), jnp.int32),
        "teacher_force_mask": jnp.zeros((batch, steps), jnp.bool_),
    }


@pytest.mark.parametrize("temporal_encoder", ["pooled", "relational"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_factorial_variants_have_the_same_output_contract(
    temporal_encoder: str, deterministic: bool  # noqa: FBT001
):
    inputs = _inputs()
    model = ExplicitEventBottleneckMemory(
        max_steps=3,
        frames=4,
        spatial_tokens=4,
        input_width=12,
        width=16,
        hidden_width=32,
        memory_tokens=12,
        temporal_encoder=temporal_encoder,
        temporal_depth=1,
        temporal_heads=4,
        deterministic_updater=deterministic,
    )
    variables = model.init(jax.random.key(0), **inputs)
    output = model.apply(variables, **inputs)
    assert output["all_tables"].shape == (2, 4, 7, 5)
    assert output["all_memories"].shape == (2, 4, 12, 16)
    assert output["event_type_logits"].shape == (2, 3, 2, 3)
    assert output["event_bottleneck"].shape == (2, 3, 2, 3)


def test_deterministic_executor_has_categorical_forward_state():
    inputs = _inputs()
    model = ExplicitEventBottleneckMemory(
        max_steps=3,
        frames=4,
        spatial_tokens=4,
        input_width=12,
        width=16,
        hidden_width=32,
        memory_tokens=12,
        deterministic_updater=True,
    )
    variables = model.init(jax.random.key(1), **inputs)
    output = model.apply(variables, **inputs)
    tables = np.asarray(output["all_tables"])
    np.testing.assert_allclose(tables.sum(axis=-1), 1.0, atol=1e-6)
    assert np.all((tables == 0.0) | (tables == 1.0))


def test_relational_encoder_supports_causal_evidence_state():
    inputs = _inputs()
    model = ExplicitEventBottleneckMemory(
        max_steps=3,
        frames=4,
        spatial_tokens=4,
        input_width=12,
        width=16,
        hidden_width=32,
        memory_tokens=12,
        temporal_encoder="relational",
        temporal_depth=1,
        temporal_heads=4,
        deterministic_updater=True,
        causal_evidence_state=True,
    )
    variables = model.init(jax.random.key(2), **inputs)
    output = model.apply(variables, **inputs)
    assert jnp.isfinite(output["all_tables"]).all()


def test_optional_proprio_evidence_preserves_output_contract():
    inputs = _inputs()
    inputs["proprio_tokens"] = jnp.ones((2, 3, 4, 8), jnp.float32)
    model = ExplicitEventBottleneckMemory(
        max_steps=3,
        frames=4,
        spatial_tokens=4,
        input_width=12,
        width=16,
        hidden_width=32,
        memory_tokens=12,
        deterministic_updater=True,
        causal_evidence_state=True,
    )
    variables = model.init(jax.random.key(3), **inputs)
    output = model.apply(variables, **inputs)
    assert output["all_tables"].shape == (2, 4, 7, 5)
    assert "proprio_evidence_hidden" in variables["params"]
    assert jnp.isfinite(output["event_type_logits"]).all()


def test_optional_proprio_evidence_checks_alignment():
    inputs = _inputs()
    inputs["proprio_tokens"] = jnp.ones((2, 3, 3, 8), jnp.float32)
    model = ExplicitEventBottleneckMemory(
        max_steps=3,
        frames=4,
        spatial_tokens=4,
        input_width=12,
        width=16,
        hidden_width=32,
        memory_tokens=12,
    )
    with pytest.raises(ValueError, match="Expected proprio tokens"):
        model.init(jax.random.key(4), **inputs)

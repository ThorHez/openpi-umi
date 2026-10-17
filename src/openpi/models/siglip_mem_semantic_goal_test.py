import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import siglip_mem_semantic_goal


def test_goal_encoder_ignores_masked_prompt_tokens():
    encoder = siglip_mem_semantic_goal.GoalTokenEncoder(
        input_width=16,
        width=8,
        num_goal_tokens=2,
        num_heads=2,
    )
    prompt = jax.random.normal(jax.random.key(0), (2, 5, 16))
    prompt_mask = jnp.asarray([[True, True, True, False, False], [True, False, False, False, False]])
    variables = encoder.init(jax.random.key(1), prompt, prompt_mask=prompt_mask)

    changed_masked_values = jnp.where(prompt_mask[..., None], prompt, prompt + 1000.0)
    original_goal = encoder.apply(variables, prompt, prompt_mask=prompt_mask)
    changed_goal = encoder.apply(variables, changed_masked_values, prompt_mask=prompt_mask)

    assert original_goal.shape == (2, 2, 8)
    np.testing.assert_allclose(original_goal, changed_goal, atol=1e-6, rtol=1e-6)


def test_goal_conditioned_memory_keeps_fixed_shape_and_respects_step_mask():
    model = siglip_mem_semantic_goal.GoalConditionedRecurrentMemory(
        prompt_width=16,
        memory_width=8,
        num_memory_tokens=6,
        num_goal_tokens=2,
        goal_heads=2,
        memory_depth=1,
        memory_heads=2,
    )
    prompt = jax.random.normal(jax.random.key(2), (2, 4, 16))
    evidence = jax.random.normal(jax.random.key(3), (2, 3, 2, 8))
    prompt_mask = jnp.asarray([[True, True, True, False], [True, True, False, False]])
    step_mask = jnp.asarray([[True, True, False], [True, False, False]])
    variables = model.init(
        jax.random.key(4),
        prompt,
        evidence,
        prompt_mask=prompt_mask,
        step_mask=step_mask,
    )

    final_memory, states, goal_tokens, initial_memory = model.apply(
        variables,
        prompt,
        evidence,
        prompt_mask=prompt_mask,
        step_mask=step_mask,
    )

    assert final_memory.shape == (2, 6, 8)
    assert states.shape == (2, 3, 6, 8)
    assert goal_tokens.shape == (2, 2, 8)
    assert initial_memory.shape == (2, 6, 8)
    np.testing.assert_allclose(states[0, 2], states[0, 1], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(states[1, 1], states[1, 0], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(states[1, 2], states[1, 1], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(final_memory, states[:, -1], atol=1e-6, rtol=1e-6)


def test_goal_tokens_are_injected_only_into_reserved_memory_slots():
    initializer = siglip_mem_semantic_goal.GoalConditionedMemoryInitializer(
        width=8,
        num_memory_tokens=5,
        num_goal_tokens=2,
    )
    zeros = jnp.zeros((1, 2, 8))
    variables = initializer.init(jax.random.key(5), zeros)
    base_memory = initializer.apply(variables, zeros)
    offset = jnp.ones((1, 2, 8))
    conditioned_memory = initializer.apply(variables, offset)

    np.testing.assert_allclose(conditioned_memory[:, :2] - base_memory[:, :2], 1.0)
    np.testing.assert_array_equal(conditioned_memory[:, 2:], base_memory[:, 2:])


def test_goal_encoder_rejects_incompatible_width():
    encoder = siglip_mem_semantic_goal.GoalTokenEncoder(input_width=16, width=8, num_heads=2)

    with pytest.raises(ValueError, match="Expected prompt_tokens"):
        encoder.init(jax.random.key(6), jnp.zeros((1, 3, 15)))

import jax
import jax.numpy as jnp
import pytest

from openpi.tasks.robomme.pickxtimes import eef_action_adapter
from openpi.tasks.robomme.pickxtimes import eef_action_chunk_adapter


@pytest.mark.parametrize("use_memory", [False, True])
def test_pickxtimes_action_chunk_adapter_shapes(use_memory):
    model = eef_action_chunk_adapter.PickXtimesEEFActionChunkAdapter(
        action_horizon=4,
        hidden_width=32,
        depth=1,
        memory_query_tokens=2,
        use_memory=use_memory,
    )
    visual = jnp.zeros((2, eef_action_adapter.VISUAL_FEATURE_DIM), dtype=jnp.float32)
    robot_goal = jnp.zeros((2, eef_action_adapter.ROBOT_GOAL_DIM), dtype=jnp.float32)
    memory = jnp.zeros(
        (2, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH), dtype=jnp.float32
    )
    outputs = model.apply(
        model.init(jax.random.key(0), visual, robot_goal, memory),
        visual,
        robot_goal,
        memory,
    )
    assert outputs["normalized_poses"].shape == (2, 4, 6)
    assert outputs["close_logits"].shape == (2, 4)
    assert outputs["phase_logits"].shape == (2, 3)


def test_pickxtimes_action_chunk_requires_positive_horizon():
    model = eef_action_chunk_adapter.PickXtimesEEFActionChunkAdapter(action_horizon=0)
    with pytest.raises(ValueError, match="action_horizon"):
        model.init(
            jax.random.key(0),
            jnp.zeros((1, eef_action_adapter.VISUAL_FEATURE_DIM)),
            jnp.zeros((1, eef_action_adapter.ROBOT_GOAL_DIM)),
            jnp.zeros((1, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH)),
        )


def test_pickxtimes_spatial_action_chunk_adapter_shapes():
    model = eef_action_chunk_adapter.PickXtimesEEFActionChunkAdapter(
        action_horizon=2,
        hidden_width=32,
        depth=1,
        memory_query_tokens=2,
        use_memory=False,
        spatial_visual_tokens=16,
    )
    visual = jnp.zeros((2, 16, eef_action_adapter.VISUAL_FEATURE_DIM), dtype=jnp.float32)
    robot_goal = jnp.zeros((2, eef_action_adapter.ROBOT_GOAL_DIM), dtype=jnp.float32)
    memory = jnp.zeros((2, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH))
    outputs = model.apply(model.init(jax.random.key(0), visual, robot_goal, memory), visual, robot_goal, memory)
    assert outputs["normalized_poses"].shape == (2, 2, 6)

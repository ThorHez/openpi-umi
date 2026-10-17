import jax
import jax.numpy as jnp
import pytest

from openpi.tasks.robomme.pickxtimes import eef_action_adapter


@pytest.mark.parametrize("use_memory", [False, True])
def test_pickxtimes_action_adapter_shapes(use_memory):
    model = eef_action_adapter.PickXtimesEEFActionAdapter(
        hidden_width=32,
        depth=1,
        memory_query_tokens=2,
        use_memory=use_memory,
    )
    visual = jnp.zeros((2, eef_action_adapter.VISUAL_FEATURE_DIM), dtype=jnp.float32)
    robot_goal = jnp.zeros((2, eef_action_adapter.ROBOT_GOAL_DIM), dtype=jnp.float32)
    memory = jnp.zeros(
        (2, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH),
        dtype=jnp.float32,
    )
    variables = model.init(jax.random.key(0), visual, robot_goal, memory, train=False)
    outputs = model.apply(variables, visual, robot_goal, memory, train=False)
    assert outputs["normalized_pose"].shape == (2, 6)
    assert outputs["close_logit"].shape == (2,)
    assert outputs["phase_logits"].shape == (2, 3)


def test_pickxtimes_action_adapter_rejects_bad_state_width():
    model = eef_action_adapter.PickXtimesEEFActionAdapter(hidden_width=32, depth=1)
    with pytest.raises(ValueError, match="robot-goal"):
        model.init(
            jax.random.key(0),
            jnp.zeros((1, eef_action_adapter.VISUAL_FEATURE_DIM)),
            jnp.zeros((1, eef_action_adapter.ROBOT_GOAL_DIM - 1)),
            jnp.zeros((1, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH)),
        )

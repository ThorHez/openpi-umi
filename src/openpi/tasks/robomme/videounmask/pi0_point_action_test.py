import jax
import jax.numpy as jnp

from openpi.tasks.robomme.videounmask import pi0_point_action


def test_target_point_conditioner_shape_and_causality():
    module = pi0_point_action.TargetPointConditioner(dtype_mm="float32")
    action_tokens = jnp.zeros((2, 16, 1024), dtype=jnp.float32)
    point_a = jnp.asarray([[0.1, 0.2], [0.1, 0.2]], dtype=jnp.float32)
    point_b = point_a.at[1].set(jnp.asarray([0.8, 0.7]))
    variables = module.init(jax.random.key(0), action_tokens, point_a)
    output_a = module.apply(variables, action_tokens, point_a)
    output_b = module.apply(variables, action_tokens, point_b)
    assert output_a.shape == action_tokens.shape
    assert jnp.allclose(output_a[0], output_a[1])
    assert not jnp.allclose(output_b[0], output_b[1])


def test_point_action_config_freeze_filter_constructs():
    config = pi0_point_action.Pi0VideoUnmaskPointActionConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        num_frames=1,
        memory_every=0,
        history_memory_tokens=1,
    )
    assert config.get_freeze_filter_action_finetune() is not None

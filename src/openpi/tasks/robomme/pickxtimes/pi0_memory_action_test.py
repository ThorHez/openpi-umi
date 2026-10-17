import jax
import jax.numpy as jnp

from openpi.tasks.robomme.pickxtimes import pi0_memory_action


def test_semantic_memory_conditioner_changes_with_memory():
    module = pi0_memory_action.SemanticMemoryActionConditioner(
        memory_tokens=4,
        memory_width=8,
        query_tokens=2,
        hidden_width=16,
        action_width=32,
        num_heads=4,
        dtype_mm="float32",
    )
    action = jnp.zeros((2, 3, 32), dtype=jnp.float32)
    memory = jnp.ones((2, 4, 8), dtype=jnp.float32)
    memory = memory.at[1, 0, 0].set(3.0)
    variables = module.init(jax.random.key(0), action, memory)
    output = module.apply(variables, action, memory)
    assert output.shape == action.shape
    assert not jnp.allclose(output[0], output[1])


def test_memory_action_config_freeze_filter_constructs():
    config = pi0_memory_action.Pi0PickXtimesMemoryActionConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        num_frames=1,
        memory_every=0,
        history_memory_tokens=1,
    )
    assert config.get_freeze_filter_action_finetune() is not None


def test_zero_memory_uses_stable_shared_null_bank():
    module = pi0_memory_action.SemanticMemoryActionConditioner(
        memory_tokens=4, memory_width=8, query_tokens=2, hidden_width=16,
        action_width=32, num_heads=4, dtype_mm="float32", use_learned_null_memory=True,
    )
    action = jnp.zeros((2, 3, 32), dtype=jnp.float32)
    memory = jnp.zeros((2, 4, 8), dtype=jnp.float32)
    variables = module.init(jax.random.key(1), action, memory)
    output = module.apply(variables, action, memory)
    assert jnp.all(jnp.isfinite(output))
    assert jnp.allclose(output[0], output[1])

    def loss(mem):
        return jnp.mean(module.apply(variables, action, mem) ** 2)

    gradient = jax.grad(loss)(memory)
    assert jnp.all(jnp.isfinite(gradient))


def test_small_residual_gate_and_training_dropout():
    module = pi0_memory_action.SemanticMemoryActionConditioner(
        memory_tokens=4,
        memory_width=8,
        query_tokens=2,
        hidden_width=16,
        action_width=32,
        num_heads=4,
        dtype_mm="float32",
        residual_gate_init=0.1,
        residual_dropout_rate=0.25,
    )
    action = jnp.zeros((2, 3, 32), dtype=jnp.float32)
    memory = jnp.arange(64, dtype=jnp.float32).reshape(2, 4, 8)
    variables = module.init(jax.random.key(2), action, memory)
    eval_output = module.apply(variables, action, memory, train=False)
    train_output = module.apply(
        variables,
        action,
        memory,
        train=True,
        dropout_rng=jax.random.key(4),
    )
    assert eval_output.shape == train_output.shape == action.shape
    assert jnp.all(jnp.isfinite(train_output))

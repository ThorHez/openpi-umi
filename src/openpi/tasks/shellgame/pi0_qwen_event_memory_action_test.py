import jax
import jax.numpy as jnp

from openpi.tasks.shellgame import pi0_qwen_event_memory_action as policy


def test_shellgame_temporal_mask_uses_future_actions_through_frame_154():
    valid = policy.shellgame_temporal_valid(
        jnp.asarray([153]), action_horizon=16, last_episode_frame=154
    )
    assert valid.shape == (1, 16)
    assert int(jnp.sum(valid)) == 1


def test_config_constructs_direct_memory_policy():
    config = policy.Pi0QwenEventMemoryActionConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        num_frames=1,
        current_frame_index=0,
        memory_every=0,
        history_memory_tokens=1,
    )
    model = config.create(jax.random.key(0))
    assert model.semantic_memory_tokens == 128
    assert model.semantic_memory_width == 64

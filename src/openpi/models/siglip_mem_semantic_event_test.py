import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip_mem_semantic_event


def test_extract_sliding_windows_uses_overlap_and_stride():
    sequence = jnp.arange(2 * 7, dtype=jnp.float32).reshape(2, 7, 1)
    windows, starts = siglip_mem_semantic_event.extract_sliding_windows(
        sequence,
        window_size=3,
        stride=2,
    )
    assert windows.shape == (2, 3, 3, 1)
    np.testing.assert_array_equal(starts[0], (0, 2, 4))
    np.testing.assert_array_equal(windows[0, :, :, 0], ((0, 1, 2), (2, 3, 4), (4, 5, 6)))


def test_causal_trigger_is_rising_edge_and_streaming_equivalent():
    logits = jnp.asarray(((-1.0, 1.0, 2.0, -1.0, 3.0, 4.0),))
    full, full_active = siglip_mem_semantic_event.causal_event_triggers(logits)
    first, active = siglip_mem_semantic_event.causal_event_triggers(logits[:, :3])
    second, final_active = siglip_mem_semantic_event.causal_event_triggers(
        logits[:, 3:],
        previous_active=active,
    )
    np.testing.assert_array_equal(full, ((False, True, False, False, True, False),))
    np.testing.assert_array_equal(jnp.concatenate((first, second), axis=1), full)
    np.testing.assert_array_equal(final_active, full_active)


def test_hysteresis_avoids_threshold_chatter():
    logits = jnp.asarray(((-1.0, 0.8, 0.1, -0.1, 0.2, -0.6, 0.9),))
    triggers, _ = siglip_mem_semantic_event.causal_event_triggers(
        logits,
        high_threshold=0.5,
        low_threshold=-0.5,
    )
    np.testing.assert_array_equal(
        triggers,
        ((False, True, False, False, False, False, True),),
    )


def test_event_mask_prevents_recurrent_memory_writes():
    module = siglip_mem_semantic_event.EventTriggeredRecurrentMemory(
        width=8,
        depth=1,
        num_heads=2,
        dtype_mm="float32",
    )
    memory = jax.random.normal(jax.random.key(0), (2, 4, 8))
    evidence = jax.random.normal(jax.random.key(1), (2, 4, 3, 8))
    logits = jnp.asarray(((-1.0, 1.0, 2.0, -1.0), (-1.0, -1.0, -1.0, -1.0)))
    variables = module.init(jax.random.key(2), memory, evidence, logits)
    outputs = module.apply(variables, memory, evidence, logits)

    np.testing.assert_array_equal(outputs["trigger_count"], (1, 0))
    np.testing.assert_allclose(outputs["memory_states"][0, 0], memory[0], atol=1e-6)
    np.testing.assert_allclose(outputs["memory_states"][0, 2], outputs["memory_states"][0, 1], atol=1e-6)
    np.testing.assert_allclose(outputs["memory"][1], memory[1], atol=1e-6)

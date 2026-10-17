import jax
import jax.numpy as jnp

from openpi.tasks.robomme.videounmask import semantic_memory_target_event


def test_target_event_memory_shapes():
    model = semantic_memory_target_event.VideoUnmaskTargetEventMemory(
        encoder_width=32,
        encoder_depth=1,
        encoder_heads=4,
        memory_width=32,
        memory_depth=1,
        memory_heads=4,
        num_memory_tokens=8,
    )
    inputs = {
        "demo_patch_tokens": jnp.zeros((1, 12, 256, 1152), dtype=jnp.float16),
        "prompt_tokens": jnp.zeros((1, 8, 2048), dtype=jnp.float16),
        "prompt_mask": jnp.ones((1, 8), dtype=jnp.bool_),
        "frame_mask": jnp.ones((1, 12), dtype=jnp.bool_),
    }
    variables = model.init(jax.random.key(0), **inputs, train=False)
    outputs = model.apply(variables, **inputs, train=False)
    assert outputs["target_point"].shape == (1, 2)
    assert outputs["locator_point"].shape == (1, 2)
    assert outputs["locator_cell_logits"].shape == (1, 64)
    assert outputs["memory_cell_logits"].shape == (1, 64)
    assert outputs["memory"].shape == (1, 8, 32)

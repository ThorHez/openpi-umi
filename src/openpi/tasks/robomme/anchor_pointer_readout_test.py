import jax
import jax.numpy as jnp

from openpi.tasks.robomme.anchor_pointer_readout import AnchorPointerReadout


def test_anchor_pointer_masks_invalid_candidates():
    model = AnchorPointerReadout(width=32, num_heads=4)
    inputs = {
        "memory": jnp.zeros((2, 128, 64), dtype=jnp.float32),
        "anchor_tokens": jnp.zeros((2, 4, 1152), dtype=jnp.float32),
        "anchor_yx": jnp.zeros((2, 4, 2), dtype=jnp.float32),
        "anchor_mask": jnp.asarray([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=jnp.bool_),
        "task_ids": jnp.asarray([0, 2], dtype=jnp.int32),
        "query_color_ids": jnp.asarray([1, 3], dtype=jnp.int32),
        "queried_ordinals": jnp.asarray([0, 2], dtype=jnp.int32),
    }
    variables = model.init(jax.random.key(0), **inputs)
    logits = model.apply(variables, **inputs)
    assert logits.shape == (2, 4)
    assert jnp.all(logits[0, 2:] < -1e20)

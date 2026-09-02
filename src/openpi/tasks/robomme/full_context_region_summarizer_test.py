import jax
import jax.numpy as jnp

from openpi.tasks.robomme.full_context_region_summarizer import FullContextRegionSummarizer


def test_full_context_region_summarizer_shapes_and_region_mask() -> None:
    model = FullContextRegionSummarizer(width=16, num_heads=4, depth=1)
    inputs = {
        "features": jnp.zeros((2, 96, 16, 3456), dtype=jnp.float16),
        "chunk_mask": jnp.asarray([[True] * 3 + [False] * 93] * 2),
        "task_ids": jnp.asarray([0, 2]),
        "query_color_ids": jnp.asarray([1, 3]),
        "queried_ordinals": jnp.asarray([0, 2]),
        "num_regions": jnp.asarray([3, 2]),
    }
    params = model.init(jax.random.key(0), **inputs)["params"]
    logits = model.apply({"params": params}, **inputs)
    assert logits.shape == (2, 4)
    assert jnp.isfinite(logits[0, :3]).all()
    assert logits[0, 3] < -1e30
    assert jnp.isfinite(logits[1, :2]).all()
    assert (logits[1, 2:] < -1e30).all()

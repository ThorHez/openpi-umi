import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip_mem_semantic as memory_core
from openpi.tasks.shellgame import semantic_memory


def test_preselected_pooled_clips_match_regular_tracker_path():
    module = semantic_memory.ThreeSwapVisualRelationMemoryTracker(
        num_frames=4,
        input_width=8,
        encoder_width=8,
        encoder_depth=1,
        encoder_heads=2,
        memory_width=8,
        memory_depth=1,
        memory_heads=2,
        adapter_heads=2,
        num_memory_tokens=4,
        num_current_tokens=4,
        current_width=8,
        dtype_mm="float32",
        swap_frame_indices=((1,), (2,), (3,)),
    )
    patch_tokens = jax.random.normal(jax.random.key(0), (2, 4, 256, 8))
    initial_slots = jnp.asarray([0, 2], dtype=jnp.int32)
    variables = module.init(jax.random.key(1), patch_tokens, initial_slots)
    regular = module.apply(variables, patch_tokens, initial_slots)

    pooled = memory_core.pool_fixed_grid(patch_tokens, pool_factor=2)
    clips = jnp.stack([pooled[:, jnp.asarray(indices)] for indices in ((1,), (2,), (3,))], axis=1)
    cached = module.apply(
        variables,
        clips,
        initial_slots,
        preselected_pooled_clips=True,
    )
    for regular_value, cached_value in zip(regular, cached, strict=True):
        np.testing.assert_allclose(regular_value, cached_value, atol=1e-6, rtol=1e-6)

    relation_only = module.apply(
        variables,
        clips,
        initial_slots,
        preselected_pooled_clips=True,
        return_relation_logits_only=True,
    )
    np.testing.assert_allclose(relation_only, regular[3], atol=1e-6, rtol=1e-6)

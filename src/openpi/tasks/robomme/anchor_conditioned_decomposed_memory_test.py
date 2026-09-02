import jax
import jax.numpy as jnp

from openpi.tasks.robomme.anchor_conditioned_decomposed_memory import (
    AnchorConditionedDecomposedMemory,
)


def test_anchor_conditioned_memory_shapes_and_masked_hold():
    model = AnchorConditionedDecomposedMemory(max_steps=3)
    inputs = {
        "patch_tokens": jnp.zeros((2, 3, 12, 16, 1152), dtype=jnp.float16),
        "sequence_mask": jnp.asarray([[True, True, False], [True, False, False]]),
        "task_ids": jnp.asarray([0, 1]),
        "goal_color_ids": jnp.asarray([[1, 0], [2, 3]]),
        "queried_ordinals": jnp.asarray([0, 0]),
        "num_regions": jnp.asarray([3, 4]),
        "anchor_yx": jnp.zeros((2, 4, 2), dtype=jnp.float32),
        "anchor_mask": jnp.asarray(
            [[True, True, True, False], [True, True, True, True]]
        ),
    }
    variables = model.init(jax.random.key(0), **inputs)
    output = model.apply(variables, **inputs)
    assert output["all_tables"].shape == (2, 4, 7, 5)
    assert output["all_memories"].shape == (2, 4, 128, 64)
    assert output["event_type_logits"].shape == (2, 3, 2, 3)
    assert output["write_region_logits"].shape == (2, 3, 2, 4)
    assert output["swap_pair_logits"].shape == (2, 3, 2, 6)
    assert jnp.all(output["write_region_logits"][0, :, :, 3] < -1e8)
    assert jnp.allclose(output["all_tables"][0, 2], output["all_tables"][0, 3])
    assert jnp.allclose(output["all_tables"][1, 1], output["all_tables"][1, 3])


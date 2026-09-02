import jax
import jax.numpy as jnp

from openpi.tasks.robomme.decomposed_region_recurrent_memory import DecomposedRegionRecurrentMemory


def test_decomposed_region_memory_contract() -> None:
    model = DecomposedRegionRecurrentMemory(
        max_steps=3,
        encoder_width=16,
        encoder_depth=1,
        encoder_heads=4,
    )
    inputs = {
        "patch_tokens": jnp.zeros((1, 3, 12, 16, 1152), dtype=jnp.float16),
        "sequence_mask": jnp.asarray([[True, True, False]]),
        "task_ids": jnp.asarray([0]),
        "goal_color_ids": jnp.asarray([[1, 0]]),
        "queried_ordinals": jnp.asarray([0]),
        "num_regions": jnp.asarray([3]),
    }
    params = model.init(jax.random.key(0), **inputs)["params"]
    output = model.apply({"params": params}, **inputs)
    assert output["all_tables"].shape == (1, 4, 7, 5)
    assert output["all_memories"].shape == (1, 4, 128, 64)
    assert output["event_type_logits"].shape == (1, 3, 2, 3)
    # Padding must preserve the preceding recurrent table exactly.
    assert jnp.allclose(output["all_tables"][:, 3], output["all_tables"][:, 2])


import jax
import jax.numpy as jnp
import pytest

from openpi.tasks.robomme.causal_visual_operation_parser import CausalVisualOperationParser


@pytest.mark.parametrize(("spatial_tokens", "recurrent"), [(16, False), (64, True)])
def test_causal_visual_operation_parser_shapes(spatial_tokens, recurrent):
    model = CausalVisualOperationParser(
        max_steps=3,
        frames=12,
        spatial_tokens=spatial_tokens,
        input_width=8,
        width=16,
        hidden_width=24,
        recurrent_event_state=recurrent,
    )
    inputs = {
        "patch_tokens": jnp.zeros(
            (2, 3, 12, spatial_tokens, 8), dtype=jnp.float32
        ),
        "sequence_mask": jnp.asarray([[True, True, False], [True, True, True]]),
        "task_ids": jnp.asarray([0, 2]),
        "goal_color_ids": jnp.asarray([[1, 0], [3, 0]]),
        "queried_ordinals": jnp.asarray([0, 2]),
        "num_regions": jnp.asarray([3, 4]),
        "anchor_yx": jnp.zeros((2, 4, 2), dtype=jnp.float32),
        "anchor_mask": jnp.asarray([[True, True, True, False], [True] * 4]),
        "previous_tables": jnp.zeros((2, 3, 7), dtype=jnp.int32),
    }
    variables = model.init(jax.random.key(0), **inputs, train=False)
    output = model.apply(variables, **inputs, train=False)
    assert output["event_type_logits"].shape == (2, 3, 2, 3)
    assert output["phase_logits"].shape == (2, 3, 4)
    assert output["completion_logits"].shape == (2, 3, 2)
    assert output["event_kind_logits"].shape == (2, 3, 2, 2)
    assert output["write_entity_logits"].shape == (2, 3, 2, 7)
    assert output["write_region_logits"].shape == (2, 3, 2, 4)
    assert output["swap_pair_logits"].shape == (2, 3, 2, 6)

import jax
import jax.numpy as jnp

from openpi.tasks.robomme.videounmask import eef_action_adapter


def test_eef_action_adapter_shapes_and_gradients():
    model = eef_action_adapter.VideoUnmaskEEFActionAdapter(hidden_width=32, depth=2)
    features = jnp.zeros((4, eef_action_adapter.ACTION_FEATURE_DIM), dtype=jnp.float32)
    crops = jnp.zeros((4, 32, 32, 3), dtype=jnp.uint8)
    variables = model.init(jax.random.key(0), features, crops, train=False)
    outputs = model.apply(variables, features, crops, train=False)

    assert outputs["normalized_pose"].shape == (4, 6)
    assert outputs["close_logit"].shape == (4,)
    leaves = jax.tree.leaves(
        jax.grad(lambda p: jnp.sum(model.apply({"params": p}, features, crops)["normalized_pose"]))(variables["params"])
    )
    assert any(jnp.any(leaf != 0) for leaf in leaves)

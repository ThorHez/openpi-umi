from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


def ema_merge_trees(decay: float, old_tree: at.PyTree, new_tree: at.PyTree) -> at.PyTree:
    """Blend ``new_tree`` into ``old_tree`` with EMA for floating-point arrays only.

    NNX graphs can include PRNG key leaves (e.g. from ``nnx.Dropout``); those must not be
    linearly mixed with parameters.
    """

    def leaf(old, new):
        if isinstance(old, jax.Array) and isinstance(new, jax.Array):
            if old.shape != new.shape:
                return new
            if jax.dtypes.issubdtype(old.dtype, jax.dtypes.prng_key) or jax.dtypes.issubdtype(
                new.dtype, jax.dtypes.prng_key
            ):
                return new
            if jnp.issubdtype(old.dtype, jnp.floating) and jnp.issubdtype(new.dtype, jnp.floating):
                return decay * old + (1 - decay) * new
        return new

    return jax.tree.map(leaf, old_tree, new_tree)


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")

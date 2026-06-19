"""Pi0MemCompress training entry point.

Byte-for-byte mirror of :mod:`scripts.mem.train_pi0_mem`, with exactly two
differences:

* The model paradigm check accepts a :class:`Pi0MemCompressConfig` instead
  of :class:`Pi0MemConfig`.
* All training helpers (``init_logging``, ``init_wandb``, ``init_train_state``,
  ``train_step``) are still imported verbatim from ``scripts/train.py``, and
  the Pi0Mem-aware data loader factory
  (``openpi.training.config_pi0_mem.create_pi0_mem_data_loader``) is reused —
  the data pipeline is identical: ``VideoFrameDataset -> BuildVideoTensor ->
  UmiInputsV4_Bimanual_Video`` produces ``[B, T, H, W, C]`` per image stream,
  which is exactly what :mod:`openpi.models.siglip_mem_compress` expects.

Launch with the standard tyro CLI::

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/mem/train_pi0_mem_compress.py \\
        pi0_mem_compress_umi_32d_60k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322 \\
        --exp_name=run_compress_T4 \\
        --data.num_frames=4 --model.num_frames=4 \\
        --model.history_memory_tokens=256 \\
        --batch_size=72 --fsdp_devices=8
"""

import os
from pathlib import Path

# Mirror train_pi0_mem.py: keep TMPDIR off / and overlay.
if "TMPDIR" not in os.environ:
    _tmp = Path(os.environ.get("HOME", "/root")) / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_tmp)

import dataclasses
import functools
import logging
import platform
import sys
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

# Pi0MemCompress-specific imports.
import openpi.models.pi0_mem_compress as pi0_mem_compress
import openpi.training.config_pi0_mem as _config_pi0_mem

# Reuse train.py helpers verbatim. Add openpi-umi/scripts to sys.path so we
# can import scripts/train.py as a top-level module. Note we deliberately do
# NOT import ``train_step`` from there — Pi0MemCompress uses its own local
# version (defined below) that surfaces the compressed-memory metrics.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train import _load_weights_and_validate, init_logging, init_wandb  # noqa: E402


# ---------------------------------------------------------------------------
# Compressed-memory training metrics (gradient health + token diversity).
# ---------------------------------------------------------------------------


def _tree_leaves_with_paths(tree):
    """Yield ``(path_str, leaf)`` tuples for every leaf in ``tree``.

    ``path_str`` is the dotted/bracketed JAX KeyPath stringification so the
    user-side substring filtering ("HistoryResampler", "memory_queries", ...)
    matches the Flax/NNX submodule names regardless of how the underlying
    pytree was registered.
    """
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        try:
            path_str = jax.tree_util.keystr(path)
        except Exception:
            path_str = str(path)
        yield path_str, leaf


def _leaf_array(leaf):
    """Return the JAX array carried by an NNX variable or a raw pytree leaf."""
    return leaf.value if hasattr(leaf, "value") else leaf


def _is_history_gate_path(path) -> bool:
    try:
        path_str = jax.tree_util.keystr(path)
    except Exception:
        path_str = str(path)
    return "history_memory_gate_logit" in path_str


def _global_l2_norm(leaves):
    """Return the global L2 norm over a list of arrays.

    Returns 0.0 (as a JAX scalar) when ``leaves`` is empty so that the
    resulting metrics dict is shape-stable even when the model has no
    matching params yet — this keeps the jit cache from being invalidated
    on the first vs. subsequent steps.
    """
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    sq_sum = sum(jnp.sum(jnp.square(jnp.asarray(_leaf_array(l), dtype=jnp.float32))) for l in leaves)
    return jnp.sqrt(sq_sum)


def history_gate_param_metrics(params):
    """Track whether the learned history gates move away from initialization."""
    leaves = [
        _leaf_array(leaf)
        for path, leaf in _tree_leaves_with_paths(params)
        if "history_memory_gate_logit" in path
    ]
    if not leaves:
        nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return {
            "gate/count": zero,
            "gate/logit_mean": nan,
            "gate/logit_min": nan,
            "gate/logit_max": nan,
            "gate/sigmoid_mean": nan,
            "gate/sigmoid_min": nan,
            "gate/sigmoid_max": nan,
        }

    logits = jnp.concatenate([jnp.ravel(jnp.asarray(x, dtype=jnp.float32)) for x in leaves])
    gates = jax.nn.sigmoid(logits)
    return {
        "gate/count": jnp.asarray(logits.size, dtype=jnp.float32),
        "gate/logit_mean": jnp.mean(logits),
        "gate/logit_min": jnp.min(logits),
        "gate/logit_max": jnp.max(logits),
        "gate/sigmoid_mean": jnp.mean(gates),
        "gate/sigmoid_min": jnp.min(gates),
        "gate/sigmoid_max": jnp.max(gates),
    }


def new_memory_param_grad_metrics(grads):
    """Compute simple gradient norms for newly added memory parameters.

    Call this inside the trainer's loss/grad function (or right after
    :func:`jax.value_and_grad` returns) so the path-string matching happens
    at trace time and the surviving JAX computation is just a few
    ``jnp.sqrt(jnp.sum(...))`` ops.

    Args:
        grads: gradient pytree with the same structure as params. It may be
            the full grads tree (NNX State) or just ``grads['params']``.

    Returns:
        ``dict[str, jnp.ndarray]`` suitable for wandb/logging.
    """
    path_leaves = list(_tree_leaves_with_paths(grads))

    def select(*needles):
        selected = []
        for path, leaf in path_leaves:
            if all(n in path for n in needles):
                selected.append(leaf)
        return selected

    # Main new module: learned history compressor.
    resampler = select("HistoryResampler")
    memory_queries = select("HistoryResampler", "memory_queries")
    current_condition = select("HistoryResampler", "current_condition")
    resampler_cross_attn = select("HistoryResampler", "CrossAttention")
    resampler_mlp = select("HistoryResampler", "MlpBlock")

    # New gates inside current-frame blocks.
    history_gate = select("history_memory_gate_logit")

    return {
        "grad/memory_new_total_l2": _global_l2_norm(resampler + history_gate),
        "grad/history_resampler_l2": _global_l2_norm(resampler),
        "grad/memory_queries_l2": _global_l2_norm(memory_queries),
        "grad/current_condition_l2": _global_l2_norm(current_condition),
        "grad/resampler_cross_attn_l2": _global_l2_norm(resampler_cross_attn),
        "grad/resampler_mlp_l2": _global_l2_norm(resampler_mlp),
        "grad/history_gate_l2": _global_l2_norm(history_gate),
    }


def history_memory_collapse_metrics(model_out, eps=1e-6):
    """Measure whether compressed history memory tokens collapse.

    Expects the model output dictionary returned by this visual encoder,
    where ``model_out['encoder']['history_mem']`` has shape ``[B, M, D]``.

    Raw vs. centered cosine — how to read them together:
    - ``memory/hist_token_cosine_offdiag_mean``: cosine over raw tokens.
      Saturates near 1 whenever the M tokens share a strong DC offset,
      *even if* the per-token residuals are diverse. So a high value here
      alone is not conclusive evidence of collapse.
    - ``memory/hist_token_centered_cosine_offdiag_mean``: same cosine,
      but computed on mean-centered tokens
      ``z = hist_mem - hist_mem.mean(axis=1, keepdims=True)``. This is the
      *true* collapse indicator — when both raw and centered are near 1,
      every token really is the same direction; when raw ~ 1 but centered
      drops to ~0.1-0.3, the model just has a shared bias and the
      residuals (which is what the gated cross-attn actually reads) are
      diverse.
    - ``memory/hist_token_centered_cosine_offdiag_max``: max counterpart of
      the centered mean.
    - ``memory/hist_token_var_mean``: near zero means token diversity is
      poor at the *value* level (raw, with bias included).
    - ``memory/hist_token_centered_var``: variance of the residuals after
      removing the shared bias. Goes to zero only when tokens really are
      identical.
    """
    hist_mem = model_out["encoder"]["history_mem"]
    hist_mem = jnp.asarray(hist_mem, dtype=jnp.float32)  # [B, M, D]
    b, m, d = hist_mem.shape

    token_norm = jnp.linalg.norm(hist_mem, axis=-1)  # [B, M]
    mem_centered = hist_mem - jnp.mean(hist_mem, axis=1, keepdims=True)
    token_var = jnp.mean(jnp.var(hist_mem, axis=1))
    centered_var = jnp.mean(jnp.square(mem_centered))

    if m <= 1:
        offdiag_mean = jnp.asarray(0.0, dtype=jnp.float32)
        offdiag_max = jnp.asarray(0.0, dtype=jnp.float32)
        centered_offdiag_mean = jnp.asarray(0.0, dtype=jnp.float32)
        centered_offdiag_max = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        eye = jnp.eye(m, dtype=jnp.bool_)[None, :, :]
        denom = b * m * (m - 1)

        # Raw cosine (kept for backwards compatibility with prior runs).
        mem_normed = hist_mem / (jnp.linalg.norm(hist_mem, axis=-1, keepdims=True) + eps)
        sim = jnp.einsum("bmd,bnd->bmn", mem_normed, mem_normed)  # [B, M, M]
        offdiag = jnp.where(eye, 0.0, sim)
        offdiag_mean = jnp.sum(jnp.abs(offdiag)) / jnp.maximum(denom, 1)
        offdiag_max = jnp.max(jnp.abs(offdiag))

        # Centered cosine (new) — the actually-meaningful "true collapse"
        # indicator. We normalize the centered residuals and look at their
        # pairwise cosine; this strips the shared-bias direction that
        # otherwise dominates the raw cosine. When both ``mem_centered``
        # is ~0 and ``hist_mem`` is non-zero (i.e. every token IS the
        # shared bias), the eps guard keeps the divide finite and the
        # resulting cosine is 0, which together with a high raw cosine
        # is the unambiguous signature of true collapse.
        z = mem_centered / (jnp.linalg.norm(mem_centered, axis=-1, keepdims=True) + eps)
        sim_c = jnp.einsum("bmd,bnd->bmn", z, z)
        offdiag_c = jnp.where(eye, 0.0, sim_c)
        centered_offdiag_mean = jnp.sum(jnp.abs(offdiag_c)) / jnp.maximum(denom, 1)
        centered_offdiag_max = jnp.max(jnp.abs(offdiag_c))

    return {
        "memory/hist_mem_norm_mean": jnp.mean(token_norm),
        "memory/hist_mem_norm_std": jnp.std(token_norm),
        "memory/hist_token_var_mean": token_var,
        "memory/hist_token_centered_var": centered_var,
        "memory/hist_token_cosine_offdiag_mean": offdiag_mean,
        "memory/hist_token_cosine_offdiag_max": offdiag_max,
        "memory/hist_token_centered_cosine_offdiag_mean": centered_offdiag_mean,
        "memory/hist_token_centered_cosine_offdiag_max": centered_offdiag_max,
    }


def simple_memory_training_metrics(model_out, grads):
    """Convenience wrapper: gradients-on-new-params + history collapse."""
    metrics = {}
    metrics.update(new_memory_param_grad_metrics(grads))
    metrics.update(history_memory_collapse_metrics(model_out))
    return metrics


def memory_diversity_loss(hist_mem, eps: float = 1e-6):
    """Push compressed-history memory tokens apart in feature space.

    Targets the failure mode we observed in early Pi0MemCompress runs: the
    sigmoid history gate starts near 0 (``sigmoid(-6.9) ~= 1e-3``), so almost
    no useful gradient reaches the resampler from the action loss, and the
    M memory tokens collapse to nearly identical directions
    (``memory/hist_token_cosine_offdiag_mean -> 0.99+``).

    The loss is the mean squared off-diagonal cosine similarity between
    mean-centered, L2-normalized memory tokens within each batch element.
    It is bounded in [0, 1] (cosine squared), zero iff every pair of tokens
    is orthogonal after centering, and large (~1) when tokens are all the
    same direction.

    Crucially, the gradient of this loss reaches the resampler **directly**
    (it bypasses the per-block sigmoid gate), so it can free the resampler
    even when the gate is still effectively closed.

    Args:
        hist_mem: ``[B, M, D]`` compressed-history tensor. ``B`` here is the
            "memory batch" — typically ``original_batch * num_image_streams``
            because :meth:`Pi0MemCompress._embed_prefix_with_history_mem`
            stacks per-stream memories along axis 0.
        eps: numerical floor for the L2 normalization.

    Returns:
        Scalar JAX array (mean squared off-diagonal cosine similarity).
        Returns 0.0 when ``M <= 1`` (no off-diagonals).
    """
    hist_mem = jnp.asarray(hist_mem, dtype=jnp.float32)
    b, m, _ = hist_mem.shape

    if m <= 1:
        return jnp.asarray(0.0, dtype=jnp.float32)

    z = hist_mem - jnp.mean(hist_mem, axis=1, keepdims=True)
    z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + eps)

    sim = jnp.einsum("bmd,bnd->bmn", z, z)
    eye = jnp.eye(m, dtype=sim.dtype)[None, :, :]
    offdiag = sim * (1.0 - eye)

    denom = jnp.asarray(b * m * (m - 1), dtype=jnp.float32)
    return jnp.sum(offdiag ** 2) / jnp.maximum(denom, 1.0)


def _copy_observation_with_images(
    observation: _model.Observation,
    images: dict[str, jnp.ndarray],
) -> _model.Observation:
    """Return ``observation`` with only the image dict replaced."""
    return dataclasses.replace(observation, images=images)


def _current_frame_index(num_frames: int, configured_index: int) -> int:
    """Resolve a possibly negative current-frame index."""
    cur_idx = int(configured_index)
    if cur_idx < 0:
        cur_idx = num_frames + cur_idx
    if not 0 <= cur_idx < num_frames:
        raise ValueError(f"current_frame_index={configured_index} is out of range for T={num_frames}")
    return cur_idx


def apply_current_frame_corruption(
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    *,
    current_frame_index: int,
    dropout_prob: float,
    mask_prob: float,
) -> tuple[_model.Observation, dict[str, jnp.ndarray]]:
    """Corrupt only the policy-relevant current frame of video observations.

    This is deliberately applied inside the train step, not in the dataset, so
    eval/ablation always sees clean inputs unless an evaluation script opts in.
    Image values are normalized to [-1, 1], so 0.0 is the neutral gray value.
    """
    dropout_prob = float(dropout_prob)
    mask_prob = float(mask_prob)
    if dropout_prob <= 0.0 and mask_prob <= 0.0:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return observation, {
            "current_frame_dropout_rate": zero,
            "current_frame_mask_rate": zero,
        }

    rng_dropout, rng_mask = jax.random.split(rng)
    images = {}
    dropout_rates = []
    mask_rates = []
    image_items = tuple(observation.images.items())
    mask_keys = jax.random.split(rng_mask, len(image_items))
    video_images = tuple(jnp.asarray(image) for _, image in image_items if jnp.asarray(image).ndim == 5)

    shared_drop = None
    if dropout_prob > 0.0 and video_images:
        batch_size = video_images[0].shape[0]
        # One decision per sample, shared by every camera stream, so no view can
        # leak the clean current-frame signal when dropout is active.
        shared_drop = jax.random.bernoulli(
            rng_dropout,
            p=jnp.asarray(dropout_prob, dtype=jnp.float32),
            shape=(batch_size, 1, 1, 1),
        )

    for i, (name, image) in enumerate(image_items):
        image = jnp.asarray(image)
        if image.ndim != 5:
            images[name] = image
            continue

        cur_idx = _current_frame_index(image.shape[1], current_frame_index)
        current = image[:, cur_idx]
        corrupted = current

        if shared_drop is not None:
            corrupted = jnp.where(shared_drop, jnp.zeros_like(corrupted), corrupted)
            dropout_rates.append(jnp.mean(shared_drop.astype(jnp.float32)))

        if mask_prob > 0.0:
            mask = jax.random.bernoulli(
                mask_keys[i],
                p=jnp.asarray(mask_prob, dtype=jnp.float32),
                shape=(*current.shape[:-1], 1),
            )
            corrupted = jnp.where(mask, jnp.zeros_like(corrupted), corrupted)
            mask_rates.append(jnp.mean(mask.astype(jnp.float32)))

        images[name] = image.at[:, cur_idx].set(corrupted)

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    return _copy_observation_with_images(observation, images), {
        "current_frame_dropout_rate": jnp.mean(jnp.stack(dropout_rates)) if dropout_rates else zero,
        "current_frame_mask_rate": jnp.mean(jnp.stack(mask_rates)) if mask_rates else zero,
    }


def _make_history_gate_mask(params: nnx.State):
    """Boolean pytree selecting only ``history_memory_gate_logit`` leaves."""
    return jax.tree_util.tree_map_with_path(
        lambda path, _leaf: _is_history_gate_path(path),
        params,
    )


def _history_gate_lr_multiplier(config: _config.TrainConfig) -> float:
    """Return the configured gate LR multiplier, with env override for quick sweeps."""
    env_value = os.environ.get("OPENPI_HISTORY_GATE_LR_MULTIPLIER")
    if env_value is not None:
        return float(env_value)
    return float(getattr(config.model, "history_gate_lr_multiplier", 1.0))


def _create_optimizer_for_trainable_params(
    config: _config.TrainConfig,
    trainable_params: nnx.State,
) -> optax.GradientTransformation:
    """Create the normal optimizer, optionally scaling history-gate updates.

    ``optax.masked(optax.scale(multiplier), gate_mask)`` is chained *after*
    AdamW, so only the final updates for ``history_memory_gate_logit`` are
    multiplied. This behaves like a parameter-group learning-rate multiplier
    without changing the shared optimizer implementation used by other scripts.
    """
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    gate_lr_multiplier = _history_gate_lr_multiplier(config)
    if gate_lr_multiplier == 1.0:
        return tx

    gate_mask = _make_history_gate_mask(trainable_params)
    return optax.chain(
        tx,
        optax.masked(optax.scale(gate_lr_multiplier), gate_mask),
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """Pi0MemCompress train-state init with optional gate LR multiplier."""

    def init_params(
        rng: at.KeyArrayLike,
        partial_params: at.Params | None = None,
    ) -> tuple[nnx.State, nnx.GraphDef]:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return params, nnx.graphdef(model)

    params_shape, _model_def_shape = jax.eval_shape(init_params, init_rng)
    trainable_params_shape = params_shape.filter(config.trainable_filter)
    tx = _create_optimizer_for_trainable_params(config, trainable_params_shape)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        params, model_def = init_params(rng, partial_params)
        trainable_params = params.filter(config.trainable_filter)

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=model_def,
            tx=tx,
            opt_state=tx.init(trainable_params),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    def count_params_from_shape(params_dict):
        total = 0
        for leaf in jax.tree.leaves(params_dict):
            arr = _leaf_array(leaf)
            if hasattr(arr, "shape"):
                total += int(np.prod(arr.shape))
        return total

    all_params_shape = params_shape
    frozen_params_shape = all_params_shape.filter(config.freeze_filter)

    total_count = count_params_from_shape(all_params_shape)
    frozen_count = count_params_from_shape(frozen_params_shape)
    trainable_count = count_params_from_shape(trainable_params_shape)
    gate_mask = _make_history_gate_mask(trainable_params_shape)
    trainable_gate_leaves = sum(bool(x) for x in jax.tree.leaves(gate_mask))
    gate_lr_multiplier = _history_gate_lr_multiplier(config)

    logging.info("=" * 60)
    logging.info("FREEZE FILTER ANALYSIS:")
    logging.info(f"  Total params:     {total_count:,} ({total_count/1e6:.2f}M)")
    logging.info(f"  Frozen params:    {frozen_count:,} ({frozen_count/total_count*100:.1f}%)")
    logging.info(f"  Trainable params: {trainable_count:,} ({trainable_count/1e6:.2f}M)")
    logging.info(
        "  Trainable history gate leaves: %d; history_gate_lr_multiplier=%.2f",
        trainable_gate_leaves,
        gate_lr_multiplier,
    )
    logging.info("=" * 60)

    trainable_flat = traverse_util.flatten_dict(trainable_params_shape.to_pure_dict())
    gate_paths = [
        "/".join(k)
        for k in trainable_flat
        if "history_memory_gate_logit" in "/".join(k)
    ]
    if gate_paths:
        logging.info("Trainable history gate paths:")
        for path in gate_paths:
            logging.info("  - %s", path)
    elif gate_lr_multiplier != 1.0:
        logging.warning(
            "history_gate_lr_multiplier=%.2f but no trainable history_memory_gate_logit "
            "params matched. Check freeze_filter/path names.",
            gate_lr_multiplier,
        )

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


# ---------------------------------------------------------------------------
# Local train_step: mirrors scripts/train.train_step but routes the loss
# through Pi0MemCompress.compute_loss_with_memory_aux so we can also log
# compressed-memory health metrics inside the same jit'd call.
# ---------------------------------------------------------------------------


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    # Diversity coefficient is a Python-static field on the model config, so
    # the ``if`` below resolves at trace time and the dead branch never
    # enters the HLO graph (no extra compute when the switch is off).
    diversity_weight = float(getattr(config.model, "diversity_weight", 0.0))
    diversity_on = diversity_weight > 0.0
    current_frame_dropout_prob = float(getattr(config.model, "current_frame_dropout_prob", 0.0))
    current_frame_mask_prob = float(getattr(config.model, "current_frame_mask_prob", 0.0))
    current_frame_corruption_on = current_frame_dropout_prob > 0.0 or current_frame_mask_prob > 0.0
    current_frame_corrupt_loss_weight = float(getattr(config.model, "current_frame_corrupt_loss_weight", 0.0))
    current_frame_corrupt_loss_on = current_frame_corrupt_loss_weight > 0.0 and current_frame_corruption_on
    current_frame_index = int(getattr(config.model, "current_frame_index", -1))

    def loss_fn(model, rng, observation, actions):
        corrupt_rng, clean_rng = jax.random.split(rng)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        corruption_metrics = {
            "current_frame_dropout_rate": zero,
            "current_frame_mask_rate": zero,
        }

        chunked_loss, aux = model.compute_loss_with_memory_aux(
            clean_rng, observation, actions, train=True
        )
        action_loss = jnp.mean(chunked_loss)

        if diversity_on:
            div_loss = memory_diversity_loss(aux["history_mem"])
        else:
            div_loss = jnp.asarray(0.0, dtype=action_loss.dtype)

        if current_frame_corrupt_loss_on:
            corrupt_observation, corruption_metrics = apply_current_frame_corruption(
                corrupt_rng,
                observation,
                current_frame_index=current_frame_index,
                dropout_prob=current_frame_dropout_prob,
                mask_prob=current_frame_mask_prob,
            )
            corrupt_chunked_loss, _ = model.compute_loss_with_memory_aux(
                clean_rng, corrupt_observation, actions, train=True
            )
            current_frame_corrupt_action_loss = jnp.mean(corrupt_chunked_loss)
            action_loss_denom = 1.0 + current_frame_corrupt_loss_weight
            normalized_action_loss = (
                action_loss
                + current_frame_corrupt_loss_weight * current_frame_corrupt_action_loss
            ) / action_loss_denom
        else:
            current_frame_corrupt_action_loss = jnp.asarray(0.0, dtype=action_loss.dtype)
            normalized_action_loss = action_loss

        total_loss = normalized_action_loss + diversity_weight * div_loss

        aux_out = {
            "history_mem": aux["history_mem"],
            "action_loss": action_loss,
            "normalized_action_loss": normalized_action_loss,
            "diversity_loss": div_loss,
            "current_frame_corrupt_action_loss": current_frame_corrupt_action_loss,
            "current_frame_dropout_rate": corruption_metrics["current_frame_dropout_rate"],
            "current_frame_mask_rate": corruption_metrics["current_frame_mask_rate"],
        }
        return total_loss, aux_out

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params (same as scripts/train.train_step).
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(
        loss_fn, argnums=diff_state, has_aux=True
    )(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=training_utils.ema_merge_trees(state.ema_decay, state.ema_params, new_params),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    # ``loss`` is the total (action + diversity); ``action_loss`` and
    # ``normalized_action_loss`` / ``diversity_loss`` are surfaced separately
    # so wandb can show the clean view, auxiliary corrupted-current view, and
    # optimizer-driving action objective without conflating their scales.
    info = {
        "loss": loss,
        "action_loss": aux["action_loss"],
        "normalized_action_loss": aux["normalized_action_loss"],
        "diversity_loss": aux["diversity_loss"],
        "diversity_weight": jnp.asarray(diversity_weight, dtype=jnp.float32),
        "current_frame_corrupt_action_loss": aux["current_frame_corrupt_action_loss"],
        "current_frame_corrupt_loss_weight": jnp.asarray(current_frame_corrupt_loss_weight, dtype=jnp.float32),
        "current_frame_dropout_rate": aux["current_frame_dropout_rate"],
        "current_frame_mask_rate": aux["current_frame_mask_rate"],
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }

    # Compressed-memory monitors. ``aux["history_mem"]`` has shape
    # [B * num_image_streams, M, D] (concatenated across image streams by
    # Pi0MemCompress._embed_prefix_with_history_mem). When M==0 or history
    # is unavailable (e.g. num_frames=1), the encoder still emits a
    # shape-stable zero tensor, so these metrics stay defined.
    info.update(new_memory_param_grad_metrics(grads))
    info.update(history_gate_param_metrics(new_params))
    info.update(
        history_memory_collapse_metrics({"encoder": {"history_mem": aux["history_mem"]}})
    )

    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # === Pi0MemCompress paradigm sanity checks ===
    if not isinstance(config.model, pi0_mem_compress.Pi0MemCompressConfig):
        raise ValueError(
            f"train_pi0_mem_compress requires a Pi0MemCompressConfig model; got "
            f"{type(config.model).__name__}. Use scripts/mem/train_pi0_mem.py for "
            "the temporal-attention Pi0Mem variant, or scripts/train.py for "
            "single-frame models."
        )
    # The data side of Pi0MemCompress is identical to Pi0Mem (same video
    # tensors, same VideoFrameConfig contract). So we keep the same
    # Pi0Mem-aware factory check and dispatch on MultiDataConfigFactory.
    if isinstance(config.data, _config.MultiDataConfigFactory):
        if not config.data.datasets:
            raise ValueError(
                "train_pi0_mem_compress requires MultiDataConfigFactory.datasets to be non-empty."
            )
        for i, child in enumerate(config.data.datasets):
            if not hasattr(child, "video_frame_config"):
                raise ValueError(
                    f"train_pi0_mem_compress requires every MultiDataConfigFactory child "
                    f"to be Pi0Mem-aware (must expose .video_frame_config()); "
                    f"datasets[{i}] is {type(child).__name__}."
                )
    elif not hasattr(config.data, "video_frame_config"):
        raise ValueError(
            "train_pi0_mem_compress requires a Pi0Mem-aware DataConfigFactory "
            "(must expose .video_frame_config()), or a MultiDataConfigFactory "
            "whose children are Pi0Mem-aware; got "
            f"{type(config.data).__name__}."
        )

    # === Everything below mirrors scripts/mem/train_pi0_mem.py.main verbatim. ===
    # The only swap is the model-config type check above; the data loader
    # factory is the same (Pi0MemCompress consumes the exact same video
    # batches as Pi0Mem).

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Pi0Mem-aware data loader: identical contract for Pi0MemCompress because
    # the visual backbone change is invisible at the data-pipeline boundary.
    data_loader = _config_pi0_mem.create_pi0_mem_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    loss_history = []
    anomaly_dir = config.checkpoint_dir / "anomalies"
    anomaly_dir.mkdir(exist_ok=True)

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        current_loss = float(info["loss"])
        loss_history.append(current_loss)
        if len(loss_history) > 100:
            loss_history.pop(0)

        is_anomaly = False
        anomaly_reason = ""
        loss_history_size = config.log_interval * 2

        if jnp.isnan(current_loss) or jnp.isinf(current_loss):
            is_anomaly = True
            anomaly_reason = "NaN or Inf loss"
        elif len(loss_history) > loss_history_size:
            recent_mean = np.mean(loss_history[-loss_history_size:])
            recent_std = np.std(loss_history[-loss_history_size:])
            if current_loss > recent_mean + 3 * recent_std and recent_std > 1e-6:
                is_anomaly = True
                anomaly_reason = f"Spike: {current_loss:.6f} vs recent {recent_mean:.6f}\u00b1{recent_std:.6f}"

        if is_anomaly:
            # import pickle
            anomaly_file = anomaly_dir / f"step_{step:06d}_{current_loss:.6f}.pkl"
            pbar.write(f"\u26a0\ufe0f  ANOMALY DETECTED at step {step}: {anomaly_reason}")
            pbar.write(f"   Saving data to {anomaly_file}")

            # batch_cpu = jax.device_get(batch)
            # anomaly_data = {
            #     "step": step,
            #     "loss": current_loss,
            #     "loss_history": loss_history[-20:],
            #     "reason": anomaly_reason,
            #     "observation": batch_cpu[0],
            #     "actions": batch_cpu[1],
            #     "info": jax.device_get(info),
            # }

            # with open(anomaly_file, "wb") as f:
            #     pickle.dump(anomaly_data, f)
            # pbar.write("   \u2713 Anomaly data saved!")

        stats_interval = 1000
        if step > 0 and step % stats_interval == 0 and len(loss_history) > 10:
            # import pickle
            stats_dir = config.checkpoint_dir / "periodic_stats"
            stats_dir.mkdir(exist_ok=True)

            # batch_cpu = jax.device_get(batch)
            # stats_file = stats_dir / f"step_{step:06d}_{current_loss:.6f}.pkl"

            # pbar.write(f"\n\U0001f4ca Saving periodic statistics at step {step} to {stats_file}")

            # stats_data = {
            #     "step": step,
            #     "loss": current_loss,
            #     "loss_history": loss_history.copy(),
            #     "reason": f"Periodic stats at step {step}",
            #     "observation": batch_cpu[0],
            #     "actions": batch_cpu[1],
            #     "info": jax.device_get(info),
            # }

            # with open(stats_file, "wb") as f:
            #     pickle.dump(stats_data, f)
            # pbar.write("   \u2713 Statistics data saved!")

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info = {k: round(float(v), 6) for k, v in reduced_info.items()}
            info_str = ", ".join(f"{k}={v:.6f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())

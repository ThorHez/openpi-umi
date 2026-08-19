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

# Keep training-generated caches and temporary files off the small root disk.
_CACHE_HOME = Path("/data2/hzl_workspace_for_pi/.cache")
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_HOME))
os.environ.setdefault("OPENPI_DATA_HOME", str(_CACHE_HOME / "openpi"))
os.environ.setdefault("HF_HOME", str(_CACHE_HOME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_HOME / "huggingface" / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_CACHE_HOME / "huggingface" / "transformers"))

if "TMPDIR" not in os.environ:
    _tmp = _CACHE_HOME / "tmp"
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_tmp)

import dataclasses
import functools
import importlib
import json
import logging
import platform
import sys
import time
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
import torch
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.multi_data_loader as _multi_data_loader
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

    # Diagnostic classifier. Keep this separate from the resampler metrics so
    # a nonzero head gradient cannot hide a disconnected memory branch.
    history_classifier = select("HistoryClassifier")
    history_classifier_head = select("HistoryClassifierHead")

    # New gates inside current-frame blocks.
    history_gate = select("history_memory_gate_logit")
    history_cross_attn = select("HistoryMultiHeadDotProductAttention_0")
    history_out_proj = select("HistoryOutProj")

    return {
        "grad/memory_new_total_l2": _global_l2_norm(resampler + history_gate + history_cross_attn + history_out_proj),
        "grad/history_resampler_l2": _global_l2_norm(resampler),
        "grad/memory_queries_l2": _global_l2_norm(memory_queries),
        "grad/current_condition_l2": _global_l2_norm(current_condition),
        "grad/resampler_cross_attn_l2": _global_l2_norm(resampler_cross_attn),
        "grad/resampler_mlp_l2": _global_l2_norm(resampler_mlp),
        "grad/history_classifier_l2": _global_l2_norm(history_classifier),
        "grad/history_classifier_head_l2": _global_l2_norm(history_classifier_head),
        "grad/history_gate_l2": _global_l2_norm(history_gate),
        "grad/history_cross_attn_l2": _global_l2_norm(history_cross_attn),
        "grad/history_out_proj_l2": _global_l2_norm(history_out_proj),
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


def memory_branch_activation_metrics(encoder_auxes, eps: float = 1e-6):
    """Track the active history branch residual relative to spatial attention."""
    mem_norms = []
    spatial_norms = []
    gate_values = []
    active_weights = []

    for encoder_aux in encoder_auxes:
        for name, block_out in encoder_aux.items():
            if not name.startswith("block") or "mem_update" not in block_out:
                continue

            mem_update = jnp.asarray(block_out["mem_update"], dtype=jnp.float32)
            y_spatial = jnp.asarray(block_out["y_spatial"], dtype=jnp.float32)
            mem_norm = jnp.mean(jnp.linalg.norm(mem_update, axis=-1))
            spatial_norm = jnp.mean(jnp.linalg.norm(y_spatial, axis=-1))
            active = (mem_norm > eps).astype(jnp.float32)

            mem_norms.append(mem_norm)
            spatial_norms.append(spatial_norm)
            gate_values.append(jnp.asarray(block_out["history_gate"], dtype=jnp.float32))
            active_weights.append(active)

    if not mem_norms:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return {
            "memory/mem_update_norm": zero,
            "memory/y_spatial_norm": zero,
            "memory/mem_update_to_spatial_ratio": zero,
            "gate/effective_mean": zero,
        }

    weights = jnp.stack(active_weights)
    denom = jnp.maximum(jnp.sum(weights), 1.0)
    mem_norm = jnp.sum(jnp.stack(mem_norms) * weights) / denom
    spatial_norm = jnp.sum(jnp.stack(spatial_norms) * weights) / denom
    gate_mean = jnp.sum(jnp.stack(gate_values) * weights) / denom
    return {
        "memory/mem_update_norm": mem_norm,
        "memory/y_spatial_norm": spatial_norm,
        "memory/mem_update_to_spatial_ratio": mem_norm / (spatial_norm + eps),
        "gate/effective_mean": gate_mean,
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
    gate_paths = []
    for key_path in trainable_flat:
        # NNX Sequential modules use integer path components (e.g. the
        # diagnostic classifier's layer 0/1), so paths cannot be joined until
        # every component is stringified.
        path = "/".join(map(str, key_path))
        if "history_memory_gate_logit" in path:
            gate_paths.append(path)
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
    *,
    class_labels_by_episode: at.Array | None = None,
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
    current_frame_corrupt_sample_prob = float(getattr(config.model, "current_frame_corrupt_sample_prob", 0.0))
    current_frame_corrupt_sample_on = current_frame_corrupt_sample_prob > 0.0 and current_frame_corruption_on
    current_frame_corrupt_loss_weight = float(getattr(config.model, "current_frame_corrupt_loss_weight", 0.0))
    current_frame_index = int(getattr(config.model, "current_frame_index", -1))
    classifier_config = config.shellgame_memory_classifier
    classifier_on = bool(classifier_config.enabled)
    classifier_weight = float(classifier_config.loss_weight)
    action_loss_weight = float(classifier_config.action_loss_weight) if classifier_on else 1.0

    def classification_loss_and_metrics(logits, observation):
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        if not classifier_on:
            return zero, zero, zero
        if class_labels_by_episode is None:
            raise ValueError("Memory classifier is enabled but no episode label table was provided.")
        if observation.episode_index is None or observation.frame_index is None:
            raise ValueError("Memory classifier requires episode_index and frame_index in every batch.")

        episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
        frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32)
        episode_valid = (episode_index >= 0) & (episode_index < class_labels_by_episode.shape[0])
        safe_episode_index = jnp.clip(episode_index, 0, class_labels_by_episode.shape[0] - 1)
        labels = class_labels_by_episode[safe_episode_index]
        valid = (
            episode_valid
            & (labels >= 0)
            & (frame_index >= classifier_config.min_frame_index)
            & (frame_index <= classifier_config.max_frame_index)
        )
        valid_f = valid.astype(jnp.float32)
        valid_count = jnp.sum(valid_f)
        safe_labels = jnp.maximum(labels, 0)
        per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(logits, safe_labels)
        classifier_loss = jnp.sum(per_sample_loss * valid_f) / jnp.maximum(valid_count, 1.0)
        accuracy = jnp.sum((jnp.argmax(logits, axis=-1) == safe_labels) * valid_f) / jnp.maximum(
            valid_count, 1.0
        )
        valid_fraction = valid_count / valid_f.size
        return classifier_loss, accuracy, valid_fraction

    def loss_fn(model, rng, observation, actions):
        sample_rng, corrupt_rng, loss_rng = jax.random.split(rng, 3)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        use_corrupt = zero
        corruption_metrics = {
            "current_frame_dropout_rate": zero,
            "current_frame_mask_rate": zero,
        }

        if current_frame_corrupt_sample_on:
            use_corrupt = jax.random.bernoulli(
                sample_rng,
                p=jnp.asarray(current_frame_corrupt_sample_prob, dtype=jnp.float32),
            ).astype(jnp.float32)
            corrupt_observation, corruption_metrics = apply_current_frame_corruption(
                corrupt_rng,
                observation,
                current_frame_index=current_frame_index,
                dropout_prob=current_frame_dropout_prob,
                mask_prob=current_frame_mask_prob,
            )
            images = {
                key: jnp.where(use_corrupt.astype(jnp.bool_), corrupt_observation.images[key], image)
                for key, image in observation.images.items()
            }
            observation = _copy_observation_with_images(observation, images)
            corruption_metrics = {
                key: use_corrupt * value
                for key, value in corruption_metrics.items()
            }

        if classifier_on and action_loss_weight == 0.0:
            history_class_logits, aux = model.compute_history_classification(
                loss_rng,
                observation,
                train=not classifier_config.disable_train_augmentation,
            )
            action_loss = zero
        else:
            chunked_loss, aux = model.compute_loss_with_memory_aux(
                loss_rng, observation, actions, train=True
            )
            action_loss = jnp.mean(chunked_loss)
            history_class_logits = aux["history_class_logits"] if classifier_on else None
        normalized_action_loss = action_loss
        current_frame_corrupt_action_loss = jnp.where(use_corrupt.astype(jnp.bool_), action_loss, zero)

        if diversity_on:
            div_loss = memory_diversity_loss(aux["history_mem"])
        else:
            div_loss = jnp.asarray(0.0, dtype=action_loss.dtype)

        if classifier_on:
            classifier_loss, classifier_accuracy, classifier_valid_fraction = classification_loss_and_metrics(
                history_class_logits, observation
            )
        else:
            classifier_loss = classifier_accuracy = classifier_valid_fraction = zero

        total_loss = (
            action_loss_weight * normalized_action_loss
            + diversity_weight * div_loss
            + classifier_weight * classifier_loss
        )

        aux_out = {
            "history_mem": aux["history_mem"],
            "encoder_auxes": aux["encoder_auxes"],
            "action_loss": action_loss,
            "normalized_action_loss": normalized_action_loss,
            "diversity_loss": div_loss,
            "history_classifier_loss": classifier_loss,
            "history_classifier_accuracy": classifier_accuracy,
            "history_classifier_valid_fraction": classifier_valid_fraction,
            "current_frame_corrupt_action_loss": current_frame_corrupt_action_loss,
            "current_frame_corrupt_sample_rate": use_corrupt,
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
    # ``loss`` is the total (action + diversity); corruption metrics show how
    # often the single forward used a corrupted-current observation.
    info = {
        "loss": loss,
        "action_loss": aux["action_loss"],
        "normalized_action_loss": aux["normalized_action_loss"],
        "diversity_loss": aux["diversity_loss"],
        "diversity_weight": jnp.asarray(diversity_weight, dtype=jnp.float32),
        "action_loss_weight": jnp.asarray(action_loss_weight, dtype=jnp.float32),
        "history_classifier_loss": aux["history_classifier_loss"],
        "history_classifier_accuracy": aux["history_classifier_accuracy"],
        "history_classifier_valid_fraction": aux["history_classifier_valid_fraction"],
        "history_classifier_weight": jnp.asarray(classifier_weight, dtype=jnp.float32),
        "current_frame_corrupt_action_loss": aux["current_frame_corrupt_action_loss"],
        "current_frame_corrupt_loss_weight": jnp.asarray(current_frame_corrupt_loss_weight, dtype=jnp.float32),
        "current_frame_corrupt_sample_prob": jnp.asarray(current_frame_corrupt_sample_prob, dtype=jnp.float32),
        "current_frame_corrupt_sample_rate": aux["current_frame_corrupt_sample_rate"],
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
    info.update(memory_branch_activation_metrics(aux["encoder_auxes"]))
    info.update(
        history_memory_collapse_metrics({"encoder": {"history_mem": aux["history_mem"]}})
    )

    return new_state, info


def _episode_split_indices(dataset, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Split a Pi0Mem dataset by episode, preventing temporal leakage."""
    current = dataset
    hf_dataset = None
    sample_indices = None
    while current is not None:
        if sample_indices is None:
            sample_indices = getattr(current, "sample_indices", None)
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            break
        current = getattr(current, "_dataset", None)

    if hf_dataset is None or "episode_index" not in getattr(hf_dataset, "column_names", ()):
        raise ValueError(
            "Validation splitting requires an episode_index column. "
            "A frame-level random split is intentionally not used because adjacent "
            "video frames would leak between training and validation."
        )

    episode_indices = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    if sample_indices is not None:
        episode_indices = episode_indices[np.asarray(sample_indices, dtype=np.int64)]
    if episode_indices.shape != (len(dataset),):
        raise ValueError(
            f"episode_index has shape {episode_indices.shape}, but dataset length is {len(dataset)}."
        )

    episodes = np.unique(episode_indices)
    if len(episodes) < 2:
        raise ValueError("Validation requires at least two episodes.")

    rng = np.random.default_rng(seed)
    shuffled_episodes = rng.permutation(episodes)
    num_val_episodes = min(max(1, round(len(episodes) * val_ratio)), len(episodes) - 1)
    val_episodes = shuffled_episodes[:num_val_episodes]
    val_mask = np.isin(episode_indices, val_episodes)
    train_indices = np.flatnonzero(~val_mask).tolist()
    val_indices = np.flatnonzero(val_mask).tolist()
    return train_indices, val_indices


def _filter_memory_classifier_frame_range(
    dataset,
    indices: list[int],
    classifier_config: _config.ShellgameMemoryClassifierConfig,
) -> list[int]:
    """Keep only rows carrying meaningful labels for classification-only runs."""
    if not classifier_config.enabled or classifier_config.action_loss_weight != 0.0:
        return indices
    current = dataset
    hf_dataset = None
    sample_indices = None
    while current is not None:
        if sample_indices is None:
            sample_indices = getattr(current, "sample_indices", None)
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            break
        current = getattr(current, "_dataset", None)
    if hf_dataset is None or "frame_index" not in getattr(hf_dataset, "column_names", ()):
        raise ValueError("Classification-only frame filtering requires a frame_index column.")
    frame_indices = np.asarray(hf_dataset["frame_index"], dtype=np.int64)
    if sample_indices is not None:
        frame_indices = frame_indices[np.asarray(sample_indices, dtype=np.int64)]
    selected = np.asarray(indices, dtype=np.int64)
    keep = (
        (frame_indices[selected] >= classifier_config.min_frame_index)
        & (frame_indices[selected] <= classifier_config.max_frame_index)
    )
    filtered = selected[keep].tolist()
    if not filtered:
        raise ValueError("Memory-classifier frame range selected no dataset rows.")
    return filtered


def _select_balanced_memory_classifier_indices(
    dataset,
    indices: list[int],
    classifier_config: _config.ShellgameMemoryClassifierConfig,
    class_labels_by_episode: at.Array,
    seed: int,
) -> list[int]:
    """Select a deterministic, class-balanced subset for an overfit probe."""
    samples_per_class = classifier_config.overfit_samples_per_class
    if samples_per_class <= 0:
        return indices

    current = dataset
    hf_dataset = None
    sample_indices = None
    while current is not None:
        if sample_indices is None:
            sample_indices = getattr(current, "sample_indices", None)
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            break
        current = getattr(current, "_dataset", None)
    if hf_dataset is None or "episode_index" not in getattr(hf_dataset, "column_names", ()):
        raise ValueError("Balanced overfit selection requires an episode_index column.")

    selected = np.asarray(indices, dtype=np.int64)
    episode_indices = np.asarray(hf_dataset["episode_index"], dtype=np.int64)
    if sample_indices is not None:
        episode_indices = episode_indices[np.asarray(sample_indices, dtype=np.int64)]
    episode_indices = episode_indices[selected]
    labels_by_episode = np.asarray(jax.device_get(class_labels_by_episode), dtype=np.int32)
    if np.any(episode_indices < 0) or np.any(episode_indices >= labels_by_episode.shape[0]):
        raise ValueError("Overfit subset contains an episode_index without a classifier label.")
    sample_labels = labels_by_episode[episode_indices]

    rng = np.random.default_rng(seed)
    balanced = []
    for class_index, class_name in enumerate(classifier_config.classes):
        candidates = selected[sample_labels == class_index]
        if candidates.size < samples_per_class:
            raise ValueError(
                f"Requested {samples_per_class} overfit samples for class {class_name!r}, "
                f"but only {candidates.size} are available."
            )
        balanced.extend(rng.permutation(candidates)[:samples_per_class].tolist())
    return rng.permutation(np.asarray(balanced, dtype=np.int64)).tolist()


def _make_split_torch_loaders(
    config: _config.TrainConfig,
    data_sharding: jax.sharding.Sharding,
    train_dataset,
    val_dataset,
):
    local_batch_size = config.batch_size // jax.process_count()
    if len(train_dataset) < local_batch_size or len(val_dataset) < local_batch_size:
        raise ValueError(
            "Both dataset splits must contain at least one full local batch: "
            f"train={len(train_dataset)}, val={len(val_dataset)}, "
            f"local_batch_size={local_batch_size}. Increase --val-ratio or reduce --batch-size."
        )

    train_loader = _data_loader.TorchDataLoader(
        train_dataset,
        local_batch_size=local_batch_size,
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )
    # Validation is deterministic and intentionally uses no worker prefetching.
    val_loader = _data_loader.TorchDataLoader(
        val_dataset,
        local_batch_size=local_batch_size,
        sharding=data_sharding,
        shuffle=False,
        num_workers=0,
        seed=config.seed,
    )
    return train_loader, val_loader


def create_train_val_data_loaders(
    config: _config.TrainConfig,
    data_sharding: jax.sharding.Sharding,
    class_labels_by_episode: at.Array | None = None,
):
    """Build Pi0Mem-aware train/validation loaders with episode-level splits."""
    if not 0.0 < config.val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1); got {config.val_ratio}.")

    if isinstance(config.data, _config.MultiDataConfigFactory):
        all_configs = config.data.create_all(config.assets_dirs, config.model)
        weights = config.data.weights or [1.0] * len(all_configs)
        if len(weights) != len(all_configs):
            raise ValueError(
                f"MultiDataConfigFactory has {len(all_configs)} datasets but {len(weights)} weights."
            )
        train_datasets = []
        val_datasets = []

        for index, (data_config, child) in enumerate(
            zip(all_configs, config.data.datasets, strict=True)
        ):
            if data_config.rlds_data_dir is not None:
                raise ValueError("Validation splitting is not supported for RLDS datasets.")
            dataset = _config_pi0_mem._build_pi0_mem_dataset(  # noqa: SLF001
                data_config,
                child.video_frame_config(),
                action_horizon=config.model.action_horizon,
                skip_norm_stats=False,
            )
            classifier_config = config.shellgame_memory_classifier
            if classifier_config.overfit_samples_per_class > 0:
                if class_labels_by_episode is None:
                    raise ValueError("Balanced overfit selection requires classifier labels.")
                eligible_indices = _filter_memory_classifier_frame_range(
                    dataset, list(range(len(dataset))), classifier_config
                )
                train_indices = _select_balanced_memory_classifier_indices(
                    dataset,
                    eligible_indices,
                    classifier_config,
                    class_labels_by_episode,
                    config.seed + index,
                )
                if not classifier_config.overfit_same_samples_for_validation:
                    raise ValueError(
                        "The overfit probe currently requires "
                        "overfit_same_samples_for_validation=True."
                    )
                val_indices = list(train_indices)
                logging.info(
                    "Dataset %d balanced overfit subset: %d classes x %d samples = %d",
                    index,
                    len(classifier_config.classes),
                    classifier_config.overfit_samples_per_class,
                    len(train_indices),
                )
            else:
                train_indices, val_indices = _episode_split_indices(
                    dataset, config.val_ratio, config.seed + index
                )
                train_indices = _filter_memory_classifier_frame_range(
                    dataset, train_indices, classifier_config
                )
                val_indices = _filter_memory_classifier_frame_range(
                    dataset, val_indices, classifier_config
                )
            train_datasets.append(torch.utils.data.Subset(dataset, train_indices))
            val_datasets.append(torch.utils.data.Subset(dataset, val_indices))
            logging.info(
                "Dataset %d episode split: train=%d, val=%d samples",
                index,
                len(train_indices),
                len(val_indices),
            )

        use_weights = len(set(weights)) > 1
        train_concat = _multi_data_loader.WeightedConcatDataset(
            train_datasets, weights=weights if use_weights else None
        )
        val_concat = _multi_data_loader.WeightedConcatDataset(val_datasets)
        train_torch, val_torch = _make_split_torch_loaders(
            config, data_sharding, train_concat, val_concat
        )
        if use_weights:
            index_weights = torch.tensor(
                train_concat.get_dataset_weights_for_sampler(), dtype=torch.double
            )
            train_torch = _data_loader.TorchDataLoader(
                train_concat,
                local_batch_size=config.batch_size // jax.process_count(),
                sharding=data_sharding,
                sampler=torch.utils.data.WeightedRandomSampler(
                    index_weights, num_samples=len(train_concat), replacement=True
                ),
                num_workers=config.num_workers,
                seed=config.seed,
                prefetch_factor=2 if config.num_workers > 0 else None,
            )
        return (
            _multi_data_loader.MultiDataLoaderImpl(all_configs, train_torch),
            _multi_data_loader.MultiDataLoaderImpl(all_configs, val_torch),
        )

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        raise ValueError("Validation splitting is not supported for RLDS datasets.")
    dataset = _config_pi0_mem._build_pi0_mem_dataset(  # noqa: SLF001
        data_config,
        config.data.video_frame_config(),
        action_horizon=config.model.action_horizon,
        skip_norm_stats=False,
    )
    classifier_config = config.shellgame_memory_classifier
    if classifier_config.overfit_samples_per_class > 0:
        if class_labels_by_episode is None:
            raise ValueError("Balanced overfit selection requires classifier labels.")
        eligible_indices = _filter_memory_classifier_frame_range(
            dataset, list(range(len(dataset))), classifier_config
        )
        train_indices = _select_balanced_memory_classifier_indices(
            dataset,
            eligible_indices,
            classifier_config,
            class_labels_by_episode,
            config.seed,
        )
        if not classifier_config.overfit_same_samples_for_validation:
            raise ValueError(
                "The overfit probe currently requires overfit_same_samples_for_validation=True."
            )
        val_indices = list(train_indices)
    else:
        train_indices, val_indices = _episode_split_indices(dataset, config.val_ratio, config.seed)
        train_indices = _filter_memory_classifier_frame_range(
            dataset, train_indices, classifier_config
        )
        val_indices = _filter_memory_classifier_frame_range(
            dataset, val_indices, classifier_config
        )
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    train_torch, val_torch = _make_split_torch_loaders(
        config, data_sharding, train_subset, val_subset
    )
    logging.info(
        "Episode-level dataset split: train=%d, val=%d samples (val_ratio=%s)",
        len(train_indices),
        len(val_indices),
        config.val_ratio,
    )
    return (
        _data_loader.DataLoaderImpl(data_config, train_torch),
        _data_loader.DataLoaderImpl(data_config, val_torch),
    )


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    class_labels_by_episode: at.Array | None = None,
) -> dict[str, at.Array]:
    """Run one gradient-free validation step using clean observations."""
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, actions = batch
    classifier_config = config.shellgame_memory_classifier
    classifier_on = bool(classifier_config.enabled)
    classifier_weight = float(classifier_config.loss_weight)
    action_loss_weight = float(classifier_config.action_loss_weight) if classifier_on else 1.0
    zero = jnp.asarray(0.0, dtype=jnp.float32)

    if classifier_on and action_loss_weight == 0.0:
        history_class_logits, aux = model.compute_history_classification(
            rng, observation, train=False
        )
        action_loss = zero
    else:
        chunked_loss, aux = model.compute_loss_with_memory_aux(
            rng, observation, actions, train=False
        )
        action_loss = jnp.mean(chunked_loss)
        history_class_logits = aux["history_class_logits"] if classifier_on else None

    diversity_loss = memory_diversity_loss(aux["history_mem"])
    diversity_weight = float(getattr(config.model, "diversity_weight", 0.0))
    if classifier_on:
        if class_labels_by_episode is None:
            raise ValueError("Memory classifier is enabled but no episode label table was provided.")
        if observation.episode_index is None or observation.frame_index is None:
            raise ValueError("Memory classifier requires episode_index and frame_index in every batch.")
        episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
        frame_index = jnp.asarray(observation.frame_index, dtype=jnp.int32)
        episode_valid = (episode_index >= 0) & (episode_index < class_labels_by_episode.shape[0])
        safe_episode_index = jnp.clip(episode_index, 0, class_labels_by_episode.shape[0] - 1)
        labels = class_labels_by_episode[safe_episode_index]
        valid = (
            episode_valid
            & (labels >= 0)
            & (frame_index >= classifier_config.min_frame_index)
            & (frame_index <= classifier_config.max_frame_index)
        )
        valid_f = valid.astype(jnp.float32)
        valid_count = jnp.sum(valid_f)
        safe_labels = jnp.maximum(labels, 0)
        per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(history_class_logits, safe_labels)
        classifier_loss_sum = jnp.sum(per_sample_loss * valid_f)
        classifier_correct_count = jnp.sum(
            (jnp.argmax(history_class_logits, axis=-1) == safe_labels) * valid_f
        )
        classifier_loss = classifier_loss_sum / jnp.maximum(valid_count, 1.0)
        classifier_accuracy = classifier_correct_count / jnp.maximum(valid_count, 1.0)
        classifier_valid_fraction = valid_count / valid_f.size
    else:
        classifier_loss = classifier_accuracy = classifier_valid_fraction = zero
        classifier_loss_sum = classifier_correct_count = valid_count = zero

    total_loss = (
        action_loss_weight * action_loss
        + diversity_weight * diversity_loss
        + classifier_weight * classifier_loss
    )
    return {
        "val/loss": total_loss,
        "val/action_loss": action_loss,
        "val/diversity_loss": diversity_loss,
        "val/history_classifier_loss": classifier_loss,
        "val/history_classifier_accuracy": classifier_accuracy,
        "val/history_classifier_valid_fraction": classifier_valid_fraction,
        "_val/history_classifier_loss_sum": classifier_loss_sum,
        "_val/history_classifier_correct_count": classifier_correct_count,
        "_val/history_classifier_valid_count": valid_count,
    }


def run_evaluation(
    peval_step,
    eval_rng: at.KeyArrayLike,
    val_iter,
    config: _config.TrainConfig,
    mesh: jax.sharding.Mesh,
    state: training_utils.TrainState,
):
    """Average validation metrics over ``config.eval_batches`` batches."""
    eval_infos = []
    for batch_index in range(config.eval_batches):
        batch = next(val_iter)
        batch_rng = jax.random.fold_in(eval_rng, batch_index)
        with sharding.set_mesh(mesh):
            eval_infos.append(peval_step(batch_rng, state, batch))

    stacked_infos = jax.device_get(common_utils.stack_forest(eval_infos))
    reduced = jax.tree.map(jnp.mean, stacked_infos)
    classifier_config = config.shellgame_memory_classifier
    if classifier_config.enabled:
        valid_count = float(jnp.sum(stacked_infos["_val/history_classifier_valid_count"]))
        loss_sum = float(jnp.sum(stacked_infos["_val/history_classifier_loss_sum"]))
        correct_count = float(jnp.sum(stacked_infos["_val/history_classifier_correct_count"]))
        classifier_loss = loss_sum / max(valid_count, 1.0)
        classifier_accuracy = correct_count / max(valid_count, 1.0)
        reduced["val/history_classifier_loss"] = classifier_loss
        reduced["val/history_classifier_accuracy"] = classifier_accuracy
        reduced["val/loss"] = (
            classifier_config.action_loss_weight * float(reduced["val/action_loss"])
            + float(getattr(config.model, "diversity_weight", 0.0)) * float(reduced["val/diversity_loss"])
            + classifier_config.loss_weight * classifier_loss
        )
    return {
        key: float(value)
        for key, value in reduced.items()
        if not key.startswith("_val/")
    }


def shellgame_cup_eval_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation: _model.Observation,
    *,
    num_steps: int,
):
    """Sample joint chunks from current EMA weights for task-level FK eval."""
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    return model.sample_actions(rng, observation, num_steps=num_steps)


def _load_shellgame_cup_eval_module():
    shellgame_dir = _SCRIPTS_DIR.parent / "examples" / "shellgame"
    if str(shellgame_dir) not in sys.path:
        sys.path.insert(0, str(shellgame_dir))
    return importlib.import_module("training_cup_eval")


def run_shellgame_cup_evaluation(
    psample_actions,
    evaluator,
    mesh: jax.sharding.Mesh,
    state: training_utils.TrainState,
    *,
    step: int,
) -> dict[str, float]:
    """Run fixed-noise sampling batches, then FK and cup classification."""
    action_batches = []
    for batch_index, observation, valid_size in evaluator.iter_batches():
        with sharding.set_mesh(mesh):
            normalized_actions = psample_actions(
                evaluator.sample_rng(batch_index), state, observation
            )
        action_batches.append(np.asarray(jax.device_get(normalized_actions))[:valid_size])
    metrics = evaluator.summarize(action_batches, step=step)
    if len(action_batches) != evaluator.num_batches:
        raise RuntimeError(
            f"ShellGame cup eval produced {len(action_batches)} batches; "
            f"expected {evaluator.num_batches}."
        )
    return metrics


def load_episode_class_labels(config: _config.TrainConfig) -> jax.Array | None:
    """Load a dense episode_index -> class-id lookup for the diagnostic head."""
    classifier_config = config.shellgame_memory_classifier
    if not classifier_config.enabled:
        return None
    metadata_path = Path(classifier_config.episodes_metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Memory-classifier episode metadata not found: {metadata_path}")
    class_to_index = {name: index for index, name in enumerate(classifier_config.classes)}
    records = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            episode_index = int(record["episode_index"])
            label_name = str(record[classifier_config.label_key])
            if label_name not in class_to_index:
                raise ValueError(
                    f"Unknown {classifier_config.label_key}={label_name!r} at "
                    f"{metadata_path}:{line_number}; expected {tuple(class_to_index)}"
                )
            records.append((episode_index, class_to_index[label_name]))
    if not records:
        raise ValueError(f"No episode labels found in {metadata_path}")
    labels = np.full(max(index for index, _ in records) + 1, -1, dtype=np.int32)
    for episode_index, class_index in records:
        if labels[episode_index] >= 0:
            raise ValueError(f"Duplicate episode_index={episode_index} in {metadata_path}")
        labels[episode_index] = class_index
    counts = np.bincount(labels[labels >= 0], minlength=len(class_to_index))
    logging.info(
        "Loaded %d memory-classifier labels from %s: %s",
        len(records),
        metadata_path,
        {name: int(counts[index]) for name, index in class_to_index.items()},
    )
    return jnp.asarray(labels)


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

    classifier_config = config.shellgame_memory_classifier
    if classifier_config.enabled:
        if config.model.history_classifier_num_classes != len(classifier_config.classes):
            raise ValueError(
                "history_classifier_num_classes must match shellgame_memory_classifier.classes: "
                f"{config.model.history_classifier_num_classes} != {len(classifier_config.classes)}"
            )
        if classifier_config.min_frame_index > classifier_config.max_frame_index:
            raise ValueError("Memory-classifier min_frame_index must not exceed max_frame_index.")
        if classifier_config.overfit_samples_per_class < 0:
            raise ValueError("overfit_samples_per_class must be nonnegative.")
        if (
            classifier_config.overfit_same_samples_for_validation
            and classifier_config.overfit_samples_per_class == 0
        ):
            raise ValueError(
                "overfit_same_samples_for_validation requires overfit_samples_per_class > 0."
            )
        if config.val_ratio <= 0.0:
            raise ValueError("Memory-classifier diagnostics require an episode-held-out validation split.")
    class_labels_by_episode = load_episode_class_labels(config)
    if classifier_config.enabled:
        logging.info(
            "Memory-classifier stochastic train augmentation: %s",
            not classifier_config.disable_train_augmentation,
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

    use_val = config.val_ratio > 0.0
    use_shellgame_cup_eval = config.shellgame_cup_eval.enabled
    if use_val:
        if config.eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive; got {config.eval_interval}.")
        if config.eval_batches <= 0:
            raise ValueError(f"eval_batches must be positive; got {config.eval_batches}.")
    if use_shellgame_cup_eval:
        if not use_val:
            raise ValueError("shellgame_cup_eval.enabled requires val_ratio > 0.")
        if config.shellgame_cup_eval.interval <= 0:
            raise ValueError(
                "shellgame_cup_eval.interval must be positive; "
                f"got {config.shellgame_cup_eval.interval}."
            )
        if config.shellgame_cup_eval.batch_size % jax.device_count() != 0:
            raise ValueError(
                "shellgame_cup_eval.batch_size must be divisible by the global JAX device count "
                f"({jax.device_count()}); got {config.shellgame_cup_eval.batch_size}."
            )

    # Pi0Mem-aware data loaders. The split is performed by episode rather than
    # frame so neighboring/history frames cannot appear on opposite sides.
    val_data_loader = None
    if use_val:
        data_loader, val_data_loader = create_train_val_data_loaders(
            config, data_sharding, class_labels_by_episode
        )
        val_iter = iter(val_data_loader)
        logging.info(
            "Validation enabled: val_ratio=%s, eval_interval=%d, eval_batches=%d",
            config.val_ratio,
            config.eval_interval,
            config.eval_batches,
        )
    else:
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

    shellgame_cup_evaluator = None
    if use_shellgame_cup_eval:
        cup_eval_module = _load_shellgame_cup_eval_module()
        shellgame_cup_evaluator = cup_eval_module.ShellgameCupEvaluator(
            config, config.shellgame_cup_eval
        )

    ptrain_step = jax.jit(
        functools.partial(
            train_step,
            config,
            class_labels_by_episode=class_labels_by_episode,
        ),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    if use_val:
        peval_step = jax.jit(
            functools.partial(
                eval_step,
                config,
                class_labels_by_episode=class_labels_by_episode,
            ),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=replicated_sharding,
        )
        best_val_loss = float("inf")
    if use_shellgame_cup_eval:
        pshellgame_cup_eval = jax.jit(
            functools.partial(
                shellgame_cup_eval_step,
                num_steps=config.shellgame_cup_eval.num_sampling_steps,
            ),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=data_sharding,
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    timing_stats = {
        "data_next_s": 0.0,
        "data_next_max_s": 0.0,
        "torch_next_s": 0.0,
        "torch_next_max_s": 0.0,
        "jax_shard_s": 0.0,
        "jax_shard_max_s": 0.0,
        "obs_from_dict_s": 0.0,
        "obs_from_dict_max_s": 0.0,
        "log_sync_s": 0.0,
        "loop_wall_s": 0.0,
        "count": 0,
    }

    for step in pbar:
        loop_start = time.perf_counter()
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        data_start = time.perf_counter()
        batch = next(data_iter)
        data_next_s = time.perf_counter() - data_start
        timing_stats["data_next_s"] += data_next_s
        timing_stats["data_next_max_s"] = max(timing_stats["data_next_max_s"], data_next_s)
        if hasattr(data_loader, "pop_timing"):
            loader_timing = data_loader.pop_timing()
            for key, value in loader_timing.items():
                if key.endswith("_max_s"):
                    timing_stats[key] = max(timing_stats.get(key, 0.0), value)
                else:
                    timing_stats[key] = timing_stats.get(key, 0.0) + value

        if step % config.log_interval == 0:
            log_start = time.perf_counter()
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info = {k: round(float(v), 6) for k, v in reduced_info.items()}
            log_sync_s = time.perf_counter() - log_start
            timing_stats["log_sync_s"] += log_sync_s

            timing_stats["loop_wall_s"] += time.perf_counter() - loop_start
            timing_stats["count"] += 1
            if config.log_perf_metrics:
                timing_count = max(timing_stats["count"], 1)
                perf_info = {}
                for key in sorted(timing_stats):
                    if key == "count":
                        continue
                    metric_key = f"perf/{key}"
                    if key.endswith("_max_s") or key == "log_sync_s":
                        perf_info[metric_key] = round(timing_stats[key], 6)
                    else:
                        perf_info[metric_key] = round(timing_stats[key] / timing_count, 6)
                reduced_info.update(perf_info)
            timing_stats = dict.fromkeys(timing_stats, 0.0)

            info_str = ", ".join(f"{k}={v:.6f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        else:
            timing_stats["loop_wall_s"] += time.perf_counter() - loop_start
            timing_stats["count"] += 1

        if use_val and step % config.eval_interval == 0 and step > start_step:
            eval_rng = jax.random.fold_in(train_rng, step)
            val_metrics = run_evaluation(
                peval_step, eval_rng, val_iter, config, mesh, train_state
            )
            val_str = ", ".join(f"{key}={value:.6f}" for key, value in val_metrics.items())
            pbar.write(f"Step {step} [eval]: {val_str}")
            wandb.log(val_metrics, step=step)
            if val_metrics["val/loss"] < best_val_loss:
                best_val_loss = val_metrics["val/loss"]
                pbar.write(f"Step {step}: new best val/loss={best_val_loss:.6f}")

        if (
            use_shellgame_cup_eval
            and step % config.shellgame_cup_eval.interval == 0
            and step > start_step
        ):
            cup_metrics = run_shellgame_cup_evaluation(
                pshellgame_cup_eval,
                shellgame_cup_evaluator,
                mesh,
                train_state,
                step=step,
            )
            cup_str = ", ".join(f"{key}={value:.6f}" for key, value in cup_metrics.items())
            pbar.write(f"Step {step} [cup eval]: {cup_str}")
            wandb.log(cup_metrics, step=step)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    if use_val:
        logging.info("Running final validation...")
        final_metrics = run_evaluation(
            peval_step,
            jax.random.fold_in(train_rng, config.num_train_steps),
            val_iter,
            config,
            mesh,
            train_state,
        )
        wandb.log(final_metrics, step=config.num_train_steps)
        logging.info("Final validation: %s; best val/loss=%.6f", final_metrics, best_val_loss)

    if use_shellgame_cup_eval:
        logging.info("Running final ShellGame cup-selection validation...")
        final_cup_metrics = run_shellgame_cup_evaluation(
            pshellgame_cup_eval,
            shellgame_cup_evaluator,
            mesh,
            train_state,
            step=config.num_train_steps,
        )
        wandb.log(final_cup_metrics, step=config.num_train_steps)
        logging.info("Final ShellGame cup validation: %s", final_cup_metrics)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if shellgame_cup_evaluator is not None:
        shellgame_cup_evaluator.close()


if __name__ == "__main__":
    main(_config.cli())

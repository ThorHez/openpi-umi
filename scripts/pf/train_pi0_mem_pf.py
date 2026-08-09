"""Pi0MemPF (Past-Future Temporal Bottleneck) training entry point.

Mirror of :mod:`scripts.mem.train_pi0_mem_compress`, adapted for the
prior/posterior dual-branch objective of :class:`openpi.models.pi0_mem_pf`:

* The model paradigm check accepts a :class:`Pi0MemPFConfig`.
* ``train_step`` routes the loss through
  ``Pi0MemPF.compute_loss_with_pf_aux`` and composes the full PF objective

      L = lambda_prior * L_prior + lambda_post * L_post
        + lambda_align * L_align + lambda_reg * L_reg
        + diversity_weight * L_div(Hmem)

  where L_prior / L_post are the two flow-matching action losses (shared
  noise / time draw), L_align pulls Zprior toward stop_gradient(Zpost) and
  L_reg bounds the latent scales.
* Monitoring adds the future-side twins of every compress metric: future
  gate statistics, Zpost / Zprior collapse metrics, prior-posterior latent
  cosine, and gradient norms for the UTR future path and the Future Latent
  Prior Encoder.

The data pipeline is the shared Pi0Mem one
(``openpi.training.config_pi0_mem.create_pi0_mem_data_loader``); the only
requirement is that the data factory is configured with
``num_future_frames > 0`` so ``VideoFrameDataset`` appends future frames
after the current one (clip layout ``[oldest_past ... current, future...]``).

Launch with the standard tyro CLI::

    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/mem/train_pi0_mem_pf.py \\
        pi0_mem_pf_umi_32d_30k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322 \\
        --exp_name=run_pf_T16F8 \\
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

# Pi0MemPF-specific imports.
import openpi.models.pi0_mem_pf as pi0_mem_pf
import openpi.training.config_pi0_mem as _config_pi0_mem

# Reuse train.py helpers verbatim. Add openpi-umi/scripts to sys.path so we
# can import scripts/train.py as a top-level module.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train import _load_weights_and_validate, init_logging, init_wandb  # noqa: E402

# Reuse the compress-side helpers that are branch-agnostic.
from mem.train_pi0_mem_compress import (  # noqa: E402
    _copy_observation_with_images,
    _leaf_array,
    _tree_leaves_with_paths,
    apply_current_frame_corruption,
    memory_diversity_loss,
)


# ---------------------------------------------------------------------------
# PF training metrics (gradient health + latent diversity + alignment).
# ---------------------------------------------------------------------------


def _global_l2_norm(leaves):
    """Global L2 norm over a list of arrays (0.0 when empty; shape-stable)."""
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    sq_sum = sum(jnp.sum(jnp.square(jnp.asarray(_leaf_array(l), dtype=jnp.float32))) for l in leaves)
    return jnp.sqrt(sq_sum)


def _is_temporal_gate_path(path) -> bool:
    """Match BOTH history_memory_gate_logit and future_memory_gate_logit."""
    try:
        path_str = jax.tree_util.keystr(path)
    except Exception:
        path_str = str(path)
    return "memory_gate_logit" in path_str


def gate_param_metrics(params, needle: str, prefix: str):
    """Statistics of one gate-logit family (``history_...`` or ``future_...``)."""
    leaves = [
        _leaf_array(leaf)
        for path, leaf in _tree_leaves_with_paths(params)
        if needle in path
    ]
    if not leaves:
        nan = jnp.asarray(jnp.nan, dtype=jnp.float32)
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return {
            f"{prefix}/count": zero,
            f"{prefix}/logit_mean": nan,
            f"{prefix}/logit_min": nan,
            f"{prefix}/logit_max": nan,
            f"{prefix}/sigmoid_mean": nan,
            f"{prefix}/sigmoid_min": nan,
            f"{prefix}/sigmoid_max": nan,
        }

    logits = jnp.concatenate([jnp.ravel(jnp.asarray(x, dtype=jnp.float32)) for x in leaves])
    gates = jax.nn.sigmoid(logits)
    return {
        f"{prefix}/count": jnp.asarray(logits.size, dtype=jnp.float32),
        f"{prefix}/logit_mean": jnp.mean(logits),
        f"{prefix}/logit_min": jnp.min(logits),
        f"{prefix}/logit_max": jnp.max(logits),
        f"{prefix}/sigmoid_mean": jnp.mean(gates),
        f"{prefix}/sigmoid_min": jnp.min(gates),
        f"{prefix}/sigmoid_max": jnp.max(gates),
    }


def gate_config_metrics(config: _config.TrainConfig):
    """Expose fixed-gate config so raw sigmoid(logit) metrics are not misread."""
    history_fixed = getattr(config.model, "history_gate_fixed", None)
    future_fixed = getattr(config.model, "future_gate_fixed", None)

    def _one(value):
        if value is None:
            return (
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(jnp.nan, dtype=jnp.float32),
            )
        return (
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray(float(value), dtype=jnp.float32),
        )

    hist_is_fixed, hist_value = _one(history_fixed)
    fut_is_fixed, fut_value = _one(future_fixed)
    return {
        "gate_history/is_fixed": hist_is_fixed,
        "gate_history/fixed_value": hist_value,
        "gate_future/is_fixed": fut_is_fixed,
        "gate_future/fixed_value": fut_value,
    }


def new_pf_param_grad_metrics(grads):
    """Gradient norms for the PF-specific parameter groups.

    Groups:
    - Unified Temporal Resampler: shared core, past queries, future queries,
      direction projections.
    - Per-block gated branches: History* / Future* cross-attn + out-proj +
      gate logits.
    - Future Latent Prior Encoder and (optional) alignment projections.
    """
    path_leaves = list(_tree_leaves_with_paths(grads))

    def select(*needles):
        selected = []
        for path, leaf in path_leaves:
            if all(n in path for n in needles):
                selected.append(leaf)
        return selected

    utr = select("UTR_0")
    resampler_core = select("UTR_0", "ResamplerCore_0")
    memory_queries = select("UTR_0", "memory_queries")
    future_queries = select("UTR_0", "future_queries")
    past_out_proj = select("UTR_0", "past_out_proj")
    future_out_proj = select("UTR_0", "future_out_proj")

    history_gate = select("history_memory_gate_logit")
    history_cross_attn = select("HistoryMultiHeadDotProductAttention_0")
    history_out_proj = select("HistoryOutProj")
    future_gate = select("future_memory_gate_logit")
    future_cross_attn = select("FutureMultiHeadDotProductAttention_0")
    future_block_out_proj = select("FutureOutProj")

    prior_encoder = select("FuturePrior")
    prior_queries = select("FuturePrior", "prior_queries")
    align_proj = select("align_proj")

    return {
        "grad/utr_total_l2": _global_l2_norm(utr),
        "grad/resampler_core_l2": _global_l2_norm(resampler_core),
        "grad/memory_queries_l2": _global_l2_norm(memory_queries),
        "grad/future_queries_l2": _global_l2_norm(future_queries),
        "grad/past_out_proj_l2": _global_l2_norm(past_out_proj),
        "grad/future_out_proj_l2": _global_l2_norm(future_out_proj),
        "grad/history_gate_l2": _global_l2_norm(history_gate),
        "grad/history_cross_attn_l2": _global_l2_norm(history_cross_attn),
        "grad/history_out_proj_l2": _global_l2_norm(history_out_proj),
        "grad/future_gate_l2": _global_l2_norm(future_gate),
        "grad/future_cross_attn_l2": _global_l2_norm(future_cross_attn),
        "grad/future_block_out_proj_l2": _global_l2_norm(future_block_out_proj),
        "grad/prior_encoder_l2": _global_l2_norm(prior_encoder),
        "grad/prior_queries_l2": _global_l2_norm(prior_queries),
        "grad/align_proj_l2": _global_l2_norm(align_proj),
    }


def latent_collapse_metrics(latent, prefix: str, eps=1e-6):
    """Collapse / diversity monitors for one ``[B, M, D]`` latent family.

    Same statistics as the compress-side ``history_memory_collapse_metrics``
    (raw + centered pairwise cosine, raw + centered variance, token norms) but
    with a configurable metric prefix so Hmem / Zpost / Zprior can be logged
    side by side.
    """
    latent = jnp.asarray(latent, dtype=jnp.float32)  # [B, M, D]
    b, m, _ = latent.shape

    token_norm = jnp.linalg.norm(latent, axis=-1)  # [B, M]
    centered = latent - jnp.mean(latent, axis=1, keepdims=True)
    token_var = jnp.mean(jnp.var(latent, axis=1))
    centered_var = jnp.mean(jnp.square(centered))

    if m <= 1:
        offdiag_mean = jnp.asarray(0.0, dtype=jnp.float32)
        offdiag_max = jnp.asarray(0.0, dtype=jnp.float32)
        centered_offdiag_mean = jnp.asarray(0.0, dtype=jnp.float32)
        centered_offdiag_max = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        eye = jnp.eye(m, dtype=jnp.bool_)[None, :, :]
        denom = b * m * (m - 1)

        normed = latent / (jnp.linalg.norm(latent, axis=-1, keepdims=True) + eps)
        sim = jnp.einsum("bmd,bnd->bmn", normed, normed)
        offdiag_abs = jnp.where(eye, 0.0, jnp.abs(sim))
        offdiag_mean = jnp.sum(offdiag_abs) / jnp.maximum(denom, 1)
        offdiag_max = jnp.max(offdiag_abs)

        z = centered / (jnp.linalg.norm(centered, axis=-1, keepdims=True) + eps)
        sim_c = jnp.einsum("bmd,bnd->bmn", z, z)
        offdiag_c_abs = jnp.where(eye, 0.0, jnp.abs(sim_c))
        centered_offdiag_mean = jnp.sum(offdiag_c_abs) / jnp.maximum(denom, 1)
        centered_offdiag_max = jnp.max(offdiag_c_abs)

    token_mean = jnp.mean(latent, axis=1)  # [B, D]
    batch_var = jnp.mean(jnp.var(token_mean, axis=0))
    if b <= 1:
        inter_sample_cosine = jnp.asarray(0.0, dtype=jnp.float32)
    else:
        sample_normed = token_mean / (jnp.linalg.norm(token_mean, axis=-1, keepdims=True) + eps)
        sample_sim = jnp.einsum("bd,ed->be", sample_normed, sample_normed)
        sample_eye = jnp.eye(b, dtype=jnp.bool_)
        sample_offdiag = jnp.where(sample_eye, 0.0, jnp.abs(sample_sim))
        inter_sample_cosine = jnp.sum(sample_offdiag) / jnp.maximum(b * (b - 1), 1)

    return {
        f"{prefix}/norm_mean": jnp.mean(token_norm),
        f"{prefix}/norm_std": jnp.std(token_norm),
        f"{prefix}/token_var_mean": token_var,
        f"{prefix}/token_centered_var": centered_var,
        f"{prefix}/cosine_offdiag_mean": offdiag_mean,
        f"{prefix}/cosine_offdiag_max": offdiag_max,
        f"{prefix}/centered_cosine_offdiag_mean": centered_offdiag_mean,
        f"{prefix}/centered_cosine_offdiag_max": centered_offdiag_max,
        f"{prefix}/batch_var_mean": batch_var,
        f"{prefix}/inter_sample_cosine_mean": inter_sample_cosine,
    }


def prior_posterior_alignment_metrics(z_prior, z_post, eps=1e-6):
    """How close the predicted Zprior is to the teacher Zpost.

    - ``align/latent_cosine_mean``: per-token cosine between matching tokens
      of Zprior and Zpost (1.0 = perfectly aligned directions).
    - ``align/latent_l2_mean``: mean per-token L2 distance.
    - ``align/post_to_prior_norm_ratio``: scale mismatch indicator.
    """
    z_prior = jnp.asarray(z_prior, dtype=jnp.float32)
    z_post = jnp.asarray(z_post, dtype=jnp.float32)

    p = z_prior / (jnp.linalg.norm(z_prior, axis=-1, keepdims=True) + eps)
    q = z_post / (jnp.linalg.norm(z_post, axis=-1, keepdims=True) + eps)
    cosine = jnp.sum(p * q, axis=-1)  # [B, M]
    shuffled_cosine = jnp.sum(p * jnp.roll(q, shift=1, axis=0), axis=-1)

    l2 = jnp.linalg.norm(z_prior - z_post, axis=-1)
    prior_norm = jnp.mean(jnp.linalg.norm(z_prior, axis=-1))
    post_norm = jnp.mean(jnp.linalg.norm(z_post, axis=-1))
    cosine_mean = jnp.mean(cosine)
    shuffled_cosine_mean = jnp.mean(shuffled_cosine)

    return {
        "align/latent_cosine_mean": cosine_mean,
        "align/latent_cosine_std": jnp.std(cosine),
        "align/latent_cosine_min": jnp.min(cosine),
        "align/shuffled_latent_cosine_mean": shuffled_cosine_mean,
        "align/latent_cosine_margin": cosine_mean - shuffled_cosine_mean,
        "align/latent_l2_mean": jnp.mean(l2),
        "align/post_to_prior_norm_ratio": post_norm / (prior_norm + eps),
    }


def temporal_branch_activation_metrics(encoder_auxes, prefix: str = "memory", eps: float = 1e-6):
    """Residual-magnitude monitors for BOTH gated branches (history + future).

    Extends the compress-side ``memory_branch_activation_metrics`` with the
    future branch: on active memory layers, compare each branch's residual
    against the spatial attention output.
    """
    mem_norms, fut_norms, spatial_norms = [], [], []
    hist_gate_values, fut_gate_values = [], []
    mem_active, fut_active = [], []

    for encoder_aux in encoder_auxes:
        for name, block_out in encoder_aux.items():
            if not name.startswith("block") or "mem_update" not in block_out:
                continue

            y_spatial = jnp.asarray(block_out["y_spatial"], dtype=jnp.float32)
            spatial_norm = jnp.mean(jnp.linalg.norm(y_spatial, axis=-1))

            mem_update = jnp.asarray(block_out["mem_update"], dtype=jnp.float32)
            mem_norm = jnp.mean(jnp.linalg.norm(mem_update, axis=-1))
            mem_on = (mem_norm > eps).astype(jnp.float32)

            fut_update = jnp.asarray(block_out.get("fut_update", jnp.zeros_like(mem_update)), dtype=jnp.float32)
            fut_norm = jnp.mean(jnp.linalg.norm(fut_update, axis=-1))
            fut_on = (fut_norm > eps).astype(jnp.float32)

            spatial_norms.append(spatial_norm)
            mem_norms.append(mem_norm)
            fut_norms.append(fut_norm)
            hist_gate_values.append(jnp.asarray(block_out["history_gate"], dtype=jnp.float32))
            fut_gate_values.append(jnp.asarray(block_out.get("future_gate", 0.0), dtype=jnp.float32))
            mem_active.append(mem_on)
            fut_active.append(fut_on)

    zero = jnp.asarray(0.0, dtype=jnp.float32)
    if not mem_norms:
        metrics = {
            f"{prefix}/mem_update_norm": zero,
            f"{prefix}/fut_update_norm": zero,
            f"{prefix}/y_spatial_norm": zero,
            f"{prefix}/mem_update_to_spatial_ratio": zero,
            f"{prefix}/fut_update_to_spatial_ratio": zero,
            f"{prefix}/gate_history_effective_mean": zero,
            f"{prefix}/gate_future_effective_mean": zero,
        }
        if prefix == "memory":
            metrics.update(
                {
                    "gate_history/effective_mean": zero,
                    "gate_future/effective_mean": zero,
                }
            )
        return metrics

    mem_w = jnp.stack(mem_active)
    fut_w = jnp.stack(fut_active)
    mem_denom = jnp.maximum(jnp.sum(mem_w), 1.0)
    fut_denom = jnp.maximum(jnp.sum(fut_w), 1.0)

    spatial = jnp.stack(spatial_norms)
    mem_norm = jnp.sum(jnp.stack(mem_norms) * mem_w) / mem_denom
    fut_norm = jnp.sum(jnp.stack(fut_norms) * fut_w) / fut_denom
    spatial_mem = jnp.sum(spatial * mem_w) / mem_denom
    spatial_fut = jnp.sum(spatial * fut_w) / fut_denom
    hist_gate_mean = jnp.sum(jnp.stack(hist_gate_values) * mem_w) / mem_denom
    fut_gate_mean = jnp.sum(jnp.stack(fut_gate_values) * fut_w) / fut_denom

    metrics = {
        f"{prefix}/mem_update_norm": mem_norm,
        f"{prefix}/fut_update_norm": fut_norm,
        f"{prefix}/y_spatial_norm": spatial_mem,
        f"{prefix}/mem_update_to_spatial_ratio": mem_norm / (spatial_mem + eps),
        f"{prefix}/fut_update_to_spatial_ratio": fut_norm / (spatial_fut + eps),
        f"{prefix}/gate_history_effective_mean": hist_gate_mean,
        f"{prefix}/gate_future_effective_mean": fut_gate_mean,
    }
    if prefix == "memory":
        metrics.update(
            {
                "gate_history/effective_mean": hist_gate_mean,
                "gate_future/effective_mean": fut_gate_mean,
            }
        )
    return metrics


def frame_validity_metrics(observation: _model.Observation, num_frames: int, num_future_frames: int):
    """Frame-level validity monitors from VideoFrameDataset metadata."""
    zero = jnp.asarray(0.0, dtype=jnp.float32)
    one = jnp.asarray(1.0, dtype=jnp.float32)
    if observation.frame_valid_masks is None:
        return {
            "data/history_valid_rate": one,
            "data/future_valid_rate": one if num_future_frames > 0 else zero,
            "data/future_padded_rate": zero,
            "data/full_future_sample_rate": one if num_future_frames > 0 else zero,
        }

    masks = [jnp.asarray(mask, dtype=jnp.float32) for mask in observation.frame_valid_masks.values()]
    if not masks:
        return {
            "data/history_valid_rate": one,
            "data/future_valid_rate": one if num_future_frames > 0 else zero,
            "data/future_padded_rate": zero,
            "data/full_future_sample_rate": one if num_future_frames > 0 else zero,
        }

    stacked = jnp.stack(masks, axis=0)  # [streams, B, T]
    history_valid = stacked[..., :num_frames]
    history_valid_rate = jnp.mean(history_valid)
    if num_future_frames <= 0:
        future_valid_rate = zero
        full_future_sample_rate = zero
    else:
        future_valid = stacked[..., num_frames : num_frames + num_future_frames]
        future_valid_rate = jnp.mean(future_valid)
        full_future_sample_rate = jnp.mean(jnp.all(future_valid > 0.5, axis=(0, 2)).astype(jnp.float32))
    return {
        "data/history_valid_rate": history_valid_rate,
        "data/future_valid_rate": future_valid_rate,
        "data/future_padded_rate": one - future_valid_rate,
        "data/full_future_sample_rate": full_future_sample_rate,
    }


def horizon_loss_metrics(loss_prior_by_t, loss_post_by_t, eps: float = 1e-6):
    """Compact per-horizon monitors without logging every action step."""
    prior = jnp.asarray(loss_prior_by_t, dtype=jnp.float32)
    post = jnp.asarray(loss_post_by_t, dtype=jnp.float32)
    horizon = prior.shape[-1]
    first = 0
    mid = horizon // 2
    last = horizon - 1
    return {
        "loss_horizon/prior_first": jnp.mean(prior[..., first]),
        "loss_horizon/prior_mid": jnp.mean(prior[..., mid]),
        "loss_horizon/prior_last": jnp.mean(prior[..., last]),
        "loss_horizon/post_first": jnp.mean(post[..., first]),
        "loss_horizon/post_mid": jnp.mean(post[..., mid]),
        "loss_horizon/post_last": jnp.mean(post[..., last]),
        "loss_horizon/advantage_first": jnp.mean(prior[..., first] - post[..., first]),
        "loss_horizon/advantage_mid": jnp.mean(prior[..., mid] - post[..., mid]),
        "loss_horizon/advantage_last": jnp.mean(prior[..., last] - post[..., last]),
        "loss_horizon/advantage_ratio": jnp.mean((prior - post) / (prior + eps)),
    }


# ---------------------------------------------------------------------------
# Optimizer: optional LR multiplier on BOTH temporal gate logits.
# ---------------------------------------------------------------------------


def _make_gate_mask(params: nnx.State):
    """Boolean pytree selecting history AND future ``*_memory_gate_logit`` leaves."""
    return jax.tree_util.tree_map_with_path(
        lambda path, _leaf: _is_temporal_gate_path(path),
        params,
    )


def _gate_lr_multiplier(config: _config.TrainConfig) -> float:
    """Configured gate LR multiplier, with env override for quick sweeps."""
    env_value = os.environ.get("OPENPI_HISTORY_GATE_LR_MULTIPLIER")
    if env_value is not None:
        return float(env_value)
    return float(getattr(config.model, "history_gate_lr_multiplier", 1.0))


def _create_optimizer_for_trainable_params(
    config: _config.TrainConfig,
    trainable_params: nnx.State,
) -> optax.GradientTransformation:
    """Normal optimizer, optionally scaling the temporal-gate updates."""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    gate_lr_multiplier = _gate_lr_multiplier(config)
    if gate_lr_multiplier == 1.0:
        return tx

    gate_mask = _make_gate_mask(trainable_params)
    return optax.chain(
        tx,
        optax.masked(optax.scale(gate_lr_multiplier), gate_mask),
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """Pi0MemPF train-state init with optional gate LR multiplier."""

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
    gate_mask = _make_gate_mask(trainable_params_shape)
    trainable_gate_leaves = sum(bool(x) for x in jax.tree.leaves(gate_mask))
    gate_lr_multiplier = _gate_lr_multiplier(config)

    logging.info("=" * 60)
    logging.info("FREEZE FILTER ANALYSIS:")
    logging.info(f"  Total params:     {total_count:,} ({total_count/1e6:.2f}M)")
    logging.info(f"  Frozen params:    {frozen_count:,} ({frozen_count/total_count*100:.1f}%)")
    logging.info(f"  Trainable params: {trainable_count:,} ({trainable_count/1e6:.2f}M)")
    logging.info(
        "  Trainable temporal gate leaves: %d; gate_lr_multiplier=%.2f",
        trainable_gate_leaves,
        gate_lr_multiplier,
    )
    logging.info("=" * 60)

    trainable_flat = traverse_util.flatten_dict(trainable_params_shape.to_pure_dict())
    gate_paths = [
        "/".join(k)
        for k in trainable_flat
        if "memory_gate_logit" in "/".join(k)
    ]
    if gate_paths:
        logging.info("Trainable temporal gate paths:")
        for path in gate_paths:
            logging.info("  - %s", path)
    elif gate_lr_multiplier != 1.0:
        logging.warning(
            "gate_lr_multiplier=%.2f but no trainable *_memory_gate_logit "
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
# Local train_step: composes the full PF objective and logs branch health.
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

    # All loss weights are Python-static config fields, so the branches below
    # resolve at trace time and disabled terms never enter the HLO graph.
    lambda_prior = float(getattr(config.model, "lambda_prior", 1.0))
    lambda_post = float(getattr(config.model, "lambda_post", 1.0))
    lambda_align = float(getattr(config.model, "lambda_align", 1.0))
    lambda_reg = float(getattr(config.model, "lambda_reg", 0.0))
    run_posterior = lambda_post > 0.0 or lambda_align > 0.0 or lambda_reg > 0.0

    diversity_weight = float(getattr(config.model, "diversity_weight", 0.0))
    diversity_on = diversity_weight > 0.0
    current_frame_dropout_prob = float(getattr(config.model, "current_frame_dropout_prob", 0.0))
    current_frame_mask_prob = float(getattr(config.model, "current_frame_mask_prob", 0.0))
    current_frame_corruption_on = current_frame_dropout_prob > 0.0 or current_frame_mask_prob > 0.0
    current_frame_corrupt_sample_prob = float(getattr(config.model, "current_frame_corrupt_sample_prob", 0.0))
    current_frame_corrupt_sample_on = current_frame_corrupt_sample_prob > 0.0 and current_frame_corruption_on
    # The current frame is NOT the last frame when future frames are appended
    # to the clip, so use the resolved (positive) index for corruption.
    current_frame_index = int(config.model.resolved_current_frame_index)

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

        chunked_prior, aux = model.compute_loss_with_pf_aux(
            loss_rng, observation, actions, train=True, run_posterior=run_posterior
        )
        loss_prior = jnp.mean(chunked_prior)
        loss_post = jnp.mean(aux["loss_post"])
        align_loss = aux["align_loss"]
        reg_loss = aux["reg_loss"]

        if diversity_on:
            div_loss = memory_diversity_loss(aux["history_mem"])
        else:
            div_loss = jnp.asarray(0.0, dtype=loss_prior.dtype)

        total_loss = (
            lambda_prior * loss_prior
            + lambda_post * loss_post
            + lambda_align * align_loss
            + lambda_reg * reg_loss
            + diversity_weight * div_loss
        )

        aux_out = {
            "loss_prior": loss_prior,
            "loss_post": loss_post,
            "loss_prior_by_t": chunked_prior,
            "loss_post_by_t": aux["loss_post"],
            "align_loss": align_loss,
            "reg_loss": reg_loss,
            "diversity_loss": div_loss,
            "history_mem": aux["history_mem"],
            "future_post": aux["future_post"],
            "future_prior": aux["future_prior"],
            "encoder_auxes": aux["encoder_auxes"],
            "post_encoder_auxes": aux["post_encoder_auxes"],
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
    info = {
        "loss": loss,
        "loss_prior": aux["loss_prior"],
        "loss_post": aux["loss_post"],
        "loss_post_minus_prior": aux["loss_post"] - aux["loss_prior"],
        "posterior_advantage": aux["loss_prior"] - aux["loss_post"],
        "posterior_advantage_ratio": (aux["loss_prior"] - aux["loss_post"]) / (aux["loss_prior"] + 1e-6),
        "prior_to_post_ratio": aux["loss_prior"] / (aux["loss_post"] + 1e-6),
        "align_loss": aux["align_loss"],
        "reg_loss": aux["reg_loss"],
        "reg/prior_norm_sq_mean": jnp.mean(jnp.square(jnp.asarray(aux["future_prior"], dtype=jnp.float32))),
        "reg/post_norm_sq_mean": jnp.mean(jnp.square(jnp.asarray(aux["future_post"], dtype=jnp.float32))),
        "diversity_loss": aux["diversity_loss"],
        "diversity_zpost_loss": memory_diversity_loss(aux["future_post"]),
        "loss_weighted/prior": lambda_prior * aux["loss_prior"],
        "loss_weighted/post": lambda_post * aux["loss_post"],
        "loss_weighted/align": lambda_align * aux["align_loss"],
        "loss_weighted/reg": lambda_reg * aux["reg_loss"],
        "loss_weighted/diversity": diversity_weight * aux["diversity_loss"],
        "lambda_prior": jnp.asarray(lambda_prior, dtype=jnp.float32),
        "lambda_post": jnp.asarray(lambda_post, dtype=jnp.float32),
        "lambda_align": jnp.asarray(lambda_align, dtype=jnp.float32),
        "lambda_reg": jnp.asarray(lambda_reg, dtype=jnp.float32),
        "diversity_weight": jnp.asarray(diversity_weight, dtype=jnp.float32),
        "current_frame_corrupt_sample_rate": aux["current_frame_corrupt_sample_rate"],
        "current_frame_dropout_rate": aux["current_frame_dropout_rate"],
        "current_frame_mask_rate": aux["current_frame_mask_rate"],
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }

    # PF health monitors. All latent tensors are stacked per image stream
    # ([B * num_streams, M, D]) and shape-stable even when a direction is
    # disabled (zeros), so the jit cache stays warm.
    info.update(new_pf_param_grad_metrics(grads))
    info.update(gate_param_metrics(new_params, "history_memory_gate_logit", "gate_history"))
    info.update(gate_param_metrics(new_params, "future_memory_gate_logit", "gate_future"))
    info.update(gate_config_metrics(config))
    info.update(temporal_branch_activation_metrics(aux["encoder_auxes"], "memory"))
    info.update(temporal_branch_activation_metrics(aux["post_encoder_auxes"], "memory_post"))
    info.update(latent_collapse_metrics(aux["history_mem"], "memory/hist"))
    info.update(latent_collapse_metrics(aux["future_post"], "memory/zpost"))
    info.update(latent_collapse_metrics(aux["future_prior"], "memory/zprior"))
    info.update(prior_posterior_alignment_metrics(aux["future_prior"], aux["future_post"]))
    info.update(horizon_loss_metrics(aux["loss_prior_by_t"], aux["loss_post_by_t"]))
    info.update(frame_validity_metrics(observation, config.model.num_frames, config.model.num_future_frames))

    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # === Pi0MemPF paradigm sanity checks ===
    if not isinstance(config.model, pi0_mem_pf.Pi0MemPFConfig):
        raise ValueError(
            f"train_pi0_mem_pf requires a Pi0MemPFConfig model; got "
            f"{type(config.model).__name__}. Use scripts/mem/train_pi0_mem_compress.py "
            "for the history-only Pi0MemCompress variant."
        )
    # The model owns the PF temporal sampling layout. Data factories only own
    # dataset-specific details; all frame/stride fields are injected here so
    # multi-dataset PF training has exactly one temporal source.
    def _with_model_frame_layout(factory, label):
        if not hasattr(factory, "video_frame_config"):
            raise ValueError(
                f"train_pi0_mem_pf requires Pi0Mem-aware DataConfigFactory "
                f"(must expose .video_frame_config()); {label} is {type(factory).__name__}."
            )
        field_names = {field.name for field in dataclasses.fields(factory)}
        missing = {"num_frames", "frame_stride", "num_future_frames", "future_frame_stride"} - field_names
        if missing:
            raise ValueError(
                f"{label} is Pi0Mem-aware but cannot accept PF temporal field(s): "
                f"{sorted(missing)}."
            )
        return dataclasses.replace(
            factory,
            num_frames=config.model.num_frames,
            frame_stride=config.model.frame_stride,
            num_future_frames=config.model.num_future_frames,
            future_frame_stride=config.model.future_frame_stride,
        )

    if isinstance(config.data, _config.MultiDataConfigFactory):
        if not config.data.datasets:
            raise ValueError("train_pi0_mem_pf requires MultiDataConfigFactory.datasets to be non-empty.")
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                datasets=[
                    _with_model_frame_layout(child, f"datasets[{i}]")
                    for i, child in enumerate(config.data.datasets)
                ],
            ),
        )
    else:
        config = dataclasses.replace(
            config,
            data=_with_model_frame_layout(config.data, "config.data"),
        )

    if config.model.num_future_frames <= 0:
        logging.warning(
            "num_future_frames=0: the posterior path sees no future frames; "
            "Zpost is all-zeros and lambda_post/lambda_align supervise nothing "
            "meaningful. This is only sensible for ablations."
        )

    # === Everything below mirrors scripts/mem/train_pi0_mem_compress.py.main. ===

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

    # Shared Pi0Mem video data loader (now emitting past+current+future clips).
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

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())

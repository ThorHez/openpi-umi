"""Train the isolated, conservative Pi0MemPF variant.

This entry point reuses the mature data loading, checkpointing and monitoring
from :mod:`pf.train_pi0_mem_pf`, but supplies a safe-specific train step with
an alignment warm-up. The original PF trainer remains behaviorally unchanged.

Example::

    uv run scripts/pf/train_pi0_mem_pf_safe.py \
        pi0_mem_pf_safe_evan_shellgame_joint_260806 \
        --exp-name=temporal_only_v1
"""

import os
from pathlib import Path
import sys

# The original PF trainer defaults to a cache directory from another
# workspace. Establish writable, repository-local defaults before importing
# JAX, HuggingFace datasets, or the base trainer; explicit user env settings
# still take precedence.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAFE_CACHE_HOME = _REPO_ROOT.parent / ".codex_tmp" / "pf_safe_cache"
_SAFE_TMP = _SAFE_CACHE_HOME / "tmp"
_EXISTING_OPENPI_DATA_HOME = Path("/data2/hzl_workspace_for_pi/.cache/openpi")
_SAFE_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_SAFE_CACHE_HOME))
os.environ.setdefault(
    "OPENPI_DATA_HOME",
    str(_EXISTING_OPENPI_DATA_HOME if _EXISTING_OPENPI_DATA_HOME.exists() else _SAFE_CACHE_HOME / "openpi"),
)
os.environ.setdefault("HF_HOME", str(_SAFE_CACHE_HOME / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_SAFE_CACHE_HOME / "huggingface" / "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_SAFE_CACHE_HOME / "huggingface" / "transformers"))
os.environ.setdefault("TMPDIR", str(_SAFE_TMP))
os.environ.setdefault("TEMP", str(_SAFE_TMP))
os.environ.setdefault("TMP", str(_SAFE_TMP))

# Adding ``scripts`` makes the original PF entry point importable both when
# this file is launched directly and when it is loaded as a module. That entry
# point also exposes the mature loader/checkpoint helpers reused below.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import dataclasses  # noqa: E402

import flax.nnx as nnx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from pf import train_pi0_mem_pf as _base  # noqa: E402

import openpi.models.model as _model  # noqa: E402
import openpi.models.pi0_mem_pf_safe as pi0_mem_pf_safe  # noqa: E402
import openpi.shared.array_typing as at  # noqa: E402
import openpi.shared.nnx_utils as nnx_utils  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.utils as training_utils  # noqa: E402


def alignment_weight(config: _config.TrainConfig, step) -> jax.Array:
    """Alignment coefficient with a static warm-up followed by linear ramp."""
    target = float(getattr(config.model, "lambda_align", 0.0))
    warmup = int(getattr(config.model, "align_warmup_steps", 0))
    ramp = int(getattr(config.model, "align_ramp_steps", 0))
    if target <= 0.0:
        return jnp.asarray(0.0, dtype=jnp.float32)
    if ramp <= 0:
        return jnp.asarray(target, dtype=jnp.float32)
    progress = jnp.clip(
        (jnp.asarray(step, dtype=jnp.float32) - float(warmup)) / float(ramp),
        0.0,
        1.0,
    )
    return jnp.asarray(target, dtype=jnp.float32) * progress


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """PF train step with conservative prior/posterior alignment scheduling."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    lambda_prior = float(getattr(config.model, "lambda_prior", 1.0))
    lambda_post = float(getattr(config.model, "lambda_post", 1.0))
    lambda_align_target = float(getattr(config.model, "lambda_align", 0.0))
    lambda_reg = float(getattr(config.model, "lambda_reg", 0.0))
    effective_lambda_align = alignment_weight(config, state.step)
    run_posterior = lambda_post > 0.0 or lambda_align_target > 0.0 or lambda_reg > 0.0

    diversity_weight = float(getattr(config.model, "diversity_weight", 0.0))
    diversity_on = diversity_weight > 0.0
    current_frame_dropout_prob = float(getattr(config.model, "current_frame_dropout_prob", 0.0))
    current_frame_mask_prob = float(getattr(config.model, "current_frame_mask_prob", 0.0))
    current_frame_corruption_on = current_frame_dropout_prob > 0.0 or current_frame_mask_prob > 0.0
    current_frame_corrupt_sample_prob = float(getattr(config.model, "current_frame_corrupt_sample_prob", 0.0))
    current_frame_corrupt_sample_on = current_frame_corrupt_sample_prob > 0.0 and current_frame_corruption_on
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
            corrupt_observation, corruption_metrics = _base.apply_current_frame_corruption(
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
            observation = _base._copy_observation_with_images(observation, images)  # noqa: SLF001
            corruption_metrics = {key: use_corrupt * value for key, value in corruption_metrics.items()}

        chunked_prior, aux = model.compute_loss_with_pf_aux(
            loss_rng,
            observation,
            actions,
            train=True,
            run_posterior=run_posterior,
        )
        loss_prior = jnp.mean(chunked_prior)
        loss_post = jnp.mean(aux["loss_post"])
        align_loss = aux["align_loss"]
        reg_loss = aux["reg_loss"]

        if diversity_on:
            div_loss = _base.memory_diversity_loss(aux["history_mem"])
        else:
            div_loss = jnp.asarray(0.0, dtype=loss_prior.dtype)

        total_loss = (
            lambda_prior * loss_prior
            + lambda_post * loss_post
            + effective_lambda_align * align_loss
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
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model,
        train_rng,
        observation,
        actions,
    )

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
        "diversity_zpost_loss": _base.memory_diversity_loss(aux["future_post"]),
        "loss_weighted/prior": lambda_prior * aux["loss_prior"],
        "loss_weighted/post": lambda_post * aux["loss_post"],
        "loss_weighted/align": effective_lambda_align * aux["align_loss"],
        "loss_weighted/reg": lambda_reg * aux["reg_loss"],
        "loss_weighted/diversity": diversity_weight * aux["diversity_loss"],
        "lambda_prior": jnp.asarray(lambda_prior, dtype=jnp.float32),
        "lambda_post": jnp.asarray(lambda_post, dtype=jnp.float32),
        "lambda_align": effective_lambda_align,
        "lambda_align_target": jnp.asarray(lambda_align_target, dtype=jnp.float32),
        "lambda_reg": jnp.asarray(lambda_reg, dtype=jnp.float32),
        "diversity_weight": jnp.asarray(diversity_weight, dtype=jnp.float32),
        "current_frame_corrupt_sample_rate": aux["current_frame_corrupt_sample_rate"],
        "current_frame_dropout_rate": aux["current_frame_dropout_rate"],
        "current_frame_mask_rate": aux["current_frame_mask_rate"],
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }

    info.update(_base.new_pf_param_grad_metrics(grads))
    info.update(_base.gate_param_metrics(new_params, "history_memory_gate_logit", "gate_history"))
    info.update(_base.gate_param_metrics(new_params, "future_memory_gate_logit", "gate_future"))
    info.update(_base.gate_config_metrics(config))
    info.update(_base.temporal_branch_activation_metrics(aux["encoder_auxes"], "memory"))
    info.update(_base.temporal_branch_activation_metrics(aux["post_encoder_auxes"], "memory_post"))
    info.update(_base.latent_collapse_metrics(aux["history_mem"], "memory/hist"))
    info.update(_base.latent_collapse_metrics(aux["future_post"], "memory/zpost"))
    info.update(_base.latent_collapse_metrics(aux["future_prior"], "memory/zprior"))
    info.update(_base.prior_posterior_alignment_metrics(aux["future_prior"], aux["future_post"]))
    info.update(_base.horizon_loss_metrics(aux["loss_prior_by_t"], aux["loss_post_by_t"]))
    info.update(_base.frame_validity_metrics(observation, config.model.num_frames, config.model.num_future_frames))
    return new_state, info


def main(config: _config.TrainConfig):
    if not isinstance(config.model, pi0_mem_pf_safe.Pi0MemPFSafeConfig):
        raise ValueError(f"train_pi0_mem_pf_safe requires Pi0MemPFSafeConfig; got {type(config.model).__name__}")
    if isinstance(config.data, _config.MultiDataConfigFactory) and config.data.state_pad_dim != config.model.action_dim:
        raise ValueError(
            "Pi0MemPFSafe FuturePrior expects state_pad_dim to equal model.action_dim; "
            f"got state_pad_dim={config.data.state_pad_dim}, action_dim={config.model.action_dim}"
        )

    # ``_base.main`` resolves its module-global train_step when constructing
    # the jitted function. Replacing it here keeps all other mature PF trainer
    # behavior while isolating the safe loss schedule in this file.
    _base.train_step = train_step
    _base.main(config)


if __name__ == "__main__":
    main(_config.cli())

"""Check whether a Pi0MemPF posterior branch causally uses Zpost.

The same batch, flow noise, and diffusion time are evaluated with normal,
batch-shuffled, zero, and random same-norm posterior latents.

Example:
    python scripts/pf/validate_future_latent_usage.py CONFIG_NAME \
        --exp-name my_experiment --num-batches 100
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
from pathlib import Path

# Keep Hugging Face/JAX caches and temporary dataset files off the root disk.
# These must be set before importing OpenPI/LeRobot/datasets.
_CACHE_HOME = Path("/data2/hzl_workspace_for_pi/.cache")
os.environ["XDG_CACHE_HOME"] = str(_CACHE_HOME)
os.environ["OPENPI_DATA_HOME"] = str(_CACHE_HOME / "openpi")
os.environ["HF_HOME"] = str(_CACHE_HOME / "huggingface")
os.environ["HF_DATASETS_CACHE"] = str(_CACHE_HOME / "huggingface" / "datasets")
os.environ["TRANSFORMERS_CACHE"] = str(_CACHE_HOME / "huggingface" / "transformers")
_tmp = _CACHE_HOME / "tmp"
_tmp.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_tmp)

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_mem_pf as pi0_mem_pf
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.config_pi0_mem as _config_pi0_mem
import openpi.training.sharding as sharding


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from pf.train_pi0_mem_pf import init_train_state  # noqa: E402
from train import init_logging  # noqa: E402


def _inject_model_frame_layout(config: _config.TrainConfig) -> _config.TrainConfig:
    """Match each data factory's temporal layout to the model config."""

    def replace_factory(factory, label: str):
        if not hasattr(factory, "video_frame_config"):
            raise ValueError(f"{label} is not Pi0Mem-aware: {type(factory).__name__}")
        fields = {field.name for field in dataclasses.fields(factory)}
        required = {"num_frames", "frame_stride", "num_future_frames", "future_frame_stride"}
        if missing := required - fields:
            raise ValueError(f"{label} cannot accept temporal fields {sorted(missing)}")
        return dataclasses.replace(
            factory,
            num_frames=config.model.num_frames,
            frame_stride=config.model.frame_stride,
            num_future_frames=config.model.num_future_frames,
            future_frame_stride=config.model.future_frame_stride,
        )

    if isinstance(config.data, _config.MultiDataConfigFactory):
        return dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                datasets=[
                    replace_factory(child, f"datasets[{i}]")
                    for i, child in enumerate(config.data.datasets)
                ],
            ),
        )
    return dataclasses.replace(config, data=replace_factory(config.data, "config.data"))


def _predict_and_loss(model, prefix, suffix, u_t, observation):
    """Run one posterior branch and return velocity prediction and loss."""
    prefix_tokens, prefix_mask, prefix_ar_mask = prefix
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = suffix
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = pi0_mem_pf.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    prediction = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    squared_error = jnp.square(prediction - u_t)
    if observation.action_loss_mask is not None:
        mask = observation.action_loss_mask[..., None, :]
        denominator = jnp.maximum(jnp.sum(mask, axis=-1), 1e-8)
        loss_by_t = jnp.sum(squared_error * mask, axis=-1) / denominator
    elif model.action_loss_mask is not None:
        mask = jnp.asarray(model.action_loss_mask)
        loss_by_t = jnp.sum(squared_error * mask, axis=-1) / jnp.sum(mask)
    else:
        loss_by_t = jnp.mean(squared_error, axis=-1)
    return prediction, loss_by_t


def _random_with_same_norm(rng, latents):
    result = {}
    for key, key_rng in zip(latents, jax.random.split(rng, len(latents))):
        latent = latents[key]
        random = jax.random.normal(key_rng, latent.shape, dtype=jnp.float32)
        random /= jnp.linalg.norm(random, axis=-1, keepdims=True) + 1e-6
        norm = jnp.linalg.norm(latent.astype(jnp.float32), axis=-1, keepdims=True)
        result[key] = (random * norm).astype(latent.dtype)
    return result


def _relative_change(value, reference, eps: float = 1e-6):
    axes = tuple(range(1, value.ndim))
    return jnp.mean(
        jnp.linalg.norm(value - reference, axis=axes)
        / (jnp.linalg.norm(reference, axis=axes) + eps)
    )


def _latent_relative_change(latents, reference):
    return jnp.mean(
        jnp.stack(
            [
                _relative_change(latents[key].astype(jnp.float32), reference[key].astype(jnp.float32))
                for key in reference
            ]
        )
    )


def _latent_max_abs_change(latents, reference):
    return jnp.max(
        jnp.stack(
            [
                jnp.max(jnp.abs(latents[key].astype(jnp.float32) - reference[key].astype(jnp.float32)))
                for key in reference
            ]
        )
    )


def _latent_batch_variance(latents):
    return jnp.mean(
        jnp.stack(
            [
                jnp.mean(jnp.var(value.astype(jnp.float32), axis=0))
                for value in latents.values()
            ]
        )
    )


def _replace_future_frames(observation, current_frame_index: int, mode: str):
    """Intervene only on frames strictly after the current frame."""
    future_start = current_frame_index + 1
    images = {}
    for key, image in observation.images.items():
        if image.ndim != 5 or future_start >= image.shape[1]:
            images[key] = image
            continue
        if mode == "shuffle":
            replacement = jnp.roll(image[:, future_start:], shift=1, axis=0)
        elif mode == "repeat_current":
            replacement = jnp.broadcast_to(
                image[:, current_frame_index : current_frame_index + 1],
                image[:, future_start:].shape,
            )
        else:
            raise ValueError(f"Unknown future-frame intervention: {mode}")
        images[key] = image.at[:, future_start:].set(replacement)
    return dataclasses.replace(observation, images=images)


def _future_frame_metrics(observation, current_frame_index: int):
    """Measure whether history/future video slots contain distinct image content.

    Also reports ``frame_valid_mask`` coverage when the observation carries it
    (P0 data-pipeline check).
    """
    metrics = {}
    aggregate = {}
    for key, image in observation.images.items():
        if image.ndim != 5 or image.shape[1] < 2:
            continue
        image = image.astype(jnp.float32)
        current = image[:, current_frame_index : current_frame_index + 1]
        # Near-black / empty-frame detector (works for [-1,1] or [0,1] images).
        abs_mean = jnp.mean(jnp.abs(image), axis=(2, 3, 4))  # [B, T]
        near_empty = abs_mean <= 1e-3

        values = {
            "clip_near_empty_rate": jnp.mean(near_empty.astype(jnp.float32)),
            "current_near_empty_rate": jnp.mean(
                near_empty[:, current_frame_index].astype(jnp.float32)
            ),
        }

        if current_frame_index > 0:
            history = image[:, :current_frame_index]
            history_vs_current = jnp.abs(history - current)
            history_consecutive = jnp.abs(history[:, 1:] - history[:, :-1])
            hist_equal_current = jnp.all(history_vs_current <= 1e-6, axis=(2, 3, 4))
            values.update(
                {
                    "history_vs_current_mae": jnp.mean(history_vs_current),
                    "history_consecutive_mae": jnp.mean(history_consecutive),
                    "history_equal_current_rate": jnp.mean(
                        hist_equal_current.astype(jnp.float32)
                    ),
                    "history_near_empty_rate": jnp.mean(
                        near_empty[:, :current_frame_index].astype(jnp.float32)
                    ),
                }
            )

        if current_frame_index + 1 < image.shape[1]:
            future = image[:, current_frame_index + 1 :]
            future_vs_current = jnp.abs(future - current)
            temporal = image[:, current_frame_index:]
            consecutive = jnp.abs(temporal[:, 1:] - temporal[:, :-1])
            batch_shuffled = jnp.abs(future - jnp.roll(future, shift=1, axis=0))
            equal_current = jnp.all(future_vs_current <= 1e-6, axis=(2, 3, 4))
            sample_all_future_eq_current = jnp.all(equal_current, axis=1)
            values.update(
                {
                    "future_vs_current_mae": jnp.mean(future_vs_current),
                    "future_consecutive_mae": jnp.mean(consecutive),
                    "future_batch_shuffle_mae": jnp.mean(batch_shuffled),
                    "future_equal_current_rate": jnp.mean(equal_current.astype(jnp.float32)),
                    "sample_all_future_eq_current_rate": jnp.mean(
                        sample_all_future_eq_current.astype(jnp.float32)
                    ),
                    "future_near_empty_rate": jnp.mean(
                        near_empty[:, current_frame_index + 1 :].astype(jnp.float32)
                    ),
                }
            )

        masks = observation.frame_valid_masks
        if masks is not None and key in masks:
            valid = jnp.asarray(masks[key], dtype=jnp.bool_)
            if current_frame_index > 0:
                history_valid = valid[..., :current_frame_index]
                history_valid_f = history_valid.astype(jnp.float32)
                values["history_valid_rate"] = jnp.mean(history_valid_f)
                if "history_equal_current_rate" in values:
                    hist_equal = jnp.all(
                        jnp.abs(image[:, :current_frame_index] - current) <= 1e-6,
                        axis=(2, 3, 4),
                    )
                    valid_hist_equal = jnp.where(history_valid, hist_equal, False)
                    hist_denom = jnp.maximum(jnp.sum(history_valid_f), 1.0)
                    values["valid_history_equal_current_rate"] = (
                        jnp.sum(valid_hist_equal.astype(jnp.float32)) / hist_denom
                    )
            if current_frame_index + 1 < image.shape[1]:
                future_valid = valid[..., current_frame_index + 1 :]
                future_valid_f = future_valid.astype(jnp.float32)
                values["future_valid_rate"] = jnp.mean(future_valid_f)
                values["full_future_sample_rate"] = jnp.mean(
                    jnp.all(future_valid, axis=-1).astype(jnp.float32)
                )
                future = image[:, current_frame_index + 1 :]
                future_vs_current = jnp.abs(future - current)
                equal_current = jnp.all(future_vs_current <= 1e-6, axis=(2, 3, 4))
                valid_equal = jnp.where(future_valid, equal_current, False)
                valid_denom = jnp.maximum(jnp.sum(future_valid_f), 1.0)
                values["valid_future_equal_current_rate"] = (
                    jnp.sum(valid_equal.astype(jnp.float32)) / valid_denom
                )
                valid_mae_num = jnp.sum(
                    future_vs_current * future_valid_f[..., None, None, None]
                )
                valid_mae_den = jnp.maximum(
                    jnp.sum(future_valid_f)
                    * future_vs_current.shape[-1]
                    * future_vs_current.shape[-2]
                    * future_vs_current.shape[-3],
                    1.0,
                )
                values["valid_future_vs_current_mae"] = valid_mae_num / valid_mae_den

        for name, value in values.items():
            metrics[f"frames/{key}/{name}"] = value
            aggregate.setdefault(name, []).append(value)

    for name, values in aggregate.items():
        metrics[f"frames/all/{name}"] = jnp.mean(jnp.stack(values))
    return metrics


def _p0_verdict(summary: dict[str, float]) -> None:
    """Print a short pass/fail style verdict for the P0 data check."""
    eq = summary.get("frames/all/future_equal_current_rate")
    mae = summary.get("frames/all/future_vs_current_mae")
    valid_rate = summary.get("frames/all/future_valid_rate")
    full_future = summary.get("frames/all/full_future_sample_rate")
    valid_eq = summary.get("frames/all/valid_future_equal_current_rate")
    hist_eq = summary.get("frames/all/history_equal_current_rate")
    hist_mae = summary.get("frames/all/history_vs_current_mae")
    hist_valid = summary.get("frames/all/history_valid_rate")
    hist_empty = summary.get("frames/all/history_near_empty_rate")
    fut_empty = summary.get("frames/all/future_near_empty_rate")
    cur_empty = summary.get("frames/all/current_near_empty_rate")
    logging.info("=" * 92)
    logging.info("P0 data-pipeline verdict")
    logging.info("=" * 92)
    if hist_eq is not None and hist_mae is not None:
        if (hist_empty is not None and hist_empty >= 0.95) or (
            cur_empty is not None and cur_empty >= 0.95
        ):
            logging.info(
                "FAIL (history): frames are near-empty "
                "(history_empty=%.4f, current_empty=%.4f).",
                hist_empty if hist_empty is not None else float("nan"),
                cur_empty if cur_empty is not None else float("nan"),
            )
        elif hist_eq >= 0.95 and hist_mae <= 1e-5:
            logging.info(
                "FAIL (history): history frames copy the current frame "
                "(equal_rate=%.4f, mae=%.6g).",
                hist_eq,
                hist_mae,
            )
        elif hist_eq >= 0.5:
            logging.info(
                "WARN (history): many history slots equal current "
                "(equal_rate=%.4f, mae=%.6g).",
                hist_eq,
                hist_mae,
            )
        else:
            logging.info(
                "PASS (history pixels): history differs from current "
                "(equal_rate=%.4f, mae=%.6g, empty=%.4f, valid=%s).",
                hist_eq,
                hist_mae,
                hist_empty if hist_empty is not None else float("nan"),
                f"{hist_valid:.4f}" if hist_valid is not None else "n/a",
            )
    if eq is None or mae is None:
        logging.info("Incomplete future metrics; cannot judge future branch.")
        return
    if (fut_empty is not None and fut_empty >= 0.95) or (
        cur_empty is not None and cur_empty >= 0.95
    ):
        logging.info(
            "FAIL (future): frames are near-empty "
            "(future_empty=%.4f, current_empty=%.4f).",
            fut_empty if fut_empty is not None else float("nan"),
            cur_empty if cur_empty is not None else float("nan"),
        )
    elif eq >= 0.95 and mae <= 1e-5:
        logging.info(
            "FAIL: future frames are effectively copies of the current frame "
            "(equal_rate=%.4f, mae=%.6g). Fix the loader / padding before "
            "changing the model.",
            eq,
            mae,
        )
    elif eq >= 0.5:
        logging.info(
            "WARN: a large fraction of future slots equal the current frame "
            "(equal_rate=%.4f, mae=%.6g). Check episode length vs "
            "num_future_frames*future_frame_stride and padding_mode.",
            eq,
            mae,
        )
    else:
        logging.info(
            "PASS (pixels): future frames differ from current "
            "(equal_rate=%.4f, mae=%.6g, empty=%.4f).",
            eq,
            mae,
            fut_empty if fut_empty is not None else float("nan"),
        )
    if valid_rate is not None:
        logging.info(
            "Mask: future_valid_rate=%.4f  full_future_sample_rate=%.4f  "
            "valid_future_equal_current_rate=%s",
            valid_rate,
            full_future if full_future is not None else float("nan"),
            f"{valid_eq:.4f}" if valid_eq is not None else "n/a",
        )
        if full_future is not None and full_future < 0.5:
            logging.info(
                "WARN: fewer than half of samples have a fully valid future "
                "window; padded/repeated futures may dominate Zpost."
            )
        if valid_eq is not None and valid_eq >= 0.95:
            logging.info(
                "FAIL: even among mask-valid future slots, pixels match "
                "current — this is not just padding."
            )
    logging.info("-" * 92)


def evaluate_batch(model_def, params, batch, rng, current_frame_index: int):
    """Evaluate all latent interventions with shared stochastic variables."""
    model = nnx.merge(model_def, params)
    model.eval()
    observation, actions = batch
    observation = _model.preprocess_observation(
        None, observation, train=False, image_keys=tuple(observation.images)
    )

    noise_rng, time_rng, random_rng = jax.random.split(rng, 3)
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    x_t = time[..., None, None] * noise + (1.0 - time[..., None, None]) * actions
    u_t = noise - actions

    _, history, zpost = model._encode_memories(observation)
    suffix = model.embed_suffix(observation, x_t, time)
    variants = {
        "normal": zpost,
        "shuffle": {key: jnp.roll(value, 1, axis=0) for key, value in zpost.items()},
        "zero": {key: jnp.zeros_like(value) for key, value in zpost.items()},
        "random": _random_with_same_norm(random_rng, zpost),
    }
    future_shuffle_observation = _replace_future_frames(
        observation, current_frame_index, "shuffle"
    )
    future_repeat_observation = _replace_future_frames(
        observation, current_frame_index, "repeat_current"
    )
    _, _, variants["future_shuffle"] = model._encode_memories(
        future_shuffle_observation
    )
    _, _, variants["future_repeat_current"] = model._encode_memories(
        future_repeat_observation
    )

    predictions = {}
    losses = {}
    sample_losses = {}
    for name, latent in variants.items():
        prefix = model._embed_prefix_with_latents(observation, history, latent)[:3]
        prediction, loss_by_t = _predict_and_loss(
            model, prefix, suffix, u_t, observation
        )
        predictions[name] = prediction
        losses[name] = jnp.mean(loss_by_t)
        sample_losses[name] = jnp.mean(loss_by_t, axis=-1)

    normal_loss = losses["normal"]
    metrics = {
        "loss/normal": normal_loss,
        "zpost/batch_variance": _latent_batch_variance(zpost),
    }
    metrics.update(_future_frame_metrics(observation, current_frame_index))
    for name in (
        "shuffle",
        "zero",
        "random",
        "future_shuffle",
        "future_repeat_current",
    ):
        delta = losses[name] - normal_loss
        metrics[f"loss/{name}"] = losses[name]
        metrics[f"delta/{name}"] = delta
        metrics[f"delta_relative/{name}"] = delta / (normal_loss + 1e-6)
        metrics[f"normal_better_rate/{name}"] = jnp.mean(
            sample_losses["normal"] < sample_losses[name]
        )
        metrics[f"prediction_relative_change/{name}"] = _relative_change(
            predictions[name], predictions["normal"]
        )
        metrics[f"zpost_relative_change/{name}"] = _latent_relative_change(
            variants[name], zpost
        )
        metrics[f"zpost_max_abs_change/{name}"] = _latent_max_abs_change(
            variants[name], zpost
        )
    return metrics


def _summarize(results: list[dict[str, float]], *, image_only: bool = False) -> None:
    logging.info("=" * 92)
    logging.info("Future-latent intervention summary over %d batches", len(results))
    logging.info("=" * 92)
    means: dict[str, float] = {}
    for key in sorted(results[0]):
        values = np.asarray([result[key] for result in results], dtype=np.float64)
        std = float(np.std(values))
        mean = float(np.mean(values))
        means[key] = mean
        logging.info(
            "%-46s mean=% .8f  std=% .8f  sem=% .8f",
            key,
            mean,
            std,
            std / math.sqrt(len(values)),
        )
    logging.info("-" * 92)
    if image_only:
        _p0_verdict(means)
    else:
        logging.info(
            "Evidence of causal use: positive delta, normal_better_rate > 0.5, "
            "and non-trivial prediction_relative_change."
        )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--fsdp-devices", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Only inspect future-frame pixel differences; do not load a checkpoint",
    )
    return parser.parse_args()


def main(args) -> None:
    init_logging()
    if args.num_batches <= 0:
        raise ValueError("--num-batches must be positive")
    replacements = {
        "exp_name": args.exp_name,
        "resume": True,
        "overwrite": False,
        "wandb_enabled": False,
    }
    for field in ("batch_size", "num_workers", "fsdp_devices"):
        if getattr(args, field) is not None:
            replacements[field] = getattr(args, field)
    config = dataclasses.replace(_config.get_config(args.config_name), **replacements)
    config = _inject_model_frame_layout(config)

    if not isinstance(config.model, pi0_mem_pf.Pi0MemPFConfig):
        raise TypeError(f"Expected Pi0MemPFConfig, got {type(config.model).__name__}")
    if config.model.num_future_frames <= 0:
        raise ValueError("The selected config has no future frames")
    if config.batch_size < 2:
        raise ValueError("Batch size must be at least 2 for batch shuffling")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by "
            f"{jax.device_count()} devices"
        )
    if not args.image_only and not config.checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {config.checkpoint_dir}")

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
    )
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = _config_pi0_mem.create_pi0_mem_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=args.num_batches,
    )

    if args.image_only:
        logging.info(
            "P0 image-only check: num_frames=%d num_future_frames=%d "
            "frame_stride=%d future_frame_stride=%d current_frame_index=%d "
            "batch_size=%d num_batches=%d",
            config.model.num_frames,
            config.model.num_future_frames,
            config.model.frame_stride,
            config.model.future_frame_stride,
            config.model.resolved_current_frame_index,
            config.batch_size,
            args.num_batches,
        )
        results = []
        with sharding.set_mesh(mesh):
            for index, (observation, _) in enumerate(loader):
                metrics = jax.device_get(
                    _future_frame_metrics(
                        observation,
                        config.model.resolved_current_frame_index,
                    )
                )
                results.append({key: float(value) for key, value in metrics.items()})
                logging.info(
                    "Batch %d/%d hist_eq=%.4f hist_mae=%.6g hist_empty=%.4f "
                    "fut_eq=%.4f fut_mae=%.6g fut_empty=%.4f fut_valid=%s",
                    index + 1,
                    args.num_batches,
                    results[-1].get("frames/all/history_equal_current_rate", float("nan")),
                    results[-1].get("frames/all/history_vs_current_mae", float("nan")),
                    results[-1].get("frames/all/history_near_empty_rate", float("nan")),
                    results[-1].get("frames/all/future_equal_current_rate", float("nan")),
                    results[-1].get("frames/all/future_vs_current_mae", float("nan")),
                    results[-1].get("frames/all/future_near_empty_rate", float("nan")),
                    (
                        f"{results[-1]['frames/all/future_valid_rate']:.4f}"
                        if "frames/all/future_valid_rate" in results[-1]
                        else "n/a"
                    ),
                )
                if index + 1 >= args.num_batches:
                    break
        if not results:
            raise RuntimeError("Data loader produced no batches")
        _summarize(results, image_only=True)
        return

    manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    if not resuming:
        raise FileNotFoundError(f"No checkpoint found in {config.checkpoint_dir}")
    state_shape, state_sharding = init_train_state(
        config, jax.random.key(config.seed), mesh, resume=True
    )
    state = _checkpoints.restore_state(
        manager, state_shape, loader, step=args.checkpoint_step
    )
    use_ema = not args.no_ema and state.ema_params is not None
    params = state.ema_params if use_ema else state.params
    params_sharding = (
        state_sharding.ema_params if use_ema else state_sharding.params
    )
    logging.info(
        "Restored step %d using %s parameters",
        int(state.step),
        "EMA" if use_ema else "online",
    )

    eval_fn = jax.jit(
        lambda p, batch, rng: evaluate_batch(
            state.model_def,
            p,
            batch,
            rng,
            config.model.resolved_current_frame_index,
        ),
        in_shardings=(params_sharding, data_sharding, replicated),
        out_shardings=replicated,
    )
    rng = jax.random.key(args.seed)
    results = []
    with sharding.set_mesh(mesh):
        for index, batch in enumerate(loader):
            metrics = jax.device_get(
                eval_fn(params, batch, jax.random.fold_in(rng, index))
            )
            result = {key: float(value) for key, value in metrics.items()}
            results.append(result)
            logging.info(
                "Batch %d/%d normal=%.6f shuffle_delta=%+.6f zero_delta=%+.6f",
                index + 1,
                args.num_batches,
                result["loss/normal"],
                result["delta/shuffle"],
                result["delta/zero"],
            )
            if index + 1 >= args.num_batches:
                break
    if not results:
        raise RuntimeError("Data loader produced no batches")
    _summarize(results, image_only=False)


if __name__ == "__main__":
    main(_parse_args())

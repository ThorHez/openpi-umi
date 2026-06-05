"""Pi0Mem training entry point.

Byte-for-byte mirror of :mod:`scripts.train` except for two paradigm-sanity
checks at the top of :func:`main` and one swapped data-loader call:

* :func:`openpi.training.data_loader.create_data_loader`
    -> :func:`openpi.training.config_pi0_mem.create_pi0_mem_data_loader`

All training helpers (``init_logging``, ``init_wandb``, ``init_train_state``,
``train_step``) are imported verbatim from ``scripts/train.py`` to avoid any
divergence in training logic.

Launch with the standard tyro CLI, identical in shape to ``scripts/train.py``::

    python scripts/mem/train_pi0_mem.py pi0_mem_umi_bimanual_horizon1 \\
        --exp_name=run_T4 \\
        --data.repo_id=/path/to/bimanual_lerobot_dataset \\
        --data.assets.assets_dir=/path/to/bimanual_lerobot_dataset \\
        --data.num_frames=4 --model.num_frames=4 \\
        --batch_size=72 --fsdp_devices=8
"""

import os
from pathlib import Path

# Avoid / or overlay filling up: use $HOME/tmp for temp files if TMPDIR not set.
# (Identical to scripts/train.py.)
if "TMPDIR" not in os.environ:
    _tmp = Path(os.environ.get("HOME", "/root")) / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_tmp)

import functools
import logging
import platform
import sys

import etils.epath as epath
from flax.training import common_utils
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model  # noqa: F401  (imported for parity with scripts/train.py)
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

# Pi0Mem-specific imports: the model config (for the paradigm check) and the
# Pi0Mem-aware data loader factory.
import openpi.models.pi0_mem as pi0_mem
import openpi.training.config_pi0_mem as _config_pi0_mem

# Reuse train.py helpers verbatim. We add openpi-umi/scripts to sys.path so we
# can import scripts/train.py as a top-level module.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from train import init_logging, init_train_state, init_wandb, train_step  # noqa: E402


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    # === Pi0Mem paradigm sanity checks ===
    if not isinstance(config.model, pi0_mem.Pi0MemConfig):
        raise ValueError(
            f"train_pi0_mem requires a Pi0MemConfig model; got "
            f"{type(config.model).__name__}. Use scripts/train.py for other models."
        )
    # Pi0Mem-aware = exposes ``video_frame_config()``. For multi-dataset
    # training (MultiDataConfigFactory) every child factory must be
    # Pi0Mem-aware; create_pi0_mem_data_loader dispatches accordingly.
    if isinstance(config.data, _config.MultiDataConfigFactory):
        if not config.data.datasets:
            raise ValueError(
                "train_pi0_mem requires MultiDataConfigFactory.datasets to be non-empty."
            )
        for i, child in enumerate(config.data.datasets):
            if not hasattr(child, "video_frame_config"):
                raise ValueError(
                    f"train_pi0_mem requires every MultiDataConfigFactory child "
                    f"to be Pi0Mem-aware (must expose .video_frame_config()); "
                    f"datasets[{i}] is {type(child).__name__}."
                )
    elif not hasattr(config.data, "video_frame_config"):
        raise ValueError(
            "train_pi0_mem requires a Pi0Mem-aware DataConfigFactory "
            "(must expose .video_frame_config()), or a MultiDataConfigFactory "
            "whose children are Pi0Mem-aware; got "
            f"{type(config.data).__name__}."
        )

    # === Everything below mirrors scripts/train.py.main byte-for-byte, with ===
    # === exactly one swap: create_data_loader -> create_pi0_mem_data_loader. ===

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

    # ▼▼▼ THE ONE LINE THAT DIFFERS FROM scripts/train.py ▼▼▼
    data_loader = _config_pi0_mem.create_pi0_mem_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    # ▲▲▲ (replaces _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)) ▲▲▲

    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    # images_to_log = [
    #     wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
    #     for i in range(min(5, len(next(iter(batch[0].images.values())))))
    # ]
    # wandb.log({"camera_views": images_to_log}, step=0)

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
    loss_history = []  # Track recent loss values for anomaly detection
    anomaly_dir = config.checkpoint_dir / "anomalies"
    anomaly_dir.mkdir(exist_ok=True)

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        # Anomaly detection
        current_loss = float(info["loss"])
        loss_history.append(current_loss)
        if len(loss_history) > 100:  # Keep last 100 steps
            loss_history.pop(0)

        # Detect anomalies: NaN, Inf, or sudden spike
        is_anomaly = False
        anomaly_reason = ""
        loss_history_size = config.log_interval * 2

        if jnp.isnan(current_loss) or jnp.isinf(current_loss):
            is_anomaly = True
            anomaly_reason = "NaN or Inf loss"
        elif len(loss_history) > loss_history_size:
            recent_mean = np.mean(loss_history[-loss_history_size:])
            recent_std = np.std(loss_history[-loss_history_size:])
            # If loss is more than 3 std deviations above recent mean
            if current_loss > recent_mean + 3 * recent_std and recent_std > 1e-6:
                is_anomaly = True
                anomaly_reason = f"Spike: {current_loss:.4f} vs recent {recent_mean:.4f}±{recent_std:.4f}"

        if is_anomaly:
            import pickle
            anomaly_file = anomaly_dir / f"step_{step:06d}_{current_loss:.4f}.pkl"
            pbar.write(f"⚠️  ANOMALY DETECTED at step {step}: {anomaly_reason}")
            pbar.write(f"   Saving data to {anomaly_file}")

            # Get data to CPU and save
            batch_cpu = jax.device_get(batch)
            anomaly_data = {
                "step": step,
                "loss": current_loss,
                "loss_history": loss_history[-20:],  # Save last 20 losses
                "reason": anomaly_reason,
                "observation": batch_cpu[0],
                "actions": batch_cpu[1],
                "info": jax.device_get(info),
            }

            with open(anomaly_file, "wb") as f:
                pickle.dump(anomaly_data, f)
            pbar.write("   ✓ Anomaly data saved!")

        # Periodic statistics logging (every 1000 steps)
        stats_interval = 1000
        if step > 0 and step % stats_interval == 0 and len(loss_history) > 10:
            import pickle
            stats_dir = config.checkpoint_dir / "periodic_stats"
            stats_dir.mkdir(exist_ok=True)

            batch_cpu = jax.device_get(batch)
            stats_file = stats_dir / f"step_{step:06d}_{current_loss:.4f}.pkl"

            pbar.write(f"\n📊 Saving periodic statistics at step {step} to {stats_file}")

            # Save data in the same format as anomaly detection
            stats_data = {
                "step": step,
                "loss": current_loss,
                "loss_history": loss_history.copy(),
                "reason": f"Periodic stats at step {step}",
                "observation": batch_cpu[0],
                "actions": batch_cpu[1],
                "info": jax.device_get(info),
            }

            with open(stats_file, "wb") as f:
                pickle.dump(stats_data, f)
            pbar.write("   ✓ Statistics data saved!")

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
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

"""
Multi-dataset training script for Pi0 Hybrid.

Same as train_hybrid.py but creates a multi-dataset data loader when config.data
is MultiDataConfigFactory (multiple LeRobot datasets with optional per-dataset weights).
Checkpoint saving stores norm_stats for each dataset under its asset_id.

Norm stats: When use_merged_norm_stats=True (default on MultiDataConfigFactory), each
dataset's norm_stats are merged (weighted by config.weights) and applied to all
datasets so state/actions are normalized consistently.

Usage:
    python scripts/train_multi_dataset.py --config pi05_umi_multi_dataset --exp_name multi_task_v1

For single-dataset configs this script behaves identically to train_hybrid.py.
"""

from __future__ import annotations

import os
from pathlib import Path

# Avoid / or overlay filling up: use $HOME/tmp for temp files if TMPDIR not set.
if "TMPDIR" not in os.environ:
    _tmp = Path(os.environ.get("HOME", "/root")) / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_tmp)

import argparse
import importlib.util
import sys
from pathlib import Path

import jax

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding

# Load train_hybrid from same directory (script is run as python scripts/train_multi_dataset.py)
_train_hybrid_path = Path(__file__).resolve().parent / "train_hybrid.py"
_spec = importlib.util.spec_from_file_location("train_hybrid", _train_hybrid_path)
train_hybrid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_hybrid)


def _create_data_loader(config: _config.TrainConfig, data_sharding):
    """Create data loader: multi-dataset if config.data is MultiDataConfigFactory, else single."""
    if isinstance(config.data, _config.MultiDataConfigFactory):
        from openpi.training.multi_data_loader import create_multi_data_loader

        multi_factory = config.data
        all_configs = multi_factory.create_all(config.assets_dirs, config.model)
        weights_list = multi_factory.weights
        dc_and_weights = [
            (dc, weights_list[i] if (weights_list and i < len(weights_list)) else 1.0)
            for i, dc in enumerate(all_configs)
        ]
        return create_multi_data_loader(
            config,
            data_configs_and_weights=dc_and_weights,
            sharding=data_sharding,
            shuffle=True,
        )
    return _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lambda_fast", type=float, default=0.1)
    parser.add_argument("--use_fast_loss", action="store_true", default=True)
    parser.add_argument("--compute_separate_grad_norms", action="store_true", default=True)
    parser.add_argument("--grad_norm_compute_interval", type=int, default=500)
    parser.add_argument("--fast_warmup_steps", type=int, default=3000)
    parser.add_argument("--adaptive_lambda", action="store_true", default=True)
    parser.add_argument("--adaptive_r", type=float, default=0.2)
    parser.add_argument("--lambda_ema_decay", type=float, default=0.99)
    parser.add_argument("--lambda_min", type=float, default=1e-4)
    parser.add_argument("--lambda_max", type=float, default=0.3)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    config = _config.cli()
    compute_grad_norms = args.compute_separate_grad_norms

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    data_loader = _create_data_loader(config, data_sharding)

    print("\n" + "=" * 60)
    print("MULTI-DATASET HYBRID TRAINING")
    print("=" * 60)
    if isinstance(config.data, _config.MultiDataConfigFactory):
        print(f"  Using multi-dataset loader: {len(config.data.datasets)} dataset(s)")
    else:
        print("  Using single-dataset loader")
    print("=" * 60 + "\n")

    train_hybrid.main(
        config,
        lambda_fast=args.lambda_fast,
        use_fast_loss=args.use_fast_loss,
        compute_separate_grad_norms=compute_grad_norms,
        grad_norm_compute_interval=args.grad_norm_compute_interval,
        fast_warmup_steps=args.fast_warmup_steps,
        adaptive_lambda=args.adaptive_lambda,
        adaptive_r=args.adaptive_r,
        lambda_ema_decay=args.lambda_ema_decay,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        data_loader=data_loader,
    )

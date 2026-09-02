#!/usr/bin/env python3
"""Train action-only or frozen-MEM Pi0.5 on PickXtimes EEF7 chunks."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

WORKSPACE_CACHE = Path("/data2/hzl_workspace_for_pi_mem/.codex_tmp")
SHARED_CACHE = Path("/data2/hzl_workspace_for_pi_mem/.cache")
os.environ.setdefault("XDG_CACHE_HOME", str(SHARED_CACHE))
os.environ.setdefault("OPENPI_DATA_HOME", str(SHARED_CACHE / "openpi"))
os.environ.setdefault("HF_HOME", str(SHARED_CACHE / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(SHARED_CACHE / "huggingface/datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(SHARED_CACHE / "huggingface/transformers"))
os.environ.setdefault("TMPDIR", str(WORKSPACE_CACHE / "tmp"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from openpi.training import config as _config  # noqa: E402
from openpi.training.mem.recipes import robomme_pickxtimes_pi_action as _recipe  # noqa: E402
from scripts.mem import train_pi0_mem_compress as _trainer  # noqa: E402


def _find_hf_dataset(dataset):
    current, sample_indices = dataset, None
    while current is not None:
        if sample_indices is None:
            sample_indices = getattr(current, "sample_indices", None)
        if (hf_dataset := getattr(current, "_hf_dataset", None)) is not None:
            return hf_dataset, sample_indices
        current = getattr(current, "_dataset", None)
    raise ValueError("Could not find underlying PickXtimes HF dataset")


def phase_balanced_full_horizon_indices(dataset, indices: list[int], _classifier_config) -> list[int]:
    hf_dataset, sample_indices = _find_hf_dataset(dataset)
    phase = np.asarray(hf_dataset["phase_id"], dtype=np.int64)
    full = np.asarray(hf_dataset["action_mask"], dtype=bool)
    if sample_indices is not None:
        source = np.asarray(sample_indices, dtype=np.int64)
        phase, full = phase[source], full[source]
    selected = np.asarray(indices, dtype=np.int64)
    selected = selected[full[selected]]
    groups = [selected[phase[selected] == value] for value in range(6)]
    if any(group.size == 0 for group in groups):
        raise ValueError(f"Empty PickXtimes phase after split: {[len(group) for group in groups]}")
    target = max(len(group) for group in groups)
    balanced = []
    for value, group in enumerate(groups):
        shuffled = np.random.default_rng(260824 + value + len(indices)).permutation(group)
        repeats, remainder = divmod(target, len(shuffled))
        balanced.append(np.concatenate((np.tile(shuffled, repeats), shuffled[:remainder])))
    merged = np.stack(balanced, axis=1).reshape(-1)
    logging.info("PickXtimes phase balance raw=%s output=%d each=%d", [len(g) for g in groups], len(merged), target)
    return merged.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--memory-mode", choices=("predicted", "action_only"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=_recipe.DEFAULT_DATASET_ROOT)
    parser.add_argument("--memory-path", type=Path, default=_recipe.DEFAULT_MEMORY_BANK)
    parser.add_argument("--init-checkpoint", type=Path, default=_recipe.DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument(
        "--keep-period",
        type=int,
        help="Permanently retain checkpoints at this interval; defaults to save-interval.",
    )
    parser.add_argument("--semantic-residual-gate-init", type=float, default=1.0)
    parser.add_argument("--semantic-residual-dropout-rate", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.dataset_root, args.memory_path, args.init_checkpoint):
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)
    WORKSPACE_CACHE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE_CACHE / "tmp").mkdir(parents=True, exist_ok=True)
    config = _recipe.make_train_config(
        config_module=_config, dataset_root=args.dataset_root, memory_path=args.memory_path,
        memory_mode=args.memory_mode, init_checkpoint=args.init_checkpoint, exp_name=args.exp_name,
        steps=args.steps, batch_size=args.batch_size, fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers, peak_lr=args.peak_lr, warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval, eval_batches=args.eval_batches,
        save_interval=args.save_interval, keep_period=args.keep_period,
        overwrite=args.overwrite, resume=args.resume,
        semantic_residual_gate_init=args.semantic_residual_gate_init,
        semantic_residual_dropout_rate=args.semantic_residual_dropout_rate,
    )
    _trainer._filter_memory_classifier_frame_range = phase_balanced_full_horizon_indices  # noqa: SLF001
    _trainer.main(config)


if __name__ == "__main__":
    main()

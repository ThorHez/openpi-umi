"""Train the oracle-point VideoUnmask Pi0.5 absolute-EEF action gate."""

from __future__ import annotations

import argparse
import dataclasses
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
from openpi.training.mem.recipes import robomme_videounmask_pi_action as _recipe  # noqa: E402
from scripts.mem import train_pi0_mem_compress as _trainer  # noqa: E402


def _find_hf_dataset(dataset):
    current = dataset
    sample_indices = None
    while current is not None:
        if sample_indices is None:
            sample_indices = getattr(current, "sample_indices", None)
        hf_dataset = getattr(current, "_hf_dataset", None)
        if hf_dataset is not None:
            return hf_dataset, sample_indices
        current = getattr(current, "_dataset", None)
    raise ValueError("Could not find the underlying VideoUnmask HF dataset")


def phase_balanced_full_horizon_indices(dataset, indices: list[int], _classifier_config) -> list[int]:
    """Balance transit/align/descend/grasp/lift rows after episode splitting."""
    hf_dataset, sample_indices = _find_hf_dataset(dataset)
    columns = set(getattr(hf_dataset, "column_names", ()) or ())
    if not {"phase_id", "action_mask"}.issubset(columns):
        raise ValueError("VideoUnmask action balancing requires phase_id and action_mask columns")
    phase = np.asarray(hf_dataset["phase_id"], dtype=np.int64)
    action_mask = np.asarray(hf_dataset["action_mask"], dtype=bool)
    if sample_indices is not None:
        source = np.asarray(sample_indices, dtype=np.int64)
        phase = phase[source]
        action_mask = action_mask[source]
    selected = np.asarray(indices, dtype=np.int64)
    selected = selected[action_mask[selected]]
    phase_values = sorted(np.unique(phase[selected]).astype(int).tolist())
    groups = [selected[phase[selected] == value] for value in phase_values]
    if any(group.size == 0 for group in groups):
        raise ValueError(f"Empty VideoUnmask action phase after split: {[len(group) for group in groups]}")
    target_size = max(group.size for group in groups)
    balanced = []
    for value, group in zip(phase_values, groups, strict=True):
        shuffled = np.random.default_rng(260823 + value + len(indices)).permutation(group)
        repeats, remainder = divmod(target_size, len(shuffled))
        balanced.append(np.concatenate((np.tile(shuffled, repeats), shuffled[:remainder])))
    merged = np.stack(balanced, axis=1).reshape(-1)
    logging.info(
        "VideoUnmask action phase balance: raw=%s full_horizon=%d output=%d each=%d",
        [len(group) for group in groups],
        len(selected),
        len(merged),
        target_size,
    )
    return merged.tolist()


def natural_temporally_masked_indices(dataset, indices: list[int], _classifier_config) -> list[int]:
    """Keep every causal row and let the model's temporal mask ignore padded suffix steps.

    This preserves the natural phase prior and, crucially, retains late grasp / lift
    observations that do not have a complete 16-step future chunk.  The converted
    frame_index and episode_T fields exactly describe how many suffix actions are real.
    """
    hf_dataset, sample_indices = _find_hf_dataset(dataset)
    columns = set(getattr(hf_dataset, "column_names", ()) or ())
    if not {"phase_id", "action_mask"}.issubset(columns):
        raise ValueError("VideoUnmask natural sampling requires phase_id and action_mask columns")
    phase = np.asarray(hf_dataset["phase_id"], dtype=np.int64)
    full_horizon = np.asarray(hf_dataset["action_mask"], dtype=bool)
    if sample_indices is not None:
        source = np.asarray(sample_indices, dtype=np.int64)
        phase = phase[source]
        full_horizon = full_horizon[source]
    selected = np.asarray(indices, dtype=np.int64)
    phase_values = sorted(np.unique(phase[selected]).astype(int).tolist())
    phase_counts = {value: int(np.sum(phase[selected] == value)) for value in phase_values}
    logging.info(
        "VideoUnmask natural temporal sampling: phase=%s rows=%d full_horizon=%d partial_horizon=%d",
        phase_counts,
        len(selected),
        int(np.sum(full_horizon[selected])),
        int(np.sum(~full_horizon[selected])),
    )
    return selected.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--dataset-root", type=Path, default=_recipe.DEFAULT_DATASET_ROOT)
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
        "--sampling-mode",
        choices=("phase_balanced_full", "natural_temporal"),
        default="phase_balanced_full",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--phase-conditioned", action="store_true")
    parser.add_argument("--goal-relative-conditioner", action="store_true")
    parser.add_argument("--phase-goal-conditioner", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(
        args.steps,
        args.warmup_steps,
        args.batch_size,
        args.num_workers + 1,
        args.fsdp_devices,
        args.eval_interval,
        args.eval_batches,
        args.save_interval,
    ) <= 0:
        raise ValueError("Training counts must be positive (num-workers may be zero)")
    return args


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.init_checkpoint = args.init_checkpoint.expanduser().resolve()
    if not args.dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset {args.dataset_root} does not exist; run "
            "scripts/mem/convert_videounmask_to_lerobot_pi_action.py first"
        )
    if not args.init_checkpoint.exists():
        raise FileNotFoundError(args.init_checkpoint)
    WORKSPACE_CACHE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE_CACHE / "tmp").mkdir(parents=True, exist_ok=True)
    config = _recipe.make_train_config(
        config_module=_config,
        dataset_root=args.dataset_root,
        init_checkpoint=args.init_checkpoint,
        exp_name=args.exp_name,
        steps=args.steps,
        batch_size=args.batch_size,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        peak_lr=args.peak_lr,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        save_interval=args.save_interval,
        overwrite=args.overwrite,
        phase_conditioned=args.phase_conditioned,
        target_point_relative_to_eef=args.goal_relative_conditioner,
        phase_goal_conditioner=args.phase_goal_conditioner,
    )
    config = dataclasses.replace(config, resume=args.resume)
    sampler = {
        "phase_balanced_full": phase_balanced_full_horizon_indices,
        "natural_temporal": natural_temporally_masked_indices,
    }[args.sampling_mode]
    _trainer._filter_memory_classifier_frame_range = sampler  # noqa: SLF001
    _trainer.main(config)


if __name__ == "__main__":
    main()

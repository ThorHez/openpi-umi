#!/usr/bin/env python3
"""Diagnose visual evidence and temporal-order use in the fixed-chunk student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import flax
import jax
import numpy as np
from scipy.optimize import linear_sum_assignment

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openpi.tasks.robomme import unified_fixed_chunk_student as student_lib  # noqa: E402
from openpi.tasks.robomme import unified_gt_teacher as teacher_lib  # noqa: E402
from scripts.mem import train_robomme_four_task_fixed_chunk_distillation as train_lib  # noqa: E402

DEFAULT_TRAINING = _ROOT / "checkpoints/robomme_four_task_fixed_chunk_student_v1_260826"
MODES = ("normal", "zero_video", "reverse_chunks", "shuffle_episode_video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional result path; defaults to <training-dir>/<split>_visual_dependence.json.",
    )
    parser.add_argument("--split", choices=train_lib.SPLITS, default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    return parser.parse_args()


def _training_args(config: dict) -> SimpleNamespace:
    values = dict(config)
    for key in (
        "sequence_dir",
        "feature_dir",
        "proprio_dir",
        "teacher_memory_dir",
        "teacher_sequence_dir",
        "teacher_training_dir",
        "teacher_checkpoint",
        "output_dir",
    ):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    return SimpleNamespace(**values)


def _within_task_length_matched_donors(
    evaluation_indices: np.ndarray,
    task_ids: np.ndarray,
    step_mask: np.ndarray,
) -> np.ndarray:
    """Build a deterministic within-task derangement minimizing length mismatch."""

    evaluation_indices = np.asarray(evaluation_indices, dtype=np.int64)
    evaluation_tasks = np.asarray(task_ids)[evaluation_indices]
    lengths = np.asarray(step_mask)[evaluation_indices].sum(axis=1).astype(np.int64)
    donors = np.full_like(evaluation_indices, -1)
    for task_id in np.unique(evaluation_tasks):
        positions = np.flatnonzero(evaluation_tasks == task_id)
        if len(positions) < 2:
            raise ValueError(f"Need at least two episodes to shuffle task {task_id}")
        task_lengths = lengths[positions]
        cost = np.abs(task_lengths[:, None] - task_lengths[None, :]).astype(np.float64)
        np.fill_diagonal(cost, 1e9)
        # A tiny deterministic tie-break keeps the assignment reproducible.
        cost += 1e-6 * np.arange(len(positions), dtype=np.float64)[None, :]
        rows, columns = linear_sum_assignment(cost)
        donors[positions[rows]] = evaluation_indices[positions[columns]]
    if np.any(donors < 0) or np.any(donors == evaluation_indices):
        raise RuntimeError("Failed to construct a complete cross-episode derangement")
    if np.any(np.asarray(task_ids)[donors] != evaluation_tasks):
        raise RuntimeError("Cross-episode donor changed task identity")
    return donors


def _perturb(
    batch: dict[str, np.ndarray],
    mode: str,
    *,
    donor_batch: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    batch = dict(batch)
    patches = np.array(batch["patch_tokens"], copy=True)
    if mode == "zero_video":
        patches.fill(0)
    elif mode == "shuffle_episode_video":
        if donor_batch is None:
            raise ValueError("shuffle_episode_video requires a donor batch")
        donor_patches = np.asarray(donor_batch["patch_tokens"])
        donor_masks = np.asarray(donor_batch["sequence_mask"])
        receiver_masks = np.asarray(batch["sequence_mask"])
        patches.fill(0)
        for index in range(len(patches)):
            receiver_length = int(receiver_masks[index].sum())
            donor_length = int(donor_masks[index].sum())
            if receiver_length <= 0 or donor_length <= 0:
                continue
            sample_indices = np.rint(
                np.linspace(0, donor_length - 1, receiver_length)
            ).astype(np.int64)
            patches[index, :receiver_length] = donor_patches[index, sample_indices]
    elif mode == "reverse_chunks":
        for index, mask in enumerate(batch["sequence_mask"]):
            length = int(mask.sum())
            patches[index, :length] = patches[index, :length][::-1]
    elif mode != "normal":
        raise ValueError(mode)
    batch["patch_tokens"] = patches
    return batch


def main() -> None:
    args = parse_args()
    config = json.loads((args.training_dir / "training_config.json").read_text())
    training_args = _training_args(config)
    training_args.batch_size = args.batch_size
    dataset = train_lib.SplitDataset(args.split, training_args)
    model = student_lib.UnifiedFixedChunkRecurrentStudent(
        proprio_dim=int(dataset.proprio_dim),
        encoder_width=int(config["encoder_width"]),
        encoder_depth=int(config["encoder_depth"]),
        encoder_heads=int(config["encoder_heads"]),
        use_write_gate=bool(config.get("write_gate", False)),
        write_gate_bias=float(config.get("write_gate_bias", -2.0)),
        use_event_gate=bool(config.get("event_gate", False)),
        event_gate_bias=float(config.get("event_gate_bias", 0.0)),
        event_gate_modulation_strength=float(
            config.get("event_gate_modulation_strength", 0.0)
        ),
        event_gate_reference=float(config.get("event_gate_reference", 0.2)),
        event_gate_modulation_min=float(config.get("event_gate_modulation_min", 0.75)),
        event_gate_modulation_max=float(config.get("event_gate_modulation_max", 1.25)),
        use_event_update_routing=bool(config.get("event_update_routing", False)),
        event_update_routing_temperature=float(
            config.get("event_update_routing_temperature", 1.0)
        ),
        event_update_routing_reference=float(
            config.get("event_update_routing_reference", 0.5)
        ),
        use_event_correction=bool(config.get("event_correction", False)),
        use_oracle_event_correction=bool(config.get("oracle_event_correction", False)),
        use_causal_evidence_state=bool(config.get("causal_evidence_state", False)),
        use_recurrent_memory=bool(config.get("recurrent_memory_carry", True)),
    )
    template_pool = np.arange(dataset.length)
    if config.get("task") is not None:
        template_task_id = teacher_lib.TASKS.index(config["task"])
        template_pool = np.flatnonzero(dataset.sequence["task_ids"] == template_task_id)
    indices = np.resize(template_pool, args.batch_size)
    template_batch = dataset.batch(
        indices,
        change_state_weight=float(config["change_state_weight"]),
        final_state_weight=float(config.get("final_state_weight", 1.0)),
    )
    template = model.init(
        jax.random.key(int(config["seed"])),
        **train_lib._student_inputs(  # noqa: SLF001
            template_batch,
            oracle_event_correction=bool(config.get("oracle_event_correction", False)),
        ),
        train=False,
    )["params"]
    checkpoint = args.checkpoint or args.training_dir / "best/params"
    params = flax.serialization.from_bytes(template, checkpoint.read_bytes())
    readout, readout_params = train_lib._load_teacher_readout(training_args)  # noqa: SLF001

    @jax.jit
    def infer(batch):
        output = model.apply(
            {"params": params},
            **train_lib._student_inputs(  # noqa: SLF001
                batch,
                oracle_event_correction=bool(config.get("oracle_event_correction", False)),
            ),
            train=False,
        )
        memory = output["all_memories"]
        flat = memory.reshape(-1, memory.shape[-2], memory.shape[-1])
        logits = readout.apply({"params": readout_params}, flat).reshape(
            *memory.shape[:2], len(teacher_lib.STATE_FIELDS), teacher_lib.MAX_FIELD_CLASSES
        )
        return logits, output["write_gates"]

    results = {
        "checkpoint": str(checkpoint.resolve()),
        "split": args.split,
        "overlapping_windows": False,
        "explicit_event_trigger": False,
        "learned_soft_write_gate": bool(config.get("write_gate", False)),
        "task": config.get("task"),
        "modes": {},
    }
    evaluation_indices = np.arange(dataset.length)
    if config.get("task") is not None:
        task_id = teacher_lib.TASKS.index(config["task"])
        evaluation_indices = np.flatnonzero(dataset.sequence["task_ids"] == task_id)
    shuffle_donors = _within_task_length_matched_donors(
        evaluation_indices,
        dataset.sequence["task_ids"],
        dataset.sequence["step_mask"],
    )
    results["shuffle_episode_video_protocol"] = {
        "within_task": True,
        "different_episode": True,
        "donor_assignment": "minimum absolute valid-chunk length mismatch",
        "length_normalization": (
            "nearest-neighbor resampling of the full donor sequence to the receiver valid length"
        ),
        "receiver_indices": evaluation_indices.tolist(),
        "donor_indices": shuffle_donors.tolist(),
    }
    try:
        for mode in args.modes:
            logits, targets, masks, changes, gates = [], [], [], [], []
            for start in range(0, len(evaluation_indices), args.batch_size):
                indices = evaluation_indices[start : start + args.batch_size]
                real_count = len(indices)
                if real_count < args.batch_size:
                    indices = np.pad(indices, (0, args.batch_size - real_count), mode="edge")
                batch = dataset.batch(
                    indices,
                    change_state_weight=float(config["change_state_weight"]),
                    final_state_weight=float(config.get("final_state_weight", 1.0)),
                )
                donor_batch = None
                if mode == "shuffle_episode_video":
                    donor_indices = shuffle_donors[start : start + real_count]
                    if real_count < args.batch_size:
                        donor_indices = np.pad(
                            donor_indices,
                            (0, args.batch_size - real_count),
                            mode="edge",
                        )
                    donor_batch = dataset.batch(
                        donor_indices,
                        change_state_weight=float(config["change_state_weight"]),
                        final_state_weight=float(config.get("final_state_weight", 1.0)),
                    )
                batch = _perturb(batch, mode, donor_batch=donor_batch)
                batch_logits, batch_gates = infer(batch)
                logits.append(np.asarray(batch_logits)[:real_count])
                gates.append(np.asarray(batch_gates)[:real_count])
                targets.append(batch["state_targets"][:real_count])
                masks.append(batch["state_field_mask"][:real_count])
                changes.append(batch["state_change_mask"][:real_count])
            summary = train_lib._host_summary(  # noqa: SLF001
                np.concatenate(logits),
                np.concatenate(targets),
                np.concatenate(masks),
                dataset.sequence["task_ids"][evaluation_indices],
                np.concatenate(changes),
            )
            dynamic_fields = tuple(
                teacher_lib.STATE_FIELDS.index(name)
                for name in ("completed_count", "holding", "ready_to_press", "done")
            )
            concatenated_logits = np.concatenate(logits)
            concatenated_targets = np.concatenate(targets)
            concatenated_masks = np.concatenate(masks)
            summary["dynamic_state"] = train_lib._host_summary(  # noqa: SLF001
                concatenated_logits[..., dynamic_fields, :],
                concatenated_targets[..., dynamic_fields],
                concatenated_masks[..., dynamic_fields],
                dataset.sequence["task_ids"][evaluation_indices],
                np.concatenate(changes),
                include_terminal_answer=False,
            )["overall"]
            all_gates = np.concatenate(gates)
            all_changes = np.concatenate(changes)
            valid_chunks = dataset.sequence["step_mask"][evaluation_indices]
            summary["write_gate"] = {
                "all": float(all_gates[valid_chunks].mean()),
                "change": float(all_gates[valid_chunks & all_changes].mean()),
                "hold": float(all_gates[valid_chunks & ~all_changes].mean()),
            }
            results["modes"][mode] = summary
            print(json.dumps({mode: summary}, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        dataset.close()
    output = args.output or args.training_dir / f"{args.split}_visual_dependence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

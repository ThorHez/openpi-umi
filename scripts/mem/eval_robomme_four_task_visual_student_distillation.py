#!/usr/bin/env python3
"""Evaluate visual dependence of the distilled four-task recurrent student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import flax
import jax
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib  # noqa: E402
from openpi.tasks.robomme import unified_visual_student as student_lib  # noqa: E402
from scripts.mem import train_robomme_four_task_visual_student_distillation as train_lib  # noqa: E402

DEFAULT_TRAINING = _ROOT / "checkpoints/robomme_four_task_visual_student_distilled_v1_260826"
MODES = ("normal", "zero_video", "reverse_event_windows", "shuffle_episode_video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=train_lib.SPLITS, default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _paths_from_config(config: dict) -> SimpleNamespace:
    values = dict(config)
    for key in (
        "sequence_dir",
        "feature_dir",
        "teacher_memory_dir",
        "teacher_training_dir",
        "teacher_checkpoint",
        "output_dir",
    ):
        values[key] = Path(values[key])
    return SimpleNamespace(**values)


def _perturb(batch: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
    batch = dict(batch)
    patches = np.array(batch["patch_tokens"], copy=True)
    if mode == "zero_video":
        patches.fill(0)
    elif mode == "shuffle_episode_video":
        patches = np.roll(patches, shift=1, axis=0)
    elif mode == "reverse_event_windows":
        for index, mask in enumerate(batch["step_mask"]):
            length = int(np.sum(mask))
            patches[index, :length] = patches[index, :length][::-1]
    elif mode != "normal":
        raise ValueError(mode)
    batch["patch_tokens"] = patches
    return batch


def main() -> None:
    args = parse_args()
    config = json.loads((args.training_dir / "training_config.json").read_text())
    training_args = _paths_from_config(config)
    training_args.batch_size = args.batch_size
    dataset = train_lib.SplitDataset(args.split, training_args)
    model = student_lib.UnifiedVisualRecurrentStudent(
        encoder_width=int(config["encoder_width"]),
        encoder_depth=int(config["encoder_depth"]),
        encoder_heads=int(config["encoder_heads"]),
    )
    indices = np.arange(args.batch_size) % dataset.length
    template_batch = dataset.batch(indices)
    template = model.init(
        jax.random.key(int(config["seed"])),
        **train_lib._student_inputs(template_batch),  # noqa: SLF001
        train=False,
    )["params"]
    checkpoint = args.checkpoint or args.training_dir / "best/params"
    params = flax.serialization.from_bytes(template, checkpoint.read_bytes())
    readout, readout_params = train_lib._load_teacher_readout(training_args)  # noqa: SLF001

    @jax.jit
    def infer(batch):
        output = model.apply(
            {"params": params},
            **train_lib._student_inputs(batch),  # noqa: SLF001
            train=False,
        )
        memory = output["all_memories"]
        flat = memory.reshape(-1, memory.shape[-2], memory.shape[-1])
        return readout.apply({"params": readout_params}, flat).reshape(
            *memory.shape[:2], len(teacher_lib.STATE_FIELDS), teacher_lib.MAX_FIELD_CLASSES
        )

    results = {
        "checkpoint": str(checkpoint.resolve()),
        "split": args.split,
        "student_receives_gt_event_or_state": False,
        "modes": {},
    }
    try:
        for mode in args.modes:
            logits = []
            for start in range(0, dataset.length, args.batch_size):
                indices = np.arange(start, min(start + args.batch_size, dataset.length))
                real_count = len(indices)
                if real_count < args.batch_size:
                    indices = np.pad(indices, (0, args.batch_size - real_count), mode="edge")
                batch = _perturb(dataset.batch(indices), mode)
                logits.append(np.asarray(infer(batch))[:real_count])
            summary = train_lib._host_summary(  # noqa: SLF001
                np.concatenate(logits),
                dataset.teacher["state_targets"],
                dataset.teacher["state_field_mask"],
                dataset.goals["task_ids"],
            )
            results["modes"][mode] = summary
            print(json.dumps({mode: summary}, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        dataset.close()
    output = args.output or args.training_dir / f"{args.split}_visual_dependence.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

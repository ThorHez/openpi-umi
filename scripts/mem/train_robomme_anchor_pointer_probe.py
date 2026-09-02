#!/usr/bin/env python3
"""Minimal frozen-MEM anchor-pointer probe for three RoboMME tasks.

This experiment freezes the existing single-task recurrent MEM checkpoints and
trains one shared pointer readout over episode-local anchor tokens. It is meant
to test the readout hypothesis before changing the recurrent updater.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import flax
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib  # noqa: E402
from openpi.tasks.robomme.anchor_pointer_readout import AnchorPointerReadout  # noqa: E402
from scripts.mem import robomme_fixed_chunk_inference as fixed_memory  # noqa: E402
from scripts.mem.build_videoplaceorder_qwen3vl_sft_manifest import _episode_metadata  # noqa: E402
from scripts.mem.build_videounmaskswap_qwen3vl_local_event_manifest import _color_centers  # noqa: E402
from scripts.mem.build_videounmaskswap_qwen3vl_local_event_manifest import _decode  # noqa: E402
from scripts.mem.build_videounmaskswap_qwen3vl_local_event_manifest import _position_centers  # noqa: E402


TASKS = (
    "videounmask_variable_demo",
    "videounmaskswap_local_event",
    "videoplaceorder_local_event",
)
TASK_TO_ENV = {
    TASKS[0]: "VideoUnmask",
    TASKS[1]: "VideoUnmaskSwap",
    TASKS[2]: "VideoPlaceOrder",
}
DEFAULT_SEQUENCE = ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_TEACHER = ROOT / "artifacts/robomme_four_task_gt_teacher_memory_v2_260826"
DEFAULT_FEATURES = ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826"
DEFAULT_DATA = ROOT / "data/robomme_extracted"
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_anchor_pointer_probe_v1_260828"
DEFAULT_RUNS = {
    TASKS[0]: ROOT / "checkpoints/robomme_single_task_unmask_equal_exposure_seed260827_260827",
    TASKS[1]: ROOT / "checkpoints/robomme_single_task_swap_equal_exposure_seed260827_260827",
    TASKS[2]: ROOT / "checkpoints/robomme_single_task_place_equal_exposure_seed260828_260827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--unmask-run", type=Path, default=DEFAULT_RUNS[TASKS[0]])
    parser.add_argument("--swap-run", type=Path, default=DEFAULT_RUNS[TASKS[1]])
    parser.add_argument("--place-run", type=Path, default=DEFAULT_RUNS[TASKS[2]])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=260828)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _sample_grid_token(tokens: np.ndarray, point_yx: tuple[float, float]) -> np.ndarray:
    """Bilinearly sample a 4x4 pooled SigLIP token grid at a 256px point."""

    grid = np.asarray(tokens, dtype=np.float32).reshape(4, 4, -1)
    gy = np.clip(float(point_yx[0]) / 256.0 * 4.0 - 0.5, 0.0, 3.0)
    gx = np.clip(float(point_yx[1]) / 256.0 * 4.0 - 0.5, 0.0, 3.0)
    y0, x0 = int(np.floor(gy)), int(np.floor(gx))
    y1, x1 = min(y0 + 1, 3), min(x0 + 1, 3)
    wy, wx = gy - y0, gx - x0
    return (
        (1.0 - wy) * (1.0 - wx) * grid[y0, x0]
        + (1.0 - wy) * wx * grid[y0, x1]
        + wy * (1.0 - wx) * grid[y1, x0]
        + wy * wx * grid[y1, x1]
    )


def _place_anchors(episode: h5py.Group, episode_index: int) -> list[tuple[float, float]]:
    item = _episode_metadata(episode, episode_index)
    anchors = [tuple(float(value) for value in drop["target_xy"]) for drop in item["drops"]]
    final = tuple(float(value) for value in item["target_xy"])
    nearest = min(float(np.linalg.norm(np.subtract(final, point))) for point in anchors)
    if nearest > 16.0:
        anchors.append(final)
    return sorted(anchors)


def _anchors(
    task: str,
    episode: h5py.Group,
    episode_index: int,
) -> list[tuple[float, float]]:
    if task == TASKS[0]:
        visible = _color_centers(episode["timestep_0/obs/front_rgb"][()])
        return sorted(visible.values())
    if task == TASKS[1]:
        visible = _color_centers(episode["timestep_0/obs/front_rgb"][()])
        difficulty = _decode(episode["setup/difficulty"][()])
        positions, valid = _position_centers(
            visible,
            episode["timestep_63/obs/front_rgb"][()],
            difficulty=difficulty,
        )
        if not valid:
            raise ValueError(f"Could not recover Swap anchors for episode {episode_index}")
        return positions
    if task == TASKS[2]:
        return _place_anchors(episode, episode_index)
    raise ValueError(task)


def _final_state(
    sequence: dict[str, np.ndarray],
    teacher: dict[str, np.ndarray],
    row: int,
) -> np.ndarray:
    length = int(sequence["step_mask"][row].sum())
    teacher_index = int(sequence["teacher_state_index"][row, length])
    return teacher["state_targets"][row, teacher_index]


def _query_rows(
    task: str,
    sequence: dict[str, np.ndarray],
    final_state: np.ndarray,
    row: int,
) -> list[tuple[int, int, int]]:
    """Return (query color, ordinal, target region) rows."""

    if task in (TASKS[0], TASKS[1]):
        result = []
        for color_id in sequence["goal_color_ids"][row]:
            color_id = int(color_id)
            if color_id == 0:
                continue
            color = teacher_lib.COLORS[color_id]
            field = teacher_lib.STATE_FIELDS.index(f"{color}_cell")
            region = int(final_state[field]) - 1
            if region >= 0:
                result.append((color_id, 0, region))
        return result
    ordinal = int(sequence["queried_ordinals"][row])
    color_id = int(sequence["goal_color_ids"][row, 0])
    field = teacher_lib.STATE_FIELDS.index(f"ordered_cell_{ordinal - 1}")
    region = int(final_state[field]) - 1
    return [(color_id, ordinal, region)] if region >= 0 else []


def _build_split(
    split: str,
    args: argparse.Namespace,
    predictors: dict[str, fixed_memory.FixedChunkMemoryPredictor],
) -> dict[str, np.ndarray]:
    sequence = _load_npz(args.sequence_dir / f"{split}.npz")
    teacher = _load_npz(args.teacher_dir / f"{split}.npz")
    feature_file = h5py.File(args.feature_dir / f"{split}.h5", "r")
    h5_files = {
        task: h5py.File(args.data_dir / f"record_dataset_{TASK_TO_ENV[task]}.h5", "r")
        for task in TASKS
    }
    values: dict[str, list[Any]] = defaultdict(list)
    try:
        for task in TASKS:
            task_id = teacher_lib.TASKS.index(task)
            rows = np.flatnonzero(sequence["task_ids"] == task_id)
            predictor = predictors[task]
            source = h5_files[task]
            for row in rows:
                row = int(row)
                episode_index = int(sequence["episode_index"][row])
                group = feature_file[f"episode_{row:06d}"]
                chunks = np.asarray(group["patch_tokens"], dtype=np.float16)
                output = predictor.predict_encoded(
                    chunks,
                    task_id=task_id,
                    goal_color_ids=tuple(int(value) for value in sequence["goal_color_ids"][row]),
                    required_count=int(sequence["required_counts"][row]),
                    queried_ordinal=int(sequence["queried_ordinals"][row]),
                    num_regions=int(sequence["num_regions"][row]),
                )
                memory = np.asarray(output["all_memories"][-1], dtype=np.float16)
                prediction = output["all_predictions"][-1]
                points = _anchors(task, source[f"episode_{episode_index}"], episode_index)
                expected_regions = int(sequence["num_regions"][row])
                if len(points) != expected_regions:
                    raise ValueError(
                        f"{split}:{task}:{episode_index} anchors={len(points)} "
                        f"but sequence num_regions={expected_regions}"
                    )
                if len(points) > 4:
                    raise ValueError(f"Pointer supports at most four anchors, got {points}")
                first_frame_tokens = chunks[0, 0]
                padded_tokens = np.zeros((4, 1152), dtype=np.float16)
                padded_yx = np.zeros((4, 2), dtype=np.float32)
                padded_mask = np.zeros((4,), dtype=np.bool_)
                for index, point in enumerate(points):
                    padded_tokens[index] = _sample_grid_token(first_frame_tokens, point)
                    padded_yx[index] = np.asarray(point, dtype=np.float32) / 127.5 - 1.0
                    padded_mask[index] = True
                final_state = _final_state(sequence, teacher, row)
                for color_id, ordinal, target_region in _query_rows(
                    task, sequence, final_state, row
                ):
                    if not 0 <= target_region < len(points):
                        raise ValueError(
                            f"Invalid target region {target_region} for {task}:{episode_index}"
                        )
                    if task in (TASKS[0], TASKS[1]):
                        color = teacher_lib.COLORS[color_id]
                        field = fixed_memory.field_index(f"{color}_cell")
                    else:
                        field = fixed_memory.field_index(f"ordered_cell_{ordinal - 1}")
                    values["memory"].append(memory)
                    values["anchor_tokens"].append(padded_tokens)
                    values["anchor_yx"].append(padded_yx)
                    values["anchor_mask"].append(padded_mask)
                    values["task_ids"].append(task_id)
                    values["query_color_ids"].append(color_id)
                    values["queried_ordinals"].append(ordinal)
                    values["targets"].append(target_region)
                    values["baseline_predictions"].append(int(prediction[field]) - 1)
                    values["episode_index"].append(episode_index)
                    values["num_regions"].append(len(points))
            print(f"cached {split} {task}: {len(rows)} episodes", flush=True)
    finally:
        feature_file.close()
        for source in h5_files.values():
            source.close()
    return {key: np.asarray(value) for key, value in values.items()}


def build_cache(args: argparse.Namespace) -> None:
    cache_dir = args.output_dir / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = {
        TASKS[0]: args.unmask_run,
        TASKS[1]: args.swap_run,
        TASKS[2]: args.place_run,
    }
    predictors = {
        task: fixed_memory.FixedChunkMemoryPredictor(path)
        for task, path in run_dirs.items()
    }
    for split in ("train", "dev", "test"):
        path = cache_dir / f"{split}.npz"
        if path.exists() and not args.rebuild_cache:
            print(f"using existing cache {path}", flush=True)
            continue
        arrays = _build_split(split, args, predictors)
        np.savez_compressed(path, **arrays)
        print(f"wrote {path}: {len(arrays['targets'])} pointer queries", flush=True)


def _batch_indices(data: dict[str, np.ndarray], rng: np.random.Generator, size: int) -> np.ndarray:
    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index, values in enumerate(
        zip(
            data["task_ids"],
            data["query_color_ids"],
            data["num_regions"],
            data["targets"],
            strict=True,
        )
    ):
        groups[tuple(int(value) for value in values)].append(index)
    keys = list(groups)
    result = []
    for _ in range(size):
        group = groups[keys[int(rng.integers(len(keys)))]]
        result.append(group[int(rng.integers(len(group)))])
    return np.asarray(result, dtype=np.int64)


def _permuted_batch(
    data: dict[str, np.ndarray], indices: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    batch = {key: value[indices] for key, value in data.items()}
    count = len(indices)
    permutations = np.stack([rng.permutation(4) for _ in range(count)])
    row = np.arange(count)[:, None]
    for key in ("anchor_tokens", "anchor_yx", "anchor_mask"):
        batch[key] = batch[key][row, permutations]
    inverse = np.argsort(permutations, axis=1)
    batch["targets"] = inverse[np.arange(count), batch["targets"]]
    return batch


def _model_inputs(data: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {
        key: jnp.asarray(data[key])
        for key in (
            "memory",
            "anchor_tokens",
            "anchor_yx",
            "anchor_mask",
            "task_ids",
            "query_color_ids",
            "queried_ordinals",
        )
    }


def _summary(predictions: np.ndarray, data: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    correct = predictions == data["targets"]
    for task in TASKS:
        task_id = teacher_lib.TASKS.index(task)
        mask = data["task_ids"] == task_id
        result[task] = {
            "queries": int(mask.sum()),
            "accuracy": float(correct[mask].mean()),
        }
    result["overall"] = {"queries": len(correct), "accuracy": float(correct.mean())}
    return result


def train(args: argparse.Namespace) -> None:
    data = {
        split: _load_npz(args.output_dir / "data" / f"{split}.npz")
        for split in ("train", "dev", "test")
    }
    model = AnchorPointerReadout()
    template = _model_inputs({key: value[:1] for key, value in data["train"].items()})
    params = model.init(jax.random.key(args.seed), **template, train=True)["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=min(100, max(args.steps // 10, 1)),
        decay_steps=args.steps,
        end_value=args.learning_rate * 0.05,
    )
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_opt_state, batch):
        def loss_fn(candidate_params):
            logits = model.apply({"params": candidate_params}, **_model_inputs(batch), train=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, jnp.asarray(batch["targets"], dtype=jnp.int32)
            ).mean()
            accuracy = jnp.mean(
                jnp.argmax(logits, axis=-1) == jnp.asarray(batch["targets"])
            )
            return loss, accuracy

        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
        next_params = optax.apply_updates(current_params, updates)
        return next_params, next_opt_state, loss, accuracy

    @jax.jit
    def predict(current_params, inputs):
        return jnp.argmax(
            model.apply({"params": current_params}, **inputs, train=False), axis=-1
        )

    def evaluate(current_params, split: str) -> dict[str, Any]:
        values = data[split]
        predictions = np.asarray(predict(current_params, _model_inputs(values)))
        return _summary(predictions, values)

    baseline = {
        split: _summary(values["baseline_predictions"], values)
        for split, values in data.items()
    }
    rng = np.random.default_rng(args.seed)
    best_score = (-1.0, -1.0)
    best_params = params
    history = []
    for step in range(1, args.steps + 1):
        indices = _batch_indices(data["train"], rng, args.batch_size)
        batch = _permuted_batch(data["train"], indices, rng)
        params, opt_state, loss, accuracy = train_step(params, opt_state, batch)
        if step % args.eval_every == 0 or step == args.steps:
            dev = evaluate(params, "dev")
            task_accuracies = [dev[task]["accuracy"] for task in TASKS]
            score = (min(task_accuracies), float(np.mean(task_accuracies)))
            if score > best_score:
                best_score = score
                best_params = jax.device_get(params)
            row = {
                "step": step,
                "train_loss": float(loss),
                "train_batch_accuracy": float(accuracy),
                "dev": dev,
                "selection_min_task_accuracy": score[0],
                "selection_mean_task_accuracy": score[1],
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    results = {
        "schema_version": 1,
        "experiment": "frozen_recurrent_mem_shared_anchor_pointer",
        "selection": "max min(task_accuracy), tie mean(task_accuracy)",
        "baseline": baseline,
        "pointer": {
            split: evaluate(best_params, split) for split in ("train", "dev", "test")
        },
        "best_score": {"min_task_accuracy": best_score[0], "mean_task_accuracy": best_score[1]},
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pointer_params.msgpack").write_bytes(
        flax.serialization.to_bytes(best_params)
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "sequence_dir": str(args.sequence_dir),
                "teacher_dir": str(args.teacher_dir),
                "feature_dir": str(args.feature_dir),
                "data_dir": str(args.data_dir),
                "output_dir": str(args.output_dir),
                "unmask_run": str(args.unmask_run),
                "swap_run": str(args.swap_run),
                "place_run": str(args.place_run),
                "jax_devices": [str(device) for device in jax.devices()],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results["pointer"], indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT.parent / ".cache/openpi"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_cache(args)
    if not args.build_cache_only:
        train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a leakage-free full-context visual region ceiling on RoboMME."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
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

from openpi.tasks.robomme import unified_gt_teacher as contract  # noqa: E402
from openpi.tasks.robomme.full_context_region_summarizer import FullContextRegionSummarizer  # noqa: E402


TASKS = contract.TASKS[:3]
DEFAULT_SEQUENCE = ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_TEACHER = ROOT / "artifacts/robomme_four_task_gt_teacher_memory_v2_260826"
DEFAULT_FEATURES = ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826"
DEFAULT_OUTPUT = ROOT / "checkpoints/robomme_full_context_visual_region_ceiling_seed260828_260828"
MAX_CHUNKS = 96
SPATIAL_TOKENS = 16
RAW_WIDTH = 1152
COMPACT_WIDTH = RAW_WIDTH * 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=260828)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _final_state(
    sequence: dict[str, np.ndarray], teacher: dict[str, np.ndarray], row: int
) -> np.ndarray:
    length = int(sequence["step_mask"][row].sum())
    teacher_index = int(sequence["teacher_state_index"][row, length])
    return teacher["state_targets"][row, teacher_index]


def _query_rows(
    task_id: int,
    sequence: dict[str, np.ndarray],
    final_state: np.ndarray,
    row: int,
) -> list[tuple[int, int, int]]:
    if task_id in (0, 1):
        result = []
        for color_id in sequence["goal_color_ids"][row]:
            color_id = int(color_id)
            if color_id == 0:
                continue
            field = contract.STATE_FIELDS.index(
                f"{contract.COLORS[color_id]}_cell"
            )
            result.append((color_id, 0, int(final_state[field]) - 1))
        return result
    ordinal = int(sequence["queried_ordinals"][row])
    field = contract.STATE_FIELDS.index(f"ordered_cell_{ordinal - 1}")
    return [
        (
            int(sequence["goal_color_ids"][row, 0]),
            ordinal,
            int(final_state[field]) - 1,
        )
    ]


def _compact(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    mean = values.mean(axis=1)
    delta = values[:, -1] - values[:, 0]
    motion = np.abs(np.diff(values, axis=1)).mean(axis=1)
    return np.concatenate((mean, delta, motion), axis=-1).astype(np.float16)


def build_cache(args: argparse.Namespace) -> None:
    cache_dir = args.output_dir / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        compact_path = cache_dir / f"{split}.h5"
        query_path = cache_dir / f"{split}.npz"
        if compact_path.exists() and query_path.exists() and not args.rebuild_cache:
            print(f"using existing cache {compact_path}", flush=True)
            continue
        sequence = _load_npz(args.sequence_dir / f"{split}.npz")
        teacher = _load_npz(args.teacher_dir / f"{split}.npz")
        values: dict[str, list[Any]] = defaultdict(list)
        with h5py.File(args.feature_dir / f"{split}.h5", "r") as source, h5py.File(
            compact_path, "w"
        ) as target:
            target.attrs.update(
                schema_version=1,
                feature_contract="mean,last-minus-first,mean-absolute-frame-difference",
                source=str((args.feature_dir / f"{split}.h5").resolve()),
                target_leakage=False,
            )
            episode_row = 0
            for row in range(len(sequence["task_ids"])):
                task_id = int(sequence["task_ids"][row])
                if task_id >= len(TASKS):
                    continue
                raw = source[f"episode_{row:06d}/patch_tokens"][()]
                target.create_dataset(
                    f"episode_{episode_row:06d}", data=_compact(raw), compression="lzf"
                )
                final = _final_state(sequence, teacher, row)
                for color_id, ordinal, region in _query_rows(
                    task_id, sequence, final, row
                ):
                    if not 0 <= region < int(sequence["num_regions"][row]):
                        raise ValueError(f"Invalid region at {split}:{row}: {region}")
                    values["episode_rows"].append(episode_row)
                    values["task_ids"].append(task_id)
                    values["query_color_ids"].append(color_id)
                    values["queried_ordinals"].append(ordinal)
                    values["num_regions"].append(int(sequence["num_regions"][row]))
                    values["targets"].append(region)
                    values["episode_index"].append(
                        int(sequence["episode_index"][row])
                    )
                episode_row += 1
        np.savez_compressed(query_path, **{k: np.asarray(v) for k, v in values.items()})
        print(
            f"cached {split}: {episode_row} episodes, {len(values['targets'])} queries",
            flush=True,
        )


def _load_compact(path: Path) -> list[np.ndarray]:
    with h5py.File(path, "r") as source:
        return [np.asarray(source[key][()], dtype=np.float16) for key in sorted(source)]


def _balanced_indices(
    data: dict[str, np.ndarray], rng: np.random.Generator, size: int
) -> np.ndarray:
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
    return np.asarray(
        [
            groups[keys[int(rng.integers(len(keys)))]] or [0]
            for _ in range(size)
        ],
        dtype=object,
    )


def _sample_indices(
    data: dict[str, np.ndarray], rng: np.random.Generator, size: int
) -> np.ndarray:
    groups: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(
        zip(
            data["task_ids"],
            data["query_color_ids"],
            data["num_regions"],
            data["targets"],
            strict=True,
        )
    ):
        groups[tuple(int(value) for value in row)].append(index)
    keys = list(groups)
    result = []
    for _ in range(size):
        group = groups[keys[int(rng.integers(len(keys)))]]
        result.append(group[int(rng.integers(len(group)))])
    return np.asarray(result, dtype=np.int32)


def _batch(
    queries: dict[str, np.ndarray],
    episodes: list[np.ndarray],
    indices: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    count = len(indices)
    features = np.zeros(
        (count, MAX_CHUNKS, SPATIAL_TOKENS, COMPACT_WIDTH), dtype=np.float16
    )
    mask = np.zeros((count, MAX_CHUNKS), dtype=np.bool_)
    for batch_row, query_index in enumerate(indices):
        episode = episodes[int(queries["episode_rows"][query_index])]
        length = len(episode)
        features[batch_row, :length] = episode
        mask[batch_row, :length] = True
        if rng is not None and length > 4:
            # Weak temporal dropout regularizes redundant full-video chunks;
            # the first and last chunks are always preserved.
            dropped = rng.random(length) < 0.05
            dropped[0] = dropped[-1] = False
            features[batch_row, :length][dropped] = 0
            mask[batch_row, :length][dropped] = False
    return {
        "features": features,
        "chunk_mask": mask,
        "task_ids": queries["task_ids"][indices],
        "query_color_ids": queries["query_color_ids"][indices],
        "queried_ordinals": queries["queried_ordinals"][indices],
        "num_regions": queries["num_regions"][indices],
        "targets": queries["targets"][indices],
    }


def _inputs(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {
        key: jnp.asarray(batch[key])
        for key in (
            "features",
            "chunk_mask",
            "task_ids",
            "query_color_ids",
            "queried_ordinals",
            "num_regions",
        )
    }


def _summary(
    predictions: np.ndarray, queries: dict[str, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    correct = predictions == queries["targets"]
    for task_id, task in enumerate(TASKS):
        mask = queries["task_ids"] == task_id
        episode_values: dict[int, list[bool]] = defaultdict(list)
        for episode, value in zip(
            queries["episode_index"][mask], correct[mask], strict=True
        ):
            episode_values[int(episode)].append(bool(value))
        result[task] = {
            "queries": int(mask.sum()),
            "query_accuracy": float(correct[mask].mean()),
            "episodes": len(episode_values),
            "episode_exact_accuracy": float(
                np.mean([all(values) for values in episode_values.values()])
            ),
        }
    result["overall"] = {
        "queries": len(correct),
        "query_accuracy": float(correct.mean()),
    }
    return result


def train(args: argparse.Namespace) -> None:
    queries = {
        split: _load_npz(args.output_dir / "data" / f"{split}.npz")
        for split in ("train", "dev", "test")
    }
    episodes = {
        split: _load_compact(args.output_dir / "data" / f"{split}.h5")
        for split in ("train", "dev", "test")
    }
    model = FullContextRegionSummarizer(width=args.width, depth=args.depth)
    template = _batch(queries["train"], episodes["train"], np.asarray([0]))
    params = model.init(jax.random.key(args.seed), **_inputs(template))["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        args.learning_rate,
        warmup_steps=min(100, max(args.steps // 10, 1)),
        decay_steps=args.steps,
        end_value=args.learning_rate * 0.05,
    )
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(current_params, current_opt_state, batch):
        def loss_fn(candidate):
            logits = model.apply({"params": candidate}, **_inputs(batch))
            target = jnp.asarray(batch["targets"], dtype=jnp.int32)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, target
            ).mean()
            accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == target)
            return loss, accuracy

        (loss, accuracy), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            current_params
        )
        updates, next_opt = optimizer.update(grads, current_opt_state, current_params)
        return optax.apply_updates(current_params, updates), next_opt, loss, accuracy

    @jax.jit
    def predict(current_params, batch):
        return jnp.argmax(
            model.apply({"params": current_params}, **_inputs(batch)), axis=-1
        )

    def evaluate(current_params, split: str) -> dict[str, Any]:
        output = []
        for start in range(0, len(queries[split]["targets"]), args.eval_batch_size):
            indices = np.arange(
                start,
                min(start + args.eval_batch_size, len(queries[split]["targets"])),
            )
            # Keep a fixed evaluation batch shape to avoid recompilation.
            if len(indices) < args.eval_batch_size:
                indices = np.pad(indices, (0, args.eval_batch_size - len(indices)), mode="edge")
                valid = len(queries[split]["targets"]) - start
            else:
                valid = len(indices)
            batch = _batch(queries[split], episodes[split], indices)
            output.extend(np.asarray(predict(current_params, batch))[:valid].tolist())
        return _summary(np.asarray(output), queries[split])

    rng = np.random.default_rng(args.seed)
    best_score = (-1.0, -1.0)
    best_step = 0
    best_params = params
    history = []
    for step_index in range(1, args.steps + 1):
        indices = _sample_indices(queries["train"], rng, args.batch_size)
        batch = _batch(queries["train"], episodes["train"], indices, rng=rng)
        params, opt_state, loss, accuracy = step(params, opt_state, batch)
        if step_index % args.eval_every == 0 or step_index == args.steps:
            dev = evaluate(params, "dev")
            task_values = [dev[task]["query_accuracy"] for task in TASKS]
            score = (min(task_values), float(np.mean(task_values)))
            if score > best_score:
                best_score = score
                best_step = step_index
                best_params = jax.device_get(params)
            row = {
                "step": step_index,
                "train_loss": float(loss),
                "train_batch_accuracy": float(accuracy),
                "selection_min_task_accuracy": score[0],
                "selection_mean_task_accuracy": score[1],
                "dev": dev,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    results = {
        "schema_version": 1,
        "experiment": "full_context_bidirectional_visual_region_ceiling",
        "input_contract": {
            "visual": "all non-overlapping RGB frames from frame 0 through demonstration end, frozen 4x4 SigLIP tokens",
            "goal": "task, target color, queried ordinal, candidate count",
            "forbidden": [
                "canonical event labels",
                "event boundaries",
                "state targets at prediction time",
                "execution GroundSG coordinates",
            ],
        },
        "selection": "max min(task dev query accuracy), tie mean task accuracy",
        "best_step": best_step,
        "best_score": {
            "min_task_accuracy": best_score[0],
            "mean_task_accuracy": best_score[1],
        },
        "metrics": {
            split: evaluate(best_params, split)
            for split in ("train", "dev", "test")
        },
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "params.msgpack").write_bytes(
        flax.serialization.to_bytes(best_params)
    )
    (args.output_dir / "result.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                **{
                    key: str(value)
                    for key, value in vars(args).items()
                    if isinstance(value, Path)
                },
                "jax_devices": [str(device) for device in jax.devices()],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results["metrics"], indent=2), flush=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT.parent / ".cache/openpi"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_cache(args)
    if not args.build_cache_only:
        train(args)


if __name__ == "__main__":
    main()


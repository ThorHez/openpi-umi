#!/usr/bin/env python3
"""Train a PickXTimes event head and evaluate deterministic count rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import flax
from flax import traverse_util
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

from openpi.tasks.robomme.pickxtimes.explicit_event_count_memory import (  # noqa: E402
    EVENT_NAMES,
    PickExplicitEventHead,
    PickTargetMotionEventHead,
    deterministic_update_numpy,
)
from scripts.mem import probe_pickxtimes_transition_rollout_decomposition as base  # noqa: E402


DEFAULT_INIT = ROOT / "checkpoints/pickxtimes_transition_decomp_rgb_proprio_seed260866_260828/best/params"
DEFAULT_OUTPUT = ROOT / "checkpoints/pickxtimes_explicit_event_count_seed260831_260831"
DEFAULT_WRIST = ROOT / "artifacts/pickxtimes_fixed_chunk_wrist_features_4x4_v1_260831"
DEFAULT_MOTION = ROOT / "artifacts/pickxtimes_target_motion_features_v1_260831"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=base.DEFAULT_SEQUENCE)
    parser.add_argument("--teacher-memory-dir", type=Path, default=base.DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=base.DEFAULT_FEATURES)
    parser.add_argument("--proprio-dir", type=Path, default=base.DEFAULT_PROPRIO)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--rollout-cache-dir", type=Path, default=None)
    parser.add_argument("--rollout-fraction", type=float, default=0.5)
    parser.add_argument(
        "--visual-mode",
        choices=("front", "front_wrist", "front_wrist_motion"),
        default="front",
    )
    parser.add_argument("--wrist-feature-dir", type=Path, default=DEFAULT_WRIST)
    parser.add_argument("--target-motion-dir", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--end-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=260831)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def inputs(batch, visual_mode: str):
    result = {
        "previous_state": jnp.asarray(batch["previous_state"]),
        "required_count": jnp.asarray(batch["required_count"]),
        "rgb": jnp.asarray(batch["rgb"]),
        "proprio": jnp.asarray(batch["proprio"]),
    }
    if visual_mode != "front":
        result.update(
            {
                "target_color_id": jnp.asarray(batch["target_color_id"]),
                "wrist": jnp.asarray(batch["wrist"]),
                "target_motion": jnp.asarray(batch["target_motion"]),
            }
        )
    return result


class PickVerifierDataset(base.PickTransitionDataset):
    def __init__(self, split: str, args: argparse.Namespace):
        super().__init__(split, args)
        self.wrist_features = h5py.File(args.wrist_feature_dir / f"{split}.h5", "r")
        self.target_motion_features = h5py.File(args.target_motion_dir / f"{split}.h5", "r")
        motion_summary = json.loads((args.target_motion_dir / "summary.json").read_text())
        self.motion_mean = np.asarray(motion_summary["normalization"]["mean"], np.float32)
        self.motion_std = np.asarray(motion_summary["normalization"]["std"], np.float32)

    def close(self) -> None:
        super().close()
        self.wrist_features.close()
        self.target_motion_features.close()

    def _extra(self, episodes: np.ndarray, steps: np.ndarray) -> dict[str, np.ndarray]:
        wrist = np.zeros((len(episodes), 12, 16, 1152), dtype=np.float16)
        motion = np.zeros((len(episodes), 12, 11), dtype=np.float32)
        for row, (episode, step) in enumerate(zip(episodes, steps, strict=True)):
            source = int(self.source_indices[int(episode)])
            name = f"episode_{source:06d}"
            wrist[row] = self.wrist_features[f"{name}/patch_tokens"][int(step)]
            raw = np.asarray(
                self.target_motion_features[f"{name}/target_motion"][int(step)],
                dtype=np.float32,
            )
            motion[row] = (raw - self.motion_mean) / self.motion_std
        return {
            "target_color_id": self.goal_color_ids[episodes],
            "wrist": wrist,
            "target_motion": motion,
        }

    def rows(self, row_indices: np.ndarray) -> dict[str, np.ndarray]:
        result = super().rows(row_indices)
        episodes = self.row_episode[row_indices]
        steps = self.row_step[row_indices]
        result.update(self._extra(episodes, steps))
        return result

    def episode_step(self, step: int, previous_state: np.ndarray):
        result, active = super().episode_step(step, previous_state)
        episodes = np.arange(len(self.lengths), dtype=np.int32)
        safe_steps = np.minimum(step, np.maximum(self.lengths - 1, 0)).astype(np.int32)
        result.update(self._extra(episodes, safe_steps))
        return result, active


def restore_matching(params, path: Path):
    source = flax.serialization.msgpack_restore(path.read_bytes())
    target_flat = traverse_util.flatten_dict(params)
    source_flat = traverse_util.flatten_dict(source)
    loaded = []
    for key, value in source_flat.items():
        if key in target_flat and np.shape(value) == np.shape(target_flat[key]):
            target_flat[key] = jnp.asarray(value, dtype=target_flat[key].dtype)
            loaded.append("/".join(key))
    print(json.dumps({"initialized_from": str(path), "loaded_leaves": len(loaded)}))
    return traverse_util.unflatten_dict(target_flat), loaded


ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}


def _oracle_state(text: str, required_count: int, previous_count: int) -> np.ndarray:
    normalized = text.lower()
    if "press" in normalized and "button" in normalized:
        return np.asarray((required_count, 0, 1, 0), dtype=np.int32)
    if "place" in normalized:
        return np.asarray((previous_count, 1, 0, 0), dtype=np.int32)
    completed = next((value for word, value in ORDINALS.items() if word in normalized), None)
    if completed is None:
        match = re.search(r"(?:pick|grasp).*(\d+)", normalized)
        completed = int(match.group(1)) - 1 if match else previous_count
    return np.asarray((completed, 0, 0, 0), dtype=np.int32)


def _event_from_states(previous: np.ndarray, current: np.ndarray) -> int:
    if np.array_equal(previous, current):
        return 0
    if current[3] > previous[3]:
        return 3
    if current[0] > previous[0] or current[2] > previous[2]:
        return 2
    if current[1] > previous[1]:
        return 1
    return 0


def load_rollout_rows(cache_dir: Path, proprio_mean: np.ndarray, proprio_std: np.ndarray):
    rows = {key: [] for key in ("previous_state", "required_count", "event_id", "rgb", "proprio", "hard_negative")}
    trace_dir = cache_dir.parent / "mem_traces"
    episodes = 0
    for path in sorted(cache_dir.glob("PickXtimes_ep*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            rgb = np.asarray(payload["rgb_tokens"], dtype=np.float16)
            proprio = np.asarray(payload["proprio"], dtype=np.float32)
            phases = np.asarray(payload["oracle_phase"]).astype(str)
            required = int(payload["required_count"])
        trace_path = trace_dir / f"{path.stem}.json"
        trace = json.loads(trace_path.read_text())["trace"]
        if not (len(rgb) == len(proprio) == len(phases) == len(trace)):
            raise ValueError(f"Mismatched rollout cache lengths for {path}")
        previous = np.zeros(4, dtype=np.int32)
        previous_count = 0
        for index, phase in enumerate(phases):
            current = _oracle_state(phase, required, previous_count)
            event_id = _event_from_states(previous, current)
            hard_negative = bool(
                event_id == 0
                and (
                    int(trace[index].get("raw_event_id", 0)) != 0
                    or int(trace[index].get("observable_gate", {}).get("candidate_event_id", 0)) != 0
                )
            )
            rows["previous_state"].append(previous.copy())
            rows["required_count"].append(required)
            rows["event_id"].append(event_id)
            rows["rgb"].append(rgb[index])
            rows["proprio"].append((proprio[index] - proprio_mean) / proprio_std)
            rows["hard_negative"].append(hard_negative)
            previous = current
            previous_count = int(current[0])
        episodes += 1
    if not rows["event_id"]:
        raise ValueError(f"No rollout caches found in {cache_dir}")
    result = {key: np.asarray(value) for key, value in rows.items()}
    result["episodes"] = episodes
    return result


def summarize(predicted_events, predicted_states, dataset):
    valid = dataset.sequence_mask
    targets = dataset.next_states
    exact = np.all(predicted_states == targets, axis=-1) & valid
    transition = dataset.state_change_mask & valid
    hold = valid & ~transition
    final_index = dataset.lengths - 1
    final = exact[np.arange(len(exact)), final_index]
    event_correct = (predicted_events == dataset.event_ids) & valid
    result = {
        "episodes": len(dataset.lengths),
        "state_exact_accuracy": float(exact.sum() / valid.sum()),
        "transition_state_exact_accuracy": float((exact & transition).sum() / transition.sum()),
        "hold_state_exact_accuracy": float((exact & hold).sum() / hold.sum()),
        "sequence_exact_accuracy": float(np.mean(np.all(exact | ~valid, axis=1))),
        "final_state_exact_accuracy": float(np.mean(final)),
        "event_accuracy": float(event_correct.sum() / valid.sum()),
    }
    recalls = []
    for event_id, name in enumerate(EVENT_NAMES):
        mask = valid & (dataset.event_ids == event_id)
        recall = float((event_correct & mask).sum() / max(mask.sum(), 1))
        result[f"event/{name}_recall"] = recall
        recalls.append(recall)
    result["event_macro_recall"] = float(np.mean(recalls))
    predicted_update = valid & (predicted_events != 0)
    true_update = predicted_update & (predicted_events == dataset.event_ids)
    result["event_update_precision"] = float(true_update.sum() / max(predicted_update.sum(), 1))
    result["event_update_recall"] = float(true_update.sum() / transition.sum())
    return result


def main() -> None:
    args = parse_args()
    if args.rollout_cache_dir is not None and args.visual_mode != "front":
        raise ValueError("Rollout augmentation currently supports --visual-mode=front only")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_args = argparse.Namespace(**vars(args), mode="rgb_proprio")
    dataset_type = base.PickTransitionDataset if args.visual_mode == "front" else PickVerifierDataset
    datasets = {split: dataset_type(split, data_args) for split in ("train", "dev", "test")}
    try:
        model = (
            PickExplicitEventHead()
            if args.visual_mode == "front"
            else PickTargetMotionEventHead(mode=args.visual_mode)
        )
        initial = datasets["train"].rows(np.arange(min(args.batch_size, len(datasets["train"]))))
        params = model.init(
            jax.random.key(args.seed), **inputs(initial, args.visual_mode), train=False
        )["params"]
        params, loaded = restore_matching(params, args.init_checkpoint)
        schedule = optax.warmup_cosine_decay_schedule(
            0.0, args.learning_rate, min(args.warmup_steps, args.steps - 1), args.steps,
            end_value=args.end_learning_rate,
        )
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
        opt_state = optimizer.init(params)
        rng = np.random.default_rng(args.seed)
        rollout = None
        if args.rollout_cache_dir is not None:
            if not 0.0 < args.rollout_fraction < 1.0:
                raise ValueError("--rollout-fraction must be in (0,1)")
            rollout = load_rollout_rows(
                args.rollout_cache_dir,
                datasets["train"].proprio_mean,
                datasets["train"].proprio_std,
            )
            print(json.dumps({
                "rollout_episodes": int(rollout["episodes"]),
                "rollout_rows": int(len(rollout["event_id"])),
                "rollout_events": np.bincount(rollout["event_id"], minlength=4).tolist(),
                "rollout_hard_negatives": int(rollout["hard_negative"].sum()),
            }), flush=True)

        @jax.jit
        def train_step(current_params, current_opt, batch):
            def objective(p):
                logits = model.apply(
                    {"params": p}, **inputs(batch, args.visual_mode), train=True
                )
                targets = jnp.asarray(batch["event_id"])
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
                accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == targets)
                return loss, {"loss": loss, "event_accuracy": accuracy}
            (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(current_params)
            updates, next_opt = optimizer.update(gradients, current_opt, current_params)
            return optax.apply_updates(current_params, updates), next_opt, metrics

        @jax.jit
        def infer(current_params, batch):
            return model.apply(
                {"params": current_params}, **inputs(batch, args.visual_mode), train=False
            )

        def evaluate(current_params, split: str):
            dataset = datasets[split]
            states = np.zeros_like(dataset.next_states)
            events = np.zeros_like(dataset.event_ids)
            current = np.array(dataset.previous_states[:, 0], copy=True)
            for step in range(dataset.sequence_mask.shape[1]):
                batch, active = dataset.episode_step(step, current)
                prediction = np.argmax(np.asarray(infer(current_params, batch)), axis=-1)
                for episode in np.flatnonzero(active):
                    current[episode] = deterministic_update_numpy(
                        current[episode], int(prediction[episode]), int(dataset.required_counts[episode])
                    )
                events[:, step] = prediction
                states[:, step] = current
            return summarize(events, states, dataset)

        def evaluate_rollout(current_params):
            if rollout is None:
                return None
            predictions = []
            for start in range(0, len(rollout["event_id"]), args.batch_size):
                stop = min(start + args.batch_size, len(rollout["event_id"]))
                batch = {key: rollout[key][start:stop] for key in (
                    "previous_state", "required_count", "event_id", "rgb", "proprio"
                )}
                predictions.append(np.argmax(np.asarray(infer(current_params, batch)), axis=-1))
            prediction = np.concatenate(predictions)
            target = rollout["event_id"]
            hard = rollout["hard_negative"].astype(bool)
            update = target != 0
            return {
                "event_accuracy": float(np.mean(prediction == target)),
                "transition_recall": float(np.mean(prediction[update] == target[update])) if np.any(update) else 1.0,
                "hold_accuracy": float(np.mean(prediction[~update] == 0)),
                "hard_negative_accuracy": float(np.mean(prediction[hard] == 0)) if np.any(hard) else 1.0,
                "predicted_updates": int(np.sum(prediction != 0)),
                "true_updates": int(np.sum(update)),
            }

        best_params = params
        best_step = 0
        best_score = (-1.0,) * 6
        history = []
        started = time.monotonic()
        event_rows = [
            np.flatnonzero(datasets["train"].row_events == event_id)
            for event_id in range(len(EVENT_NAMES))
        ]
        if rollout is not None:
            rollout_groups = {
                "hard": np.flatnonzero(rollout["hard_negative"]),
                "event": np.flatnonzero(rollout["event_id"] != 0),
                "hold": np.flatnonzero((rollout["event_id"] == 0) & ~rollout["hard_negative"]),
            }

        def sample_expert(count: int):
            hold_count = count // 2
            remaining = count - hold_count
            rows = [rng.choice(event_rows[0], hold_count, replace=True)]
            per_event = [remaining // 3] * 3
            for index in range(remaining % 3):
                per_event[index] += 1
            rows.extend(
                rng.choice(event_rows[event_id], number, replace=True)
                for event_id, number in zip(range(1, 4), per_event, strict=True)
                if number
            )
            selected = np.concatenate(rows)
            rng.shuffle(selected)
            return datasets["train"].rows(selected)

        def sample_rollout(count: int):
            hard_count = count // 2
            event_count = (count - hard_count) // 2
            hold_count = count - hard_count - event_count
            indices = np.concatenate((
                rng.choice(rollout_groups["hard"], hard_count, replace=True),
                rng.choice(rollout_groups["event"], event_count, replace=True),
                rng.choice(rollout_groups["hold"], hold_count, replace=True),
            ))
            rng.shuffle(indices)
            return {key: rollout[key][indices] for key in (
                "previous_state", "required_count", "event_id", "rgb", "proprio"
            )}

        def concatenate_batches(parts):
            return {key: np.concatenate([part[key] for part in parts], axis=0) for key in (
                "previous_state", "required_count", "event_id", "rgb", "proprio"
            )}

        for step in range(1, args.steps + 1):
            if rollout is None:
                batch = sample_expert(args.batch_size)
            else:
                rollout_count = int(round(args.batch_size * args.rollout_fraction))
                batch = concatenate_batches((
                    sample_expert(args.batch_size - rollout_count),
                    sample_rollout(rollout_count),
                ))
            params, opt_state, train_metrics = train_step(params, opt_state, batch)
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate(params, "dev")
                balance = min(
                    dev["transition_state_exact_accuracy"],
                    dev["hold_state_exact_accuracy"],
                    dev["final_state_exact_accuracy"],
                )
                score = (
                    balance,
                    dev["final_state_exact_accuracy"],
                    dev["transition_state_exact_accuracy"],
                    dev["hold_state_exact_accuracy"],
                    dev["event_update_precision"],
                    dev["event_update_recall"],
                )
                if score > best_score:
                    best_score, best_step, best_params = score, step, jax.device_get(params)
                row = {"step": step, "score": score, "train": {k: float(v) for k, v in train_metrics.items()}, "dev_free_rollout": dev}
                history.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        result = {
            "schema_version": 1,
            "experiment": "pickxtimes_learned_event_head_deterministic_count_updater",
            "deployment_inputs": ["RGB", "gripper", "gripper command", "EEF Z", "required count", "previous recurrent state"],
            "training_only_teacher": "privileged completed-event trajectory",
            "test_used_for_selection": False,
            "best_step": best_step,
            "best_score": best_score,
            "loaded_initialization_leaves": len(loaded),
            "metrics": {"train": evaluate(best_params, "train"), "dev": evaluate(best_params, "dev"), "test": evaluate(best_params, "test")},
            "rollout_training_metrics": evaluate_rollout(best_params),
            "elapsed_seconds": time.monotonic() - started,
            "history": history,
        }
        (args.output_dir / "params.msgpack").write_bytes(flax.serialization.to_bytes(best_params))
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update({"model": type(model).__name__, "event_names": EVENT_NAMES})
        (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()

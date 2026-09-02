#!/usr/bin/env python3
"""Decompose PickXTimes observation, transition, and free-rollout errors.

The probe is deliberately structured: it predicts the four dynamic Pick state
fields from the previous symbolic state plus either a fixed 12-frame
RGB/proprio chunk or the GT event type.  It is trained with GT previous states,
then evaluated both teacher-forced and autoregressively.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import flax
import flax.linen as nn
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib
from openpi.tasks.robomme.unified_visual_student import VisualWindowEncoder


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE = _ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_TEACHER = _ROOT / "artifacts/robomme_four_task_gt_teacher_memory_v2_260826"
DEFAULT_FEATURES = _ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826"
DEFAULT_PROPRIO = _ROOT / "artifacts/pickxtimes_fixed_chunk_proprio_v1_260828"
MODES = ("rgb_proprio", "gt_event")
DYNAMIC_FIELDS = ("completed_count", "holding", "ready_to_press", "done")
DYNAMIC_FIELD_INDICES = tuple(teacher_lib.STATE_FIELDS.index(name) for name in DYNAMIC_FIELDS)
DYNAMIC_CLASS_COUNTS = (6, 2, 2, 2)
EVENT_NAMES = ("no_change", "pick_complete", "place_complete", "press_complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--teacher-memory-dir", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--proprio-dir", type=Path, default=DEFAULT_PROPRIO)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260864)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def derive_event_ids(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Derive no-change/pick/place/press semantics from adjacent GT states."""

    if previous.shape != current.shape or previous.shape[-1] != len(DYNAMIC_FIELDS):
        raise ValueError(f"Expected matching [...,{len(DYNAMIC_FIELDS)}] states")
    changed = np.any(previous != current, axis=-1)
    event = np.zeros(changed.shape, dtype=np.int32)
    completed_increased = current[..., 0] > previous[..., 0]
    done_increased = current[..., 3] > previous[..., 3]
    pick = changed & ~completed_increased & ~done_increased & (current[..., 1] == 1)
    place = changed & completed_increased
    press = changed & done_increased
    event[pick] = 1
    event[place] = 2
    event[press] = 3
    unresolved = changed & (event == 0)
    if np.any(unresolved):
        examples = np.argwhere(unresolved)[:5].tolist()
        raise ValueError(f"Could not derive Pick event type at {examples}")
    return event


class PickTransitionDataset:
    def __init__(self, split: str, args: argparse.Namespace):
        self.split = split
        sequence = _load_npz(args.sequence_dir / f"{split}.npz")
        teacher = _load_npz(args.teacher_memory_dir / f"{split}.npz")
        pick_id = teacher_lib.TASKS.index("pickxtimes_local_event")
        self.source_indices = np.flatnonzero(sequence["task_ids"] == pick_id)
        self.required_counts = sequence["required_counts"][self.source_indices].astype(np.int32)
        self.goal_color_ids = sequence["goal_color_ids"][self.source_indices, 0].astype(np.int32)
        self.episode_indices = sequence["episode_index"][self.source_indices].astype(np.int32)
        self.sequence_mask = sequence["step_mask"][self.source_indices].astype(bool)
        self.state_change_mask = sequence["state_change_mask"][self.source_indices].astype(bool)
        state_index = sequence["teacher_state_index"][self.source_indices]
        source_targets = teacher["state_targets"][self.source_indices]
        mapped = np.take_along_axis(source_targets, state_index[..., None], axis=1)
        self.states = mapped[..., DYNAMIC_FIELD_INDICES].astype(np.int32)
        self.previous_states = self.states[:, :-1]
        self.next_states = self.states[:, 1:]
        self.event_ids = derive_event_ids(self.previous_states, self.next_states)
        if not np.array_equal((self.event_ids > 0) & self.sequence_mask, self.state_change_mask):
            mismatch = int(np.sum(((self.event_ids > 0) & self.sequence_mask) != self.state_change_mask))
            raise ValueError(f"Derived event/state-change mismatch on {split}: {mismatch}")
        self.lengths = self.sequence_mask.sum(axis=1).astype(np.int32)
        self.row_episode, self.row_step = np.nonzero(self.sequence_mask)
        self.row_events = self.event_ids[self.row_episode, self.row_step]
        self.transition_rows = np.flatnonzero(self.row_events > 0)
        self.hold_rows = np.flatnonzero(self.row_events == 0)
        self.features = None
        self.proprio = None
        self.proprio_mean = None
        self.proprio_std = None
        if args.mode == "rgb_proprio":
            self.features = h5py.File(args.feature_dir / f"{split}.h5", "r")
            self.proprio = h5py.File(args.proprio_dir / f"{split}.h5", "r")
            summary = json.loads((args.proprio_dir / "summary.json").read_text())
            self.proprio_mean = np.asarray(summary["normalization"]["mean"], dtype=np.float32)
            self.proprio_std = np.asarray(summary["normalization"]["std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.row_episode)

    def close(self) -> None:
        if self.features is not None:
            self.features.close()
        if self.proprio is not None:
            self.proprio.close()

    def _observations(
        self, episodes: np.ndarray, steps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        count = len(episodes)
        if self.features is None or self.proprio is None:
            return (
                np.zeros((count, 12, 16, 1), dtype=np.float16),
                np.zeros((count, 12, 0), dtype=np.float32),
            )
        rgb = np.zeros((count, 12, 16, 1152), dtype=np.float16)
        proprio = np.zeros((count, 12, len(self.proprio_mean)), dtype=np.float32)
        for row, (episode, step) in enumerate(zip(episodes, steps, strict=True)):
            source = int(self.source_indices[int(episode)])
            rgb[row] = self.features[f"episode_{source:06d}/patch_tokens"][int(step)]
            values = np.asarray(
                self.proprio[f"episode_{source:06d}/proprio"][int(step)], dtype=np.float32
            )
            proprio[row] = (values - self.proprio_mean) / self.proprio_std
        return rgb, proprio

    def rows(self, row_indices: np.ndarray) -> dict[str, np.ndarray]:
        episodes = self.row_episode[row_indices]
        steps = self.row_step[row_indices]
        rgb, proprio = self._observations(episodes, steps)
        return {
            "previous_state": self.previous_states[episodes, steps],
            "next_state": self.next_states[episodes, steps],
            "required_count": self.required_counts[episodes],
            "event_id": self.event_ids[episodes, steps],
            "rgb": rgb,
            "proprio": proprio,
        }

    def episode_step(
        self, step: int, previous_state: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        episodes = np.arange(len(self.lengths), dtype=np.int32)
        active = step < self.lengths
        safe_steps = np.minimum(step, np.maximum(self.lengths - 1, 0)).astype(np.int32)
        rgb, proprio = self._observations(episodes, safe_steps)
        batch = {
            "previous_state": previous_state,
            "required_count": self.required_counts,
            "event_id": self.event_ids[episodes, safe_steps],
            "rgb": rgb,
            "proprio": proprio,
        }
        return batch, active


class LowDimTemporalEncoder(nn.Module):
    width: int = 64

    @nn.compact
    def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.width, name="input")(values)
        position = self.param(
            "position", nn.initializers.normal(stddev=0.02), (1, 12, self.width), jnp.float32
        )
        x = x + position
        residual = nn.gelu(nn.Dense(self.width * 2, name="hidden")(x))
        x = nn.LayerNorm(name="output_ln")(x + nn.Dense(self.width, name="out")(residual))
        early = jnp.mean(x[:, :6], axis=1)
        late = jnp.mean(x[:, 6:], axis=1)
        delta = late - early
        return nn.gelu(
            nn.Dense(self.width * 2, name="summary")(
                jnp.concatenate((jnp.mean(x, axis=1), early, late, delta, jnp.abs(delta)), axis=-1)
            )
        )


class StructuredTransitionProbe(nn.Module):
    mode: str
    width: int = 64
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8

    @nn.compact
    def __call__(
        self,
        previous_state: jnp.ndarray,
        required_count: jnp.ndarray,
        event_id: jnp.ndarray,
        rgb: jnp.ndarray,
        proprio: jnp.ndarray,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        if previous_state.ndim != 2 or previous_state.shape[-1] != len(DYNAMIC_FIELDS):
            raise ValueError(f"Expected previous state [B,{len(DYNAMIC_FIELDS)}]")
        state_embed = nn.Embed(6, self.width, name="state_value_embedding")(previous_state)
        state_type = self.param(
            "state_field_type",
            nn.initializers.normal(stddev=0.02),
            (1, len(DYNAMIC_FIELDS), self.width),
            jnp.float32,
        )
        state_summary = (state_embed + state_type).reshape(previous_state.shape[0], -1)
        required = nn.Embed(6, self.width, name="required_count_embedding")(required_count)
        evidence: list[jnp.ndarray] = [state_summary, required]
        if self.mode == "gt_event":
            evidence.append(nn.Embed(len(EVENT_NAMES), self.width, name="event_embedding")(event_id))
        elif self.mode == "rgb_proprio":
            visual = VisualWindowEncoder(
                name="visual_window_encoder",
                frames=12,
                spatial_tokens=16,
                input_width=1152,
                width=self.width,
                encoder_width=self.encoder_width,
                depth=self.encoder_depth,
                num_heads=self.encoder_heads,
                dtype_mm="bfloat16",
            )(rgb, train=train).reshape(rgb.shape[0], 12, 16, self.width)
            visual = nn.LayerNorm(name="visual_output_ln")(visual.astype(jnp.float32))
            early = jnp.mean(visual[:, :6], axis=(1, 2))
            late = jnp.mean(visual[:, 6:], axis=(1, 2))
            delta = late - early
            evidence.append(
                nn.gelu(
                    nn.Dense(self.width * 2, name="visual_summary")(
                        jnp.concatenate(
                            (jnp.mean(visual, axis=(1, 2)), early, late, delta, jnp.abs(delta)),
                            axis=-1,
                        )
                    )
                )
            )
            evidence.append(LowDimTemporalEncoder(width=self.width, name="proprio_encoder")(proprio))
        else:
            raise ValueError(self.mode)
        fused = jnp.concatenate(evidence, axis=-1)
        hidden = nn.gelu(nn.Dense(self.width * 4, name="transition_hidden")(fused))
        hidden = nn.LayerNorm(name="transition_ln")(
            hidden + nn.Dense(self.width * 4, name="transition_residual")(
                nn.gelu(nn.Dense(self.width * 4, name="transition_residual_hidden")(hidden))
            )
        )
        logits = nn.Dense(len(DYNAMIC_FIELDS) * 6, name="state_out")(hidden)
        logits = logits.reshape(previous_state.shape[0], len(DYNAMIC_FIELDS), 6)
        valid = jnp.arange(6)[None, :] < jnp.asarray(DYNAMIC_CLASS_COUNTS)[:, None]
        return jnp.where(valid[None], logits, jnp.asarray(-1e9, dtype=logits.dtype))


def _model_inputs(batch: dict[str, np.ndarray]) -> dict[str, jax.Array]:
    return {
        key: jnp.asarray(batch[key])
        for key in ("previous_state", "required_count", "event_id", "rgb", "proprio")
    }


def _prediction_summary(
    predictions: np.ndarray,
    dataset: PickTransitionDataset,
) -> dict[str, Any]:
    valid = dataset.sequence_mask
    targets = dataset.next_states
    exact = np.all(predictions == targets, axis=-1) & valid
    transition = dataset.state_change_mask & valid
    hold = valid & ~transition
    final_indices = dataset.lengths - 1
    final = exact[np.arange(len(exact)), final_indices]
    result: dict[str, Any] = {
        "episodes": len(exact),
        "state_exact_accuracy": float(exact.sum() / valid.sum()),
        "transition_state_exact_accuracy": float((exact & transition).sum() / transition.sum()),
        "no_change_state_exact_accuracy": float((exact & hold).sum() / hold.sum()),
        "sequence_exact_accuracy": float(np.mean(np.all(exact | ~valid, axis=1))),
        "final_state_exact_accuracy": float(np.mean(final)),
    }
    for field, name in enumerate(DYNAMIC_FIELDS):
        result[f"field/{name}_accuracy"] = float(
            ((predictions[..., field] == targets[..., field]) & valid).sum() / valid.sum()
        )
    event_metrics = {}
    for event_id, name in enumerate(EVENT_NAMES):
        mask = valid & (dataset.event_ids == event_id)
        event_metrics[name] = {
            "states": int(mask.sum()),
            "state_exact_accuracy": float((exact & mask).sum() / max(mask.sum(), 1)),
        }
    result["by_event"] = event_metrics
    first_errors = []
    for episode, length in enumerate(dataset.lengths):
        incorrect = np.flatnonzero(~exact[episode, :length])
        first_errors.append(int(incorrect[0] + 1) if len(incorrect) else int(length + 1))
    result["mean_first_error_step"] = float(np.mean(first_errors))
    return result


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: PickTransitionDataset(split, args) for split in ("train", "dev", "test")}
    rng = np.random.default_rng(args.seed)
    model = StructuredTransitionProbe(
        mode=args.mode,
        width=args.width,
        encoder_width=args.encoder_width,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
    )
    initial = datasets["train"].rows(np.arange(min(args.batch_size, len(datasets["train"]))))
    params = model.init(jax.random.key(args.seed), **_model_inputs(initial), train=False)["params"]
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        args.learning_rate,
        min(args.warmup_steps, max(args.steps - 1, 0)),
        args.steps,
        end_value=args.end_learning_rate,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4)
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, batch):
        def objective(current_params):
            logits = model.apply({"params": current_params}, **_model_inputs(batch), train=True)
            targets = jnp.asarray(batch["next_state"])
            token_loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            loss = token_loss.mean()
            exact = jnp.mean(jnp.all(jnp.argmax(logits, axis=-1) == targets, axis=-1))
            return loss, {"loss": loss, "state_exact_accuracy": exact}

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, next_opt_state = optimizer.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), next_opt_state, metrics

    @jax.jit
    def infer(params, batch):
        return model.apply({"params": params}, **_model_inputs(batch), train=False)

    def teacher_forced_predictions(current_params, dataset: PickTransitionDataset):
        predictions = np.zeros_like(dataset.next_states)
        inference_batch = max(args.batch_size, 64)
        for start in range(0, len(dataset), inference_batch):
            rows = np.arange(start, min(start + inference_batch, len(dataset)))
            batch = dataset.rows(rows)
            predicted = np.argmax(np.asarray(infer(current_params, batch)), axis=-1)
            predictions[dataset.row_episode[rows], dataset.row_step[rows]] = predicted
        return predictions

    def free_rollout_predictions(current_params, dataset: PickTransitionDataset):
        predictions = np.zeros_like(dataset.next_states)
        current = np.array(dataset.previous_states[:, 0], copy=True)
        for step in range(dataset.sequence_mask.shape[1]):
            batch, active = dataset.episode_step(step, current)
            predicted = np.argmax(np.asarray(infer(current_params, batch)), axis=-1)
            current = np.where(active[:, None], predicted, current)
            predictions[:, step] = current
        return predictions

    def evaluate_teacher_forced(current_params, split: str) -> dict[str, Any]:
        dataset = datasets[split]
        return _prediction_summary(teacher_forced_predictions(current_params, dataset), dataset)

    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dynamic_fields": DYNAMIC_FIELDS,
        "event_names": EVENT_NAMES,
        "train_rows": len(datasets["train"]),
        "dev_rows": len(datasets["dev"]),
        "test_rows": len(datasets["test"]),
        "teacher_latent_used": False,
        "training_previous_state": "GT",
        "selection": "min(dev teacher-forced transition, no-change)",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    best_params = params
    best_score = (-1.0, -1.0, -1.0, -1.0)
    best_step = 0
    started = time.monotonic()
    try:
        with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as metrics_file:
            for step in range(args.steps + 1):
                if step % args.eval_every == 0 or step == args.steps:
                    dev = evaluate_teacher_forced(params, "dev")
                    score = (
                        min(
                            dev["transition_state_exact_accuracy"],
                            dev["no_change_state_exact_accuracy"],
                        ),
                        dev["transition_state_exact_accuracy"],
                        dev["no_change_state_exact_accuracy"],
                        dev["state_exact_accuracy"],
                    )
                    row = {"split": "dev_teacher_forced", "step": step, **dev}
                    metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                    metrics_file.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)
                    if score > best_score:
                        best_score = score
                        best_step = step
                        best_params = jax.device_get(params)
                if step == args.steps:
                    break
                half = args.batch_size // 2
                transition_rows = rng.choice(
                    datasets["train"].transition_rows, half, replace=True
                )
                hold_rows = rng.choice(
                    datasets["train"].hold_rows, args.batch_size - half, replace=True
                )
                rows = np.concatenate((transition_rows, hold_rows))
                rng.shuffle(rows)
                batch = datasets["train"].rows(rows)
                params, opt_state, train_metrics = train_step(params, opt_state, batch)
                if step == 0 or (step + 1) % 50 == 0:
                    row = {
                        "split": "train",
                        "step": step + 1,
                        "learning_rate": float(schedule(step + 1)),
                        **{key: float(value) for key, value in train_metrics.items()},
                    }
                    metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
                    metrics_file.flush()
                    print(json.dumps(row, sort_keys=True), flush=True)
        (args.output_dir / "best").mkdir(exist_ok=True)
        (args.output_dir / "best/params").write_bytes(flax.serialization.to_bytes(best_params))
        dev_teacher = evaluate_teacher_forced(best_params, "dev")
        dev_free = _prediction_summary(
            free_rollout_predictions(best_params, datasets["dev"]), datasets["dev"]
        )
        result: dict[str, Any] = {
            "mode": args.mode,
            "best_step": best_step,
            "best_dev_score": best_score,
            "dev": {"teacher_forced": dev_teacher, "free_rollout": dev_free},
            "dev_rollout_gap": {
                key: dev_teacher[key] - dev_free[key]
                for key in (
                    "state_exact_accuracy",
                    "transition_state_exact_accuracy",
                    "no_change_state_exact_accuracy",
                    "final_state_exact_accuracy",
                )
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        if not args.skip_test:
            test_teacher = evaluate_teacher_forced(best_params, "test")
            test_free = _prediction_summary(
                free_rollout_predictions(best_params, datasets["test"]), datasets["test"]
            )
            result["test"] = {
                "teacher_forced": test_teacher,
                "free_rollout": test_free,
            }
            result["test_rollout_gap"] = {
                key: test_teacher[key] - test_free[key]
                for key in (
                    "state_exact_accuracy",
                    "transition_state_exact_accuracy",
                    "no_change_state_exact_accuracy",
                    "final_state_exact_accuracy",
                )
            }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()

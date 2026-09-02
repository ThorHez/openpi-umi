#!/usr/bin/env python3
"""Train trigger-free RoboMME memory on fixed non-overlapping visual chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import flax
from flax import traverse_util
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.tasks.robomme import unified_fixed_chunk_student as student_lib
from openpi.tasks.robomme import unified_gt_teacher as teacher_lib

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEQUENCE = _ROOT / "artifacts/robomme_four_task_fixed_chunk_sequences_v1_260826"
DEFAULT_FEATURES = _ROOT / "artifacts/robomme_four_task_fixed_chunk_features_4x4_v1_260826"
DEFAULT_MEMORY = _ROOT / "artifacts/robomme_four_task_gt_teacher_memory_v2_260826"
DEFAULT_TEACHER_SEQUENCE = _ROOT / "artifacts/robomme_four_task_gt_teacher_sequences_v1_260826"
DEFAULT_TEACHER_TRAINING = _ROOT / "checkpoints/robomme_four_task_unified_gt_teacher_canonical_v2_260826"
DEFAULT_TEACHER_CHECKPOINT = DEFAULT_TEACHER_TRAINING / "best/params"
DEFAULT_OUTPUT = _ROOT / "checkpoints/robomme_four_task_fixed_chunk_student_v1_260826"
SPLITS = ("train", "dev", "test")
INPUT_KEYS = (
    "task_ids",
    "goal_color_ids",
    "required_counts",
    "queried_ordinals",
    "num_regions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--proprio-dir",
        type=Path,
        help="Optional fixed-chunk proprio cache aligned with --sequence-dir.",
    )
    parser.add_argument("--teacher-memory-dir", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--teacher-sequence-dir", type=Path, default=DEFAULT_TEACHER_SEQUENCE)
    parser.add_argument("--teacher-training-dir", type=Path, default=DEFAULT_TEACHER_TRAINING)
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", choices=teacher_lib.TASKS)
    parser.add_argument(
        "--balance-goal-region",
        action="store_true",
        help="For VideoUnmask, sample goal-color/target-region combinations uniformly.",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--end-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--memory-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--supervision-mode",
        choices=("full", "terminal_answer_only"),
        default="full",
        help=(
            "full uses dense teacher-memory and state-trajectory targets; "
            "terminal_answer_only removes both trajectory losses and supervises only "
            "the queried fields at the final valid state through the shared frozen readout."
        ),
    )
    parser.add_argument("--semantic-token-weight", type=float, default=4.0)
    parser.add_argument("--change-state-weight", type=float, default=6.0)
    parser.add_argument("--final-state-weight", type=float, default=1.0)
    parser.add_argument("--decoupled-state-loss", action="store_true")
    parser.add_argument("--change-readout-loss-weight", type=float, default=0.75)
    parser.add_argument("--no-change-readout-loss-weight", type=float, default=0.0)
    parser.add_argument("--final-readout-loss-weight", type=float, default=1.0)
    parser.add_argument("--keep-loss-weight", type=float, default=0.05)
    parser.add_argument("--direct-teacher-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--pick-transition-balanced-loss", action="store_true")
    parser.add_argument("--pick-transition-field-weighting", action="store_true")
    parser.add_argument(
        "--pick-no-change-field-weighting",
        action="store_true",
        help="Apply the PickXTimes transition field weights to the no-change readout loss too.",
    )
    parser.add_argument("--encoder-width", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=8)
    parser.add_argument("--write-gate", action="store_true")
    parser.add_argument("--write-gate-bias", type=float, default=-2.0)
    parser.add_argument("--event-gate", action="store_true")
    parser.add_argument(
        "--causal-evidence-state",
        action="store_true",
        help="Carry one shared visual-evidence state across fixed chunks.",
    )
    parser.add_argument(
        "--no-recurrent-carry",
        action="store_true",
        help="Reset latent memory to the goal-initialized state before every chunk.",
    )
    parser.add_argument("--event-gate-bias", type=float, default=0.0)
    parser.add_argument("--event-gate-modulation-strength", type=float, default=0.0)
    parser.add_argument("--event-gate-reference", type=float, default=0.2)
    parser.add_argument("--event-gate-modulation-min", type=float, default=0.75)
    parser.add_argument("--event-gate-modulation-max", type=float, default=1.25)
    parser.add_argument("--event-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--event-gate-rank-weight", type=float, default=0.0)
    parser.add_argument("--event-gate-sigma", type=float, default=0.5)
    parser.add_argument("--event-gate-rank-margin", type=float, default=0.3)
    parser.add_argument("--event-gate-label-smoothing", type=float, default=0.02)
    parser.add_argument(
        "--event-update-routing",
        action="store_true",
        help="Route zero-initialized event/hold update residuals with the independent event gate.",
    )
    parser.add_argument("--event-update-routing-temperature", type=float, default=1.0)
    parser.add_argument("--event-update-routing-reference", type=float, default=0.5)
    parser.add_argument(
        "--event-correction",
        action="store_true",
        help="Add a zero-initialized correction after the base gated update.",
    )
    parser.add_argument(
        "--oracle-event-correction",
        action="store_true",
        help="Use privileged state-change masks to activate event correction.",
    )
    parser.add_argument("--event-correction-delta-loss-weight", type=float, default=0.0)
    parser.add_argument("--privileged-soft-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--privileged-soft-gate-rank-weight", type=float, default=0.0)
    parser.add_argument("--privileged-soft-gate-floor", type=float, default=0.05)
    parser.add_argument("--privileged-soft-gate-peak", type=float, default=0.8)
    parser.add_argument("--privileged-soft-gate-sigma", type=float, default=0.75)
    parser.add_argument("--privileged-soft-gate-rank-margin", type=float, default=0.2)
    parser.add_argument(
        "--gate-only-training",
        action="store_true",
        help=(
            "Freeze every parameter except the recurrent updater's gate_* modules. "
            "This is intended as a privileged-label gate-separability diagnostic."
        ),
    )
    parser.add_argument(
        "--gate-proprio-only-training",
        action="store_true",
        help=(
            "Freeze everything except proprio_* and recurrent gate_* parameters, and "
            "select checkpoints by the dev change-minus-far-hold write-gate margin."
        ),
    )
    parser.add_argument(
        "--freeze-gate-proprio",
        action="store_true",
        help="Freeze the calibrated proprio_* and recurrent gate_* parameters.",
    )
    parser.add_argument(
        "--event-gate-only-training",
        action="store_true",
        help="Freeze all parameters except the independent event_* head.",
    )
    parser.add_argument(
        "--freeze-event-gate",
        action="store_true",
        help="Freeze event_* parameters while adapting the recurrent memory around modulation.",
    )
    parser.add_argument(
        "--route-only-training",
        action="store_true",
        help="Freeze all parameters except the route_* event/hold update branches.",
    )
    parser.add_argument(
        "--event-correction-only-training",
        action="store_true",
        help="Freeze all parameters except correction_* and optimize direct teacher correction.",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260826)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-progress-dir", type=Path)
    parser.add_argument("--no-progress-fraction", type=float, default=0.0)
    parser.add_argument("--no-progress-weight", type=float, default=4.0)
    parser.add_argument("--no-progress-gate-loss-weight", type=float, default=0.0)
    parser.add_argument("--no-progress-keep-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--no-progress-exclude-episodes",
        default="",
        help="Comma-separated episode ids reserved from no-progress training.",
    )
    parser.add_argument("--online-phase-dir", type=Path)
    parser.add_argument("--online-phase-fraction", type=float, default=0.0)
    parser.add_argument("--online-phase-weight", type=float, default=2.0)
    parser.add_argument("--online-hold-weight", type=float, default=2.0)
    parser.add_argument("--online-transition-weight", type=float, default=4.0)
    parser.add_argument("--online-hold-readout-loss-weight", type=float, default=1.0)
    parser.add_argument("--online-transition-readout-loss-weight", type=float, default=1.0)
    parser.add_argument("--online-hold-keep-loss-weight", type=float, default=0.1)
    parser.add_argument("--groundsg-action-teacher-dir", type=Path)
    parser.add_argument("--groundsg-action-loss-weight", type=float, default=0.0)
    parser.add_argument("--groundsg-action-min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--groundsg-action-gradient-projection",
        action="store_true",
        help="Remove GroundSG gradient components that conflict with canonical GT losses.",
    )
    parser.add_argument(
        "--groundsg-action-protect-transition-gradient",
        action="store_true",
        help="Additionally remove action-gradient components that hurt transition readout CE.",
    )
    parser.add_argument(
        "--groundsg-action-sample-fraction",
        type=float,
        default=0.0,
        help="Fraction of regular offline rows drawn from episodes with GroundSG targets.",
    )
    parser.add_argument(
        "--online-aligned-selection",
        action="store_true",
        help="Select checkpoints by transition, hold, trajectory, then final accuracy.",
    )
    parser.add_argument(
        "--min-transition-no-change-selection",
        action="store_true",
        help=(
            "Select checkpoints by min(transition,no-change), then transition, "
            "no-change, state and final accuracy."
        ),
    )
    parser.add_argument(
        "--terminal-answer-selection",
        action="store_true",
        help=(
            "Select checkpoints by dev terminal Answer, then final-state and all-state "
            "accuracy. This avoids privileged intermediate-trajectory model selection."
        ),
    )
    parser.add_argument(
        "--gate-margin-selection-floor",
        type=float,
        default=0.0,
        help=(
            "With min-transition-no-change selection, subtract a penalty when the dev "
            "change-minus-far-hold write-gate margin is below this floor."
        ),
    )
    parser.add_argument(
        "--gate-margin-selection-penalty",
        type=float,
        default=0.0,
        help="Multiplier for the gate-margin shortfall in checkpoint selection.",
    )
    return parser.parse_args()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    for singular, plural in (
        ("task_id", "task_ids"),
        ("required_count", "required_counts"),
        ("queried_ordinal", "queried_ordinals"),
    ):
        if singular in arrays:
            arrays[plural] = arrays.pop(singular)
    return arrays


class SplitDataset:
    def __init__(self, split: str, args: argparse.Namespace):
        self.split = split
        self.task = args.task
        self.pick_transition_balanced_loss = args.pick_transition_balanced_loss
        self.sequence = _load_npz(args.sequence_dir / f"{split}.npz")
        self.teacher = _load_npz(args.teacher_memory_dir / f"{split}.npz")
        self.features = h5py.File(args.feature_dir / f"{split}.h5", "r")
        self.proprio = None
        self.proprio_dim = 0
        self.proprio_mean = None
        self.proprio_std = None
        if args.proprio_dir is not None:
            self.proprio = h5py.File(args.proprio_dir / f"{split}.h5", "r")
            summary = json.loads((args.proprio_dir / "summary.json").read_text())
            self.proprio_mean = np.asarray(
                summary["normalization"]["mean"], dtype=np.float32
            )
            self.proprio_std = np.asarray(
                summary["normalization"]["std"], dtype=np.float32
            )
            self.proprio_dim = len(self.proprio_mean)
        self.length = len(self.sequence["task_ids"])
        if self.length != len(self.teacher["teacher_memory"]):
            raise ValueError(f"Sequence/teacher length mismatch on {split}")
        self.max_steps = self.sequence["step_mask"].shape[1]
        self.action_teacher = None
        self.action_teacher_rows = np.asarray([], dtype=np.int64)
        self.action_min_confidence = float(args.groundsg_action_min_confidence)
        if args.groundsg_action_teacher_dir is not None:
            path = args.groundsg_action_teacher_dir / f"{split}.npz"
            if path.exists():
                self.action_teacher = _load_npz(path)
                if not np.array_equal(
                    self.action_teacher["episode_index"], self.sequence["episode_index"]
                ):
                    raise ValueError(f"GroundSG action teacher episode mismatch on {split}")
                self.action_teacher_rows = np.flatnonzero(
                    np.any(self.action_teacher["action_phase_mask"], axis=1)
                )
        for index in range(self.length):
            group = self.features.get(f"episode_{index:06d}")
            if group is None or not bool(group.attrs.get("complete", False)):
                raise ValueError(f"Missing feature episode {split}:{index}")
            if int(group.attrs["episode_index"]) != int(self.teacher["episode_index"][index]):
                raise ValueError(f"Feature/teacher alignment mismatch on {split}:{index}")

    def close(self) -> None:
        self.features.close()
        if self.proprio is not None:
            self.proprio.close()

    def batch(
        self,
        indices: np.ndarray,
        *,
        change_state_weight: float,
        final_state_weight: float = 1.0,
    ) -> dict[str, np.ndarray]:
        batch_size = len(indices)
        patches = np.zeros((batch_size, self.max_steps, 12, 16, 1152), dtype=np.float16)
        proprio = np.zeros(
            (batch_size, self.max_steps, 12, self.proprio_dim), dtype=np.float32
        )
        for batch_index, episode_index in enumerate(indices):
            tokens = self.features[f"episode_{int(episode_index):06d}/patch_tokens"][:]
            patches[batch_index, : len(tokens)] = tokens
            if self.proprio is not None:
                path = f"episode_{int(episode_index):06d}/proprio"
                if path not in self.proprio:
                    raise ValueError(
                        f"Missing proprio episode {self.split}:{int(episode_index)}"
                    )
                values = np.asarray(self.proprio[path], dtype=np.float32)
                values = (values - self.proprio_mean) / self.proprio_std
                if len(values) != len(tokens):
                    raise ValueError(
                        f"Proprio/visual chunk mismatch {self.split}:{int(episode_index)}"
                    )
                proprio[batch_index, : len(values)] = values
        state_index = self.sequence["teacher_state_index"][indices]
        source_memory = self.teacher["teacher_memory"][indices]
        source_targets = self.teacher["state_targets"][indices]
        source_mask = self.teacher["state_field_mask"][indices]
        teacher_memory = np.take_along_axis(
            source_memory,
            state_index[..., None, None],
            axis=1,
        )
        state_targets = np.take_along_axis(
            source_targets,
            state_index[..., None],
            axis=1,
        )
        state_field_mask = np.take_along_axis(
            source_mask,
            state_index[..., None],
            axis=1,
        )
        sequence_mask = self.sequence["step_mask"][indices]
        valid_states = np.concatenate(
            (np.ones((batch_size, 1), dtype=np.bool_), sequence_mask), axis=1
        )
        state_field_mask &= valid_states[..., None]
        change_mask = self.sequence["state_change_mask"][indices]
        memory_state_weights = valid_states.astype(np.float32)
        memory_state_weights[:, 1:] += change_mask.astype(np.float32) * (
            change_state_weight - 1.0
        )
        state_weights = memory_state_weights.copy()
        final_indices = valid_states.sum(axis=1).astype(np.int64) - 1
        state_weights[np.arange(batch_size), final_indices] *= final_state_weight
        transition_state_mask = np.concatenate(
            (np.zeros((batch_size, 1), dtype=np.bool_), change_mask), axis=1
        ) & valid_states
        no_change_state_mask = valid_states & ~transition_state_mask
        transition_state_weights = transition_state_mask.astype(np.float32)
        if self.pick_transition_balanced_loss:
            if self.task != "pickxtimes_local_event":
                raise ValueError(
                    "--pick-transition-balanced-loss requires PickXTimes single-task training"
                )
            completed = state_targets[..., teacher_lib.STATE_FIELDS.index("completed_count")]
            holding = state_targets[..., teacher_lib.STATE_FIELDS.index("holding")]
            ready = state_targets[..., teacher_lib.STATE_FIELDS.index("ready_to_press")]
            done = state_targets[..., teacher_lib.STATE_FIELDS.index("done")]
            subtype = np.full(transition_state_mask.shape, -1, dtype=np.int32)
            subtype[transition_state_mask & (holding == 1)] = 0
            subtype[
                transition_state_mask
                & (holding == 0)
                & (ready == 0)
                & (done == 0)
                & (completed > 0)
            ] = 1
            subtype[transition_state_mask & (ready == 1)] = 2
            subtype[transition_state_mask & (done == 1)] = 3
            present = subtype >= 0
            counts = np.bincount(subtype[present], minlength=4).astype(np.float32)
            inverse = np.zeros_like(counts)
            np.divide(
                counts.sum(),
                4.0 * counts,
                out=inverse,
                where=counts > 0,
            )
            transition_state_weights[present] = inverse[subtype[present]]
        final_state_mask = np.zeros_like(valid_states)
        final_state_mask[np.arange(batch_size), final_indices] = True
        action_phase_probs = np.zeros((batch_size, self.max_steps + 1, 3), dtype=np.float32)
        action_phase_mask = np.zeros((batch_size, self.max_steps + 1), dtype=np.bool_)
        if self.action_teacher is not None:
            action_phase_probs = self.action_teacher["action_phase_probs"][indices]
            action_phase_mask = self.action_teacher["action_phase_mask"][indices]
            action_phase_mask &= (
                np.max(action_phase_probs, axis=-1) >= self.action_min_confidence
            )
        return {
            "patch_tokens": patches,
            "proprio": proprio,
            "sequence_mask": sequence_mask,
            "teacher_memory": teacher_memory,
            "state_targets": state_targets,
            "state_field_mask": state_field_mask,
            "state_weights": state_weights,
            "memory_state_weights": memory_state_weights,
            "valid_states": valid_states,
            "transition_state_mask": transition_state_mask,
            "transition_state_weights": transition_state_weights,
            "no_change_state_mask": no_change_state_mask,
            "final_state_mask": final_state_mask,
            "state_change_mask": change_mask,
            "no_progress_chunk_mask": np.zeros_like(sequence_mask),
            "online_hold_chunk_mask": np.zeros_like(sequence_mask),
            "online_transition_state_mask": np.zeros_like(valid_states),
            "action_phase_probs": action_phase_probs,
            "action_phase_mask": action_phase_mask,
            **{key: self.sequence[key][indices] for key in INPUT_KEYS},
        }


class NoProgressDataset:
    """On-policy sequences where the simulator confirms that no event completed."""

    def __init__(
        self,
        path: Path,
        reference: SplitDataset,
        *,
        excluded_episodes: set[int],
    ):
        files = sorted(path.expanduser().resolve().glob("episode_*.npz"))
        self.rows = []
        episode_lookup = {
            int(episode): index
            for index, episode in enumerate(reference.sequence["episode_index"])
            if int(reference.sequence["task_ids"][index])
            == teacher_lib.TASKS.index("pickxtimes_local_event")
        }
        for file in files:
            with np.load(file, allow_pickle=False) as payload:
                episode = int(payload["episode_index"])
            if episode in excluded_episodes:
                continue
            if episode not in episode_lookup:
                raise ValueError(f"No train reference for no-progress episode {episode}")
            self.rows.append((file, episode_lookup[episode]))
        if not self.rows:
            raise ValueError(f"No no-progress files selected from {path}")
        self.reference = reference

    def __len__(self) -> int:
        return len(self.rows)

    def batch(self, indices: np.ndarray, *, weight: float) -> dict[str, np.ndarray]:
        batch_size = len(indices)
        max_steps = self.reference.max_steps
        patches = np.zeros(
            (batch_size, max_steps, 12, 16, 1152), dtype=np.float16
        )
        sequence_mask = np.zeros((batch_size, max_steps), dtype=np.bool_)
        reference_indices = []
        for batch_index, row_index in enumerate(indices):
            file, reference_index = self.rows[int(row_index)]
            with np.load(file, allow_pickle=False) as payload:
                tokens = np.asarray(payload["patch_tokens"], dtype=np.float16)
            count = min(len(tokens) // 12, max_steps)
            patches[batch_index, :count] = tokens[: count * 12].reshape(
                count, 12, 16, 1152
            )
            sequence_mask[batch_index, :count] = True
            reference_indices.append(reference_index)
        reference_indices = np.asarray(reference_indices, dtype=np.int64)
        source = self.reference.batch(
            reference_indices, change_state_weight=1.0, final_state_weight=1.0
        )
        states = max_steps + 1
        valid_states = np.concatenate(
            (np.ones((batch_size, 1), dtype=np.bool_), sequence_mask), axis=1
        )
        teacher_memory = np.repeat(source["teacher_memory"][:, :1], states, axis=1)
        state_targets = np.repeat(source["state_targets"][:, :1], states, axis=1)
        state_field_mask = np.repeat(source["state_field_mask"][:, :1], states, axis=1)
        state_field_mask &= valid_states[..., None]
        state_weights = valid_states.astype(np.float32) * float(weight)
        final_state_mask = np.zeros_like(valid_states)
        final_indices = valid_states.sum(axis=1).astype(np.int64) - 1
        final_state_mask[np.arange(batch_size), final_indices] = True
        return {
            "patch_tokens": patches,
            "proprio": np.zeros(
                (batch_size, max_steps, 12, self.reference.proprio_dim), dtype=np.float32
            ),
            "sequence_mask": sequence_mask,
            "teacher_memory": teacher_memory,
            "state_targets": state_targets,
            "state_field_mask": state_field_mask,
            "state_weights": state_weights,
            "memory_state_weights": state_weights,
            "valid_states": valid_states,
            "transition_state_mask": np.zeros_like(valid_states),
            "transition_state_weights": np.zeros_like(valid_states, dtype=np.float32),
            "no_change_state_mask": valid_states.copy(),
            "final_state_mask": final_state_mask,
            "state_change_mask": np.zeros_like(sequence_mask),
            "no_progress_chunk_mask": sequence_mask.copy(),
            "online_hold_chunk_mask": np.zeros_like(sequence_mask),
            "online_transition_state_mask": np.zeros_like(valid_states),
            "action_phase_probs": np.zeros((batch_size, states, 3), dtype=np.float32),
            "action_phase_mask": np.zeros((batch_size, states), dtype=np.bool_),
            **{key: source[key] for key in INPUT_KEYS},
        }


class OnlinePhaseDataset:
    """Chunk-level Pick states from official-controller closed-loop rollouts."""

    STATE_NAMES = ("completed_count", "holding", "ready_to_press", "done")

    def __init__(self, path: Path, reference: SplitDataset):
        self.files = sorted(path.expanduser().resolve().glob("episode_*.npz"))
        if not self.files:
            raise ValueError(f"No online phase files found in {path}")
        self.reference = reference
        self.pick_task_id = teacher_lib.TASKS.index("pickxtimes_local_event")
        self.state_fields = np.asarray(
            [teacher_lib.STATE_FIELDS.index(name) for name in self.STATE_NAMES],
            dtype=np.int64,
        )
        self.pick_indices = np.flatnonzero(
            reference.sequence["task_ids"] == self.pick_task_id
        )

    def __len__(self) -> int:
        return len(self.files)

    def _prototype_memory(
        self,
        goal_color_id: int,
        required_count: int,
        state: np.ndarray,
    ) -> np.ndarray:
        candidates = self.pick_indices[
            (self.reference.sequence["goal_color_ids"][self.pick_indices, 0] == goal_color_id)
            & (self.reference.sequence["required_counts"][self.pick_indices] == required_count)
        ]
        if not len(candidates):
            raise ValueError(
                f"No canonical Pick prototype for color={goal_color_id}, count={required_count}"
            )
        for episode_index in candidates:
            length = int(self.reference.sequence["step_mask"][episode_index].sum())
            state_indices = self.reference.sequence["teacher_state_index"][
                episode_index, : length + 1
            ]
            states = self.reference.teacher["state_targets"][episode_index, state_indices]
            matches = np.all(states[:, self.state_fields] == state[None], axis=1)
            if np.any(matches):
                position = int(np.flatnonzero(matches)[0])
                return self.reference.teacher["teacher_memory"][
                    episode_index, state_indices[position]
                ]
        raise ValueError(
            f"No canonical Pick memory for color={goal_color_id}, count={required_count}, "
            f"state={state.tolist()}"
        )

    def batch(
        self,
        indices: np.ndarray,
        *,
        base_weight: float,
        hold_weight: float,
        transition_weight: float,
    ) -> dict[str, np.ndarray]:
        batch_size = len(indices)
        max_steps = self.reference.max_steps
        memory_shape = self.reference.teacher["teacher_memory"].shape[-2:]
        patches = np.zeros((batch_size, max_steps, 12, 16, 1152), dtype=np.float16)
        sequence_mask = np.zeros((batch_size, max_steps), dtype=np.bool_)
        state_targets = np.zeros(
            (batch_size, max_steps + 1, len(teacher_lib.STATE_FIELDS)), dtype=np.int32
        )
        state_field_mask = np.zeros_like(state_targets, dtype=np.bool_)
        teacher_memory = np.zeros(
            (batch_size, max_steps + 1, *memory_shape),
            dtype=self.reference.teacher["teacher_memory"].dtype,
        )
        task_ids = np.full((batch_size,), self.pick_task_id, dtype=np.int32)
        goal_color_ids = np.zeros((batch_size, 2), dtype=np.int32)
        required_counts = np.zeros((batch_size,), dtype=np.int32)
        queried_ordinals = np.zeros((batch_size,), dtype=np.int32)
        num_regions = np.zeros((batch_size,), dtype=np.int32)
        change_mask = np.zeros((batch_size, max_steps), dtype=np.bool_)

        for batch_index, row_index in enumerate(indices):
            with np.load(self.files[int(row_index)], allow_pickle=False) as payload:
                tokens = np.asarray(payload["patch_tokens"], dtype=np.float16)
                states = np.asarray(payload["state_values"], dtype=np.int32)
                color = int(payload["goal_color_id"])
                required = int(payload["required_count"])
            count = min(len(tokens), len(states), max_steps)
            if count < 1:
                raise ValueError(f"Empty online phase cache: {self.files[int(row_index)]}")
            patches[batch_index, :count] = tokens[:count]
            sequence_mask[batch_index, :count] = True
            goal_color_ids[batch_index, 0] = color
            required_counts[batch_index] = required
            initial = np.asarray([0, 0, 0, 0], dtype=np.int32)
            phase_states = np.concatenate((initial[None], states[:count]), axis=0)
            state_rows = np.arange(count + 1, dtype=np.int64)[:, None]
            state_cols = self.state_fields[None, :]
            state_targets[batch_index, state_rows, state_cols] = phase_states
            state_field_mask[batch_index, state_rows, state_cols] = True
            for state_index, state in enumerate(phase_states):
                teacher_memory[batch_index, state_index] = self._prototype_memory(
                    color, required, state
                )
            change_mask[batch_index, :count] = np.any(
                phase_states[1:] != phase_states[:-1], axis=1
            )

        valid_states = np.concatenate(
            (np.ones((batch_size, 1), dtype=np.bool_), sequence_mask), axis=1
        )
        transition_state_mask = np.concatenate(
            (np.zeros((batch_size, 1), dtype=np.bool_), change_mask), axis=1
        ) & valid_states
        no_change_state_mask = valid_states & ~transition_state_mask
        online_hold_chunk_mask = sequence_mask & ~change_mask
        state_weights = valid_states.astype(np.float32) * float(base_weight)
        state_weights[:, 1:] *= np.where(
            change_mask,
            float(transition_weight),
            float(hold_weight),
        )
        final_state_mask = np.zeros_like(valid_states)
        final_indices = valid_states.sum(axis=1).astype(np.int64) - 1
        final_state_mask[np.arange(batch_size), final_indices] = True
        return {
            "patch_tokens": patches,
            "proprio": np.zeros(
                (batch_size, max_steps, 12, self.reference.proprio_dim), dtype=np.float32
            ),
            "sequence_mask": sequence_mask,
            "teacher_memory": teacher_memory,
            "state_targets": state_targets,
            "state_field_mask": state_field_mask,
            "state_weights": state_weights,
            "memory_state_weights": state_weights,
            "valid_states": valid_states,
            "transition_state_mask": transition_state_mask,
            "transition_state_weights": transition_state_mask.astype(np.float32),
            "no_change_state_mask": no_change_state_mask,
            "final_state_mask": final_state_mask,
            "state_change_mask": change_mask,
            "no_progress_chunk_mask": np.zeros_like(sequence_mask),
            "online_hold_chunk_mask": online_hold_chunk_mask,
            "online_transition_state_mask": transition_state_mask,
            "action_phase_probs": np.zeros((batch_size, max_steps + 1, 3), dtype=np.float32),
            "action_phase_mask": np.zeros((batch_size, max_steps + 1), dtype=np.bool_),
            "task_ids": task_ids,
            "goal_color_ids": goal_color_ids,
            "required_counts": required_counts,
            "queried_ordinals": queried_ordinals,
            "num_regions": num_regions,
        }


def _concatenate_batches(*batches: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in batches[0]
    }


def _student_inputs(
    batch: dict[str, np.ndarray | jax.Array], *, oracle_event_correction: bool = False
) -> dict[str, jax.Array]:
    inputs = {
        "patch_tokens": jnp.asarray(batch["patch_tokens"]),
        "proprio": jnp.asarray(batch["proprio"]),
        "sequence_mask": jnp.asarray(batch["sequence_mask"]),
        **{key: jnp.asarray(batch[key]) for key in INPUT_KEYS},
    }
    if oracle_event_correction:
        inputs["oracle_event_mask"] = jnp.asarray(batch["state_change_mask"])
    return inputs


def _balanced_indices(
    dataset: SplitDataset,
    rng: np.random.Generator,
    batch_size: int,
    *,
    task: str | None = None,
    balance_goal_region: bool = False,
):
    if task is not None:
        task_id = teacher_lib.TASKS.index(task)
        candidates = np.flatnonzero(dataset.sequence["task_ids"] == task_id)
        if balance_goal_region:
            if task != "videounmask_variable_demo":
                raise ValueError("--balance-goal-region currently supports VideoUnmask only")
            colors = dataset.sequence["goal_color_ids"][candidates, 0].astype(np.int64)
            lengths = dataset.sequence["step_mask"][candidates].sum(axis=1).astype(np.int64)
            final_state_indices = dataset.sequence["teacher_state_index"][candidates, lengths]
            final_states = dataset.teacher["state_targets"][candidates, final_state_indices]
            cell_fields = np.asarray(
                [
                    teacher_lib.STATE_FIELDS.index("red_cell"),
                    teacher_lib.STATE_FIELDS.index("green_cell"),
                    teacher_lib.STATE_FIELDS.index("blue_cell"),
                ],
                dtype=np.int64,
            )
            regions = final_states[np.arange(len(candidates)), cell_fields[colors - 1]] - 1
            keys = colors * 4 + regions
            groups = [candidates[keys == key] for key in np.unique(keys)]
            # Draw the group and member separately so every observed
            # goal-color/region combination has equal marginal probability.
            result = []
            for _ in range(batch_size):
                group = groups[int(rng.integers(len(groups)))]
                result.append(group[int(rng.integers(len(group)))])
            return np.asarray(result, dtype=np.int64)
        return np.asarray(
            rng.choice(candidates, batch_size, replace=len(candidates) < batch_size),
            dtype=np.int64,
        )
    if batch_size % len(teacher_lib.TASKS):
        raise ValueError("batch-size must be divisible by four")
    per_task = batch_size // len(teacher_lib.TASKS)
    result = []
    for task_id in range(len(teacher_lib.TASKS)):
        candidates = np.flatnonzero(dataset.sequence["task_ids"] == task_id)
        result.extend(rng.choice(candidates, per_task, replace=len(candidates) < per_task))
    rng.shuffle(result)
    return np.asarray(result, dtype=np.int64)


def _load_teacher_readout(args: argparse.Namespace):
    config = json.loads((args.teacher_training_dir / "training_config.json").read_text())
    sequence = _load_npz(args.teacher_sequence_dir / "train.npz")
    teacher = teacher_lib.UnifiedRoboMMEGTTeacher(
        width=int(config["memory_width"]),
        num_memory_tokens=int(config["memory_tokens"]),
        memory_depth=int(config["memory_depth"]),
        memory_heads=int(config["memory_heads"]),
        readout_heads=int(config["memory_heads"]),
    )
    one = np.asarray([0])
    teacher_inputs = {
        key: jnp.asarray(sequence[key][one])
        for key in (
            "task_ids",
            "goal_color_ids",
            "required_counts",
            "queried_ordinals",
            "num_regions",
            "event_ids",
            "entity_ids",
            "region_a_ids",
            "region_b_ids",
            "step_mask",
        )
    }
    template = teacher.init(
        jax.random.key(int(config["seed"])),
        **teacher_inputs,
        teacher_state_targets=jnp.asarray(sequence["state_targets"][one]),
        teacher_state_field_mask=jnp.asarray(sequence["state_field_mask"][one]),
    )["params"]
    params = flax.serialization.from_bytes(template, args.teacher_checkpoint.read_bytes())
    readout = teacher_lib.UnifiedStateReadout(
        width=int(config["memory_width"]), num_heads=int(config["memory_heads"])
    )
    return readout, params["unified_state_readout"]


def _host_summary(
    logits: np.ndarray,
    targets: np.ndarray,
    field_mask: np.ndarray,
    task_ids: np.ndarray,
    state_change_mask: np.ndarray,
    *,
    include_terminal_answer: bool = True,
) -> dict[str, Any]:
    predictions = np.argmax(logits, axis=-1)
    valid = np.any(field_mask, axis=-1)
    exact = np.all((predictions == targets) | ~field_mask, axis=-1) & valid
    transition = np.concatenate(
        (np.zeros((len(task_ids), 1), dtype=np.bool_), state_change_mask), axis=1
    ) & valid
    no_change = valid & ~transition

    def score(indices: np.ndarray) -> dict[str, float | int]:
        subset_valid = valid[indices]
        subset_exact = exact[indices]
        subset_transition = transition[indices]
        subset_no_change = no_change[indices]
        lengths = subset_valid.sum(axis=1)
        final = subset_exact[np.arange(len(indices)), np.maximum(lengths - 1, 0)]
        fields = field_mask[indices]
        result = {
            "episodes": len(indices),
            "field_accuracy": float(
                (((predictions[indices] == targets[indices]) & fields).sum()) / max(fields.sum(), 1)
            ),
            "state_exact_accuracy": float(subset_exact.sum() / max(subset_valid.sum(), 1)),
            "transition_state_exact_accuracy": float(
                (subset_exact & subset_transition).sum() / max(subset_transition.sum(), 1)
            ),
            "no_change_state_exact_accuracy": float(
                (subset_exact & subset_no_change).sum() / max(subset_no_change.sum(), 1)
            ),
            "sequence_exact_accuracy": float(np.mean(np.all(subset_exact | ~subset_valid, axis=1))),
            "final_state_exact_accuracy": float(np.mean(final)),
        }
        if include_terminal_answer:
            terminal_answers = []
            for subset_index, episode_index in enumerate(indices):
                final_index = max(int(lengths[subset_index]) - 1, 0)
                task_id = int(task_ids[episode_index])
                final_targets = targets[episode_index, final_index]
                final_predictions = predictions[episode_index, final_index]
                final_fields = field_mask[episode_index, final_index]
                if task_id in (0, 1):
                    # The goal identifies one or two queried colors.  The action
                    # answer is the final region of every queried color, not the
                    # auxiliary coverage/count fields in the full state schema.
                    color_fields = []
                    for raw_color_id in final_targets[1:3]:
                        color_id = int(raw_color_id)
                        if 1 <= color_id <= 3:
                            color_fields.append(2 + color_id)
                    answer_fields = tuple(dict.fromkeys(color_fields))
                elif task_id == 2:
                    queried_ordinal = int(final_targets[18])
                    answer_fields = (5 + queried_ordinal,) if 1 <= queried_ordinal <= 4 else ()
                elif task_id == 3:
                    answer_fields = (14, 15, 16, 17)
                else:
                    raise ValueError(f"Unsupported task id {task_id}")
                terminal_answers.append(
                    bool(answer_fields)
                    and all(bool(final_fields[field]) for field in answer_fields)
                    and all(
                        int(final_predictions[field]) == int(final_targets[field])
                        for field in answer_fields
                    )
                )
            result["terminal_answer_exact_accuracy"] = float(np.mean(terminal_answers))
        return result

    result = {"overall": score(np.arange(len(task_ids)))}
    for task_id, task_name in enumerate(teacher_lib.TASKS):
        task_indices = np.flatnonzero(task_ids == task_id)
        if len(task_indices):
            result[task_name] = score(task_indices)
    return result


def _binary_event_metrics(
    scores: np.ndarray, labels: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    scores = np.asarray(scores)[np.asarray(valid, dtype=bool)].astype(np.float64)
    labels = np.asarray(labels)[np.asarray(valid, dtype=bool)].astype(bool)
    positives = scores[labels]
    negatives = scores[~labels]
    if not len(positives) or not len(negatives):
        return {"event_gate_auprc": 0.0, "event_gate_auroc": 0.5}
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(sorted_labels) + 1)
    average_precision = float(np.sum(precision * sorted_labels) / len(positives))
    pairwise = positives[:, None] - negatives[None, :]
    auroc = float(np.mean((pairwise > 0).astype(np.float64) + 0.5 * (pairwise == 0)))
    return {"event_gate_auprc": average_precision, "event_gate_auroc": auroc}


def _save_params(params, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "params").write_bytes(flax.serialization.to_bytes(jax.device_get(params)))


def _restore_matching_params(params, checkpoint: Path):
    """Restore matching leaves while retaining newly introduced parameters."""

    restored = flax.serialization.msgpack_restore(checkpoint.read_bytes())
    flat_params = traverse_util.flatten_dict(params)
    flat_restored = traverse_util.flatten_dict(restored)
    loaded = 0
    skipped_shape = []
    for path, value in flat_restored.items():
        if path not in flat_params:
            continue
        if np.shape(value) != np.shape(flat_params[path]):
            skipped_shape.append("/".join(path))
            continue
        flat_params[path] = jnp.asarray(value, dtype=flat_params[path].dtype)
        loaded += 1
    missing = ["/".join(path) for path in flat_params if path not in flat_restored]
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "restored_parameter_leaves": loaded,
                "new_parameter_leaves": missing,
                "shape_mismatch_leaves": skipped_shape,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return traverse_util.unflatten_dict(flat_params)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.no_progress_fraction < 1.0:
        raise ValueError("--no-progress-fraction must be in [0,1)")
    if not 0.0 <= args.online_phase_fraction < 1.0:
        raise ValueError("--online-phase-fraction must be in [0,1)")
    if args.no_progress_fraction + args.online_phase_fraction >= 1.0:
        raise ValueError("no-progress and online-phase fractions must sum to less than one")
    if not 0.0 <= args.groundsg_action_sample_fraction <= 1.0:
        raise ValueError("--groundsg-action-sample-fraction must be in [0,1]")
    if not 0.0 <= args.groundsg_action_min_confidence <= 1.0:
        raise ValueError("--groundsg-action-min-confidence must be in [0,1]")
    if args.groundsg_action_loss_weight and args.groundsg_action_teacher_dir is None:
        raise ValueError("GroundSG action loss requires --groundsg-action-teacher-dir")
    if (
        args.groundsg_action_protect_transition_gradient
        and not args.groundsg_action_gradient_projection
    ):
        raise ValueError("Transition-gradient protection requires action gradient projection")
    if (
        args.privileged_soft_gate_loss_weight or args.privileged_soft_gate_rank_weight
    ) and not args.write_gate:
        raise ValueError("Privileged soft-gate supervision requires --write-gate")
    if (args.gate_only_training or args.gate_proprio_only_training) and not (
        args.privileged_soft_gate_loss_weight or args.privileged_soft_gate_rank_weight
    ):
        raise ValueError("Gate/proprio-only training requires a privileged soft-gate loss")
    if sum(
        (
            args.gate_only_training,
            args.gate_proprio_only_training,
            args.event_gate_only_training,
            args.route_only_training,
            args.event_correction_only_training,
        )
    ) > 1:
        raise ValueError("Choose only one gate/route-only training mode")
    if args.gate_proprio_only_training and args.proprio_dir is None:
        raise ValueError("--gate-proprio-only-training requires --proprio-dir")
    if args.freeze_gate_proprio and not (args.write_gate and args.proprio_dir is not None):
        raise ValueError("--freeze-gate-proprio requires --write-gate and --proprio-dir")
    if args.freeze_gate_proprio and any(
        (
            args.gate_only_training,
            args.gate_proprio_only_training,
            args.event_gate_only_training,
            args.route_only_training,
            args.event_correction_only_training,
        )
    ):
        raise ValueError("Cannot freeze gate/proprio while using an exclusive training mode")
    if args.pick_no_change_field_weighting and args.task != "pickxtimes_local_event":
        raise ValueError(
            "--pick-no-change-field-weighting requires PickXTimes single-task training"
        )
    if args.gate_margin_selection_floor < 0 or args.gate_margin_selection_penalty < 0:
        raise ValueError("Gate-margin selection floor and penalty must be non-negative")
    if (
        args.gate_margin_selection_floor or args.gate_margin_selection_penalty
    ) and not args.min_transition_no_change_selection:
        raise ValueError(
            "Gate-margin checkpoint constraints require --min-transition-no-change-selection"
        )
    if sum(
        (
            args.online_aligned_selection,
            args.min_transition_no_change_selection,
            args.terminal_answer_selection,
        )
    ) > 1:
        raise ValueError("Choose only one checkpoint-selection objective")
    if args.supervision_mode == "terminal_answer_only" and any(
        (
            args.decoupled_state_loss,
            args.direct_teacher_delta_loss_weight,
            args.privileged_soft_gate_loss_weight,
            args.privileged_soft_gate_rank_weight,
            args.event_gate_loss_weight,
            args.event_gate_rank_weight,
            args.event_correction_delta_loss_weight,
            args.no_progress_gate_loss_weight,
            args.no_progress_keep_loss_weight,
            args.online_hold_readout_loss_weight,
            args.online_transition_readout_loss_weight,
            args.online_hold_keep_loss_weight,
            args.groundsg_action_loss_weight,
        )
    ):
        raise ValueError(
            "terminal_answer_only must not receive trajectory, event, gate, online-phase, "
            "or action-distillation losses"
        )
    if args.freeze_event_gate and args.event_gate_only_training:
        raise ValueError("Cannot freeze and exclusively train the event gate together")
    if args.freeze_event_gate and not args.event_gate:
        raise ValueError("--freeze-event-gate requires --event-gate")
    if (args.event_gate_loss_weight or args.event_gate_rank_weight) and not args.event_gate:
        raise ValueError("Event-gate supervision requires --event-gate")
    if args.event_gate_only_training and not (
        args.event_gate_loss_weight or args.event_gate_rank_weight
    ):
        raise ValueError("--event-gate-only-training requires an event-gate loss")
    if args.event_gate_modulation_strength and not args.event_gate:
        raise ValueError("Event modulation requires --event-gate")
    if args.event_update_routing and not args.event_gate:
        raise ValueError("Event update routing requires --event-gate")
    if args.event_update_routing and args.event_gate_modulation_strength:
        raise ValueError("Do not combine update routing with multiplicative gate modulation")
    if args.route_only_training and not args.event_update_routing:
        raise ValueError("--route-only-training requires --event-update-routing")
    if args.oracle_event_correction and not args.event_correction:
        raise ValueError("--oracle-event-correction requires --event-correction")
    if args.event_correction and not (args.oracle_event_correction or args.event_gate):
        raise ValueError("Event correction requires an oracle mask or --event-gate")
    if args.event_correction and (
        args.event_update_routing or args.event_gate_modulation_strength
    ):
        raise ValueError("Isolate event correction from routing and gate modulation")
    if args.event_correction_delta_loss_weight and not args.event_correction:
        raise ValueError("Correction delta loss requires --event-correction")
    if args.event_correction_only_training and not (
        args.event_correction and args.event_correction_delta_loss_weight > 0
    ):
        raise ValueError(
            "--event-correction-only-training requires correction and positive delta loss"
        )
    if args.event_update_routing_temperature <= 0:
        raise ValueError("--event-update-routing-temperature must be positive")
    if not 0.0 < args.event_update_routing_reference < 1.0:
        raise ValueError("--event-update-routing-reference must be in (0,1)")
    if args.event_gate_sigma <= 0:
        raise ValueError("--event-gate-sigma must be positive")
    if not 0.0 <= args.event_gate_label_smoothing < 0.5:
        raise ValueError("--event-gate-label-smoothing must be in [0,0.5)")
    if not 0.0 <= args.event_gate_reference <= 1.0:
        raise ValueError("--event-gate-reference must be in [0,1]")
    if not 0.0 < args.event_gate_modulation_min <= 1.0 <= args.event_gate_modulation_max:
        raise ValueError("Event modulation bounds must straddle 1")
    if not 0.0 <= args.privileged_soft_gate_floor < args.privileged_soft_gate_peak <= 1.0:
        raise ValueError("Expected 0 <= soft-gate floor < peak <= 1")
    if args.privileged_soft_gate_sigma <= 0:
        raise ValueError("--privileged-soft-gate-sigma must be positive")
    if args.no_progress_fraction and args.no_progress_dir is None:
        raise ValueError("--no-progress-fraction requires --no-progress-dir")
    if args.online_phase_fraction and args.online_phase_dir is None:
        raise ValueError("--online-phase-fraction requires --online-phase-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: SplitDataset(split, args) for split in SPLITS}
    no_progress = None
    if args.no_progress_dir is not None:
        excluded = {
            int(value)
            for value in args.no_progress_exclude_episodes.split(",")
            if value.strip()
        }
        no_progress = NoProgressDataset(
            args.no_progress_dir, datasets["train"], excluded_episodes=excluded
        )
    online_phase = (
        OnlinePhaseDataset(args.online_phase_dir, datasets["train"])
        if args.online_phase_dir is not None
        else None
    )
    if any(dataset.max_steps != 96 for dataset in datasets.values()):
        raise ValueError("This experiment expects max_steps=96")
    rng = np.random.default_rng(args.seed)
    model = student_lib.UnifiedFixedChunkRecurrentStudent(
        proprio_dim=datasets["train"].proprio_dim,
        encoder_width=args.encoder_width,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        use_write_gate=args.write_gate,
        write_gate_bias=args.write_gate_bias,
        use_event_gate=args.event_gate,
        event_gate_bias=args.event_gate_bias,
        event_gate_modulation_strength=args.event_gate_modulation_strength,
        event_gate_reference=args.event_gate_reference,
        event_gate_modulation_min=args.event_gate_modulation_min,
        event_gate_modulation_max=args.event_gate_modulation_max,
        use_event_update_routing=args.event_update_routing,
        event_update_routing_temperature=args.event_update_routing_temperature,
        event_update_routing_reference=args.event_update_routing_reference,
        use_event_correction=args.event_correction,
        use_oracle_event_correction=args.oracle_event_correction,
        use_causal_evidence_state=args.causal_evidence_state,
        use_recurrent_memory=not args.no_recurrent_carry,
    )
    def sample_training_batch() -> dict[str, np.ndarray]:
        no_progress_count = (
            max(1, int(round(args.batch_size * args.no_progress_fraction)))
            if no_progress is not None and args.no_progress_fraction > 0
            else 0
        )
        online_phase_count = (
            max(1, int(round(args.batch_size * args.online_phase_fraction)))
            if online_phase is not None and args.online_phase_fraction > 0
            else 0
        )
        regular_count = args.batch_size - no_progress_count - online_phase_count
        if regular_count < 1:
            raise ValueError("Training batch must retain at least one offline episode")
        action_count = (
            min(
                regular_count,
                max(1, int(round(regular_count * args.groundsg_action_sample_fraction))),
            )
            if args.groundsg_action_sample_fraction > 0
            and len(datasets["train"].action_teacher_rows)
            else 0
        )
        ordinary_count = regular_count - action_count
        parts = []
        if ordinary_count:
            parts.append(
                _balanced_indices(
                    datasets["train"],
                    rng,
                    ordinary_count,
                    task=args.task,
                    balance_goal_region=args.balance_goal_region,
                )
            )
        if action_count:
            candidates = datasets["train"].action_teacher_rows
            if args.task is not None:
                task_id = teacher_lib.TASKS.index(args.task)
                candidates = candidates[
                    datasets["train"].sequence["task_ids"][candidates] == task_id
                ]
            if not len(candidates):
                raise ValueError("No GroundSG action-teacher rows match the selected task")
            parts.append(
                np.asarray(
                    rng.choice(candidates, action_count, replace=len(candidates) < action_count),
                    dtype=np.int64,
                )
            )
        regular_indices = np.concatenate(parts)
        rng.shuffle(regular_indices)
        regular = datasets["train"].batch(
            regular_indices,
            change_state_weight=args.change_state_weight,
            final_state_weight=args.final_state_weight,
        )
        batches = [regular]
        if no_progress_count:
            negative_indices = rng.choice(
                len(no_progress), no_progress_count, replace=len(no_progress) < no_progress_count
            )
            batches.append(
                no_progress.batch(
                    np.asarray(negative_indices, dtype=np.int64), weight=args.no_progress_weight
                )
            )
        if online_phase_count:
            phase_indices = rng.choice(
                len(online_phase),
                online_phase_count,
                replace=len(online_phase) < online_phase_count,
            )
            batches.append(
                online_phase.batch(
                    np.asarray(phase_indices, dtype=np.int64),
                    base_weight=args.online_phase_weight,
                    hold_weight=args.online_hold_weight,
                    transition_weight=args.online_transition_weight,
                )
            )
        return _concatenate_batches(*batches)

    initial_batch = sample_training_batch()
    params = model.init(
        jax.random.key(args.seed),
        **_student_inputs(
            initial_batch, oracle_event_correction=args.oracle_event_correction
        ),
        train=False,
    )["params"]
    if args.resume_checkpoint is not None:
        params = _restore_matching_params(params, args.resume_checkpoint)
    readout, readout_params = _load_teacher_readout(args)
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        args.learning_rate,
        min(args.warmup_steps, max(args.steps - 1, 0)),
        args.steps,
        end_value=args.end_learning_rate,
    )
    trainable_optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    if (
        args.gate_only_training
        or args.gate_proprio_only_training
        or args.event_gate_only_training
        or args.route_only_training
        or args.event_correction_only_training
    ):
        flat_params = traverse_util.flatten_dict(params)
        trainable_prefix = (
            "correction_"
            if args.event_correction_only_training
            else "route_"
            if args.route_only_training
            else "event_"
            if args.event_gate_only_training
            else "gate_"
        )
        def exclusively_trainable(path):
            recurrent_gate = path[0] == "shared_visual_memory_updater" and any(
                str(part).startswith(trainable_prefix) for part in path[1:]
            )
            proprio = args.gate_proprio_only_training and str(path[0]).startswith("proprio_")
            return recurrent_gate or proprio

        flat_labels = {
            path: "gate" if exclusively_trainable(path) else "frozen" for path in flat_params
        }
        gate_parameter_count = sum(
            int(np.prod(np.asarray(flat_params[path]).shape))
            for path, label in flat_labels.items()
            if label == "gate"
        )
        if gate_parameter_count == 0:
            raise ValueError(f"Gate-only training found no {trainable_prefix}* parameters")
        optimizer = optax.multi_transform(
            {"gate": trainable_optimizer, "frozen": optax.set_to_zero()},
            traverse_util.unflatten_dict(flat_labels),
        )
    elif args.freeze_gate_proprio:
        flat_params = traverse_util.flatten_dict(params)

        def calibrated_gate_or_proprio(path):
            recurrent_gate = path[0] == "shared_visual_memory_updater" and any(
                str(part).startswith("gate_") for part in path[1:]
            )
            return recurrent_gate or str(path[0]).startswith("proprio_")

        flat_labels = {
            path: "frozen" if calibrated_gate_or_proprio(path) else "trainable"
            for path in flat_params
        }
        gate_parameter_count = sum(
            int(np.prod(np.asarray(flat_params[path]).shape))
            for path, label in flat_labels.items()
            if label == "frozen"
        )
        optimizer = optax.multi_transform(
            {"trainable": trainable_optimizer, "frozen": optax.set_to_zero()},
            traverse_util.unflatten_dict(flat_labels),
        )
    elif args.freeze_event_gate:
        flat_params = traverse_util.flatten_dict(params)
        flat_labels = {
            path: (
                "frozen"
                if path[0] == "shared_visual_memory_updater"
                and any(str(part).startswith("event_") for part in path[1:])
                else "trainable"
            )
            for path in flat_params
        }
        gate_parameter_count = sum(
            int(np.prod(np.asarray(flat_params[path]).shape))
            for path, label in flat_labels.items()
            if label == "frozen"
        )
        optimizer = optax.multi_transform(
            {"trainable": trainable_optimizer, "frozen": optax.set_to_zero()},
            traverse_util.unflatten_dict(flat_labels),
        )
    else:
        gate_parameter_count = 0
        optimizer = trainable_optimizer
    opt_state = optimizer.init(params)

    holding_field = teacher_lib.STATE_FIELDS.index("holding")
    completed_field = teacher_lib.STATE_FIELDS.index("completed_count")
    ready_field = teacher_lib.STATE_FIELDS.index("ready_to_press")
    done_field = teacher_lib.STATE_FIELDS.index("done")
    transition_field_weights = jnp.ones(
        (len(teacher_lib.STATE_FIELDS),), dtype=jnp.float32
    )
    if args.pick_transition_field_weighting:
        if args.task != "pickxtimes_local_event":
            raise ValueError(
                "--pick-transition-field-weighting requires PickXTimes single-task training"
            )
        transition_field_weights = transition_field_weights.at[completed_field].set(3.0)
        transition_field_weights = transition_field_weights.at[holding_field].set(4.0)
        transition_field_weights = transition_field_weights.at[ready_field].set(2.0)
    no_change_field_weights = (
        transition_field_weights
        if args.pick_no_change_field_weighting
        else jnp.ones_like(transition_field_weights)
    )

    def objective(current_params, batch, *, train: bool):
        output = model.apply(
            {"params": current_params},
            **_student_inputs(
                batch, oracle_event_correction=args.oracle_event_correction
            ),
            train=train,
        )
        memory = output["all_memories"]
        state_weights = jnp.asarray(
            batch["memory_state_weights"]
            if args.decoupled_state_loss
            else batch["state_weights"]
        )
        memory_loss, memory_metrics = student_lib.weighted_memory_distillation_loss(
            memory,
            jnp.asarray(batch["teacher_memory"]),
            state_weights,
            semantic_token_weight=args.semantic_token_weight,
        )
        direct_delta_loss, direct_delta_metrics = student_lib.direct_teacher_delta_loss(
            memory,
            jnp.asarray(batch["teacher_memory"]),
            jnp.asarray(batch["sequence_mask"]),
            jnp.asarray(batch["state_change_mask"]),
        )
        correction_delta_loss, correction_metrics = (
            student_lib.oracle_event_correction_delta_loss(
                output["event_corrections"],
                output["base_chunk_memories"],
                jnp.asarray(batch["teacher_memory"])[:, 1:],
                jnp.asarray(batch["sequence_mask"]),
                jnp.asarray(batch["state_change_mask"]),
            )
        )
        flat = memory.reshape(-1, memory.shape[-2], memory.shape[-1])
        logits = readout.apply({"params": readout_params}, flat).reshape(
            *memory.shape[:2], len(teacher_lib.STATE_FIELDS), teacher_lib.MAX_FIELD_CLASSES
        )
        field_log_probs = jax.nn.log_softmax(logits, axis=-1)
        # Project canonical state probabilities onto the three action-relevant
        # phases. Count remains under the original GT/state supervision.
        phase_scores = jnp.stack(
            (
                field_log_probs[..., holding_field, 0]
                + field_log_probs[..., ready_field, 0]
                + field_log_probs[..., done_field, 0],
                field_log_probs[..., holding_field, 1]
                + field_log_probs[..., done_field, 0],
                field_log_probs[..., ready_field, 1]
                + field_log_probs[..., done_field, 0],
            ),
            axis=-1,
        )
        action_phase_log_probs = jax.nn.log_softmax(phase_scores, axis=-1)
        action_phase_targets = jnp.asarray(batch["action_phase_probs"], dtype=jnp.float32)
        action_phase_mask = jnp.asarray(batch["action_phase_mask"], dtype=jnp.float32)
        action_phase_ce = -jnp.sum(action_phase_targets * action_phase_log_probs, axis=-1)
        action_phase_ce_sum = jnp.sum(action_phase_ce * action_phase_mask)
        action_phase_count = jnp.sum(action_phase_mask)
        action_phase_loss = action_phase_ce_sum / jnp.maximum(action_phase_count, 1.0)
        action_phase_correct_count = jnp.sum(
            (
                jnp.argmax(action_phase_log_probs, axis=-1)
                == jnp.argmax(action_phase_targets, axis=-1)
            ).astype(jnp.float32)
            * action_phase_mask
        )
        targets = jnp.asarray(batch["state_targets"])
        field_mask = jnp.asarray(batch["state_field_mask"])
        if args.supervision_mode == "terminal_answer_only":
            # Only the final fields used by the common terminal Answer metric
            # enter this loss. Intermediate state labels remain evaluation-only.
            field_count = len(teacher_lib.STATE_FIELDS)
            field_ids = jnp.arange(field_count)[None, :]
            task_ids = jnp.asarray(batch["task_ids"], dtype=jnp.int32)
            goal_colors = jnp.asarray(batch["goal_color_ids"], dtype=jnp.int32)
            queried_ordinals = jnp.asarray(batch["queried_ordinals"], dtype=jnp.int32)

            color_fields = 2 + goal_colors
            color_valid = (goal_colors >= 1) & (goal_colors <= 3)
            color_answer_fields = jnp.any(
                (field_ids[:, None, :] == color_fields[:, :, None])
                & color_valid[:, :, None],
                axis=1,
            )
            ordinal_field = 5 + queried_ordinals
            ordinal_answer_fields = (
                (field_ids == ordinal_field[:, None])
                & (queried_ordinals[:, None] >= 1)
                & (queried_ordinals[:, None] <= 4)
            )
            pick_answer_fields = (field_ids >= 14) & (field_ids <= 17)
            answer_fields = jnp.where(
                task_ids[:, None] <= 1,
                color_answer_fields,
                jnp.where(task_ids[:, None] == 2, ordinal_answer_fields, pick_answer_fields),
            )
            terminal_field_mask = (
                field_mask
                & jnp.asarray(batch["final_state_mask"])[..., None]
                & answer_fields[:, None, :]
            )
            state_loss = student_lib.weighted_state_cross_entropy(
                logits,
                targets,
                terminal_field_mask,
                jnp.asarray(batch["final_state_mask"]),
            )
            change_readout_loss = jnp.asarray(0.0, dtype=jnp.float32)
            no_change_readout_loss = jnp.asarray(0.0, dtype=jnp.float32)
            final_readout_loss = state_loss
            keep_loss = jnp.asarray(0.0, dtype=jnp.float32)
        elif args.decoupled_state_loss:
            state_loss = student_lib.weighted_state_cross_entropy(
                logits,
                targets,
                field_mask,
                jnp.asarray(batch["valid_states"]),
            )
            change_readout_loss = student_lib.weighted_state_cross_entropy(
                logits,
                targets,
                field_mask,
                jnp.asarray(batch["transition_state_weights"]),
                transition_field_weights,
            )
            no_change_readout_loss = student_lib.weighted_state_cross_entropy(
                logits,
                targets,
                field_mask,
                jnp.asarray(batch["no_change_state_mask"]),
                no_change_field_weights,
            )
            final_readout_loss = student_lib.weighted_state_cross_entropy(
                logits,
                targets,
                field_mask,
                jnp.asarray(batch["final_state_mask"]),
            )
            keep_loss = student_lib.no_change_memory_consistency_loss(
                memory,
                jnp.asarray(batch["sequence_mask"]),
                jnp.asarray(batch["state_change_mask"]),
            )
        else:
            state_loss = student_lib.weighted_state_cross_entropy(
                logits, targets, field_mask, state_weights
            )
            change_readout_loss = jnp.asarray(0.0, dtype=jnp.float32)
            no_change_readout_loss = jnp.asarray(0.0, dtype=jnp.float32)
            final_readout_loss = jnp.asarray(0.0, dtype=jnp.float32)
            keep_loss = jnp.asarray(0.0, dtype=jnp.float32)
        _, strict_metrics = teacher_lib.compute_teacher_losses(
            {"state_logits": logits, "all_memories": memory}, targets, field_mask
        )
        transition_protection_loss = student_lib.weighted_state_cross_entropy(
            logits,
            targets,
            field_mask,
            jnp.asarray(batch["transition_state_weights"]),
            transition_field_weights,
        )
        if args.supervision_mode == "terminal_answer_only":
            loss = args.state_loss_weight * state_loss
        else:
            loss = args.memory_loss_weight * memory_loss + args.state_loss_weight * state_loss
        if args.decoupled_state_loss and args.supervision_mode == "full":
            loss = (
                loss
                + args.change_readout_loss_weight * change_readout_loss
                + args.no_change_readout_loss_weight * no_change_readout_loss
                + args.final_readout_loss_weight * final_readout_loss
                + args.keep_loss_weight * keep_loss
            )
        loss = loss + args.direct_teacher_delta_loss_weight * direct_delta_loss
        gates = output["write_gates"].astype(jnp.float32)
        event_gates = output["event_gates"].astype(jnp.float32)
        gate_modulations = output["gate_modulations"].astype(jnp.float32)
        effective_gates = output["effective_write_gates"].astype(jnp.float32)
        event_update_residual_norms = output["event_update_residual_norms"].astype(jnp.float32)
        hold_update_residual_norms = output["hold_update_residual_norms"].astype(jnp.float32)
        routed_update_residual_norms = output["routed_update_residual_norms"].astype(jnp.float32)
        routing_probabilities = output["event_update_routing_probabilities"].astype(jnp.float32)
        valid_chunks = jnp.asarray(batch["sequence_mask"]).astype(jnp.float32)
        change_chunks = (
            jnp.asarray(batch["state_change_mask"]).astype(jnp.float32) * valid_chunks
        )
        hold_chunks = (1.0 - jnp.asarray(batch["state_change_mask"]).astype(jnp.float32)) * valid_chunks
        chunk_positions = jnp.arange(gates.shape[1], dtype=jnp.float32)
        chunk_distances = chunk_positions[:, None] - chunk_positions[None, :]
        gate_kernel = jnp.exp(
            -0.5 * jnp.square(chunk_distances / args.privileged_soft_gate_sigma)
        )
        gate_event_influence = jnp.max(
            change_chunks[:, None, :] * gate_kernel[None, :, :], axis=-1
        )
        soft_gate_targets = (
            args.privileged_soft_gate_floor
            + (args.privileged_soft_gate_peak - args.privileged_soft_gate_floor)
            * gate_event_influence
        )
        clipped_gates = jnp.clip(gates, 1e-6, 1.0 - 1e-6)
        soft_gate_bce = -(
            soft_gate_targets * jnp.log(clipped_gates)
            + (1.0 - soft_gate_targets) * jnp.log(1.0 - clipped_gates)
        )
        privileged_soft_gate_loss = jnp.sum(soft_gate_bce * valid_chunks) / jnp.maximum(
            jnp.sum(valid_chunks), 1.0
        )
        far_hold_chunks = (
            (gate_event_influence < 0.1).astype(jnp.float32) * hold_chunks
        )
        event_gate_mean = jnp.sum(gates * change_chunks) / jnp.maximum(
            jnp.sum(change_chunks), 1.0
        )
        far_hold_gate_mean = jnp.sum(gates * far_hold_chunks) / jnp.maximum(
            jnp.sum(far_hold_chunks), 1.0
        )
        privileged_soft_gate_rank_loss = jax.nn.relu(
            args.privileged_soft_gate_rank_margin
            - event_gate_mean
            + far_hold_gate_mean
        )
        event_gate_kernel = jnp.exp(
            -0.5 * jnp.square(chunk_distances / args.event_gate_sigma)
        )
        event_gate_influence = jnp.max(
            change_chunks[:, None, :] * event_gate_kernel[None, :, :], axis=-1
        )
        event_gate_targets = (
            args.event_gate_label_smoothing
            + (1.0 - 2.0 * args.event_gate_label_smoothing) * event_gate_influence
        )
        clipped_event_gates = jnp.clip(event_gates, 1e-6, 1.0 - 1e-6)
        # Balance soft positive and negative mass.  Otherwise the many hold
        # chunks let a constant low-confidence predictor minimize BCE.
        event_positive_loss = -jnp.sum(
            event_gate_targets * jnp.log(clipped_event_gates) * valid_chunks
        ) / jnp.maximum(jnp.sum(event_gate_targets * valid_chunks), 1.0)
        event_negative_loss = -jnp.sum(
            (1.0 - event_gate_targets)
            * jnp.log(1.0 - clipped_event_gates)
            * valid_chunks
        ) / jnp.maximum(jnp.sum((1.0 - event_gate_targets) * valid_chunks), 1.0)
        event_gate_loss = 0.5 * (event_positive_loss + event_negative_loss)
        event_far_hold_chunks = (
            (event_gate_influence < 0.1).astype(jnp.float32) * hold_chunks
        )
        event_confidence_mean = jnp.sum(event_gates * change_chunks) / jnp.maximum(
            jnp.sum(change_chunks), 1.0
        )
        far_hold_event_confidence_mean = jnp.sum(
            event_gates * event_far_hold_chunks
        ) / jnp.maximum(jnp.sum(event_far_hold_chunks), 1.0)
        event_gate_rank_loss = jax.nn.relu(
            args.event_gate_rank_margin
            - event_confidence_mean
            + far_hold_event_confidence_mean
        )
        no_progress_chunks = jnp.asarray(batch["no_progress_chunk_mask"]).astype(jnp.float32)
        no_progress_gate_loss = jnp.sum(gates * no_progress_chunks) / jnp.maximum(
            jnp.sum(no_progress_chunks), 1.0
        )
        initial_memory = jax.lax.stop_gradient(memory[:, :1].astype(jnp.float32))
        no_progress_memory_error = jnp.mean(
            jnp.square(memory[:, 1:].astype(jnp.float32) - initial_memory), axis=(-2, -1)
        )
        no_progress_keep_loss = jnp.sum(
            no_progress_memory_error * no_progress_chunks
        ) / jnp.maximum(jnp.sum(no_progress_chunks), 1.0)
        online_hold_chunks = jnp.asarray(batch["online_hold_chunk_mask"]).astype(jnp.float32)
        online_hold_states = jnp.concatenate(
            (jnp.zeros((online_hold_chunks.shape[0], 1), dtype=jnp.float32), online_hold_chunks),
            axis=1,
        )
        online_transition_states = jnp.asarray(
            batch["online_transition_state_mask"]
        ).astype(jnp.float32)
        online_hold_readout_loss = student_lib.weighted_state_cross_entropy(
            logits, targets, field_mask, online_hold_states
        )
        online_transition_readout_loss = student_lib.weighted_state_cross_entropy(
            logits, targets, field_mask, online_transition_states
        )
        recurrent_memory_error = jnp.mean(
            jnp.square(memory[:, 1:].astype(jnp.float32) - memory[:, :-1].astype(jnp.float32)),
            axis=(-2, -1),
        )
        online_hold_keep_loss = jnp.sum(
            recurrent_memory_error * online_hold_chunks
        ) / jnp.maximum(jnp.sum(online_hold_chunks), 1.0)
        canonical_task_loss = (
            loss
            + args.no_progress_gate_loss_weight * no_progress_gate_loss
            + args.no_progress_keep_loss_weight * no_progress_keep_loss
            + args.online_hold_readout_loss_weight * online_hold_readout_loss
            + args.online_transition_readout_loss_weight * online_transition_readout_loss
            + args.online_hold_keep_loss_weight * online_hold_keep_loss
        )
        gate_supervision_loss = (
            args.privileged_soft_gate_loss_weight * privileged_soft_gate_loss
            + args.privileged_soft_gate_rank_weight * privileged_soft_gate_rank_loss
        )
        event_gate_supervision_loss = (
            args.event_gate_loss_weight * event_gate_loss
            + args.event_gate_rank_weight * event_gate_rank_loss
        )
        event_correction_supervision_loss = (
            args.event_correction_delta_loss_weight * correction_delta_loss
        )
        if args.gate_only_training or args.gate_proprio_only_training:
            loss = gate_supervision_loss
        elif args.event_gate_only_training:
            loss = event_gate_supervision_loss
        elif args.event_correction_only_training:
            loss = event_correction_supervision_loss
        else:
            loss = (
                canonical_task_loss
                + gate_supervision_loss
                + event_gate_supervision_loss
                + event_correction_supervision_loss
            )
        base_loss = loss
        loss = base_loss + args.groundsg_action_loss_weight * action_phase_loss

        def masked_mean(values, mask):
            return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        def masked_gate_mean(mask):
            return masked_mean(gates, mask)

        metrics = {
            "loss": loss,
            "base_loss": base_loss,
            "canonical_task_loss": canonical_task_loss,
            "gate_supervision_loss": gate_supervision_loss,
            "event_gate_supervision_loss": event_gate_supervision_loss,
            "event_correction_supervision_loss": event_correction_supervision_loss,
            **memory_metrics,
            **direct_delta_metrics,
            **correction_metrics,
            "state_loss": state_loss,
            "transition_protection_loss": transition_protection_loss,
            "change_readout_loss": change_readout_loss,
            "no_change_readout_loss": no_change_readout_loss,
            "final_readout_loss": final_readout_loss,
            "keep_loss": keep_loss,
            "field_accuracy": strict_metrics["field_accuracy"],
            "state_exact_accuracy": strict_metrics["state_exact_accuracy"],
            "sequence_exact_accuracy": strict_metrics["sequence_exact_accuracy"],
            "final_state_exact_accuracy": strict_metrics["final_state_exact_accuracy"],
            "write_gate_mean": masked_gate_mean(valid_chunks),
            "change_write_gate_mean": masked_gate_mean(change_chunks),
            "hold_write_gate_mean": masked_gate_mean(hold_chunks),
            "far_hold_write_gate_mean": far_hold_gate_mean,
            "event_to_far_hold_gate_ratio": event_gate_mean
            / jnp.maximum(far_hold_gate_mean, 1e-6),
            "event_gate_loss": event_gate_loss,
            "event_gate_rank_loss": event_gate_rank_loss,
            "event_gate_target_mean": jnp.sum(event_gate_targets * valid_chunks)
            / jnp.maximum(jnp.sum(valid_chunks), 1.0),
            "event_confidence_mean": event_confidence_mean,
            "far_hold_event_confidence_mean": far_hold_event_confidence_mean,
            "event_confidence_margin": (
                event_confidence_mean - far_hold_event_confidence_mean
            ),
            "event_confidence_ratio": event_confidence_mean
            / jnp.maximum(far_hold_event_confidence_mean, 1e-6),
            "gate_modulation_mean": masked_mean(gate_modulations, valid_chunks),
            "effective_write_gate_mean": masked_mean(effective_gates, valid_chunks),
            "event_update_residual_norm_mean": masked_mean(
                event_update_residual_norms, valid_chunks
            ),
            "hold_update_residual_norm_mean": masked_mean(
                hold_update_residual_norms, valid_chunks
            ),
            "routed_update_residual_norm_mean": masked_mean(
                routed_update_residual_norms, valid_chunks
            ),
            "routed_update_residual_norm_on_change": masked_mean(
                routed_update_residual_norms, change_chunks
            ),
            "routed_update_residual_norm_on_hold": masked_mean(
                routed_update_residual_norms, hold_chunks
            ),
            "event_update_routing_probability_mean": masked_mean(
                routing_probabilities, valid_chunks
            ),
            "event_update_routing_probability_on_change": masked_mean(
                routing_probabilities, change_chunks
            ),
            "event_update_routing_probability_on_far_hold": masked_mean(
                routing_probabilities, event_far_hold_chunks
            ),
            "privileged_soft_gate_loss": privileged_soft_gate_loss,
            "privileged_soft_gate_rank_loss": privileged_soft_gate_rank_loss,
            "privileged_soft_gate_target_mean": jnp.sum(
                soft_gate_targets * valid_chunks
            )
            / jnp.maximum(jnp.sum(valid_chunks), 1.0),
            "no_progress_gate_loss": no_progress_gate_loss,
            "no_progress_keep_loss": no_progress_keep_loss,
            "online_hold_readout_loss": online_hold_readout_loss,
            "online_transition_readout_loss": online_transition_readout_loss,
            "online_hold_keep_loss": online_hold_keep_loss,
            "action_phase_loss": action_phase_loss,
            "action_phase_ce_sum": action_phase_ce_sum,
            "action_phase_count": action_phase_count,
            "action_phase_correct_count": action_phase_correct_count,
            "action_phase_accuracy": action_phase_correct_count
            / jnp.maximum(action_phase_count, 1.0),
        }
        return loss, (metrics, logits)

    def _tree_dot(left, right):
        return sum(
            jnp.vdot(x.astype(jnp.float32), y.astype(jnp.float32))
            for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right), strict=True)
        )

    if args.groundsg_action_gradient_projection:

        @jax.jit
        def train_step(params, opt_state, batch):
            def canonical_objective(current_params):
                _, (metrics, _) = objective(current_params, batch, train=True)
                return metrics["base_loss"], metrics

            (_, metrics), canonical_gradients = jax.value_and_grad(
                canonical_objective, has_aux=True
            )(params)

            def action_objective(current_params):
                _, (metrics, _) = objective(current_params, batch, train=True)
                return metrics["action_phase_loss"]

            action_gradients = jax.grad(action_objective)(params)
            gradient_dot = _tree_dot(action_gradients, canonical_gradients)
            canonical_norm_sq = _tree_dot(canonical_gradients, canonical_gradients)
            action_norm_sq = _tree_dot(action_gradients, action_gradients)
            projection_coefficient = jnp.where(
                gradient_dot < 0.0,
                gradient_dot / jnp.maximum(canonical_norm_sq, 1e-12),
                0.0,
            )
            projected_action_gradients = jax.tree_util.tree_map(
                lambda action, canonical: action - projection_coefficient * canonical,
                action_gradients,
                canonical_gradients,
            )
            transition_gradient_cosine = jnp.asarray(0.0, dtype=jnp.float32)
            transition_projection_applied = jnp.asarray(0.0, dtype=jnp.float32)
            if args.groundsg_action_protect_transition_gradient:

                def transition_objective(current_params):
                    _, (metrics, _) = objective(current_params, batch, train=True)
                    return metrics["transition_protection_loss"]

                transition_gradients = jax.grad(transition_objective)(params)
                transition_dot = _tree_dot(projected_action_gradients, transition_gradients)
                transition_norm_sq = _tree_dot(transition_gradients, transition_gradients)
                projected_action_norm_sq = _tree_dot(
                    projected_action_gradients, projected_action_gradients
                )
                transition_coefficient = jnp.where(
                    transition_dot < 0.0,
                    transition_dot / jnp.maximum(transition_norm_sq, 1e-12),
                    0.0,
                )
                projected_action_gradients = jax.tree_util.tree_map(
                    lambda action, transition: action
                    - transition_coefficient * transition,
                    projected_action_gradients,
                    transition_gradients,
                )
                transition_gradient_cosine = transition_dot / jnp.maximum(
                    jnp.sqrt(transition_norm_sq * projected_action_norm_sq), 1e-12
                )
                transition_projection_applied = (transition_dot < 0.0).astype(jnp.float32)
            gradients = jax.tree_util.tree_map(
                lambda canonical, action: canonical
                + args.groundsg_action_loss_weight * action,
                canonical_gradients,
                projected_action_gradients,
            )
            gradient_cosine = gradient_dot / jnp.maximum(
                jnp.sqrt(canonical_norm_sq * action_norm_sq), 1e-12
            )
            updates, next_opt_state = optimizer.update(gradients, opt_state, params)
            return (
                optax.apply_updates(params, updates),
                next_opt_state,
                {
                    **metrics,
                    "gradient_norm": optax.global_norm(gradients),
                    "canonical_gradient_norm": jnp.sqrt(canonical_norm_sq),
                    "action_gradient_norm": jnp.sqrt(action_norm_sq),
                    "canonical_action_gradient_cosine": gradient_cosine,
                    "action_gradient_projection_applied": (gradient_dot < 0.0).astype(
                        jnp.float32
                    ),
                    "transition_action_gradient_cosine": transition_gradient_cosine,
                    "transition_action_gradient_projection_applied": transition_projection_applied,
                },
            )

    else:

        @jax.jit
        def train_step(params, opt_state, batch):
            (_, (metrics, _)), gradients = jax.value_and_grad(objective, has_aux=True)(
                params, batch, train=True
            )
            updates, next_opt_state = optimizer.update(gradients, opt_state, params)
            return (
                optax.apply_updates(params, updates),
                next_opt_state,
                {**metrics, "gradient_norm": optax.global_norm(gradients)},
            )

    @jax.jit
    def eval_step(params, batch):
        _, (metrics, logits) = objective(params, batch, train=False)
        output = model.apply(
            {"params": params},
            **_student_inputs(
                batch, oracle_event_correction=args.oracle_event_correction
            ),
            train=False,
        )
        return metrics, logits, output["event_gates"]

    def evaluate(split: str, params):
        dataset = datasets[split]
        evaluation_indices = np.arange(dataset.length)
        if args.task is not None:
            task_id = teacher_lib.TASKS.index(args.task)
            evaluation_indices = np.flatnonzero(dataset.sequence["task_ids"] == task_id)
        logits = []
        metric_rows = []
        targets, masks, changes = [], [], []
        event_scores, chunk_validity = [], []
        for start in range(0, len(evaluation_indices), args.batch_size):
            indices = evaluation_indices[start : start + args.batch_size]
            real_count = len(indices)
            if real_count < args.batch_size:
                indices = np.pad(indices, (0, args.batch_size - real_count), mode="edge")
            batch = dataset.batch(
                indices,
                change_state_weight=args.change_state_weight,
                final_state_weight=args.final_state_weight,
            )
            metrics, batch_logits, batch_event_scores = eval_step(params, batch)
            logits.append(np.asarray(batch_logits)[:real_count])
            event_scores.append(np.asarray(batch_event_scores)[:real_count])
            chunk_validity.append(batch["sequence_mask"][:real_count])
            targets.append(batch["state_targets"][:real_count])
            masks.append(batch["state_field_mask"][:real_count])
            changes.append(batch["state_change_mask"][:real_count])
            metric_rows.append({key: float(value) for key, value in metrics.items()})
        summary = _host_summary(
            np.concatenate(logits),
            np.concatenate(targets),
            np.concatenate(masks),
            dataset.sequence["task_ids"][evaluation_indices],
            np.concatenate(changes),
        )
        averaged = {key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]}
        action_count = sum(row["action_phase_count"] for row in metric_rows)
        averaged["action_phase_loss"] = float(
            sum(row["action_phase_ce_sum"] for row in metric_rows) / max(action_count, 1.0)
        )
        averaged["action_phase_accuracy"] = float(
            sum(row["action_phase_correct_count"] for row in metric_rows)
            / max(action_count, 1.0)
        )
        averaged["action_phase_count"] = float(action_count)
        averaged.update(
            _binary_event_metrics(
                np.concatenate(event_scores),
                np.concatenate(changes),
                np.concatenate(chunk_validity),
            )
        )
        return summary, averaged

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        {
            "overlapping_windows": False,
            "explicit_event_trigger": False,
            "chunk_frames": 12,
            "stride_frames": 12,
            "student_receives_gt_event_or_state": bool(args.oracle_event_correction),
            "supervision_mode": args.supervision_mode,
            "privileged_trajectory_teacher_used": args.supervision_mode == "full",
            "terminal_answer_training_fields": (
                "queried colors, queried ordinal, or Pick control tuple"
                if args.supervision_mode == "terminal_answer_only"
                else None
            ),
            "mask_semantics": "variable-length sequence padding only",
            "write_gate_supervision": bool(
                args.privileged_soft_gate_loss_weight
                or args.privileged_soft_gate_rank_weight
            ),
            "write_gate_target": "privileged event Gaussian soft target",
            "causal_evidence_state": bool(args.causal_evidence_state),
            "recurrent_memory_carry": not args.no_recurrent_carry,
            "gate_trainable_parameter_count": gate_parameter_count,
            "independent_event_gate": bool(args.event_gate),
            "event_gate_target": "balanced privileged Gaussian transition confidence",
            "event_gate_affects_memory": bool(args.event_gate_modulation_strength),
            "event_update_routing": bool(args.event_update_routing),
            "event_gate_affects_update_content": bool(args.event_update_routing),
            "event_correction": bool(args.event_correction),
            "oracle_event_correction": bool(args.oracle_event_correction),
            "event_correction_target": "teacher_memory_after - base_student_memory",
            "online_phase_supervision": bool(args.online_phase_fraction),
            "online_phase_target": "simulator simple_subgoal decoded at the final frame of each 12-frame chunk",
            "online_persistence_objective": "state CE plus recurrent-memory consistency on oracle no-change chunks",
            "groundsg_action_distillation": bool(args.groundsg_action_loss_weight),
            "groundsg_action_target": (
                "three-way Pick/Place/Press soft target from official-policy action-chunk RMSE"
            ),
            "groundsg_action_count_supervision": "retained canonical GT state/count losses",
            "checkpoint_selection_objective": (
                "change_minus_far_hold_gate_margin"
                if args.gate_proprio_only_training
                else "min_transition_no_change_then_transition_no_change_state_final"
                if args.min_transition_no_change_selection
                else "terminal_answer_then_final_state"
                if args.terminal_answer_selection
                else "transition_no_change_trajectory_final"
                if args.online_aligned_selection
                else "final_transition_sequence_state"
            ),
            "checkpoint_gate_margin_constraint": {
                "floor": args.gate_margin_selection_floor,
                "shortfall_penalty": args.gate_margin_selection_penalty,
                "margin": "change_write_gate_mean - far_hold_write_gate_mean",
            },
            "memory_trajectory_final_weight": (
                1.0 if args.decoupled_state_loss else args.final_state_weight
            ),
        }
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    best_params = params
    best_score = (
        (-1.0, -1.0, -1.0, -1.0)
        if args.gate_proprio_only_training
        else (-1.0, -1.0, -1.0)
        if args.terminal_answer_selection
        else (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
        if args.min_transition_no_change_selection
        and args.gate_margin_selection_penalty
        else (-1.0, -1.0, -1.0, -1.0, -1.0)
        if args.min_transition_no_change_selection
        else (-1.0, -1.0, -1.0, -1.0)
    )
    best_step = 0
    started = time.perf_counter()
    try:
        with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
            if (
                args.groundsg_action_teacher_dir is not None
                or args.event_gate
                or args.event_correction
                or args.min_transition_no_change_selection
                or args.terminal_answer_selection
                or args.gate_proprio_only_training
            ):
                dev_summary, dev_metrics = evaluate("dev", params)
                overall = dev_summary["overall"]
                if args.gate_proprio_only_training:
                    best_score = (
                        dev_metrics["change_write_gate_mean"]
                        - dev_metrics["far_hold_write_gate_mean"],
                        dev_metrics["change_write_gate_mean"],
                        -dev_metrics["far_hold_write_gate_mean"],
                        -dev_metrics["privileged_soft_gate_loss"],
                    )
                elif args.event_gate_only_training:
                    best_score = (
                        dev_metrics["event_gate_auprc"],
                        dev_metrics["event_gate_auroc"],
                        dev_metrics["event_confidence_margin"],
                        -dev_metrics["event_gate_loss"],
                    )
                elif args.min_transition_no_change_selection:
                    transition = overall["transition_state_exact_accuracy"]
                    no_change = overall["no_change_state_exact_accuracy"]
                    balance = min(transition, no_change)
                    if args.gate_margin_selection_penalty:
                        gate_margin = (
                            dev_metrics["change_write_gate_mean"]
                            - dev_metrics["far_hold_write_gate_mean"]
                        )
                        constrained_balance = balance - args.gate_margin_selection_penalty * max(
                            args.gate_margin_selection_floor - gate_margin, 0.0
                        )
                        best_score = (
                            constrained_balance,
                            balance,
                            gate_margin,
                            transition,
                            no_change,
                            overall["state_exact_accuracy"],
                            overall["final_state_exact_accuracy"],
                        )
                    else:
                        best_score = (
                            balance,
                            transition,
                            no_change,
                            overall["state_exact_accuracy"],
                            overall["final_state_exact_accuracy"],
                        )
                elif args.terminal_answer_selection:
                    best_score = (
                        overall["terminal_answer_exact_accuracy"],
                        overall["final_state_exact_accuracy"],
                        overall["state_exact_accuracy"],
                    )
                else:
                    best_score = (
                        overall["final_state_exact_accuracy"],
                        overall["transition_state_exact_accuracy"],
                        overall["sequence_exact_accuracy"],
                        overall["state_exact_accuracy"],
                    )
                best_params = jax.device_get(params)
                _save_params(best_params, args.output_dir / "best")
                record = {
                    "step": 0,
                    "split": "dev",
                    "elapsed_seconds": time.perf_counter() - started,
                    **dev_metrics,
                    "strict": dev_summary,
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
            for step in range(1, args.steps + 1):
                batch = sample_training_batch()
                params, opt_state, metrics = train_step(params, opt_state, batch)
                if step == 1 or step % 10 == 0:
                    record = {
                        "step": step,
                        "split": "train",
                        "elapsed_seconds": time.perf_counter() - started,
                        "learning_rate": float(schedule(step)),
                        **{key: float(value) for key, value in metrics.items()},
                    }
                    log.write(json.dumps(record, sort_keys=True) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                if step % args.eval_every == 0 or step == args.steps:
                    dev_summary, dev_metrics = evaluate("dev", params)
                    overall = dev_summary["overall"]
                    if args.gate_proprio_only_training:
                        score = (
                            dev_metrics["change_write_gate_mean"]
                            - dev_metrics["far_hold_write_gate_mean"],
                            dev_metrics["change_write_gate_mean"],
                            -dev_metrics["far_hold_write_gate_mean"],
                            -dev_metrics["privileged_soft_gate_loss"],
                        )
                    elif args.event_gate_only_training:
                        score = (
                            dev_metrics["event_gate_auprc"],
                            dev_metrics["event_gate_auroc"],
                            dev_metrics["event_confidence_margin"],
                            -dev_metrics["event_gate_loss"],
                        )
                    elif args.min_transition_no_change_selection:
                        transition = overall["transition_state_exact_accuracy"]
                        no_change = overall["no_change_state_exact_accuracy"]
                        balance = min(transition, no_change)
                        if args.gate_margin_selection_penalty:
                            gate_margin = (
                                dev_metrics["change_write_gate_mean"]
                                - dev_metrics["far_hold_write_gate_mean"]
                            )
                            constrained_balance = (
                                balance
                                - args.gate_margin_selection_penalty
                                * max(args.gate_margin_selection_floor - gate_margin, 0.0)
                            )
                            score = (
                                constrained_balance,
                                balance,
                                gate_margin,
                                transition,
                                no_change,
                                overall["state_exact_accuracy"],
                                overall["final_state_exact_accuracy"],
                            )
                        else:
                            score = (
                                balance,
                                transition,
                                no_change,
                                overall["state_exact_accuracy"],
                                overall["final_state_exact_accuracy"],
                            )
                    elif args.terminal_answer_selection:
                        score = (
                            overall["terminal_answer_exact_accuracy"],
                            overall["final_state_exact_accuracy"],
                            overall["state_exact_accuracy"],
                        )
                    elif args.online_aligned_selection:
                        score = (
                            overall["transition_state_exact_accuracy"],
                            overall["no_change_state_exact_accuracy"],
                            overall["state_exact_accuracy"],
                            overall["final_state_exact_accuracy"],
                        )
                    else:
                        score = (
                            overall["final_state_exact_accuracy"],
                            overall["transition_state_exact_accuracy"],
                            overall["sequence_exact_accuracy"],
                            overall["state_exact_accuracy"],
                        )
                    record = {
                        "step": step,
                        "split": "dev",
                        "elapsed_seconds": time.perf_counter() - started,
                        **dev_metrics,
                        "strict": dev_summary,
                    }
                    log.write(json.dumps(record, sort_keys=True) + "\n")
                    log.flush()
                    print(json.dumps(record, sort_keys=True), flush=True)
                    if score > best_score:
                        best_score, best_step, best_params = score, step, jax.device_get(params)
                        _save_params(best_params, args.output_dir / "best")
                if step % args.save_every == 0 or step == args.steps:
                    _save_params(params, args.output_dir / str(step))
        result = {
            "best_step": best_step,
            "best_dev_score": best_score,
            "elapsed_seconds": time.perf_counter() - started,
        }
        if not args.skip_test:
            test_summary, test_metrics = evaluate("test", best_params)
            result.update({"test": test_summary, "test_losses": test_metrics})
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        for dataset in datasets.values():
            dataset.close()


if __name__ == "__main__":
    main()

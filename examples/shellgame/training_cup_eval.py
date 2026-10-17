"""In-training ShellGame cup-selection evaluation for absolute-joint policies.

The evaluator owns only data preparation, FK, metrics, and JSON reporting.  It
does not load a checkpoint or a second model: the training loop supplies action
chunks sampled from its current EMA parameters.
"""

# The implementation deliberately reuses the standalone evaluator's private
# helpers so both paths share exactly the same FK and cup-classification logic.
# ruff: noqa: SLF001

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import joint_fk_selection_eval as fk_eval
import numpy as np

from openpi import transforms
from openpi.models import model as model_api
from openpi.training import config as training_config

_SLOTS = ("left", "middle", "right")
_OBSERVATION_KEYS = (
    "image",
    "image_mask",
    "state",
    "tokenized_prompt",
    "tokenized_prompt_mask",
    "token_ar_mask",
    "token_loss_mask",
    "action_loss_mask",
    "frame_valid_mask",
    "fast_tokenized_prompt",
    "fast_tokenized_prompt_mask",
    "fast_token_ar_mask",
    "fast_token_loss_mask",
)


def _data_and_video_configs(config: training_config.TrainConfig):
    if isinstance(config.data, training_config.MultiDataConfigFactory):
        if len(config.data.datasets) != 1:
            raise ValueError(
                "ShellGame cup evaluation currently requires exactly one dataset; "
                f"got {len(config.data.datasets)}."
            )
        data_config = config.data.create_all(config.assets_dirs, config.model)[0]
        video_config = config.data.datasets[0].video_frame_config()
    else:
        if not hasattr(config.data, "video_frame_config"):
            raise ValueError("ShellGame cup evaluation requires a Pi0Mem video data config.")
        data_config = config.data.create(config.assets_dirs, config.model)
        video_config = config.data.video_frame_config()

    # Match data_loader.transform_dataset exactly.  This is especially
    # important for pi0.5 configs whose prompt carries a robot-type suffix.
    if data_config.robot_type is not None:
        data_config = training_config._set_robot_type(
            data_config, data_config.robot_type
        )
    if data_config.norm_stats is None:
        raise ValueError("ShellGame cup evaluation requires action/state normalization statistics.")
    return data_config, video_config


def _balanced_validation_ids(
    episode_dirs: list[Path],
    *,
    val_ratio: float,
    split_seed: int,
    sample_seed: int,
    num_episodes: int,
) -> np.ndarray:
    if num_episodes <= 0 or num_episodes % len(_SLOTS) != 0:
        raise ValueError(
            f"shellgame_cup_eval.num_episodes must be a positive multiple of {len(_SLOTS)}; "
            f"got {num_episodes}."
        )
    val_ids = fk_eval._validation_episode_ids(len(episode_dirs), val_ratio, split_seed)
    per_slot = num_episodes // len(_SLOTS)
    grouped: dict[str, list[int]] = {slot: [] for slot in _SLOTS}
    for episode_id in np.random.default_rng(sample_seed).permutation(val_ids):
        metadata_path = episode_dirs[int(episode_id)] / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        slot = str(metadata["final_ball_cup"])
        if slot not in grouped:
            raise ValueError(f"Unexpected final_ball_cup={slot!r} in {metadata_path}.")
        if len(grouped[slot]) < per_slot:
            grouped[slot].append(int(episode_id))
        if all(len(values) == per_slot for values in grouped.values()):
            break

    missing = {slot: per_slot - len(values) for slot, values in grouped.items() if len(values) < per_slot}
    if missing:
        raise ValueError(
            f"Validation split does not contain enough balanced ShellGame episodes: missing={missing}."
        )
    return np.asarray(sorted(value for values in grouped.values() for value in values), dtype=np.int64)


def _copy_tree(tree):
    return jax.tree.map(
        lambda value: value.copy() if isinstance(value, np.ndarray) else value,
        tree,
    )


class ShellgameCupEvaluator:
    """Fixed, balanced held-out task evaluator used by the training loop."""

    def __init__(
        self,
        config: training_config.TrainConfig,
        settings: training_config.ShellgameCupEvalConfig,
    ) -> None:
        self._config = config
        self._settings = settings
        self._closed = False

        dataset_root = Path(settings.raw_dataset_root).expanduser().resolve()
        self._episode_dirs = sorted(path for path in dataset_root.glob("episode_*") if path.is_dir())
        if not self._episode_dirs:
            raise FileNotFoundError(f"No episode_* directories found under {dataset_root}.")
        if not 0.0 < config.val_ratio < 1.0:
            raise ValueError(
                "ShellGame cup evaluation uses the held-out episode split and requires val_ratio in (0, 1); "
                f"got {config.val_ratio}."
            )
        if settings.batch_size <= 0:
            raise ValueError(f"shellgame_cup_eval.batch_size must be positive; got {settings.batch_size}.")
        if settings.num_sampling_steps <= 0:
            raise ValueError(
                "shellgame_cup_eval.num_sampling_steps must be positive; "
                f"got {settings.num_sampling_steps}."
            )

        data_config, video_config = _data_and_video_configs(config)
        if int(video_config.num_frames) != int(config.model.num_frames):
            raise ValueError(
                "ShellGame cup evaluation found inconsistent frame counts: "
                f"data={video_config.num_frames}, model={config.model.num_frames}."
            )
        self.num_frames = int(video_config.num_frames)
        self.frame_stride = int(video_config.frame_stride)
        self.num_sampling_steps = int(settings.num_sampling_steps)
        self.batch_size = int(settings.batch_size)

        self.selected_episode_ids = _balanced_validation_ids(
            self._episode_dirs,
            val_ratio=config.val_ratio,
            split_seed=config.seed,
            sample_seed=settings.sample_seed,
            num_episodes=settings.num_episodes,
        )
        raw_args = SimpleNamespace(num_frames=self.num_frames, frame_stride=self.frame_stride)
        self._records = [
            fk_eval._load_episode(self._episode_dirs[int(index)], raw_args)
            for index in self.selected_episode_ids
        ]

        input_transform = transforms.compose(
            [
                transforms.InjectDefaultPrompt(fk_eval.GRASP_PROMPT),
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    data_config.norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                    key_masks=data_config.normalize_masks,
                ),
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = transforms.compose(
            [
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    data_config.norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                    key_masks=data_config.normalize_masks,
                ),
                *data_config.data_transforms.outputs,
            ]
        )

        self._prepared = []
        for record in self._records:
            transformed = input_transform(_copy_tree(record["obs"]))
            self._prepared.append(
                {key: transformed[key] for key in _OBSERVATION_KEYS if key in transformed}
            )
            # BuildVideoTensor copied the selected frames into the prepared
            # sample, so retaining the per-frame raw dictionary only wastes RAM.
            record.pop("obs", None)
            record.pop("reveal_wrist", None)
            record.pop("reveal_third_person", None)

        shell = fk_eval.base._import_shellgame_tools(settings.robosuite_root)
        self._shell = shell
        self._fk_env = fk_eval._make_fk_env(shell, self._records[0])
        for record in self._records:
            reference_fk, _ = fk_eval._fk_chunk(self._shell, self._fk_env, record["reference_joint"])
            reference_class = fk_eval._classify(
                reference_fk[:, :2], record["cup_positions"], settings.selection_radius
            )
            record["reference_endpoint_correct"] = reference_class["endpoint_cup"] == record["target_cup"]

        self.output_dir = config.checkpoint_dir / "cup_eval"
        logging.info(
            "ShellGame cup eval ready: episodes=%d (balanced %d/slot), frames=%d, stride=%d, "
            "batch=%d, diffusion_steps=%d, output=%s",
            len(self._records),
            len(self._records) // len(_SLOTS),
            self.num_frames,
            self.frame_stride,
            self.batch_size,
            self.num_sampling_steps,
            self.output_dir,
        )

    def iter_batches(self):
        """Yield fixed-shape model observations and the number of real rows."""
        for batch_index, start in enumerate(range(0, len(self._prepared), self.batch_size)):
            items = list(self._prepared[start : start + self.batch_size])
            valid_size = len(items)
            while len(items) < self.batch_size:
                items.append(items[-1])
            stacked = jax.tree.map(lambda *values: np.stack(values, axis=0), *items)
            observation = model_api.Observation.from_dict(stacked)
            yield batch_index, observation, valid_size

    def sample_rng(self, batch_index: int):
        # Reuse exactly the same diffusion noise at every training step so the
        # metric measures model changes rather than Monte-Carlo noise.
        return jax.random.fold_in(jax.random.key(self._settings.sample_seed), batch_index)

    def summarize(self, normalized_action_batches: list[np.ndarray], *, step: int) -> dict[str, float]:
        normalized_actions = np.concatenate(normalized_action_batches, axis=0)[: len(self._records)]
        if normalized_actions.shape[0] != len(self._records):
            raise ValueError(
                f"Expected {len(self._records)} predictions, got {normalized_actions.shape[0]}."
            )

        samples: list[dict[str, Any]] = []
        for record, prepared, normalized in zip(
            self._records, self._prepared, normalized_actions, strict=True
        ):
            output = self._output_transform(
                {
                    "state": np.asarray(prepared["state"]),
                    "actions": np.asarray(normalized),
                }
            )
            predicted = np.asarray(output["actions"], dtype=np.float32)
            predicted_eef, clipped_values = fk_eval._fk_chunk(self._shell, self._fk_env, predicted)
            classification = fk_eval._classify(
                predicted_eef[:, :2], record["cup_positions"], self._settings.selection_radius
            )
            endpoint_cup = classification["endpoint_cup"]
            endpoint_slot = str(record["cup_slots"][endpoint_cup])
            nearest_distance = min(classification["endpoint_distances_m"].values())
            samples.append(
                {
                    "episode": record["episode"],
                    "target_cup": record["target_cup"],
                    "target_slot": record["final_ball_slot"],
                    "endpoint_cup": endpoint_cup,
                    "endpoint_slot": endpoint_slot,
                    "endpoint_correct": endpoint_cup == record["target_cup"],
                    "endpoint_reaches_any_cup": nearest_distance <= self._settings.selection_radius,
                    "endpoint_nearest_distance_m": float(nearest_distance),
                    "endpoint_distances_m": classification["endpoint_distances_m"],
                    "joint_mse": float(
                        np.mean(
                            np.square(
                                predicted[:, : fk_eval.joint_eval.JOINT_DIM]
                                - record["reference_joint"]
                            )
                        )
                    ),
                    "clipped_joint_values": clipped_values,
                }
            )

        predicted_slots = collections.Counter(sample["endpoint_slot"] for sample in samples)
        target_slots = collections.Counter(sample["target_slot"] for sample in samples)
        confusion: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for sample in samples:
            confusion[sample["target_slot"]][sample["endpoint_slot"]] += 1

        accuracy = float(np.mean([sample["endpoint_correct"] for sample in samples]))
        numeric_metrics = {
            "val/cup_endpoint_accuracy": accuracy,
            "val/cup_accuracy_above_chance": accuracy - 1.0 / len(_SLOTS),
            "val/cup_reach_any_rate": float(
                np.mean([sample["endpoint_reaches_any_cup"] for sample in samples])
            ),
            "val/cup_mean_endpoint_distance_m": float(
                np.mean([sample["endpoint_nearest_distance_m"] for sample in samples])
            ),
            "val/cup_mean_joint_mse": float(np.mean([sample["joint_mse"] for sample in samples])),
            "val/cup_reference_endpoint_accuracy": float(
                np.mean([record["reference_endpoint_correct"] for record in self._records])
            ),
            "val/cup_clipped_joint_values_per_prediction": float(
                np.mean([sample["clipped_joint_values"] for sample in samples])
            ),
        }
        for slot in _SLOTS:
            numeric_metrics[f"val/cup_pred_{slot}_rate"] = predicted_slots[slot] / len(samples)
            numeric_metrics[f"val/cup_target_{slot}_rate"] = target_slots[slot] / len(samples)

        output = {
            "step": int(step),
            "config": self._config.name,
            "raw_dataset_root": str(Path(self._settings.raw_dataset_root).expanduser().resolve()),
            "settings": dataclasses.asdict(self._settings),
            "num_frames": self.num_frames,
            "frame_stride": self.frame_stride,
            "fixed_diffusion_noise": True,
            "selected_episode_ids": self.selected_episode_ids.tolist(),
            "metrics": numeric_metrics,
            "slot_confusion": {
                target: dict(predicted) for target, predicted in confusion.items()
            },
            "samples": samples,
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"step_{int(step):08d}.json"
        output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
        logging.info("Wrote ShellGame cup eval details to %s", output_path)
        return numeric_metrics

    @property
    def num_batches(self) -> int:
        return math.ceil(len(self._prepared) / self.batch_size)

    def close(self) -> None:
        if not self._closed:
            self._fk_env.close()
            self._closed = True

"""Frozen-memory probe after one ShellGame swap.

Frames 0..29 are compressed as history (reveal, cover, and swap_0). Frame 30
is present only because Pi0MemCompress separates history from a current frame;
the classifier reads *only* the Base-camera ``history_mem``. Labels are
derived from each raw episode's initial target slot and first recorded swap.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.nnx as nnx
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders
from scripts.mem import train_pi0_mem_compress as _trainer


DATASET_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_lerobot_absolute_joint"
)
RAW_DATASET_ROOT = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_absolute_joint_dataset"
)
DERIVED_LABELS_PATH = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame/"
    "one_swap_history_probe/episode_labels.jsonl"
)
SOURCE_CHECKPOINT = (
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
    "pi0_mem_compress_evan_shellgame_openpi_joint_260727/"
    "my_experiment_30f_s2_6gpu/23000/params"
)


def apply_slot_swap(slot: str, swap: list[str]) -> str:
    if len(swap) != 2 or swap[0] == swap[1]:
        raise ValueError(f"Invalid swap: {swap!r}")
    if slot == swap[0]:
        return swap[1]
    if slot == swap[1]:
        return swap[0]
    return slot


def build_one_swap_labels() -> pathlib.Path:
    """Derive and cross-check episode_index -> target slot after swap_0."""
    with (DATASET_ROOT / "meta/episodes.jsonl").open("r", encoding="utf-8") as handle:
        lerobot_records = {
            int(record["episode_index"]): record
            for line in handle
            if line.strip()
            for record in (json.loads(line),)
        }

    records = []
    raw_paths = sorted(RAW_DATASET_ROOT.glob("episode_*/metadata.json"))
    if len(raw_paths) != len(lerobot_records):
        raise ValueError(
            f"Raw/LeRobot episode count mismatch: {len(raw_paths)} != {len(lerobot_records)}"
        )

    for raw_path in raw_paths:
        episode_index = int(raw_path.parent.name.split("_")[-1])
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        lerobot = lerobot_records[episode_index]
        initial = str(raw["initial_ball_cup"])
        swaps = raw["swaps"]
        if raw["phase_ranges"]["swap_0"] != [20, 29]:
            raise ValueError(f"Unexpected swap_0 range in {raw_path}")
        if initial != str(lerobot["initial_ball_cup"]):
            raise ValueError(f"Initial-label mismatch for episode {episode_index}")

        after_first = apply_slot_swap(initial, swaps[0])
        reconstructed_final = initial
        for swap in swaps:
            reconstructed_final = apply_slot_swap(reconstructed_final, swap)
        if reconstructed_final != str(raw["final_ball_cup"]):
            raise ValueError(f"Raw final-label reconstruction failed for episode {episode_index}")
        if reconstructed_final != str(lerobot["final_ball_cup"]):
            raise ValueError(f"Raw/LeRobot final-label mismatch for episode {episode_index}")

        records.append(
            {
                "episode_index": episode_index,
                "initial_ball_cup": initial,
                "first_swap": swaps[0],
                "after_first_swap_ball_cup": after_first,
                "final_ball_cup": reconstructed_final,
            }
        )

    DERIVED_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DERIVED_LABELS_PATH.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return DERIVED_LABELS_PATH


@dataclasses.dataclass(frozen=True)
class OneSwapHistoryProbeConfig(_base_model.Pi0MemCompressConfig):
    probe_stream: str = "base_rgb"

    def create(self, rng: at.KeyArrayLike) -> OneSwapHistoryProbe:
        return OneSwapHistoryProbe(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_classifier_only(self) -> nnx.filterlib.Filter:
        classifier = nnx_utils.PathRegex(r".*HistoryClassifier.*")
        return nnx.Not(classifier)


class OneSwapHistoryProbe(_base_model.Pi0MemCompress):
    """Linear probe over the checkpoint's frozen Base history memory."""

    def __init__(self, config: OneSwapHistoryProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.probe_stream = config.probe_stream

    def compute_history_classification(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        observation = _model.preprocess_observation(rng, observation, train=train)
        image = observation.images[self.probe_stream]
        if image.ndim == 4:
            image = image[:, None, ...]
        if image.ndim != 5 or image.shape[1] != 31:
            raise ValueError(f"One-swap probe expects [B,31,H,W,C], got {image.shape}")

        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_mem = encoder_out["encoder"]["history_mem"]
        normalized_mem = self.HistoryClassifierNorm(history_mem)
        logits = self.HistoryClassifierHead(
            normalized_mem.reshape(normalized_mem.shape[0], -1)
        )
        return logits, {
            "history_mem": jnp.asarray(history_mem),
            "encoder_auxes": (encoder_out["encoder"],),
        }


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = OneSwapHistoryProbeConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        # At anchor frame 30, frames 0..29 are history and frame 30 is current.
        num_frames=31,
        memory_every=1,
        current_frame_index=-1,
        history_memory_tokens=256,
        history_resampler_depth=1,
        history_use_current_condition=True,
        history_gate_fixed=1.0,
        diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=3,
        probe_stream="base_rgb",
    )

    if args.probe_mode == "frozen":
        freeze_filter = model.get_freeze_filter_classifier_only()
    elif args.probe_mode == "resampler":
        freeze_filter = model.get_freeze_filter_history_classifier_probe()
    else:
        raise ValueError(f"Unknown probe mode: {args.probe_mode}")

    return _config.TrainConfig(
        name=f"pi0_shellgame_one_swap_{args.probe_mode}_history_probe_260807",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=freeze_filter,
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=str(DATASET_ROOT),
                    assets=_config.AssetsConfig(
                        asset_id=".", assets_dir=str(DATASET_ROOT)
                    ),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=31,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            SOURCE_CHECKPOINT
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=20,
            peak_lr=args.peak_lr,
            decay_steps=args.steps,
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
        ema_decay=None,
        num_train_steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fsdp_devices=args.fsdp_devices,
        log_interval=10,
        save_interval=args.steps,
        keep_period=args.steps,
        val_ratio=0.1,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        wandb_enabled=False,
        overwrite=args.overwrite,
        resume=args.resume,
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=str(labels_path),
            label_key="after_first_swap_ball_cup",
            classes=("left", "middle", "right"),
            min_frame_index=30,
            max_frame_index=30,
            loss_weight=1.0,
            action_loss_weight=0.0,
            disable_train_augmentation=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", default="one_swap_frozen_history_linear")
    parser.add_argument(
        "--probe-mode",
        choices=("frozen", "resampler"),
        default="frozen",
        help="Train only the linear head, or the HistoryResampler plus head.",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--peak-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    _trainer.main(build_config(parsed_args, build_one_swap_labels()))

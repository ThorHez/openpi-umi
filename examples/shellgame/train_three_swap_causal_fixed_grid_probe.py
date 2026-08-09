"""Diagnose compositional ShellGame tracking over all three swaps.

The model receives raw frames 0..59 as history (frame 60 is loaded only as the
Pi0 current-frame placeholder).  A shared causal fixed-grid temporal encoder
predicts the hidden-ball slot after swap_0, swap_1, and swap_2.  The three
3-way logits are represented as a factorized 27-way joint distribution so the
existing classification trainer optimizes the sum of the three endpoint CEs.

This is an isolated visual-tracking diagnostic: Pi0 action loss, the learned
final memory compressor, and the Pi0 history reader are all bypassed.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import flax.traverse_util
import jax.numpy as jnp
import numpy as np

from examples.shellgame.train_one_swap_history_probe import DATASET_ROOT
from examples.shellgame.train_one_swap_history_probe import RAW_DATASET_ROOT
from examples.shellgame.train_one_swap_history_probe import SOURCE_CHECKPOINT
from examples.shellgame.train_one_swap_history_probe import apply_slot_swap
from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base_model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import utils as training_utils
from scripts.mem import train_pi0_mem_compress as _trainer

LABELS_PATH = pathlib.Path(
    "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame/"
    "three_swap_causal_fixed_grid_probe/episode_labels.jsonl"
)
SLOTS = ("left", "middle", "right")
JOINT_CLASSES = tuple("|".join(values) for values in itertools.product(SLOTS, repeat=3))


def build_three_swap_labels() -> pathlib.Path:
    """Create and cross-check the three endpoint labels for every episode."""
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
        slot = str(raw["initial_ball_cup"])
        endpoint_slots = []
        for swap in raw["swaps"]:
            slot = apply_slot_swap(slot, swap)
            endpoint_slots.append(slot)
        if len(endpoint_slots) != 3:
            raise ValueError(f"Expected three swaps in {raw_path}, got {len(endpoint_slots)}")
        if endpoint_slots[-1] != str(raw["final_ball_cup"]):
            raise ValueError(f"Raw final-label reconstruction failed for episode {episode_index}")
        if endpoint_slots[-1] != str(lerobot["final_ball_cup"]):
            raise ValueError(f"Raw/LeRobot final-label mismatch for episode {episode_index}")
        records.append(
            {
                "episode_index": episode_index,
                "after_swap_0": endpoint_slots[0],
                "after_swap_1": endpoint_slots[1],
                "after_swap_2": endpoint_slots[2],
                "swap_track_code": "|".join(endpoint_slots),
            }
        )

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return LABELS_PATH


class CausalFactorizedSpaceTimeBlock(nn.Module):
    """Causal temporal attention per grid cell, then spatial attention per frame."""

    width: int = 256
    num_heads: int = 8
    mlp_ratio: int = 4
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, x):
        b, t, n, d = x.shape
        y = nn.LayerNorm(name="temporal_ln", dtype=self.dtype_mm)(x)
        y = jnp.transpose(y, (0, 2, 1, 3)).reshape(b * n, t, d)
        causal_mask = jnp.tril(jnp.ones((1, 1, t, t), dtype=jnp.bool_))
        y = nn.MultiHeadDotProductAttention(
            name="temporal_attn",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(y, y, mask=causal_mask)
        y = y.reshape(b, n, t, d).transpose(0, 2, 1, 3)
        x = x + y

        y = nn.LayerNorm(name="spatial_ln", dtype=self.dtype_mm)(x)
        y = y.reshape(b * t, n, d)
        y = nn.MultiHeadDotProductAttention(
            name="spatial_attn",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(y, y)
        x = x + y.reshape(b, t, n, d)

        y = nn.LayerNorm(name="mlp_ln", dtype=self.dtype_mm)(x)
        y = nn.Dense(self.width * self.mlp_ratio, name="mlp_in", dtype=self.dtype_mm)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.width, name="mlp_out", dtype=self.dtype_mm)(y)
        return x + y


class EndpointReadout(nn.Module):
    width: int = 256
    num_heads: int = 8
    num_classes: int = 3
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, tokens):
        b = tokens.shape[0]
        query = self.param(
            "query",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.width),
            tokens.dtype,
        )
        query = jnp.tile(query, (b, 1, 1))
        pooled = nn.MultiHeadDotProductAttention(
            name="attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=True,
            dtype=self.dtype_mm,
        )(query, tokens)
        pooled = nn.LayerNorm(name="output_ln", dtype=self.dtype_mm)(pooled[:, 0])
        return nn.Dense(self.num_classes, name="classifier", dtype=jnp.float32)(pooled)


class ThreeSwapCausalTracker(nn.Module):
    """Shared tracker with causal endpoint readouts at raw frames 29, 39, and 49."""

    num_frames: int = 60
    input_width: int = 1152
    width: int = 256
    depth: int = 2
    num_heads: int = 8
    spatial_pool_factor: int = 2
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, patch_tokens):
        b, t, n, d = patch_tokens.shape
        if (t, n, d) != (self.num_frames, 256, self.input_width):
            raise ValueError(
                f"Expected [B,{self.num_frames},256,{self.input_width}], got {patch_tokens.shape}"
            )
        input_grid = 16
        output_grid = input_grid // self.spatial_pool_factor
        x = patch_tokens.reshape(
            b,
            t,
            output_grid,
            self.spatial_pool_factor,
            output_grid,
            self.spatial_pool_factor,
            d,
        )
        x = jnp.mean(x, axis=(3, 5)).reshape(b, t, output_grid**2, d)
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(self.width, name="input_projection", dtype=self.dtype_mm)(x)
        temporal_pos = self.param(
            "temporal_pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_frames, 1, self.width),
            x.dtype,
        )
        x = x + temporal_pos
        for block_index in range(self.depth):
            x = CausalFactorizedSpaceTimeBlock(
                name=f"block_{block_index}",
                width=self.width,
                num_heads=self.num_heads,
                dtype_mm=self.dtype_mm,
            )(x)
        x = nn.LayerNorm(name="shared_output_ln", dtype=self.dtype_mm)(x)

        stage_logits = []
        for stage, endpoint in enumerate((29, 39, 49)):
            prefix = x[:, : endpoint + 1].reshape(b, -1, self.width)
            stage_logits.append(
                EndpointReadout(
                    name=f"endpoint_{stage}",
                    width=self.width,
                    num_heads=self.num_heads,
                    dtype_mm=self.dtype_mm,
                )(prefix)
            )
        logits_0, logits_1, logits_2 = stage_logits
        joint_logits = (
            logits_0[:, :, None, None]
            + logits_1[:, None, :, None]
            + logits_2[:, None, None, :]
        ).reshape(b, 27)
        return joint_logits, jnp.stack(stage_logits, axis=1)


@dataclasses.dataclass(frozen=True)
class ThreeSwapCausalProbeConfig(_base_model.Pi0MemCompressConfig):
    temporal_width: int = 256
    temporal_depth: int = 2
    temporal_heads: int = 8
    spatial_pool_factor: int = 2

    def create(self, rng: at.KeyArrayLike) -> ThreeSwapCausalProbeModel:
        return ThreeSwapCausalProbeModel(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_tracker_only(self) -> nnx.filterlib.Filter:
        tracker = nnx_utils.PathRegex(r".*HistoryThreeSwapCausalTracker.*")
        return nnx.Not(tracker)


class ThreeSwapCausalProbeModel(_base_model.Pi0MemCompress):
    def __init__(self, config: ThreeSwapCausalProbeConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.HistoryThreeSwapCausalTracker = nnx_bridge.ToNNX(
            ThreeSwapCausalTracker(
                num_frames=60,
                input_width=1152,
                width=config.temporal_width,
                depth=config.temporal_depth,
                num_heads=config.temporal_heads,
                spatial_pool_factor=config.spatial_pool_factor,
                dtype_mm=config.dtype,
            )
        )
        fake_tokens = jnp.zeros((1, 60, 256, 1152), dtype=jnp.bfloat16)
        self.HistoryThreeSwapCausalTracker.lazy_init(fake_tokens, rngs=rngs)

    def compute_history_classification(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        observation = _model.preprocess_observation(rng, observation, train=train)
        image = observation.images["base_rgb"]
        if image.ndim == 4:
            image = image[:, None]
        if image.ndim != 5 or image.shape[1] != 61:
            raise ValueError(f"Three-swap probe expects [B,61,H,W,C], got {image.shape}")
        _, encoder_out = self.PaliGemma.img(image, train=False)
        history_patches = encoder_out["with_posemb"][:, :60]
        joint_logits, stage_logits = self.HistoryThreeSwapCausalTracker(history_patches)
        return joint_logits, {
            "history_mem": stage_logits,
            "encoder_auxes": (),
        }


@dataclasses.dataclass(frozen=True)
class ProbeCheckpointLoader:
    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(self.params_path, restore_type=np.ndarray)
        target_flat = flax.traverse_util.flatten_dict(params, sep="/")
        source_flat = flax.traverse_util.flatten_dict(loaded, sep="/")
        result = {}
        restored = 0
        initialized = []
        for key, reference in target_flat.items():
            candidate = source_flat.get(key)
            if candidate is not None and np.shape(candidate) == np.shape(reference):
                result[key] = np.asarray(candidate, dtype=np.dtype(reference.dtype))
                restored += 1
            else:
                result[key] = reference
                initialized.append(key)
        print(
            "ProbeCheckpointLoader: "
            f"restored={restored}, initialized={len(initialized)}, "
            f"examples={initialized[:5]}"
        )
        return flax.traverse_util.unflatten_dict(result, sep="/")


_BASE_EVAL_STEP = _trainer.eval_step


def multistage_eval_step(
    config,
    rng,
    state: training_utils.TrainState,
    batch,
    *,
    class_labels_by_episode=None,
):
    """Add per-swap accuracies while preserving the trainer's joint metrics."""
    result = _BASE_EVAL_STEP(
        config,
        rng,
        state,
        batch,
        class_labels_by_episode=class_labels_by_episode,
    )
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, _ = batch
    joint_logits, _ = model.compute_history_classification(rng, observation, train=False)
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    safe_episode = jnp.clip(episode_index, 0, class_labels_by_episode.shape[0] - 1)
    labels = class_labels_by_episode[safe_episode]
    predictions = jnp.argmax(joint_logits, axis=-1)
    pred_stages = jnp.stack(
        (predictions // 9, (predictions // 3) % 3, predictions % 3),
        axis=-1,
    )
    label_stages = jnp.stack((labels // 9, (labels // 3) % 3, labels % 3), axis=-1)
    for stage in range(3):
        result[f"val/swap_{stage}_accuracy"] = jnp.mean(
            pred_stages[:, stage] == label_stages[:, stage]
        )
    return result


def build_config(args: argparse.Namespace, labels_path: pathlib.Path) -> _config.TrainConfig:
    model = ThreeSwapCausalProbeConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=61,
        memory_every=0,
        current_frame_index=-1,
        # The base Pi0MemCompress constructor builds its unused legacy
        # classifier with M*D input features, so M must remain non-zero.
        # One token is the cheapest inert placeholder; memory_every=0 keeps
        # it out of the current-frame ViT and our tracker bypasses it.
        history_memory_tokens=1,
        history_resampler_depth=1,
        history_use_current_condition=False,
        history_gate_fixed=0.0,
        diversity_weight=0.0,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=27,
        temporal_width=args.temporal_width,
        temporal_depth=args.temporal_depth,
        temporal_heads=args.temporal_heads,
        spatial_pool_factor=2,
    )
    return _config.TrainConfig(
        name=f"pi0_shellgame_three_swap_causal_fixed_grid_d{args.temporal_depth}_260808",
        exp_name=args.exp_name,
        model=model,
        freeze_filter=model.get_freeze_filter_tracker_only(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=str(DATASET_ROOT),
                    assets=_config.AssetsConfig(asset_id=".", assets_dir=str(DATASET_ROOT)),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=61,
                    frame_stride=1,
                )
            ],
        ),
        weight_loader=ProbeCheckpointLoader(SOURCE_CHECKPOINT),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
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
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=str(labels_path),
            label_key="swap_track_code",
            classes=JOINT_CLASSES,
            min_frame_index=60,
            max_frame_index=60,
            loss_weight=1.0,
            action_loss_weight=0.0,
            disable_train_augmentation=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--peak-lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--temporal-width", type=int, default=256)
    parser.add_argument("--temporal-depth", type=int, choices=(2, 6), required=True)
    parser.add_argument("--temporal-heads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _trainer.eval_step = multistage_eval_step
    _trainer.main(build_config(args, build_three_swap_labels()))


if __name__ == "__main__":
    main()

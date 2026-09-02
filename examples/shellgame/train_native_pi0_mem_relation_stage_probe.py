"""Train native Pi0Mem with final-slot plus dense three-swap supervision.

This is the controlled positive-supervision counterpart of
``train_native_pi0_mem_final_slot_probe.py``.  The native SigLIP-MEM video
encoder, 60-frame input, temporal schedule, base initialization, trainable
visual parameters, and episode-heldout split are unchanged.  The only task
change is that causal encoder tokens at frames 29, 39, and 49 additionally
predict:

* which cup pair was swapped at that stage; and
* the ball slot immediately after that swap.

The original frame-59 final-slot head remains the primary diagnostic.  There
is no action loss, recurrent semantic updater, memory compressor, teacher
forcing, or relation input to the model.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import optax

from examples.shellgame import train_native_pi0_mem_final_slot_probe as _final_probe
from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain as _memory_recipe
from scripts.mem import train_pi0_mem_compress as _classifier_trainer
from scripts.mem import train_semantic_memory as _trainer


NUM_STAGES = 3
NUM_CLASSES = 3
STAGE_ENDPOINTS = (29, 39, 49)
ABLATION_MODES = _final_probe.ABLATION_MODES


class NativeEndpointRelationStageHead(nn.Module):
    """Shared learned-query readout for the three causal endpoints."""

    input_width: int = 1152
    width: int = 256
    num_heads: int = 8
    dtype_mm: str = "bfloat16"

    @nn.compact
    def __call__(self, endpoint_tokens, *, train: bool):
        if endpoint_tokens.ndim != 4:
            raise ValueError(
                "Expected endpoint tokens [B,3,N,D], got "
                f"{endpoint_tokens.shape}"
            )
        b, stages, tokens, width = endpoint_tokens.shape
        if stages != NUM_STAGES or width != self.input_width:
            raise ValueError(
                f"Expected [B,{NUM_STAGES},N,{self.input_width}], got "
                f"{endpoint_tokens.shape}"
            )
        x = endpoint_tokens.reshape(b * stages, tokens, width)
        x = nn.LayerNorm(name="input_ln", dtype=self.dtype_mm)(x)
        x = nn.Dense(
            self.width, name="input_projection", dtype=self.dtype_mm
        )(x)
        query = self.param(
            "readout_query",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.width),
            x.dtype,
        )
        query = jnp.tile(query, (b * stages, 1, 1))
        pooled = nn.MultiHeadDotProductAttention(
            name="readout_attention",
            num_heads=self.num_heads,
            dropout_rate=0.0,
            deterministic=not train,
            dtype=self.dtype_mm,
        )(query, x)[:, 0]
        pooled = nn.LayerNorm(name="readout_ln", dtype=self.dtype_mm)(pooled)
        pooled = pooled.astype(jnp.float32)
        relation_logits = nn.Dense(
            NUM_CLASSES,
            name="relation_classifier",
            dtype=jnp.float32,
            kernel_init=nn.initializers.xavier_uniform(),
        )(pooled)
        stage_slot_logits = nn.Dense(
            NUM_CLASSES,
            name="stage_slot_classifier",
            dtype=jnp.float32,
            kernel_init=nn.initializers.xavier_uniform(),
        )(pooled)
        return (
            relation_logits.reshape(b, stages, NUM_CLASSES),
            stage_slot_logits.reshape(b, stages, NUM_CLASSES),
        )


@dataclasses.dataclass(frozen=True)
class NativePi0MemDenseSupervisionConfig(
    _final_probe.NativePi0MemFinalSlotProbeConfig
):
    """Trainer-compatible config for the dense-supervision control."""

    final_loss_weight: float = 1.0
    endpoint_head_width: int = 256
    endpoint_head_heads: int = 8

    def create(self, rng: at.KeyArrayLike) -> "NativePi0MemDenseSupervisionProbe":
        return NativePi0MemDenseSupervisionProbe(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_native_video_and_heads(self) -> nnx.filterlib.Filter:
        trainable = nnx_utils.PathRegex(
            r".*(PaliGemma/img|NativeFinalSlotHead|NativeEndpointAuxHead).*"
        )
        return nnx.Not(trainable)


class NativePi0MemDenseSupervisionProbe(
    _final_probe.NativePi0MemFinalSlotProbe
):
    """Native causal video encoder with final and endpoint readouts."""

    def __init__(
        self, config: NativePi0MemDenseSupervisionConfig, rngs: nnx.Rngs
    ):
        super().__init__(config, rngs)
        self.NativeEndpointAuxHead = nnx_bridge.ToNNX(
            NativeEndpointRelationStageHead(
                input_width=1152,
                width=config.endpoint_head_width,
                num_heads=config.endpoint_head_heads,
                dtype_mm=config.dtype,
            )
        )
        self.NativeEndpointAuxHead.lazy_init(
            jnp.zeros((1, NUM_STAGES, 256, 1152), dtype=jnp.bfloat16),
            train=False,
            rngs=rngs,
        )

    def _supervised_outputs_from_video(self, video, *, train: bool):
        current_tokens, encoder_out = self.PaliGemma.img(video, train=train)
        final_logits, _pooled = self.NativeFinalSlotHead(
            current_tokens, train=train
        )
        encoded_video = encoder_out["encoded_video"]
        if encoded_video.shape[1] != _final_probe.NUM_FRAMES:
            raise ValueError(
                "Expected the complete 60-frame causal encoder output, got "
                f"{encoded_video.shape}"
            )
        endpoint_tokens = jnp.stack(
            [encoded_video[:, endpoint] for endpoint in STAGE_ENDPOINTS],
            axis=1,
        )
        relation_logits, stage_slot_logits = self.NativeEndpointAuxHead(
            endpoint_tokens, train=train
        )
        return {
            "final_logits": final_logits,
            "relation_logits": relation_logits,
            "stage_slot_logits": stage_slot_logits,
        }

    def compute_dense_supervision_outputs(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        train: bool,
    ):
        observation = _model.preprocess_observation(
            rng, observation, train=train
        )
        return self._supervised_outputs_from_video(
            self._base_video(observation), train=train
        )

    def compute_final_ablation_logits(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
    ):
        observation = _model.preprocess_observation(
            rng, observation, train=False
        )
        video = self._base_video(observation)
        return {
            mode: self._supervised_outputs_from_video(
                self._ablate_video(video, mode), train=False
            )["final_logits"]
            for mode in ABLATION_MODES
            if mode != "normal"
        }


@dataclasses.dataclass(frozen=True)
class NativeDenseSupervisionTrainConfig(
    _memory_recipe.ShellGameSemanticMemoryPretrainConfig
):
    """Semantic trainer contract with an explicit final-label loss weight."""

    final_loss_weight: float = 1.0


def compute_objective(
    config: NativeDenseSupervisionTrainConfig,
    model: NativePi0MemDenseSupervisionProbe,
    rng,
    observation,
    label_table,
    *,
    train: bool,
):
    """Final CE plus three relation and three stage-slot CEs."""
    if observation.episode_index is None:
        raise ValueError("Dense native supervision requires episode_index")
    episode_index = jnp.asarray(observation.episode_index, dtype=jnp.int32)
    labels = label_table[episode_index]
    relation_labels = labels[:, 1 : 1 + NUM_STAGES]
    stage_labels = labels[:, 1 + NUM_STAGES :]
    final_labels = stage_labels[:, -1]

    outputs = model.compute_dense_supervision_outputs(
        rng, observation, train=train
    )
    final_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            outputs["final_logits"].astype(jnp.float32), final_labels
        )
    )
    relation_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            outputs["relation_logits"].astype(jnp.float32), relation_labels
        )
    )
    stage_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            outputs["stage_slot_logits"].astype(jnp.float32), stage_labels
        )
    )
    loss = (
        config.final_loss_weight * final_loss
        + config.relation_loss_weight * relation_loss
        + config.stage_memory_loss_weight * stage_loss
    )
    final_predictions = jnp.argmax(outputs["final_logits"], axis=-1)
    relation_predictions = jnp.argmax(outputs["relation_logits"], axis=-1)
    stage_predictions = jnp.argmax(outputs["stage_slot_logits"], axis=-1)
    metrics = {
        "loss": loss,
        "final_loss": final_loss,
        "final_accuracy": jnp.mean(final_predictions == final_labels),
        "relation_loss": relation_loss,
        "relation_accuracy": jnp.mean(relation_predictions == relation_labels),
        "relation_sequence_accuracy": jnp.mean(
            jnp.all(relation_predictions == relation_labels, axis=1)
        ),
        "stage_memory_loss": stage_loss,
        "stage_memory_accuracy": jnp.mean(stage_predictions == stage_labels),
        "stage_sequence_accuracy": jnp.mean(
            jnp.all(stage_predictions == stage_labels, axis=1)
        ),
        "endpoint_final_accuracy": jnp.mean(
            stage_predictions[:, -1] == final_labels
        ),
    }
    for stage in range(NUM_STAGES):
        metrics[f"relation_{stage}_accuracy"] = jnp.mean(
            relation_predictions[:, stage] == relation_labels[:, stage]
        )
        metrics[f"slot_{stage}_accuracy"] = jnp.mean(
            stage_predictions[:, stage] == stage_labels[:, stage]
        )

    if not train:
        ablation_logits = model.compute_final_ablation_logits(rng, observation)
        for mode, logits in ablation_logits.items():
            metrics[f"final_{mode}_accuracy"] = jnp.mean(
                jnp.argmax(logits, axis=-1) == final_labels
            )
            metrics[f"final_{mode}_loss"] = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(
                    logits.astype(jnp.float32), final_labels
                )
            )
    return loss, metrics


def build_config(args: argparse.Namespace) -> NativeDenseSupervisionTrainConfig:
    model = NativePi0MemDenseSupervisionConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        max_token_len=256,
        num_frames=_final_probe.NUM_FRAMES,
        current_frame_index=-1,
        temporal_every=args.temporal_every,
        history_classifier_num_classes=NUM_CLASSES,
        diversity_weight=0.0,
        native_head_width=args.head_width,
        native_head_heads=args.head_heads,
        endpoint_head_width=args.head_width,
        endpoint_head_heads=args.head_heads,
        siglip_remat_policy=args.remat_policy,
    )
    data_config_cls = _config.LeRobotUmiDataConfig_shellgame_Pi0Mem_AbsoluteEEF7
    return NativeDenseSupervisionTrainConfig(
        name="pi0_shellgame_native_mem_relation_stage_probe_260824",
        exp_name=args.exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        model=model,
        freeze_filter=model.get_freeze_filter_native_video_and_heads(),
        data=_config.MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                data_config_cls(
                    repo_id=str(_final_probe.DATASET_ROOT),
                    assets=_config.AssetsConfig(
                        asset_id=".", assets_dir=str(_final_probe.DATASET_ROOT)
                    ),
                    base_config=_config.UmiDataConfig(
                        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=_final_probe.NUM_FRAMES,
                    frame_stride=1,
                    video_layout="sliding",
                    fixed_prefix_frames=0,
                    tokenize_prompt=False,
                    min_frame_index=_final_probe.NUM_FRAMES - 1,
                    max_frame_index=_final_probe.NUM_FRAMES - 1,
                )
            ],
        ),
        weight_loader=_final_probe.NativeProbeCheckpointLoader(
            args.init_checkpoint
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(args.warmup_steps, max(args.steps - 1, 0)),
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, 1),
            decay_lr=args.peak_lr * 0.1,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
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
        raw_metadata_root=args.raw_metadata_root,
        initial_loss_weight=0.0,
        final_loss_weight=args.final_loss_weight,
        relation_loss_weight=args.relation_loss_weight,
        stage_memory_loss_weight=args.stage_loss_weight,
        memory_train_augmentation=False,
        # Reuse the already validated classification-only loader solely for
        # its explicit frame_index==59 filtering.  The semantic trainer's
        # default loader otherwise retains every action row in each episode.
        shellgame_memory_classifier=_config.ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=str(_final_probe.LABELS_PATH),
            label_key="final_ball_cup",
            classes=_memory_recipe.SLOTS,
            min_frame_index=_final_probe.NUM_FRAMES - 1,
            max_frame_index=_final_probe.NUM_FRAMES - 1,
            loss_weight=1.0,
            action_loss_weight=0.0,
            disable_train_augmentation=True,
        ),
    )


def run_self_test() -> None:
    head = NativeEndpointRelationStageHead(
        input_width=32, width=16, num_heads=4, dtype_mm="float32"
    )
    tokens = jax.random.normal(jax.random.key(1), (2, NUM_STAGES, 8, 32))
    variables = head.init(jax.random.key(0), tokens, train=False)
    relation, slots = head.apply(variables, tokens, train=False)
    expected = (2, NUM_STAGES, NUM_CLASSES)
    if relation.shape != expected or slots.shape != expected:
        raise AssertionError(
            f"Unexpected auxiliary shapes: {relation.shape}, {slots.shape}"
        )
    print(
        "Native endpoint auxiliary-head self-test passed: "
        f"relation={relation.shape}, stage_slots={slots.shape}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "--init-checkpoint", default=_final_probe.DEFAULT_INIT_CHECKPOINT
    )
    parser.add_argument(
        "--raw-metadata-root",
        default=(
            "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
            "shellgame_absolute_eef_phase_instruction_dataset"
        ),
    )
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument(
        "--checkpoint-base-dir",
        default="/tmp/pi0_native_dense_probe_checkpoints_260824",
        help="Use /tmp by default because /data2 may not have checkpoint space.",
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--fsdp-devices", type=int, default=6)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--temporal-every", type=int, default=4)
    parser.add_argument("--head-width", type=int, default=256)
    parser.add_argument("--head-heads", type=int, default=8)
    parser.add_argument("--final-loss-weight", type=float, default=1.0)
    parser.add_argument("--relation-loss-weight", type=float, default=1.0)
    parser.add_argument("--stage-loss-weight", type=float, default=1.0)
    parser.add_argument("--remat-policy", default="nothing_saveable")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    # The generic semantic trainer calls the selected recipe module.  Override
    # only these task seams; its optimizer, heldout split, sharding, eval, and
    # checkpoint loops remain unchanged.
    _memory_recipe.compute_objective = compute_objective
    _trainer.create_train_val_data_loaders = (
        lambda config, data_sharding: _classifier_trainer.create_train_val_data_loaders(
            config, data_sharding, None
        )
    )
    _trainer.main(build_config(args))


if __name__ == "__main__":
    main()

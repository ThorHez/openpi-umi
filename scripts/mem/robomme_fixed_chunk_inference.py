"""Reusable inference helpers for the trigger-free RoboMME fixed-chunk MEM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np

from openpi.tasks.robomme import unified_fixed_chunk_student as student_lib
from openpi.tasks.robomme import unified_gt_teacher as teacher_lib
from scripts.mem import cache_robomme_four_task_fixed_chunk_features as feature_cache
from scripts.mem import train_robomme_four_task_fixed_chunk_distillation as train_lib


CHUNK_FRAMES = 12
MAX_CHUNKS = 96
SPATIAL_TOKENS = 16
PATCH_WIDTH = 1152
COLOR_IDS = {"none": 0, "red": 1, "green": 2, "blue": 3}


class FixedChunkMemoryPredictor:
    """Restore one trained student and expose padded encoded-chunk inference."""

    def __init__(self, training_dir: str | Path, checkpoint: str | Path | None = None):
        self.training_dir = Path(training_dir).expanduser().resolve()
        self.config = json.loads(
            (self.training_dir / "training_config.json").read_text(encoding="utf-8")
        )
        training_args = train_lib._training_args(self.config) if hasattr(train_lib, "_training_args") else None
        if training_args is None:
            from types import SimpleNamespace

            values = dict(self.config)
            for key in (
                "sequence_dir",
                "feature_dir",
                "teacher_memory_dir",
                "teacher_sequence_dir",
                "teacher_training_dir",
                "teacher_checkpoint",
                "output_dir",
            ):
                values[key] = Path(values[key])
            training_args = SimpleNamespace(**values)
        self.model = student_lib.UnifiedFixedChunkRecurrentStudent(
            encoder_width=int(self.config["encoder_width"]),
            encoder_depth=int(self.config["encoder_depth"]),
            encoder_heads=int(self.config["encoder_heads"]),
            use_write_gate=bool(self.config.get("write_gate", False)),
            write_gate_bias=float(self.config.get("write_gate_bias", -2.0)),
        )
        dummy = {
            "patch_tokens": jnp.zeros(
                (1, MAX_CHUNKS, CHUNK_FRAMES, SPATIAL_TOKENS, PATCH_WIDTH),
                dtype=jnp.float16,
            ),
            "sequence_mask": jnp.zeros((1, MAX_CHUNKS), dtype=jnp.bool_),
            "task_ids": jnp.zeros((1,), dtype=jnp.int32),
            "goal_color_ids": jnp.zeros((1, 2), dtype=jnp.int32),
            "required_counts": jnp.zeros((1,), dtype=jnp.int32),
            "queried_ordinals": jnp.zeros((1,), dtype=jnp.int32),
            "num_regions": jnp.zeros((1,), dtype=jnp.int32),
        }
        template = self.model.init(
            jax.random.key(int(self.config["seed"])), **dummy, train=False
        )["params"]
        checkpoint_path = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else self.training_dir / "best/params"
        )
        self.checkpoint = checkpoint_path
        self.params = flax.serialization.from_bytes(template, checkpoint_path.read_bytes())
        self.readout, self.readout_params = train_lib._load_teacher_readout(training_args)

        @jax.jit
        def infer(inputs: dict[str, jax.Array]):
            output = self.model.apply({"params": self.params}, **inputs, train=False)
            memories = output["all_memories"]
            flat = memories.reshape(-1, memories.shape[-2], memories.shape[-1])
            logits = self.readout.apply({"params": self.readout_params}, flat).reshape(
                *memories.shape[:2], len(teacher_lib.STATE_FIELDS), teacher_lib.MAX_FIELD_CLASSES
            )
            return memories, logits, output["write_gates"]

        self._infer = infer

    def predict_encoded(
        self,
        chunks: np.ndarray,
        *,
        task_id: int,
        goal_color_ids: tuple[int, int] | list[int],
        required_count: int,
        queried_ordinal: int,
        num_regions: int,
    ) -> dict[str, np.ndarray]:
        chunks = np.asarray(chunks, dtype=np.float16)
        expected_tail = (CHUNK_FRAMES, SPATIAL_TOKENS, PATCH_WIDTH)
        if chunks.ndim != 4 or chunks.shape[1:] != expected_tail:
            raise ValueError(f"Expected chunks [T,{expected_tail}], got {chunks.shape}")
        if len(chunks) > MAX_CHUNKS:
            raise ValueError(f"Chunk count {len(chunks)} exceeds {MAX_CHUNKS}")
        padded = np.zeros((1, MAX_CHUNKS, *expected_tail), dtype=np.float16)
        padded[0, : len(chunks)] = chunks
        mask = np.zeros((1, MAX_CHUNKS), dtype=np.bool_)
        mask[0, : len(chunks)] = True
        inputs = {
            "patch_tokens": jnp.asarray(padded),
            "sequence_mask": jnp.asarray(mask),
            "task_ids": jnp.asarray([task_id], dtype=jnp.int32),
            "goal_color_ids": jnp.asarray([goal_color_ids], dtype=jnp.int32),
            "required_counts": jnp.asarray([required_count], dtype=jnp.int32),
            "queried_ordinals": jnp.asarray([queried_ordinal], dtype=jnp.int32),
            "num_regions": jnp.asarray([num_regions], dtype=jnp.int32),
        }
        memories, logits, gates = jax.device_get(self._infer(inputs))
        valid_states = len(chunks) + 1
        return {
            "all_memories": np.asarray(memories[0, :valid_states], dtype=np.float32),
            "all_logits": np.asarray(logits[0, :valid_states], dtype=np.float32),
            "all_predictions": np.argmax(np.asarray(logits[0, :valid_states]), axis=-1),
            "write_gates": np.asarray(gates[0, : len(chunks)], dtype=np.float32),
        }


def load_backbone(checkpoint: str | Path | None = None):
    path = Path(checkpoint).expanduser().resolve() if checkpoint else feature_cache.DEFAULT_CHECKPOINT
    return feature_cache.load_backbone(path)


def encode_frames(backbone: Any, frames: np.ndarray, *, batch_size: int = 64) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 1:
        raise ValueError(f"Expected non-empty uint8 frames [T,H,W,3], got {frames.shape}")
    rows = []
    for start in range(0, len(frames), batch_size):
        real = frames[start : start + batch_size]
        count = len(real)
        if count < batch_size:
            real = np.concatenate((real, np.repeat(real[-1:], batch_size - count, axis=0)))
        encoded = np.asarray(feature_cache.encode_images(backbone, jnp.asarray(real)))
        rows.append(encoded[:count])
    return np.concatenate(rows).astype(np.float16)


def tokens_to_chunks(tokens: np.ndarray) -> np.ndarray:
    tokens = np.asarray(tokens, dtype=np.float16)
    if tokens.ndim != 3 or tokens.shape[1:] != (SPATIAL_TOKENS, PATCH_WIDTH):
        raise ValueError(f"Expected frame tokens [T,{SPATIAL_TOKENS},{PATCH_WIDTH}], got {tokens.shape}")
    if len(tokens) < 1:
        return np.zeros((0, CHUNK_FRAMES, SPATIAL_TOKENS, PATCH_WIDTH), dtype=np.float16)
    count = (len(tokens) + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    if count > MAX_CHUNKS:
        raise ValueError(f"Frame sequence requires {count} chunks, maximum is {MAX_CHUNKS}")
    padded_length = count * CHUNK_FRAMES
    if padded_length > len(tokens):
        tokens = np.concatenate((tokens, np.repeat(tokens[-1:], padded_length - len(tokens), axis=0)))
    return tokens.reshape(count, CHUNK_FRAMES, SPATIAL_TOKENS, PATCH_WIDTH)


def field_index(name: str) -> int:
    return teacher_lib.STATE_FIELDS.index(name)


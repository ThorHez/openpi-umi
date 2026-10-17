"""Online inference for the causal explicit-event RoboMME region memory."""

from __future__ import annotations

import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from openpi.tasks.robomme.explicit_event_bottleneck_memory import ExplicitEventBottleneckMemory
from scripts.mem.cache_robomme_event_rgb_grid_features import _grid_features_batch
from scripts.mem.train_robomme_explicit_event_bottleneck_ablation import VARIANTS

CHUNK_FRAMES = 12
MAX_CHUNKS = 96
GRID_SIZE = 8
SPATIAL_TOKENS = GRID_SIZE**2
PATCH_WIDTH = 12
MAX_ANCHORS = 4
COLOR_IDS = {"none": 0, "red": 1, "green": 2, "blue": 3}
TABLE_FIELDS = (
    "red_cell",
    "green_cell",
    "blue_cell",
    "ordered_cell_0",
    "ordered_cell_1",
    "ordered_cell_2",
    "ordered_cell_3",
)


def _chunks_from_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 1:
        raise ValueError(f"Expected non-empty frames [T,H,W,3], got {frames.shape}")
    features = _grid_features_batch(frames, GRID_SIZE).astype(np.float16)
    chunk_count = (len(features) + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    if chunk_count > MAX_CHUNKS:
        raise ValueError(f"Frame sequence requires {chunk_count} chunks, max is {MAX_CHUNKS}")
    padded = chunk_count * CHUNK_FRAMES
    if padded > len(features):
        features = np.concatenate((features, np.repeat(features[-1:], padded - len(features), axis=0)))
    return features.reshape(chunk_count, CHUNK_FRAMES, SPATIAL_TOKENS, PATCH_WIDTH)


def _grid_features_batch_device(images: jax.Array) -> jax.Array:
    """Compute the 8x8 RGB statistics directly on the active JAX device."""
    height, width = images.shape[1:3]
    usable_height = (height // GRID_SIZE) * GRID_SIZE
    usable_width = (width // GRID_SIZE) * GRID_SIZE
    cell_height = usable_height // GRID_SIZE
    cell_width = usable_width // GRID_SIZE
    images = images[:, :usable_height, :usable_width].astype(jnp.float32) / 255.0
    cells = images.reshape(images.shape[0], GRID_SIZE, cell_height, GRID_SIZE, cell_width, 3).transpose(
        0, 1, 3, 2, 4, 5
    )
    mean = cells.mean(axis=(3, 4))
    std = cells.std(axis=(3, 4))
    gradient_x = jnp.abs(jnp.diff(cells, axis=4)).mean(axis=(3, 4))
    gradient_y = jnp.abs(jnp.diff(cells, axis=3)).mean(axis=(3, 4))
    return jnp.concatenate((mean, std, gradient_x, gradient_y), axis=-1).reshape(
        images.shape[0], SPATIAL_TOKENS, PATCH_WIDTH
    )


def _chunks_from_frames_device_impl(frames: jax.Array) -> jax.Array:
    features = _grid_features_batch_device(frames).astype(jnp.float16)
    chunk_count = (features.shape[0] + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    padded_frames = chunk_count * CHUNK_FRAMES
    if padded_frames > features.shape[0]:
        features = jnp.concatenate((features, jnp.repeat(features[-1:], padded_frames - features.shape[0], axis=0)))
    return features.reshape(chunk_count, CHUNK_FRAMES, SPATIAL_TOKENS, PATCH_WIDTH)


_chunks_from_frames_device = jax.jit(_chunks_from_frames_device_impl)


def _chunks_from_frames_gpu(frames: np.ndarray) -> jax.Array:
    """Validate host frames and return device-resident grid chunks."""
    shape = np.shape(frames)
    if len(shape) != 4 or shape[-1] != 3 or shape[0] < 1:
        raise ValueError(f"Expected non-empty frames [T,H,W,3], got {shape}")
    chunk_count = (shape[0] + CHUNK_FRAMES - 1) // CHUNK_FRAMES
    if chunk_count > MAX_CHUNKS:
        raise ValueError(f"Frame sequence requires {chunk_count} chunks, max is {MAX_CHUNKS}")
    return _chunks_from_frames_device(jnp.asarray(frames, dtype=jnp.uint8))


def _normalized_anchors(anchors_yx: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    anchors = np.asarray(anchors_yx, dtype=np.float32)
    if anchors.ndim != 2 or anchors.shape[1] != 2 or not 1 <= len(anchors) <= MAX_ANCHORS:
        raise ValueError(f"Expected 1-{MAX_ANCHORS} anchors [A,2], got {anchors.shape}")
    height, width = image_shape
    normalized = np.zeros((MAX_ANCHORS, 2), dtype=np.float32)
    normalized[: len(anchors), 0] = anchors[:, 0] / ((height - 1) / 2.0) - 1.0
    normalized[: len(anchors), 1] = anchors[:, 1] / ((width - 1) / 2.0) - 1.0
    mask = np.zeros(MAX_ANCHORS, dtype=np.bool_)
    mask[: len(anchors)] = True
    return normalized, mask


class ExplicitEventMemoryPredictor:
    """Restore a trained causal MEM and infer its episode-local region table."""

    def __init__(self, training_dir: str | Path, checkpoint: str | Path | None = None):
        self.training_dir = Path(training_dir).expanduser().resolve()
        self.config = json.loads((self.training_dir / "training_config.json").read_text(encoding="utf-8"))
        variant = str(self.config["variant"])
        self.use_proprio = variant == "pooled_hard_causal_proprio"
        if self.use_proprio:
            encoder, deterministic, causal = ("pooled", True, True)
        elif variant not in VARIANTS:
            raise ValueError(f"Unsupported explicit-event variant: {variant}")
        else:
            encoder, deterministic, causal = VARIANTS[variant]
        self.model = ExplicitEventBottleneckMemory(
            max_steps=MAX_CHUNKS,
            spatial_tokens=SPATIAL_TOKENS,
            input_width=PATCH_WIDTH,
            temporal_encoder=encoder,
            temporal_depth=int(self.config.get("temporal_depth", 2)),
            temporal_heads=int(self.config.get("temporal_heads", 4)),
            deterministic_updater=deterministic,
            causal_evidence_state=causal,
            gate_temperature=float(self.config.get("gate_temperature", 0.25)),
        )
        dummy = self._inputs(
            np.zeros((0, CHUNK_FRAMES, SPATIAL_TOKENS, PATCH_WIDTH), np.float16),
            task_id=0,
            goal_color_ids=(0, 0),
            queried_ordinal=0,
            num_regions=0,
            anchor_yx=np.zeros((MAX_ANCHORS, 2), np.float32),
            anchor_mask=np.zeros(MAX_ANCHORS, np.bool_),
            proprio_chunks=(np.zeros((0, CHUNK_FRAMES, 8), np.float32) if self.use_proprio else None),
        )
        template = self.model.init(jax.random.key(int(self.config["seed"])), **dummy)["params"]
        checkpoint_path = (
            Path(checkpoint).expanduser().resolve() if checkpoint is not None else self.training_dir / "params.msgpack"
        )
        self.checkpoint = checkpoint_path
        self.params = flax.serialization.from_bytes(template, checkpoint_path.read_bytes())

        @jax.jit
        def infer(inputs):
            return self.model.apply({"params": self.params}, **inputs)

        self._infer = infer

        @jax.jit
        def infer_frames_device(
            frames,
            proprio_states,
            task_ids,
            goal_color_ids,
            queried_ordinals,
            num_regions,
            anchor_yx,
            anchor_mask,
        ):
            chunks = _chunks_from_frames_device_impl(frames)
            chunk_count = chunks.shape[0]
            patches = (
                jnp.zeros(
                    (1, MAX_CHUNKS, CHUNK_FRAMES, SPATIAL_TOKENS, PATCH_WIDTH),
                    dtype=jnp.float16,
                )
                .at[0, :chunk_count]
                .set(chunks)
            )
            sequence_mask = jnp.zeros((1, MAX_CHUNKS), dtype=jnp.bool_)
            sequence_mask = sequence_mask.at[0, :chunk_count].set(True)
            inputs = {
                "patch_tokens": patches,
                "sequence_mask": sequence_mask,
                "task_ids": task_ids,
                "goal_color_ids": goal_color_ids,
                "queried_ordinals": queried_ordinals,
                "num_regions": num_regions,
                "anchor_yx": anchor_yx,
                "anchor_mask": anchor_mask,
                "teacher_previous_tables": jnp.zeros((1, MAX_CHUNKS, 7), jnp.int32),
                "teacher_force_mask": jnp.zeros((1, MAX_CHUNKS), jnp.bool_),
            }
            if self.use_proprio:
                padded_frames = chunk_count * CHUNK_FRAMES
                if padded_frames > proprio_states.shape[0]:
                    proprio_states = jnp.concatenate(
                        (
                            proprio_states,
                            jnp.repeat(
                                proprio_states[-1:],
                                padded_frames - proprio_states.shape[0],
                                axis=0,
                            ),
                        )
                    )
                proprio_chunks = proprio_states.reshape(chunk_count, CHUNK_FRAMES, 8)
                proprio = (
                    jnp.zeros((1, MAX_CHUNKS, CHUNK_FRAMES, 8), dtype=jnp.float32)
                    .at[0, :chunk_count]
                    .set(proprio_chunks)
                )
                inputs["proprio_tokens"] = proprio
            return self.model.apply({"params": self.params}, **inputs)

        self._infer_frames_device = infer_frames_device

    @staticmethod
    def _inputs(
        chunks: np.ndarray,
        *,
        task_id: int,
        goal_color_ids: tuple[int, int] | list[int],
        queried_ordinal: int,
        num_regions: int,
        anchor_yx: np.ndarray,
        anchor_mask: np.ndarray,
        proprio_chunks: np.ndarray | None = None,
    ) -> dict[str, jax.Array]:
        chunks = np.asarray(chunks, dtype=np.float16)
        expected = (CHUNK_FRAMES, SPATIAL_TOKENS, PATCH_WIDTH)
        if chunks.ndim != 4 or chunks.shape[1:] != expected:
            raise ValueError(f"Expected chunks [T,{expected}], got {chunks.shape}")
        if len(chunks) > MAX_CHUNKS:
            raise ValueError(f"Chunk count {len(chunks)} exceeds {MAX_CHUNKS}")
        patches = np.zeros((1, MAX_CHUNKS, *expected), dtype=np.float16)
        patches[0, : len(chunks)] = chunks
        mask = np.zeros((1, MAX_CHUNKS), dtype=np.bool_)
        mask[0, : len(chunks)] = True
        result = {
            "patch_tokens": jnp.asarray(patches),
            "sequence_mask": jnp.asarray(mask),
            "task_ids": jnp.asarray([task_id], dtype=jnp.int32),
            "goal_color_ids": jnp.asarray([goal_color_ids], dtype=jnp.int32),
            "queried_ordinals": jnp.asarray([queried_ordinal], dtype=jnp.int32),
            "num_regions": jnp.asarray([num_regions], dtype=jnp.int32),
            "anchor_yx": jnp.asarray(anchor_yx[None]),
            "anchor_mask": jnp.asarray(anchor_mask[None]),
            "teacher_previous_tables": jnp.zeros((1, MAX_CHUNKS, 7), jnp.int32),
            "teacher_force_mask": jnp.zeros((1, MAX_CHUNKS), jnp.bool_),
        }
        if proprio_chunks is not None:
            proprio_chunks = np.asarray(proprio_chunks, dtype=np.float32)
            if proprio_chunks.shape != (len(chunks), CHUNK_FRAMES, 8):
                raise ValueError(
                    f"Expected proprio chunks {(len(chunks), CHUNK_FRAMES, 8)}, got {proprio_chunks.shape}"
                )
            proprio = np.zeros((1, MAX_CHUNKS, CHUNK_FRAMES, 8), dtype=np.float32)
            proprio[0, : len(chunks)] = proprio_chunks
            result["proprio_tokens"] = jnp.asarray(proprio)
        return result

    def predict_frames(
        self,
        frames: np.ndarray,
        anchors_yx: np.ndarray,
        *,
        task_id: int,
        goal_color_ids: tuple[int, int] | list[int],
        queried_ordinal: int,
        proprio_states: np.ndarray | None = None,
        preprocess_backend: str = "gpu",
    ) -> dict[str, np.ndarray]:
        frames = np.asarray(frames, dtype=np.uint8)
        if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) < 1:
            raise ValueError(f"Expected non-empty frames [T,H,W,3], got {frames.shape}")
        chunk_count = (len(frames) + CHUNK_FRAMES - 1) // CHUNK_FRAMES
        if chunk_count > MAX_CHUNKS:
            raise ValueError(f"Frame sequence requires {chunk_count} chunks, max is {MAX_CHUNKS}")
        if preprocess_backend not in {"gpu", "cpu"}:
            raise ValueError(f"preprocess_backend must be 'gpu' or 'cpu', got {preprocess_backend!r}")
        anchor_yx, anchor_mask = _normalized_anchors(anchors_yx, tuple(int(value) for value in frames.shape[1:3]))
        states = None
        if self.use_proprio:
            if proprio_states is None:
                raise ValueError("This checkpoint requires joint+gripper proprio_states")
            states = np.asarray(proprio_states, dtype=np.float32)
            if states.ndim != 2 or states.shape[1] != 8 or len(states) != len(frames):
                raise ValueError(f"Expected proprio states [T,8] aligned with frames, got {states.shape}")
        if preprocess_backend == "gpu":
            if states is None:
                states = np.zeros((len(frames), 8), dtype=np.float32)
            output = jax.device_get(
                self._infer_frames_device(
                    jnp.asarray(frames, dtype=jnp.uint8),
                    jnp.asarray(states, dtype=jnp.float32),
                    jnp.asarray([task_id], dtype=jnp.int32),
                    jnp.asarray([goal_color_ids], dtype=jnp.int32),
                    jnp.asarray([queried_ordinal], dtype=jnp.int32),
                    jnp.asarray([int(anchor_mask.sum())], dtype=jnp.int32),
                    jnp.asarray(anchor_yx[None], dtype=jnp.float32),
                    jnp.asarray(anchor_mask[None], dtype=jnp.bool_),
                )
            )
        else:
            chunks = _chunks_from_frames(frames)
            proprio_chunks = None
            if states is not None:
                padded = chunk_count * CHUNK_FRAMES
                if padded > len(states):
                    states = np.concatenate((states, np.repeat(states[-1:], padded - len(states), axis=0)))
                proprio_chunks = states.reshape(chunk_count, CHUNK_FRAMES, 8)
            inputs = self._inputs(
                chunks,
                task_id=task_id,
                goal_color_ids=goal_color_ids,
                queried_ordinal=queried_ordinal,
                num_regions=int(anchor_mask.sum()),
                anchor_yx=anchor_yx,
                anchor_mask=anchor_mask,
                proprio_chunks=proprio_chunks,
            )
            output = jax.device_get(self._infer(inputs))
        valid_states = chunk_count + 1
        tables = np.asarray(output["all_tables"][0, :valid_states], dtype=np.float32)
        return {
            "all_tables": tables,
            "all_predictions": np.argmax(tables, axis=-1),
            "all_memories": np.asarray(output["all_memories"][0, :valid_states], dtype=np.float32),
            "event_type": np.argmax(np.asarray(output["event_type_logits"][0, :chunk_count]), axis=-1),
            "event_bottleneck": np.asarray(output["event_bottleneck"][0, :chunk_count], dtype=np.float32),
            "chunks": np.asarray(chunk_count, dtype=np.int32),
        }


def field_index(name: str) -> int:
    return TABLE_FIELDS.index(name)

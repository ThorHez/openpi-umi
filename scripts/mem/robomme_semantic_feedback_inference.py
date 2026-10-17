"""Online inference for the unified PickXTimes semantic-feedback MEM."""

from __future__ import annotations

import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from openpi.tasks.robomme import unified_gt_teacher as teacher_lib
from openpi.tasks.robomme import unified_semantic_feedback_student as student_lib
from scripts.mem import robomme_fixed_chunk_inference as fixed_lib


PICK_ACTIVE_FIELDS = (
    "task",
    "target_color_0",
    "target_color_1",
    "required_count",
    "completed_count",
    "holding",
    "ready_to_press",
    "done",
)


class SemanticFeedbackPredictor:
    """Restore a semantic-feedback checkpoint and run causal padded rollouts."""

    def __init__(self, training_dir: str | Path, checkpoint: str | Path | None = None):
        self.training_dir = Path(training_dir).expanduser().resolve()
        self.config = json.loads(
            (self.training_dir / "training_config.json").read_text(encoding="utf-8")
        )
        self.max_steps = fixed_lib.MAX_CHUNKS
        self.proprio_dir = Path(self.config["proprio_dir"])
        proprio_summary = json.loads(
            (self.proprio_dir / "summary.json").read_text(encoding="utf-8")
        )
        self.proprio_mean = np.asarray(
            proprio_summary["normalization"]["mean"], dtype=np.float32
        )
        self.proprio_std = np.asarray(
            proprio_summary["normalization"]["std"], dtype=np.float32
        )
        self.proprio_fields = tuple(proprio_summary["fields"])
        self.model = student_lib.UnifiedSemanticFeedbackStudent(
            max_steps=self.max_steps,
            proprio_dim=len(self.proprio_mean),
            width=int(self.config["width"]),
            encoder_width=int(self.config["encoder_width"]),
            encoder_depth=int(self.config["encoder_depth"]),
            encoder_heads=int(self.config["encoder_heads"]),
            straight_through_hard_feedback=bool(
                self.config.get("straight_through_hard_feedback", False)
            ),
        )
        fields = len(teacher_lib.STATE_FIELDS)
        dummy = {
            "patch_tokens": jnp.zeros(
                (
                    1,
                    self.max_steps,
                    fixed_lib.CHUNK_FRAMES,
                    fixed_lib.SPATIAL_TOKENS,
                    fixed_lib.PATCH_WIDTH,
                ),
                dtype=jnp.float16,
            ),
            "proprio": jnp.zeros(
                (1, self.max_steps, fixed_lib.CHUNK_FRAMES, len(self.proprio_mean)),
                dtype=jnp.float32,
            ),
            "sequence_mask": jnp.zeros((1, self.max_steps), dtype=jnp.bool_),
            "initial_state_targets": jnp.zeros((1, fields), dtype=jnp.int32),
            "state_field_mask": jnp.zeros(
                (1, self.max_steps, fields), dtype=jnp.bool_
            ),
            "teacher_previous_targets": jnp.zeros(
                (1, self.max_steps, fields), dtype=jnp.int32
            ),
            "teacher_force_mask": jnp.zeros((1, self.max_steps), dtype=jnp.bool_),
        }
        template = self.model.init(
            jax.random.key(int(self.config["seed"])), **dummy, train=False
        )["params"]
        self.checkpoint = (
            Path(checkpoint).expanduser().resolve()
            if checkpoint is not None
            else self.training_dir / "best/params"
        )
        self.params = flax.serialization.from_bytes(template, self.checkpoint.read_bytes())

        @jax.jit
        def infer(inputs):
            return self.model.apply(
                {"params": self.params}, **inputs, train=False
            )["state_logits"]

        self._infer = infer

    def predict_encoded(
        self,
        chunks: np.ndarray,
        proprio: np.ndarray,
        *,
        target_color_id: int,
        required_count: int,
    ) -> dict[str, np.ndarray]:
        chunks = np.asarray(chunks, dtype=np.float16)
        proprio = np.asarray(proprio, dtype=np.float32)
        chunk_tail = (
            fixed_lib.CHUNK_FRAMES,
            fixed_lib.SPATIAL_TOKENS,
            fixed_lib.PATCH_WIDTH,
        )
        proprio_tail = (fixed_lib.CHUNK_FRAMES, len(self.proprio_mean))
        if chunks.ndim != 4 or chunks.shape[1:] != chunk_tail:
            raise ValueError(f"Expected chunks [T,{chunk_tail}], got {chunks.shape}")
        if proprio.shape != (len(chunks), *proprio_tail):
            raise ValueError(
                f"Expected proprio [{len(chunks)},{proprio_tail}], got {proprio.shape}"
            )
        if len(chunks) > self.max_steps:
            raise ValueError(f"Chunk count {len(chunks)} exceeds {self.max_steps}")

        fields = len(teacher_lib.STATE_FIELDS)
        patch_pad = np.zeros((1, self.max_steps, *chunk_tail), dtype=np.float16)
        proprio_pad = np.zeros(
            (1, self.max_steps, *proprio_tail), dtype=np.float32
        )
        sequence_mask = np.zeros((1, self.max_steps), dtype=np.bool_)
        patch_pad[0, : len(chunks)] = chunks
        proprio_pad[0, : len(chunks)] = (
            proprio - self.proprio_mean[None, None]
        ) / self.proprio_std[None, None]
        sequence_mask[0, : len(chunks)] = True

        initial = np.zeros((1, fields), dtype=np.int32)
        initial[0, teacher_lib.STATE_FIELDS.index("task")] = teacher_lib.TASKS.index(
            "pickxtimes_local_event"
        )
        initial[0, teacher_lib.STATE_FIELDS.index("target_color_0")] = target_color_id
        initial[0, teacher_lib.STATE_FIELDS.index("required_count")] = required_count
        field_mask = np.zeros((1, self.max_steps, fields), dtype=np.bool_)
        field_mask[..., [teacher_lib.STATE_FIELDS.index(name) for name in PICK_ACTIVE_FIELDS]] = True
        inputs = {
            "patch_tokens": jnp.asarray(patch_pad),
            "proprio": jnp.asarray(proprio_pad),
            "sequence_mask": jnp.asarray(sequence_mask),
            "initial_state_targets": jnp.asarray(initial),
            "state_field_mask": jnp.asarray(field_mask),
            "teacher_previous_targets": jnp.zeros(
                (1, self.max_steps, fields), dtype=jnp.int32
            ),
            "teacher_force_mask": jnp.zeros((1, self.max_steps), dtype=jnp.bool_),
        }
        logits = np.asarray(jax.device_get(self._infer(inputs))[0, : len(chunks) + 1])
        return {
            "all_logits": logits.astype(np.float32),
            "all_probabilities": jax.device_get(jax.nn.softmax(logits, axis=-1)),
            "all_predictions": np.argmax(logits, axis=-1).astype(np.int32),
        }


def pick_semantic_key(prediction: np.ndarray) -> tuple[int, ...]:
    """Convert a unified 19-field prediction to the MME Pick codebook key."""

    prediction = np.asarray(prediction)
    return tuple(
        int(prediction[teacher_lib.STATE_FIELDS.index(name)])
        for name in (
            "target_color_0",
            "required_count",
            "completed_count",
            "holding",
            "ready_to_press",
            "done",
        )
    )


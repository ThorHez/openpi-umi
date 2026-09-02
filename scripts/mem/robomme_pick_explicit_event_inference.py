"""Online inference for PickXTimes explicit events and deterministic count."""

from __future__ import annotations

import json
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np

from openpi.tasks.robomme.pickxtimes.explicit_event_count_memory import (
    PickExplicitEventHead,
    PickObjectSuccessHead,
    PickTargetMotionEventHead,
    deterministic_update_numpy,
)


class PickExplicitEventPredictor:
    def __init__(self, training_dir: str | Path):
        self.training_dir = Path(training_dir).expanduser().resolve()
        self.config = json.loads((self.training_dir / "training_config.json").read_text())
        summary = json.loads(
            (Path(self.config["proprio_dir"]) / "summary.json").read_text()
        )
        self.proprio_mean = np.asarray(summary["normalization"]["mean"], np.float32)
        self.proprio_std = np.asarray(summary["normalization"]["std"], np.float32)
        self.visual_mode = self.config.get("visual_mode", "front")
        self.model = (
            PickExplicitEventHead()
            if self.visual_mode == "front"
            else PickTargetMotionEventHead(mode=self.visual_mode)
        )
        self.motion_mean = None
        self.motion_std = None
        dummy = {
            "previous_state": jnp.zeros((1, 4), jnp.int32),
            "required_count": jnp.ones((1,), jnp.int32),
            "rgb": jnp.zeros((1, 12, 16, 1152), jnp.float16),
            "proprio": jnp.zeros((1, 12, 6), jnp.float32),
        }
        if self.visual_mode != "front":
            motion_summary = json.loads(
                (Path(self.config["target_motion_dir"]) / "summary.json").read_text()
            )
            self.motion_mean = np.asarray(motion_summary["normalization"]["mean"], np.float32)
            self.motion_std = np.asarray(motion_summary["normalization"]["std"], np.float32)
            dummy.update(
                {
                    "target_color_id": jnp.ones((1,), jnp.int32),
                    "wrist": jnp.zeros((1, 12, 16, 1152), jnp.float16),
                    "target_motion": jnp.zeros((1, 12, 11), jnp.float32),
                }
            )
        template = self.model.init(
            jax.random.key(int(self.config["seed"])), **dummy, train=False
        )["params"]
        self.checkpoint = self.training_dir / "params.msgpack"
        self.params = flax.serialization.from_bytes(template, self.checkpoint.read_bytes())

        @jax.jit
        def infer(batch):
            return self.model.apply({"params": self.params}, **batch, train=False)

        self._infer = infer

    def predict_chunk(
        self,
        rgb: np.ndarray,
        proprio: np.ndarray,
        previous_state: np.ndarray,
        required_count: int,
        *,
        wrist: np.ndarray | None = None,
        target_motion: np.ndarray | None = None,
        target_color_id: int | None = None,
    ) -> dict[str, np.ndarray | int]:
        rgb = np.asarray(rgb, dtype=np.float16)
        proprio = np.asarray(proprio, dtype=np.float32)
        if rgb.shape != (12, 16, 1152):
            raise ValueError(f"Expected RGB tokens [12,16,1152], got {rgb.shape}")
        if proprio.shape != (12, 6):
            raise ValueError(f"Expected proprio [12,6], got {proprio.shape}")
        normalized = (proprio - self.proprio_mean) / self.proprio_std
        batch = {
            "previous_state": jnp.asarray(previous_state[None], jnp.int32),
            "required_count": jnp.asarray([required_count], jnp.int32),
            "rgb": jnp.asarray(rgb[None]),
            "proprio": jnp.asarray(normalized[None]),
        }
        if self.visual_mode != "front":
            wrist = np.asarray(wrist, dtype=np.float16)
            target_motion = np.asarray(target_motion, dtype=np.float32)
            if wrist.shape != (12, 16, 1152):
                raise ValueError(f"Expected wrist tokens [12,16,1152], got {wrist.shape}")
            if target_motion.shape != (12, 11):
                raise ValueError(f"Expected target motion [12,11], got {target_motion.shape}")
            if target_color_id is None:
                raise ValueError("target_color_id is required for verifier modes")
            batch.update(
                {
                    "target_color_id": jnp.asarray([target_color_id], jnp.int32),
                    "wrist": jnp.asarray(wrist[None]),
                    "target_motion": jnp.asarray(
                        ((target_motion - self.motion_mean) / self.motion_std)[None]
                    ),
                }
            )
        logits = np.asarray(jax.device_get(self._infer(batch))[0], np.float32)
        event_id = int(np.argmax(logits))
        next_state = deterministic_update_numpy(previous_state, event_id, required_count)
        return {"event_logits": logits, "event_id": event_id, "state": next_state}


class PickObjectSuccessPredictor:
    """Serve the non-privileged student of simulator object-success labels."""

    def __init__(self, training_dir: str | Path):
        self.training_dir = Path(training_dir).expanduser().resolve()
        self.config = json.loads((self.training_dir / "training_config.json").read_text())
        self.proprio_mean = np.asarray(self.config["proprio_mean"], dtype=np.float32)
        self.proprio_std = np.asarray(self.config["proprio_std"], dtype=np.float32)
        self.threshold = float(self.config.get("threshold", 0.5))
        self.model = PickObjectSuccessHead()
        dummy = {
            "target_color_id": jnp.ones((1,), jnp.int32),
            "rgb": jnp.zeros((1, 12, 16, 1152), jnp.float16),
            "wrist": jnp.zeros((1, 12, 16, 1152), jnp.float16),
            "proprio": jnp.zeros((1, 12, 6), jnp.float32),
        }
        template = self.model.init(
            jax.random.key(int(self.config["seed"])), **dummy, train=False
        )["params"]
        self.checkpoint = self.training_dir / "params.msgpack"
        self.params = flax.serialization.from_bytes(template, self.checkpoint.read_bytes())

        @jax.jit
        def infer(batch):
            return self.model.apply({"params": self.params}, **batch, train=False)

        self._infer = infer

    def predict_chunk(
        self,
        rgb: np.ndarray,
        wrist: np.ndarray,
        proprio: np.ndarray,
        target_color_id: int,
    ) -> dict[str, np.ndarray]:
        normalized = (
            np.asarray(proprio, dtype=np.float32) - self.proprio_mean
        ) / self.proprio_std
        batch = {
            "target_color_id": jnp.asarray([target_color_id], jnp.int32),
            "rgb": jnp.asarray(np.asarray(rgb, dtype=np.float16)[None]),
            "wrist": jnp.asarray(np.asarray(wrist, dtype=np.float16)[None]),
            "proprio": jnp.asarray(normalized[None]),
        }
        logits = np.asarray(jax.device_get(self._infer(batch))[0], dtype=np.float32)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return {"logits": logits, "probabilities": probabilities}


def _target_image_features(image: np.ndarray, color_id: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    red, green, blue = (image[..., index] for index in range(3))
    masks = (
        (red > 180) & (green < 70) & (blue < 70),
        (green > 180) & (red < 70) & (blue < 70),
        (blue > 180) & (red < 70) & (green < 70),
    )
    mask = masks[int(color_id) - 1]
    y, x = np.nonzero(mask)
    if len(y) < 6:
        return np.zeros(4, dtype=np.float32)
    height, width = mask.shape
    return np.asarray(
        (
            2.0 * float(np.median(y)) / max(height - 1, 1) - 1.0,
            2.0 * float(np.median(x)) / max(width - 1, 1) - 1.0,
            np.sqrt(float(len(y)) / float(height * width)),
            1.0,
        ),
        dtype=np.float32,
    )


def target_motion_frame(
    front: np.ndarray, wrist: np.ndarray, eef_xyz: np.ndarray, target_color_id: int
) -> np.ndarray:
    return np.concatenate(
        (
            _target_image_features(front, target_color_id),
            _target_image_features(wrist, target_color_id),
            np.asarray(eef_xyz, dtype=np.float32).reshape(3),
        )
    )

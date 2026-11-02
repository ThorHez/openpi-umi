"""Inference adapters for the single-arm 16-step elec_meta training contract."""

import dataclasses
import re

import numpy as np


@dataclasses.dataclass(frozen=True)
class ElecMetaInputs:
    """Accept the prepared observations used by ``get_real_umi_obs_dict``.

    Match UmiInputsV4's per-frame 16-value state and newest wrist image while
    synthesizing the training-only action field for ordinary inference. An
    explicitly supplied action field is preserved for RTC normalization.
    """

    action_horizon: int = 16

    def __post_init__(self):
        if type(self.action_horizon) is not int or self.action_horizon <= 0:
            raise ValueError("elec_meta action_horizon must be a positive integer")

    def __call__(self, data: dict) -> dict:
        for key in data:
            match = re.match(r"robot(\d+)_", key)
            if match and int(match[1]) != 0:
                raise ValueError("elec_meta RTC supports exactly one arm (robot0)")
        result = dict(data)
        states = []
        for name, width in (
            ("robot0_eef_pos", 3),
            ("robot0_eef_rot_axis_angle", 6),
            ("robot0_eef_rot_axis_angle_wrt_start", 6),
            ("robot0_gripper_width", 1),
        ):
            value = np.asarray(data.get(name), dtype=np.float32)
            if value.shape != (2, width) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape (2, {width})")
            result[name] = value.copy()
            states.append(result[name])

        if "left_wrist_0_rgb_1" in data:
            image = np.asarray(data["left_wrist_0_rgb_1"])
        else:
            key = "left_wrist_0_rgb" if "left_wrist_0_rgb" in data else "camera0_rgb"
            images = np.asarray(data.get(key))
            if images.ndim != 4 or images.shape[0] < 1 or images.shape[1:] != (3, 224, 224):
                raise ValueError("elec_meta requires wrist image history with shape (T, 3, 224, 224)")
            image = images[-1]
        if image.shape != (3, 224, 224):
            raise ValueError("elec_meta requires a newest wrist image with shape (3, 224, 224)")
        if image.dtype == np.uint8:
            image = image.copy()
        elif image.dtype.kind == "f" and np.all(np.isfinite(image)) and np.all((image >= 0) & (image <= 1)):
            image = (255 * image).astype(np.uint8)
        else:
            raise ValueError("Wrist images must be uint8 or finite floating values in [0, 1]")
        result["left_wrist_0_rgb_1"] = image
        actions = np.asarray(data.get("actions", np.zeros((self.action_horizon, 10))), dtype=np.float32)
        if actions.shape != (self.action_horizon, 10) or not np.all(np.isfinite(actions)):
            raise ValueError(f"elec_meta actions must be finite with shape ({self.action_horizon}, 10)")
        result["actions"] = actions.copy()
        result["state"] = np.concatenate(states, axis=-1)
        result["image"] = {
            "base_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.moveaxis(image, 0, -1),
            "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
        }
        result["image_mask"] = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        }
        return result


@dataclasses.dataclass(frozen=True)
class ElecMetaOutputs:
    """Select ten physical action dimensions before output unnormalization.

    The flattened model state is deliberately dropped: it has width 32 while
    the checkpoint's state statistics and normalization mask have width 16.
    """

    action_dim: int = 10

    def __post_init__(self):
        if self.action_dim != 10:
            raise ValueError("elec_meta uses exactly ten action dimensions")

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data.get("actions"))
        if (
            actions.ndim != 2
            or len(actions) == 0
            or actions.shape[1] < self.action_dim
            or not np.all(np.isfinite(actions))
        ):
            raise ValueError("elec_meta model output must be a finite (H, D >= 10) action array")
        return {"actions": actions[:, : self.action_dim].copy()}

"""Lightweight frozen-memory target to EEF7 action adapter for VideoUnmask."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

ACTION_FEATURE_DIM = 18
EEF_POSE_DIM = 6
TARGET_CROP_SIZE = 32


class VideoUnmaskEEFActionAdapter(nn.Module):
    """Predict a normalized EEF pose target and binary gripper command.

    Inputs are normalized outside the module and contain the frozen memory's
    target point, current EEF/gripper/joint state, and rollout progress.  The
    target may represent an absolute pose or a wrapped adjacent-frame delta;
    that contract is handled by the dataset/evaluator.  The adapter deliberately
    has no path back into the memory model.
    """

    hidden_width: int = 256
    depth: int = 3
    feature_dim: int = ACTION_FEATURE_DIM

    @nn.compact
    def __call__(self, features, target_crop, *, train: bool = False):
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected [B,{self.feature_dim}] features, got {features.shape}")
        expected_crop = (TARGET_CROP_SIZE, TARGET_CROP_SIZE, 3)
        if target_crop.ndim != 4 or target_crop.shape[1:] != expected_crop:
            raise ValueError(f"Expected target crop [B,{expected_crop}], got {target_crop.shape}")
        visual = target_crop.astype(jnp.float32) / 127.5 - 1.0
        visual = nn.gelu(nn.Conv(32, (5, 5), strides=(2, 2), name="crop_conv_0", dtype=jnp.float32)(visual))
        visual = nn.gelu(nn.Conv(64, (3, 3), strides=(2, 2), name="crop_conv_1", dtype=jnp.float32)(visual))
        visual = nn.gelu(nn.Conv(128, (3, 3), strides=(2, 2), name="crop_conv_2", dtype=jnp.float32)(visual))
        visual = jnp.mean(visual, axis=(1, 2))
        state = nn.gelu(nn.Dense(128, name="state_projection", dtype=jnp.float32)(features))
        x = nn.Dense(self.hidden_width, name="input_projection", dtype=jnp.float32)(state)
        x = nn.gelu(x)
        for index in range(self.depth):
            residual = x
            x = nn.LayerNorm(name=f"block_{index}_ln", dtype=jnp.float32)(x)
            x = nn.Dense(self.hidden_width * 2, name=f"block_{index}_expand", dtype=jnp.float32)(x)
            x = nn.gelu(x)
            x = nn.Dropout(0.05, name=f"block_{index}_dropout")(x, deterministic=not train)
            x = nn.Dense(self.hidden_width, name=f"block_{index}_project", dtype=jnp.float32)(x)
            x = x + residual
        x = nn.LayerNorm(name="output_ln", dtype=jnp.float32)(x)
        orientation = jnp.concatenate((x, visual), axis=-1)
        orientation = nn.gelu(
            nn.Dense(self.hidden_width, name="orientation_projection", dtype=jnp.float32)(orientation)
        )
        return {
            "normalized_pose": jnp.concatenate(
                (
                    nn.Dense(3, name="position_head", dtype=jnp.float32)(x),
                    nn.Dense(3, name="orientation_head", dtype=jnp.float32)(orientation),
                ),
                axis=-1,
            ),
            "close_logit": nn.Dense(1, name="gripper_head", dtype=jnp.float32)(x)[..., 0],
        }

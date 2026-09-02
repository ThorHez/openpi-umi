"""Loss recipe for VideoUnmask demo-to-target semantic memory."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import optax


@dataclasses.dataclass(frozen=True)
class VideoUnmaskLossWeights:
    final_point: float = 2.0
    stage_point: float = 1.0
    target_color: float = 0.25


def _huber(predictions, targets, delta: float = 0.05):
    error = jnp.abs(predictions - targets)
    return jnp.where(error <= delta, 0.5 * error**2 / delta, error - 0.5 * delta)


def compute_losses(outputs, targets, weights: VideoUnmaskLossWeights | None = None):
    if weights is None:
        weights = VideoUnmaskLossWeights()
    target_point = targets["target_point"].astype(jnp.float32)
    final_point_loss = jnp.mean(_huber(outputs["target_point"], target_point))
    stage_targets = jnp.broadcast_to(target_point[:, None], outputs["stage_target_points"].shape)
    stage_point_loss = jnp.mean(_huber(outputs["stage_target_points"], stage_targets))
    color_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            outputs["target_color_logits"].astype(jnp.float32), targets["target_color"]
        )
    )
    loss = (
        weights.final_point * final_point_loss
        + weights.stage_point * stage_point_loss
        + weights.target_color * color_loss
    )
    pixel_error = jnp.linalg.norm((outputs["target_point"] - target_point) * 255.0, axis=-1)
    return loss, {
        "loss": loss,
        "final_point_loss": final_point_loss,
        "stage_point_loss": stage_point_loss,
        "target_color_loss": color_loss,
        "target_color_accuracy": jnp.mean(
            jnp.argmax(outputs["target_color_logits"], axis=-1) == targets["target_color"]
        ),
        "point_mae_y_px": jnp.mean(jnp.abs(outputs["target_point"][:, 0] - target_point[:, 0])) * 255.0,
        "point_mae_x_px": jnp.mean(jnp.abs(outputs["target_point"][:, 1] - target_point[:, 1])) * 255.0,
        "point_distance_px": jnp.mean(pixel_error),
        "within_10px": jnp.mean(pixel_error <= 10.0),
        "within_20px": jnp.mean(pixel_error <= 20.0),
        "within_30px": jnp.mean(pixel_error <= 30.0),
    }

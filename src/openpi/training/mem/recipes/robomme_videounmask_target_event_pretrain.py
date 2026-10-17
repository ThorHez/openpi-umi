"""Loss recipe for VideoUnmask target-localization semantic-event memory."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import optax


@dataclasses.dataclass(frozen=True)
class LossWeights:
    locator_cell: float = 1.0
    memory_cell: float = 1.0
    locator_point: float = 1.0
    memory_point: float = 2.0
    target_color: float = 0.25


def _huber(predictions, targets, delta: float = 0.05):
    error = jnp.abs(predictions - targets)
    return jnp.where(error <= delta, 0.5 * error**2 / delta, error - 0.5 * delta)


def compute_losses(outputs, targets, weights: LossWeights | None = None):
    if weights is None:
        weights = LossWeights()
    point = targets["target_point"].astype(jnp.float32)
    cell = targets["target_cell"]
    locator_cell_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs["locator_cell_logits"], cell)
    )
    memory_cell_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs["memory_cell_logits"], cell)
    )
    locator_point_loss = jnp.mean(_huber(outputs["locator_point"], point))
    memory_point_loss = jnp.mean(_huber(outputs["target_point"], point))
    color_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(outputs["target_color_logits"], targets["target_color"])
    )
    loss = (
        weights.locator_cell * locator_cell_loss
        + weights.memory_cell * memory_cell_loss
        + weights.locator_point * locator_point_loss
        + weights.memory_point * memory_point_loss
        + weights.target_color * color_loss
    )
    pixel_error = jnp.linalg.norm((outputs["target_point"] - point) * 255.0, axis=-1)
    locator_pixel_error = jnp.linalg.norm((outputs["locator_point"] - point) * 255.0, axis=-1)
    return loss, {
        "loss": loss,
        "locator_cell_loss": locator_cell_loss,
        "memory_cell_loss": memory_cell_loss,
        "locator_point_loss": locator_point_loss,
        "memory_point_loss": memory_point_loss,
        "target_color_loss": color_loss,
        "locator_cell_accuracy": jnp.mean(jnp.argmax(outputs["locator_cell_logits"], axis=-1) == cell),
        "memory_cell_accuracy": jnp.mean(jnp.argmax(outputs["memory_cell_logits"], axis=-1) == cell),
        "target_color_accuracy": jnp.mean(
            jnp.argmax(outputs["target_color_logits"], axis=-1) == targets["target_color"]
        ),
        "locator_distance_px": jnp.mean(locator_pixel_error),
        "point_distance_px": jnp.mean(pixel_error),
        "within_10px": jnp.mean(pixel_error <= 10.0),
        "within_20px": jnp.mean(pixel_error <= 20.0),
        "within_30px": jnp.mean(pixel_error <= 30.0),
    }

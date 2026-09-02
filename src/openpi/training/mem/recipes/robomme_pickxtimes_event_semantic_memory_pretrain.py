"""Loss recipe for PickXtimes goal-conditioned event-memory pretraining."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import optax


@dataclasses.dataclass(frozen=True)
class PickXtimesLossWeights:
    event: float = 0.5
    event_type: float = 0.5
    goal: float = 0.5
    completed_count: float = 1.0
    remaining_count: float = 1.0
    holding: float = 0.5
    should_press: float = 0.5
    done: float = 0.25
    next_event: float = 0.25
    pick_type: float = 1.0
    place_type: float = 1.0
    press_type: float = 1.0


def _masked_mean(values, mask):
    mask = mask.astype(jnp.float32)
    return jnp.sum(values * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def _categorical_loss(logits, targets, mask):
    losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), targets)
    return _masked_mean(losses, mask)


def _binary_loss(logits, targets, mask):
    losses = optax.sigmoid_binary_cross_entropy(logits.astype(jnp.float32), targets.astype(jnp.float32))
    return _masked_mean(losses, mask)


def compute_losses(outputs, targets, weights: PickXtimesLossWeights | None = None):
    """Compute balanced event, goal, and per-stage recurrent-state losses."""
    if weights is None:
        weights = PickXtimesLossWeights()
    candidate_valid = targets["candidate_valid_mask"].astype(jnp.bool_)
    positive = (targets["event_targets"] > 0) & candidate_valid
    negative = (targets["event_targets"] <= 0) & candidate_valid
    event_losses = optax.sigmoid_binary_cross_entropy(
        outputs["event_logits"].astype(jnp.float32),
        targets["event_targets"].astype(jnp.float32),
    )
    event_loss = 0.5 * (_masked_mean(event_losses, positive) + _masked_mean(event_losses, negative))
    event_type_losses = optax.softmax_cross_entropy_with_integer_labels(
        outputs["event_type_logits"].astype(jnp.float32),
        targets["event_type_targets"],
    )
    event_type_class_weights = jnp.asarray(
        [weights.pick_type, weights.place_type, weights.press_type], dtype=jnp.float32
    )
    event_type_sample_weights = event_type_class_weights[targets["event_type_targets"]]
    event_type_mask = targets["event_type_mask"].astype(jnp.float32)
    weighted_type_mask = event_type_mask * event_type_sample_weights
    event_type_loss = jnp.sum(event_type_losses * weighted_type_mask) / jnp.maximum(
        jnp.sum(weighted_type_mask), 1.0
    )
    goal_color_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            outputs["goal_color_logits"].astype(jnp.float32),
            targets["goal_color"],
        )
    )
    goal_count_loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(
            outputs["goal_required_count_logits"].astype(jnp.float32),
            targets["goal_required_count"],
        )
    )
    goal_loss = 0.5 * (goal_color_loss + goal_count_loss)

    stage_mask = targets["sequence_mask"]
    completed_count_loss = _categorical_loss(
        outputs["completed_count_logits"], targets["completed_count_targets"], stage_mask
    )
    remaining_count_loss = _categorical_loss(
        outputs["remaining_count_logits"], targets["remaining_count_targets"], stage_mask
    )
    holding_loss = _binary_loss(outputs["holding_logits"], targets["holding_targets"], stage_mask)
    should_press_loss = _binary_loss(outputs["should_press_logits"], targets["should_press_targets"], stage_mask)
    done_loss = _binary_loss(outputs["done_logits"], targets["done_targets"], stage_mask)
    next_event_loss = _categorical_loss(
        outputs["next_event_logits"], targets["next_event_targets"], targets["next_event_mask"]
    )
    loss = (
        weights.event * event_loss
        + weights.event_type * event_type_loss
        + weights.goal * goal_loss
        + weights.completed_count * completed_count_loss
        + weights.remaining_count * remaining_count_loss
        + weights.holding * holding_loss
        + weights.should_press * should_press_loss
        + weights.done * done_loss
        + weights.next_event * next_event_loss
    )

    event_predictions = outputs["event_logits"] > 0
    event_type_predictions = jnp.argmax(outputs["event_type_logits"], axis=-1)
    completed_predictions = jnp.argmax(outputs["completed_count_logits"], axis=-1)
    remaining_predictions = jnp.argmax(outputs["remaining_count_logits"], axis=-1)
    return loss, {
        "loss": loss,
        "event_gate_loss": event_loss,
        "event_type_loss": event_type_loss,
        "goal_loss": goal_loss,
        "completed_count_loss": completed_count_loss,
        "remaining_count_loss": remaining_count_loss,
        "holding_loss": holding_loss,
        "should_press_loss": should_press_loss,
        "done_loss": done_loss,
        "next_event_loss": next_event_loss,
        "complete_event_recall": _masked_mean(event_predictions, positive),
        "no_event_rejection": _masked_mean(~event_predictions, negative),
        "event_type_accuracy": _masked_mean(
            event_type_predictions == targets["event_type_targets"], targets["event_type_mask"]
        ),
        "goal_color_accuracy": jnp.mean(jnp.argmax(outputs["goal_color_logits"], axis=-1) == targets["goal_color"]),
        "goal_count_accuracy": jnp.mean(
            jnp.argmax(outputs["goal_required_count_logits"], axis=-1) == targets["goal_required_count"]
        ),
        "stage_count_accuracy": _masked_mean(completed_predictions == targets["completed_count_targets"], stage_mask),
        "stage_remaining_accuracy": _masked_mean(
            remaining_predictions == targets["remaining_count_targets"], stage_mask
        ),
        "should_press_accuracy": _masked_mean(
            (outputs["should_press_logits"] > 0) == targets["should_press_targets"], stage_mask
        ),
    }

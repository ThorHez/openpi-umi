"""Deployable PickXTimes event head with a deterministic recurrent counter."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from openpi.tasks.robomme.unified_visual_student import VisualWindowEncoder


EVENT_NAMES = ("hold", "pick_complete", "place_complete", "press_complete")
DYNAMIC_FIELDS = ("completed_count", "holding", "ready_to_press", "done")


class LowDimTemporalEncoder(nn.Module):
    width: int = 64

    @nn.compact
    def __call__(self, values: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.width, name="input")(values)
        position = self.param(
            "position", nn.initializers.normal(stddev=0.02), (1, 12, self.width), jnp.float32
        )
        x = x + position
        residual = nn.gelu(nn.Dense(self.width * 2, name="hidden")(x))
        x = nn.LayerNorm(name="output_ln")(x + nn.Dense(self.width, name="out")(residual))
        early = jnp.mean(x[:, :6], axis=1)
        late = jnp.mean(x[:, 6:], axis=1)
        delta = late - early
        return nn.gelu(
            nn.Dense(self.width * 2, name="summary")(
                jnp.concatenate(
                    (jnp.mean(x, axis=1), early, late, delta, jnp.abs(delta)), axis=-1
                )
            )
        )


class PickExplicitEventHead(nn.Module):
    """Predict one causal completed event from a fixed 12-frame chunk."""

    width: int = 64
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8

    @nn.compact
    def __call__(
        self,
        previous_state: jnp.ndarray,
        required_count: jnp.ndarray,
        rgb: jnp.ndarray,
        proprio: jnp.ndarray,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        if previous_state.ndim != 2 or previous_state.shape[-1] != 4:
            raise ValueError(f"Expected previous state [B,4], got {previous_state.shape}")
        if proprio.ndim != 3 or proprio.shape[1:] != (12, 6):
            raise ValueError(f"Expected proprio [B,12,6], got {proprio.shape}")
        state_embed = nn.Embed(6, self.width, name="state_value_embedding")(previous_state)
        state_type = self.param(
            "state_field_type",
            nn.initializers.normal(stddev=0.02),
            (1, 4, self.width),
            jnp.float32,
        )
        state_summary = (state_embed + state_type).reshape(previous_state.shape[0], -1)
        required = nn.Embed(6, self.width, name="required_count_embedding")(required_count)

        visual = VisualWindowEncoder(
            name="visual_window_encoder",
            frames=12,
            spatial_tokens=16,
            input_width=1152,
            width=self.width,
            encoder_width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            dtype_mm="bfloat16",
        )(rgb, train=train).reshape(rgb.shape[0], 12, 16, self.width)
        visual = nn.LayerNorm(name="visual_output_ln")(visual.astype(jnp.float32))
        early = jnp.mean(visual[:, :6], axis=(1, 2))
        late = jnp.mean(visual[:, 6:], axis=(1, 2))
        delta = late - early
        visual_summary = nn.gelu(
            nn.Dense(self.width * 2, name="visual_summary")(
                jnp.concatenate(
                    (jnp.mean(visual, axis=(1, 2)), early, late, delta, jnp.abs(delta)),
                    axis=-1,
                )
            )
        )
        proprio_summary = LowDimTemporalEncoder(
            width=self.width, name="proprio_encoder"
        )(proprio)
        fused = jnp.concatenate(
            (state_summary, required, visual_summary, proprio_summary), axis=-1
        )
        hidden = nn.gelu(nn.Dense(self.width * 4, name="event_hidden")(fused))
        hidden = nn.LayerNorm(name="event_ln")(
            hidden
            + nn.Dense(self.width * 4, name="event_residual")(
                nn.gelu(nn.Dense(self.width * 4, name="event_residual_hidden")(hidden))
            )
        )
        return nn.Dense(len(EVENT_NAMES), name="event_out")(hidden)


class PickTargetMotionEventHead(nn.Module):
    """Ablation head adding wrist tokens and target-conditioned motion evidence."""

    mode: str = "front_wrist"
    width: int = 64
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8

    @nn.compact
    def __call__(
        self,
        previous_state: jnp.ndarray,
        required_count: jnp.ndarray,
        target_color_id: jnp.ndarray,
        rgb: jnp.ndarray,
        wrist: jnp.ndarray,
        proprio: jnp.ndarray,
        target_motion: jnp.ndarray,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        if self.mode not in {"front_wrist", "front_wrist_motion"}:
            raise ValueError(f"Unsupported verifier mode {self.mode!r}")
        if previous_state.ndim != 2 or previous_state.shape[-1] != 4:
            raise ValueError(f"Expected previous state [B,4], got {previous_state.shape}")
        if proprio.ndim != 3 or proprio.shape[1:] != (12, 6):
            raise ValueError(f"Expected proprio [B,12,6], got {proprio.shape}")
        if target_motion.ndim != 3 or target_motion.shape[1:] != (12, 11):
            raise ValueError(f"Expected target motion [B,12,11], got {target_motion.shape}")

        state_embed = nn.Embed(6, self.width, name="state_value_embedding")(previous_state)
        state_type = self.param(
            "state_field_type", nn.initializers.normal(stddev=0.02), (1, 4, self.width), jnp.float32
        )
        state_summary = (state_embed + state_type).reshape(previous_state.shape[0], -1)
        required = nn.Embed(6, self.width, name="required_count_embedding")(required_count)

        def encode_view(tokens, name: str):
            visual = VisualWindowEncoder(
                name=name,
                frames=12,
                spatial_tokens=16,
                input_width=1152,
                width=self.width,
                encoder_width=self.encoder_width,
                depth=self.encoder_depth,
                num_heads=self.encoder_heads,
                dtype_mm="bfloat16",
            )(tokens, train=train).reshape(tokens.shape[0], 12, 16, self.width)
            visual = nn.LayerNorm(name=f"{name}_output_ln")(visual.astype(jnp.float32))
            early = jnp.mean(visual[:, :6], axis=(1, 2))
            late = jnp.mean(visual[:, 6:], axis=(1, 2))
            delta = late - early
            return nn.gelu(
                nn.Dense(self.width * 2, name=f"{name}_summary")(
                    jnp.concatenate(
                        (jnp.mean(visual, axis=(1, 2)), early, late, delta, jnp.abs(delta)),
                        axis=-1,
                    )
                )
            )

        # Preserve the base head parameter names so its front representation
        # can initialize both ablations exactly.
        front = VisualWindowEncoder(
            name="visual_window_encoder",
            frames=12,
            spatial_tokens=16,
            input_width=1152,
            width=self.width,
            encoder_width=self.encoder_width,
            depth=self.encoder_depth,
            num_heads=self.encoder_heads,
            dtype_mm="bfloat16",
        )(rgb, train=train).reshape(rgb.shape[0], 12, 16, self.width)
        front = nn.LayerNorm(name="visual_output_ln")(front.astype(jnp.float32))
        front_early = jnp.mean(front[:, :6], axis=(1, 2))
        front_late = jnp.mean(front[:, 6:], axis=(1, 2))
        front_delta = front_late - front_early
        front_summary = nn.gelu(
            nn.Dense(self.width * 2, name="visual_summary")(
                jnp.concatenate(
                    (
                        jnp.mean(front, axis=(1, 2)),
                        front_early,
                        front_late,
                        front_delta,
                        jnp.abs(front_delta),
                    ),
                    axis=-1,
                )
            )
        )
        wrist_summary = encode_view(wrist, "wrist_window_encoder")
        proprio_summary = LowDimTemporalEncoder(width=self.width, name="proprio_encoder")(proprio)
        evidence = [state_summary, required, front_summary, wrist_summary, proprio_summary]
        if self.mode == "front_wrist_motion":
            evidence.extend(
                (
                    nn.Embed(4, self.width, name="target_color_embedding")(target_color_id),
                    LowDimTemporalEncoder(width=self.width, name="target_motion_encoder")(
                        target_motion
                    ),
                )
            )
        fused = jnp.concatenate(evidence, axis=-1)
        hidden = nn.gelu(nn.Dense(self.width * 4, name="event_hidden")(fused))
        hidden = nn.LayerNorm(name="event_ln")(
            hidden
            + nn.Dense(self.width * 4, name="event_residual")(
                nn.gelu(nn.Dense(self.width * 4, name="event_residual_hidden")(hidden))
            )
        )
        return nn.Dense(len(EVENT_NAMES), name="event_out")(hidden)


class PickObjectSuccessHead(nn.Module):
    """Predict privileged-teacher pick/place success from deployable inputs.

    The two logits are independent binary predicates.  Simulator object pose
    and contact are used only to create their training targets; this module
    consumes front/wrist visual tokens, proprioception, and the instructed
    target color at deployment.
    """

    width: int = 64
    encoder_width: int = 128
    encoder_depth: int = 2
    encoder_heads: int = 8

    @nn.compact
    def __call__(
        self,
        target_color_id: jnp.ndarray,
        rgb: jnp.ndarray,
        wrist: jnp.ndarray,
        proprio: jnp.ndarray,
        *,
        train: bool = False,
    ) -> jnp.ndarray:
        if proprio.ndim != 3 or proprio.shape[1:] != (12, 6):
            raise ValueError(f"Expected proprio [B,12,6], got {proprio.shape}")

        def encode_view(tokens: jnp.ndarray, name: str, *, base_names: bool) -> jnp.ndarray:
            encoder_name = "visual_window_encoder" if base_names else name
            output_name = "visual_output_ln" if base_names else f"{name}_output_ln"
            summary_name = "visual_summary" if base_names else f"{name}_summary"
            visual = VisualWindowEncoder(
                name=encoder_name,
                frames=12,
                spatial_tokens=16,
                input_width=1152,
                width=self.width,
                encoder_width=self.encoder_width,
                depth=self.encoder_depth,
                num_heads=self.encoder_heads,
                dtype_mm="bfloat16",
            )(tokens, train=train).reshape(tokens.shape[0], 12, 16, self.width)
            visual = nn.LayerNorm(name=output_name)(visual.astype(jnp.float32))
            early = jnp.mean(visual[:, :6], axis=(1, 2))
            late = jnp.mean(visual[:, 6:], axis=(1, 2))
            delta = late - early
            return nn.gelu(
                nn.Dense(self.width * 2, name=summary_name)(
                    jnp.concatenate(
                        (jnp.mean(visual, axis=(1, 2)), early, late, delta, jnp.abs(delta)),
                        axis=-1,
                    )
                )
            )

        front = encode_view(rgb, "front_window_encoder", base_names=True)
        wrist_summary = encode_view(wrist, "wrist_window_encoder", base_names=False)
        proprio_summary = LowDimTemporalEncoder(width=self.width, name="proprio_encoder")(
            proprio
        )
        target_color = nn.Embed(4, self.width, name="target_color_embedding")(
            target_color_id
        )
        fused = jnp.concatenate(
            (target_color, front, wrist_summary, proprio_summary), axis=-1
        )
        hidden = nn.gelu(nn.Dense(self.width * 4, name="success_hidden")(fused))
        hidden = nn.LayerNorm(name="success_ln")(
            hidden
            + nn.Dense(self.width * 4, name="success_residual")(
                nn.gelu(
                    nn.Dense(self.width * 4, name="success_residual_hidden")(hidden)
                )
            )
        )
        return nn.Dense(2, name="success_out")(hidden)


def deterministic_update_numpy(previous, event_id: int, required_count: int):
    """Execute a legal event; illegal predictions are conservatively held."""

    import numpy as np

    state = np.asarray(previous, dtype=np.int32).copy()
    completed, holding, ready, done = (int(value) for value in state)
    if event_id == 1 and not done and not holding and completed < required_count:
        holding = 1
    elif event_id == 2 and not done and holding and completed < required_count:
        completed += 1
        holding = 0
        ready = int(completed == required_count)
    elif event_id == 3 and not done and ready and not holding:
        ready = 0
        done = 1
    return np.asarray((completed, holding, ready, done), dtype=np.int32)

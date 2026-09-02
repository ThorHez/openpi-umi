"""Memory-only Pi0 shell for supervised causal event-memory pretraining."""

from __future__ import annotations

import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0_mem_compress as _base
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.tasks.shellgame import pi0_mem_semantic_action as _semantic_action
from openpi.tasks.shellgame import semantic_memory
from openpi.tasks.shellgame import semantic_memory_event


@dataclasses.dataclass(frozen=True)
class Pi0MemSemanticEventMemoryConfig(_semantic_action.Pi0MemSemanticActionConfig):
    """Configuration for the short-window event-memory training shell."""

    event_window_size: int = semantic_memory_event.WINDOW_SIZE

    def create(self, rng: at.KeyArrayLike) -> Pi0MemSemanticEventMemory:
        return Pi0MemSemanticEventMemory(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter_event_memory_pretrain(self) -> nnx.filterlib.Filter:
        """Train initial perception and the window event/relation encoder."""
        initial = nnx_utils.PathRegex(r".*HistoryFrame0InitialCupClassifier.*")
        window_encoder = nnx_utils.PathRegex(r".*HistorySlidingWindowRelationMemoryTracker/window_classifier.*")
        return nnx.Not(nnx.Any(initial, window_encoder))


class Pi0MemSemanticEventMemory(_base.Pi0MemCompress):
    """Expose only the visual event-memory path to the standalone trainer."""

    def __init__(self, config: Pi0MemSemanticEventMemoryConfig, rngs: nnx.Rngs):
        if config.num_frames not in (config.history_frames, config.history_frames + 1):
            raise ValueError(
                "Event-memory pretraining requires fixed history and optionally one current frame: "
                f"num_frames={config.num_frames}, history_frames={config.history_frames}"
            )
        super().__init__(config, rngs)
        self.history_frames = int(config.history_frames)
        self.event_window_size = int(config.event_window_size)
        self.HistoryFrame0InitialCupClassifier = nnx_bridge.ToNNX(
            semantic_memory.FrozenFrame0InitialCupClassifier(input_width=1152)
        )
        self.HistoryFrame0InitialCupClassifier.lazy_init(jnp.zeros((1, 256, 1152), dtype=jnp.bfloat16), rngs=rngs)
        self.HistorySlidingWindowRelationMemoryTracker = nnx_bridge.ToNNX(
            semantic_memory_event.ShellGameSlidingWindowEventMemoryTracker(
                num_frames=config.history_frames,
                window_size=config.event_window_size,
                input_width=1152,
                encoder_width=config.encoder_width,
                encoder_depth=config.encoder_depth,
                encoder_heads=config.encoder_heads,
                memory_width=config.semantic_memory_width,
                memory_depth=config.semantic_memory_depth,
                memory_heads=config.semantic_memory_heads,
                adapter_heads=config.diagnostic_adapter_heads,
                num_memory_tokens=config.semantic_memory_tokens,
                num_current_tokens=config.diagnostic_current_tokens,
                current_width=1152,
                residual_scale=config.diagnostic_residual_scale,
                dtype_mm=config.dtype,
            )
        )
        self.HistorySlidingWindowRelationMemoryTracker.lazy_init(
            jnp.zeros((1, config.history_frames, 256, 1152), dtype=jnp.bfloat16),
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.asarray([[20, 30, 40, 25, 35, 45, 0, 5]], dtype=jnp.int32),
            causal_selection=False,
            train=False,
            rngs=rngs,
        )

    def compute_event_memory_outputs(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        initial_slots,
        window_starts,
        causal_selection: bool,
        train: bool,
    ):
        observation = _model.preprocess_observation(rng, observation, train=train)
        image = observation.images.get("base_rgb")
        valid_frame_counts = (self.history_frames, self.history_frames + 1)
        if image is None or image.ndim != 5 or image.shape[1] not in valid_frame_counts:
            raise ValueError(
                f"Expected base_rgb frame count in {valid_frame_counts}, got {None if image is None else image.shape}"
            )
        history = image[:, : self.history_frames]
        _, initial_encoder_out = self.PaliGemma.img(history[:, :1], train=False)
        initial_logits = self.HistoryFrame0InitialCupClassifier(initial_encoder_out["encoded"])
        if initial_slots is None:
            initial_slots = jnp.argmax(initial_logits, axis=-1)

        _, history_encoder_out = self.PaliGemma.img(history, train=False)
        tracker = self.HistorySlidingWindowRelationMemoryTracker(
            history_encoder_out["with_posemb"][:, : self.history_frames],
            initial_slots.astype(jnp.int32),
            window_starts.astype(jnp.int32),
            causal_selection=causal_selection,
            train=train,
        )
        return {"initial_logits": initial_logits, **tracker}

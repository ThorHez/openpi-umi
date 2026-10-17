"""Local Qwen3-VL wrapper for low-frequency structured event proposals.

The imports of Torch and Transformers are deliberately lazy.  OpenPI can
import the shared schema without installing Qwen dependencies, while this
wrapper is executed only inside the isolated ``qwen3vl_shellgame`` Conda
environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any
from typing import Sequence

from openpi.planning.qwenvl_event_schema import PlannerPatch


SYSTEM_PROMPT = """You are a high-level visual event interpreter for a robot memory system.
You do not output robot joint, end-effector, or gripper actions. You inspect only the supplied
chronological frames and the explicit previous task memory. Propose at most one state-changing
event. Never infer from future frames. Return exactly one JSON object and no prose or markdown.
Use only the operation and decision vocabularies specified in the user request. If evidence is
insufficient, use operation no_state_change, decision request_reobservation, confidence <= 0.5,
request_reobservation true, and state_delta {"operation":"no_state_change"}. Do not invent entities."""


@dataclass(frozen=True)
class QwenVLPlannerConfig:
    model_path: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    max_new_tokens: int = 320
    min_pixels: int = 224 * 224
    max_pixels: int = 224 * 224
    attn_implementation: str = "sdpa"
    video_fps: float = 10.0


@dataclass(frozen=True)
class PlannerGeneration:
    patch: PlannerPatch
    raw_text: str
    latency_seconds: float


class PlannerGenerationError(ValueError):
    """A generation completed, but its text failed deterministic validation."""

    def __init__(self, message: str, *, raw_text: str, latency_seconds: float):
        super().__init__(message)
        self.raw_text = raw_text
        self.latency_seconds = latency_seconds


class LocalQwen3VLPlanner:
    """One-process, frozen local Qwen3-VL inference wrapper."""

    def __init__(self, config: QwenVLPlannerConfig):
        import torch
        from transformers import AutoProcessor
        from transformers import Qwen3VLForConditionalGeneration
        from transformers.video_utils import VideoMetadata

        self.config = config
        dtype = getattr(torch, config.dtype)
        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            local_files_only=True,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        # The checkpoint defaults to fps-based subsampling.  Our input is
        # already a selected causal keyframe window with no external video
        # metadata, so preserve every supplied frame via ``num_frames``.
        self.processor.video_processor.fps = None
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_path,
            local_files_only=True,
            dtype=dtype,
            device_map={"": config.device},
            attn_implementation=config.attn_implementation,
        )
        self.model.eval()
        # The checkpoint's sampling defaults are irrelevant for deterministic
        # greedy decoding and otherwise produce noisy warnings.
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.torch = torch
        self.video_metadata_type = VideoMetadata

    def generate(
        self,
        images: Sequence[Any],
        *,
        request: str,
        previous_task_memory: dict[str, Any],
        request_id: str,
    ) -> PlannerGeneration:
        if not images:
            raise ValueError("At least one chronological frame is required")
        memory_json = json.dumps(previous_task_memory, ensure_ascii=False, sort_keys=True)
        user_text = (
            f"Request id: {request_id}\n"
            f"Previous task memory (trusted state before these frames): {memory_json}\n\n"
            f"{request}\n\n"
            "The following images are chronological, oldest to newest."
        )
        # Use Qwen3-VL's native video path so temporal patches and video RoPE
        # are constructed.  Treating frames as unrelated images was observed
        # to lose the single-exchange trajectory in the ShellGame smoke test.
        content: list[dict[str, Any]] = [
            {"type": "video", "video": list(images)},
            {"type": "text", "text": user_text},
        ]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            num_frames=len(images),
            video_metadata=[
                self.video_metadata_type(
                    total_num_frames=len(images),
                    fps=self.config.video_fps,
                    width=int(images[0].width),
                    height=int(images[0].height),
                    duration=len(images) / self.config.video_fps,
                    frames_indices=list(range(len(images))),
                )
            ],
        )
        inputs = inputs.to(self.config.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        latency = time.perf_counter() - started
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        raw_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        try:
            patch = PlannerPatch.from_text(raw_text)
            if patch.request_id != request_id:
                raise ValueError(f"Qwen returned request_id={patch.request_id!r}, expected {request_id!r}")
        except Exception as exc:
            raise PlannerGenerationError(
                f"{type(exc).__name__}: {exc}",
                raw_text=raw_text,
                latency_seconds=latency,
            ) from exc
        return PlannerGeneration(patch=patch, raw_text=raw_text, latency_seconds=latency)

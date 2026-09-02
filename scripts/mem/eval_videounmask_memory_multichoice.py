#!/usr/bin/env python3
"""Run VideoUnmask memory predictions in RoboMME's multi-choice environment.

The evaluator uses the reset demonstration as the only visual conditioning,
predicts a front-camera pixel with the trained target-event memory, and sends
the real high-level action ``{"choice": "A", "point": [y, x]}`` to RoboMME.
Only easy/medium (single-target) episodes are evaluated.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import time
from typing import Any

import flax
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.tasks.robomme.videounmask import semantic_memory_target_event
from openpi.training.mem.recipes import shellgame_semantic_action

DEFAULT_RUN_DIR = pathlib.Path("evaluation/robomme/videounmask_semantic_memory_v1_260823/target_event_seed260823_2k")
DEFAULT_BACKBONE_CHECKPOINT = pathlib.Path(
    "checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/"
    "absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params"
)
CONTROLS = ("full", "oracle", "late_only", "zero_video", "wrong_video")
NUM_DEMO_FRAMES = semantic_memory_target_event.NUM_DEMO_FRAMES
IMAGE_MAX = 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--backbone-checkpoint", type=pathlib.Path, default=DEFAULT_BACKBONE_CHECKPOINT)
    parser.add_argument("--dataset", choices=("val", "test"), required=True)
    parser.add_argument(
        "--controls",
        default="full",
        help=f"Comma-separated subset of: {','.join(CONTROLS)}",
    )
    parser.add_argument("--episodes", help="Comma-separated episode indices; still filters out hard episodes")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--gui-render", action="store_true")
    return parser.parse_args()


@nnx.jit
def encode_images(model, uint8_images):
    images = uint8_images.astype(jnp.float32) / 127.5 - 1.0
    images = jax.image.resize(images, (images.shape[0], 224, 224, 3), method="linear", antialias=True)
    _, encoder_out = model.PaliGemma.img(images[:, None], train=False)
    return encoder_out["with_posemb"][:, 0]


@nnx.jit
def embed_prompts(model, token_ids):
    return model.PaliGemma.llm(token_ids, method="embed")


def _controls(value: str) -> list[str]:
    controls = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(controls) - set(CONTROLS))
    if invalid:
        raise ValueError(f"Unknown controls: {invalid}; expected a subset of {CONTROLS}")
    if not controls:
        raise ValueError("--controls must not be empty")
    return list(dict.fromkeys(controls))


def _episode_indices(builder, requested: str | None, maximum: int | None) -> list[int]:
    if requested:
        candidates = [int(item.strip()) for item in requested.split(",") if item.strip()]
    else:
        candidates = sorted(episode for task, episode in builder.metadata_index if task == "VideoUnmask")
    single_target = []
    for episode in candidates:
        metadata = builder.metadata_index.get(("VideoUnmask", episode))
        if metadata is None:
            raise ValueError(f"Episode {episode} is absent from {builder.dataset} metadata")
        if str(metadata.get("difficulty", "")).lower() != "hard":
            single_target.append(episode)
    if maximum is not None:
        if maximum < 1:
            raise ValueError("--max-episodes must be positive")
        single_target = single_target[:maximum]
    if not single_target:
        raise ValueError("No single-target episodes selected")
    return single_target


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_bool(value: Any) -> bool:
    array = _as_numpy(value)
    return bool(array.reshape(-1)[-1]) if array.size else False


def _prompt_from_info(info: dict[str, Any]) -> str:
    prompt = info.get("task_goal")
    while isinstance(prompt, list | tuple) and prompt:
        prompt = prompt[-1]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing task_goal in reset info: {type(prompt).__name__}")
    return prompt.strip()


def _reset_frames(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    frames_like = obs.get("front_rgb_list")
    if not isinstance(frames_like, list) or len(frames_like) < 2:
        raise ValueError("reset observation did not contain a demonstration plus current front frame")
    frames = [_as_numpy(frame).astype(np.uint8, copy=False) for frame in frames_like]
    demo = frames[:-1]
    indices = np.rint(np.linspace(0, len(demo) - 1, NUM_DEMO_FRAMES)).astype(np.int32)
    return np.stack([demo[index] for index in indices]), np.asarray(indices)


def _target_color(prompt: str) -> str:
    match = re.search(r"\b(red|green|blue)\b", prompt.lower())
    if match is None:
        raise ValueError(f"Could not identify target color in prompt: {prompt!r}")
    return match.group(1)


def _cube_centers(image: np.ndarray) -> dict[str, list[float]]:
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    masks = {
        "red": (red > 200) & (green < 50) & (blue < 50),
        "green": (green > 200) & (red < 50) & (blue < 50),
        "blue": (blue > 200) & (red < 50) & (green < 50),
    }
    centers: dict[str, list[float]] = {}
    for color, mask in masks.items():
        y, x = np.nonzero(mask)
        if len(y) < 4:
            raise ValueError(f"Could not segment {color} cube; pixels={len(y)}")
        centers[color] = [float(np.mean(y)), float(np.mean(x))]
    return centers


def _load_memory(run_dir: pathlib.Path, checkpoint: pathlib.Path | None):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = checkpoint or run_dir / "checkpoints/step_1000.msgpack"
    model = semantic_memory_target_event.VideoUnmaskTargetEventMemory(
        encoder_width=int(config["encoder_width"]),
        encoder_depth=int(config["encoder_depth"]),
        memory_width=int(config["memory_width"]),
        memory_depth=int(config["memory_depth"]),
        num_memory_tokens=int(config["memory_tokens"]),
    )
    dummy = {
        "demo_patch_tokens": jnp.zeros((1, 12, 256, 1152), dtype=jnp.float16),
        "prompt_tokens": jnp.zeros((1, 256, 2048), dtype=jnp.float16),
        "prompt_mask": jnp.ones((1, 256), dtype=jnp.bool_),
        "frame_mask": jnp.ones((1, 12), dtype=jnp.bool_),
    }
    variables = model.init(jax.random.key(0), **dummy, train=False)
    params = flax.serialization.from_bytes(variables["params"], checkpoint.read_bytes())

    @jax.jit
    def predict(inputs):
        return model.apply({"params": params}, **inputs, train=False)["target_point"]

    return checkpoint, predict


def _load_backbone(checkpoint: pathlib.Path):
    params = _model.restore_params(checkpoint.expanduser().resolve(), dtype=jnp.bfloat16)
    model = shellgame_semantic_action.MODEL_CONFIG.load(params)
    model.eval()
    tokenizer = _tokenizer.PaligemmaTokenizer(shellgame_semantic_action.MODEL_CONFIG.max_token_len)
    return model, tokenizer


def _conditioning(backbone, tokenizer, obs, info) -> dict[str, Any]:
    frames, indices = _reset_frames(obs)
    prompt = _prompt_from_info(info)
    token_ids, prompt_mask = tokenizer.tokenize(prompt)
    patch_tokens = np.asarray(encode_images(backbone, jnp.asarray(frames))).astype(np.float16)
    prompt_tokens = np.asarray(
        embed_prompts(backbone, jnp.asarray(np.asarray(token_ids, dtype=np.int32)[None]))
    ).astype(np.float16)[0]
    return {
        "frames": frames,
        "demo_indices": indices,
        "prompt": prompt,
        "demo_patch_tokens": patch_tokens,
        "prompt_tokens": prompt_tokens,
        "prompt_mask": np.asarray(prompt_mask, dtype=np.bool_),
    }


def _predict_point(predict, conditioning: dict[str, Any], control: str, wrong_video=None) -> list[float]:
    video = conditioning["demo_patch_tokens"]
    if control == "late_only":
        video = np.repeat(video[-1:], NUM_DEMO_FRAMES, axis=0)
    elif control == "zero_video":
        video = np.zeros_like(video)
    elif control == "wrong_video":
        if wrong_video is None:
            raise ValueError("wrong_video control requires conditioning from another episode")
        video = wrong_video["demo_patch_tokens"]
    elif control == "oracle":
        centers = _cube_centers(conditioning["frames"][0])
        return centers[_target_color(conditioning["prompt"])]
    elif control != "full":
        raise ValueError(control)

    inputs = {
        "demo_patch_tokens": jnp.asarray(video[None]),
        "prompt_tokens": jnp.asarray(conditioning["prompt_tokens"][None]),
        "prompt_mask": jnp.asarray(conditioning["prompt_mask"][None]),
        "frame_mask": jnp.ones((1, NUM_DEMO_FRAMES), dtype=jnp.bool_),
    }
    normalized = np.asarray(predict(inputs))[0]
    return np.clip(normalized * IMAGE_MAX, 0.0, IMAGE_MAX).astype(float).tolist()


def _close(env) -> None:
    if env is not None:
        with contextlib.suppress(Exception):
            env.close()


def _write_result(path: pathlib.Path, result: dict[str, Any]) -> None:
    for control_result in result["controls"].values():
        records = control_result["episodes"]
        valid = [record for record in records if record.get("status") != "reset_error"]
        successes = sum(record.get("success", False) for record in valid)
        control_result["summary"] = {
            "attempted": len(valid),
            "successes": successes,
            "success_rate": successes / len(valid) if valid else None,
            "errors": sum(record.get("status") in ("error", "reset_error") for record in records),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OPENPI_DATA_HOME", "/data2/hzl_workspace_for_pi_mem/.cache/openpi")
    controls = _controls(args.controls)

    from robomme.env_record_wrapper.episode_config_resolver import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id="VideoUnmask",
        dataset=args.dataset,
        action_space="multi_choice",
        gui_render=args.gui_render,
    )
    episodes = _episode_indices(builder, args.episodes, args.max_episodes)
    checkpoint, predict = _load_memory(args.run_dir, args.checkpoint)
    print(f"Restoring frozen backbone from {args.backbone_checkpoint.resolve()}", flush=True)
    backbone, tokenizer = _load_backbone(args.backbone_checkpoint)

    result: dict[str, Any] = {
        "schema_version": 1,
        "dataset": args.dataset,
        "single_target_difficulties": ["easy", "medium"],
        "episode_indices": episodes,
        "memory_checkpoint": str(checkpoint.resolve()),
        "backbone_checkpoint": str(args.backbone_checkpoint.resolve()),
        "controls": {control: {"episodes": []} for control in controls},
    }
    started = time.monotonic()
    previous_conditioning = None

    # The first wrong-video sample uses the final selected episode as its
    # mismatched source; later samples use the preceding episode.
    if "wrong_video" in controls:
        preload_env = None
        try:
            preload_env = builder.make_env_for_episode(episodes[-1])
            preload_obs, preload_info = preload_env.reset()
            previous_conditioning = _conditioning(backbone, tokenizer, preload_obs, preload_info)
        finally:
            _close(preload_env)

    for ordinal, episode in enumerate(episodes, start=1):
        metadata = builder.metadata_index[("VideoUnmask", episode)]
        env = None
        current_conditioning = None
        try:
            env = builder.make_env_for_episode(episode)
            obs, info = env.reset()
            current_conditioning = _conditioning(backbone, tokenizer, obs, info)
            predicted = {
                control: _predict_point(
                    predict,
                    current_conditioning,
                    control,
                    wrong_video=previous_conditioning,
                )
                for control in controls
            }
        except Exception as exc:
            _close(env)
            for control in controls:
                result["controls"][control]["episodes"].append(
                    {
                        "episode": episode,
                        "difficulty": metadata.get("difficulty"),
                        "status": "reset_error",
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            _write_result(args.output, result)
            print(f"[{ordinal}/{len(episodes)}] episode {episode}: reset_error: {exc}", flush=True)
            continue

        for control_index, control in enumerate(controls):
            control_env = env if control_index == 0 else None
            try:
                if control_env is None:
                    control_env = builder.make_env_for_episode(episode)
                    control_env.reset()
                point = predicted[control]
                _, reward, terminated, truncated, step_info = control_env.step(
                    {"choice": "A", "point": [int(np.rint(point[0])), int(np.rint(point[1]))]}
                )
                status = str(step_info.get("status", "unknown"))
                record = {
                    "episode": episode,
                    "difficulty": metadata.get("difficulty"),
                    "prompt": current_conditioning["prompt"],
                    "demo_frame_count": int(current_conditioning["demo_indices"][-1] + 1),
                    "sampled_demo_indices": current_conditioning["demo_indices"].tolist(),
                    "predicted_point_yx": point,
                    "action_point_yx": [int(np.rint(point[0])), int(np.rint(point[1]))],
                    "status": status,
                    "success": status == "success",
                    "reward": float(_as_numpy(reward).reshape(-1)[-1]),
                    "terminated": _as_bool(terminated),
                    "truncated": _as_bool(truncated),
                }
                if step_info.get("error_message"):
                    record["error"] = str(step_info["error_message"])
            except Exception as exc:
                record = {
                    "episode": episode,
                    "difficulty": metadata.get("difficulty"),
                    "prompt": current_conditioning["prompt"],
                    "predicted_point_yx": predicted[control],
                    "status": "error",
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                _close(control_env)
                if control_index == 0:
                    env = None
            result["controls"][control]["episodes"].append(record)

        previous_conditioning = current_conditioning
        _write_result(args.output, result)
        statuses = ", ".join(
            f"{control}={result['controls'][control]['episodes'][-1]['status']}" for control in controls
        )
        elapsed = (time.monotonic() - started) / 60.0
        print(f"[{ordinal}/{len(episodes)}] episode {episode}: {statuses} ({elapsed:.1f} min)", flush=True)

    _write_result(args.output, result)
    for control in controls:
        summary = result["controls"][control]["summary"]
        print(
            f"{control}: {summary['successes']}/{summary['attempted']} "
            f"({summary['success_rate']:.3f}) errors={summary['errors']}",
            flush=True,
        )


if __name__ == "__main__":
    main()

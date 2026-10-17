#!/usr/bin/env python3
"""Closed-loop ShellGame evaluation for Qwen events -> recurrent MEM -> Pi action.

The Qwen reveal prediction and causal sliding-window event triggers are read
from audited inference artifacts.  Simulator metadata is used only to rebuild
the exact episode and to score the rollout; it is never used to construct the
policy memory.  Since robot control begins after the scripted observation,
replaying the cached upstream Qwen result is causally equivalent to keeping the
Qwen process resident while the Pi policy acts.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "shellgame"))

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import image_tools

from examples.shellgame import main as shell_main
from examples.shellgame import oracle_joint_replay as oracle_replay
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import tokenizer as tokenizer_lib
from openpi.policies import policy as policy_lib
from openpi.tasks.shellgame import qwenvl_event_adapter
from openpi.training import checkpoints
from openpi.training.mem.recipes import shellgame_qwen_event_memory_action as action_recipe


DEFAULT_CHECKPOINT = Path(
    "checkpoints/pi0_shellgame_qwen_event_memory_action_eef7_260825/"
    "qwen_event_mem_v10init_action250_6gpu_260825/249"
)
DEFAULT_RAW_ROOT = Path(
    "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
    "shellgame_absolute_eef_phase_instruction_dataset"
)
DEFAULT_INITIAL_CACHE = Path(
    "evaluation/shellgame/qwenvl_event_cache/qwen3vl_lora_step375_val20.jsonl"
)
DEFAULT_TRIGGER_SUMMARY = Path(
    "evaluation/shellgame/"
    "qwen3vl_gt_event_lora_v1_step375_sliding_trigger_val20.summary.json"
)
DEFAULT_OUTPUT = Path(
    "evaluation/shellgame/qwen_event_mem_action_closed_loop_260825/result.json"
)
DEFAULT_VIDEO_DIR = Path(
    "evaluation/shellgame/qwen_event_mem_action_closed_loop_260825/videos"
)
DEFAULT_EPISODES = "31,16,8,56,122,17,102,47,72"
# Match the task string stored in the LeRobot action datasets exactly.  The
# previous extra prefix was an avoidable train/eval contract mismatch.
PROMPT = "Grasp and lift the cup containing the ball."
POLICY_IMAGE_SIZE = 224
SLOTS = ("left", "middle", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--initial-cache", type=Path, default=DEFAULT_INITIAL_CACHE)
    parser.add_argument("--trigger-summary", type=Path, default=DEFAULT_TRIGGER_SUMMARY)
    parser.add_argument("--memory-bank", type=Path, default=action_recipe.DEFAULT_MEMORY_BANK)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--episodes", default=DEFAULT_EPISODES)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--remote-policy", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8025)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--num-sampling-steps", type=int, default=4)
    parser.add_argument("--max-policy-steps", type=int, default=95)
    parser.add_argument("--selection-skip", type=int, default=10)
    parser.add_argument("--selection-window", type=int, default=30)
    parser.add_argument("--selection-radius", type=float, default=0.06)
    parser.add_argument("--lift-success-height", type=float, default=0.08)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size == 0 or float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return image_tools.resize_with_pad(image, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE)


@dataclasses.dataclass(frozen=True)
class OnlineShellGameInputs(transforms.DataTransformFn):
    """Construct the exact single-frame state/image/external-MEM contract."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = np.asarray(data["state_raw"], dtype=np.float32).reshape(-1)
        memory = np.asarray(data["semantic_memory_raw"], dtype=np.float32)
        if state.shape != (10,):
            raise ValueError(f"Expected absolute EEF state10, got {state.shape}")
        if memory.shape != (128, 64) or not np.all(np.isfinite(memory)):
            raise ValueError(f"Invalid semantic memory: {memory.shape}")
        return {
            "state": state,
            "semantic_memory": memory,
            "image": {
                "base_rgb": _resize(data["base_rgb"])[None],
                "wrist_rgb": _resize(data["wrist_rgb"])[None],
            },
            "image_mask": {"base_rgb": np.True_, "wrist_rgb": np.True_},
            "frame_valid_mask": {
                "base_rgb": np.ones(1, dtype=np.bool_),
                "wrist_rgb": np.ones(1, dtype=np.bool_),
            },
            "prompt": str(data["prompt"]),
            "episode_index": np.int32(data["episode_index"]),
            "frame_index": np.int32(data["frame_index"]),
            "episode_T": np.float32(data["episode_T"]),
        }


def _load_policy(checkpoint: Path, num_sampling_steps: int) -> policy_lib.Policy:
    model_config = action_recipe.make_model_config()
    model = model_config.load(model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    model.eval()
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", ".")
    normalize_masks = {
        "actions": transforms.make_bool_mask(7),
        "state": transforms.make_bool_mask(10),
    }
    return policy_lib.Policy(
        model,
        transforms=[
            OnlineShellGameInputs(),
            # UmiDataConfig trains this policy with quantile normalization.
            # Inference must use the same contract; mean/std here compresses
            # left/right EEF targets toward the center by roughly 40 mm.
            transforms.Normalize(norm_stats, use_quantiles=True, key_masks=normalize_masks),
            transforms.TokenizePrompt(
                tokenizer_lib.PaligemmaTokenizer(model_config.max_token_len),
                discrete_state_input=True,
                robot_type="ARM=1 G=0 H=0",
            ),
            transforms.PadActionsOnly(model_config.action_dim),
            transforms.FlattenState(),
            transforms.KeepModelKeys(),
        ],
        output_transforms=[
            transforms.ChunkActions(target_dim=7),
            transforms.DropKeys(keys=("state",)),
            transforms.Unnormalize(norm_stats, use_quantiles=True, key_masks=normalize_masks),
        ],
        sample_kwargs={"num_steps": num_sampling_steps},
    )


def _qwen_sequences(initial_cache: Path, trigger_summary: Path) -> dict[int, dict[str, Any]]:
    initial_by_episode: dict[int, str] = {}
    for line in initial_cache.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("query_key") == "reveal" and row.get("schema_valid") and row.get("adapter_valid"):
            initial_by_episode[int(row["episode_index"])] = str(row["prediction"])
    summary = json.loads(trigger_summary.read_text(encoding="utf-8"))
    result = {}
    for episode_result in summary["event_trigger"]["per_episode"]:
        episode = int(episode_result["episode_index"])
        triggers = episode_result["triggers"]
        if episode not in initial_by_episode or len(triggers) != 3:
            continue
        initial = qwenvl_event_adapter.normalize_cup_entity(initial_by_episode[episode])
        pairs = []
        relation_ids = []
        for trigger in triggers:
            pair = tuple(
                qwenvl_event_adapter.normalize_cup_entity(str(value))
                for value in trigger["pair"]
            )
            pair = tuple(sorted(pair, key=SLOTS.index))
            pairs.append(pair)
            relation_ids.append(qwenvl_event_adapter.SWAP_PAIRS.index(pair))
        result[episode] = {
            "initial_slot": initial,
            "pairs": [list(pair) for pair in pairs],
            "relation_ids": relation_ids,
            "event_sequence": [SLOTS.index(initial), *relation_ids],
            "triggers": triggers,
        }
    return result


def _memory_lookup(path: Path):
    with np.load(path, allow_pickle=False) as source:
        memories = np.asarray(source["memory_templates"], dtype=np.float32)
        if "event_sequences" in source:
            sequences = np.asarray(source["event_sequences"], dtype=np.int32)
            lookup = {
                tuple(sequence.tolist()): memory
                for sequence, memory in zip(sequences, memories, strict=True)
            }
            return "event_sequence", lookup
        episode_to_template = np.asarray(source["episode_template_index"], dtype=np.int32)
    if episode_to_template.ndim != 1 or np.any(episode_to_template < 0):
        raise ValueError("Invalid episode_template_index in direct visual memory bank")
    if np.any(episode_to_template >= len(memories)):
        raise ValueError("episode_template_index exceeds memory template count")
    return "episode", {
        episode: memories[template]
        for episode, template in enumerate(episode_to_template.tolist())
    }


def _apply_sequence(initial_slot: str, pairs: list[list[str]]) -> str:
    slot = initial_slot
    for raw_pair in pairs:
        slot = qwenvl_event_adapter.apply_swap(slot, tuple(raw_pair))
    return slot


def _policy_state(shell, env) -> np.ndarray:
    observation = env._get_observations(force_update=True)
    position = np.asarray(shell.get_eef_pos(env), dtype=np.float32)
    quaternion = np.asarray(shell.get_eef_quat(env), dtype=np.float32)
    rot6d = shell_main._quat_to_rot6d(quaternion, "openpi")  # noqa: SLF001
    width = np.asarray(
        [shell_main._gripper_width(shell.obs_vector(observation, "robot0_gripper_qpos"))],  # noqa: SLF001
        dtype=np.float32,
    )
    state = np.concatenate((position, rot6d, width))
    if state.shape != (10,) or not np.all(np.isfinite(state)):
        raise ValueError(f"Invalid online state: {state}")
    return state


def _image_total_variation(image: np.ndarray) -> float:
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(
        (np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1)))) / 2
    )


def _current_images(
    shell,
    env,
    ep_args,
    wrist_camera: str,
    *,
    max_read_attempts: int = 8,
    corruption_tv_threshold: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Read owned camera arrays and reject the known transient EGL snow frame."""
    last_tv = None
    for attempt in range(max_read_attempts):
        observation = env._get_observations(force_update=True)
        base = shell.image_from_obs(observation, ep_args.camera, image_rotation=ep_args.image_rotation)
        wrist = shell.optional_image_from_obs(observation, wrist_camera, image_rotation=0)
        if wrist is None:
            raise RuntimeError("ShellGame wrist camera unavailable")
        base = np.array(base, dtype=np.uint8, order="C", copy=True)
        wrist = np.array(wrist, dtype=np.uint8, order="C", copy=True)
        last_tv = _image_total_variation(base)
        if last_tv < corruption_tv_threshold:
            return base, wrist, attempt
    raise RuntimeError(
        f"EGL camera remained corrupted for {max_read_attempts} reads; base total variation={last_tv:.1f}"
    )


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for frame in frames:
            current = np.asarray(frame, dtype=np.uint8)
            if current.shape[:2] != (height, width):
                current = cv2.resize(current, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(current, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _episode_args(command_args: dict[str, Any], robosuite_root: str) -> shell_main.Args:
    args = shell_main.Args()
    for field in dataclasses.fields(args):
        if field.name in command_args:
            setattr(args, field.name, command_args[field.name])
    args.robosuite_root = robosuite_root
    args.width = POLICY_IMAGE_SIZE
    args.height = POLICY_IMAGE_SIZE
    args.resize_size = POLICY_IMAGE_SIZE
    args.gpu_id = -1
    args.osc_input_type = "absolute"
    args.control_during_scripted_observation = False
    args.observe_eef_frames = 0
    return args


def _selected_cup(votes: dict[str, int], distance_sums: dict[str, float]) -> str | None:
    maximum = max(votes.values(), default=0)
    if maximum <= 0:
        return None
    candidates = [cup for cup, count in votes.items() if count == maximum]
    return min(candidates, key=lambda cup: distance_sums[cup] / votes[cup])


def _run_episode(
    episode: int,
    policy: policy_lib.Policy,
    memory: np.ndarray,
    qwen: dict[str, Any],
    shell,
    args: argparse.Namespace,
) -> dict[str, Any]:
    episode_dir = args.raw_root / f"episode_{episode:06d}"
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    command_args = metadata["command_args"]
    policy_args = _episode_args(command_args, args.robosuite_root)
    ep_args = shell_main._episode_namespace(  # noqa: SLF001
        policy_args,
        seed=int(command_args["seed"]),
        initial_ball_cup=str(command_args["initial_ball_cup"]),
        num_swaps=int(command_args["num_swaps"]),
    )
    env = shell.make_env(ep_args)
    replay: list[np.ndarray] = []
    inference_ms = []
    try:
        scene = oracle_replay._prepare_scripted_state(shell, env, ep_args)  # noqa: SLF001
        if scene["swaps"] != metadata["swaps"] or scene["final_ball_cup"] != metadata["final_ball_cup"]:
            raise RuntimeError("Reconstructed simulator episode does not match dataset metadata")
        wrist_camera = shell.resolve_wrist_camera_name(env, ep_args.wrist_camera)
        with np.load(episode_dir / "vla_trajectory.npz", allow_pickle=False) as source:
            recorded_prefix = np.asarray(source["third_person_images"][:60], dtype=np.uint8)
        replay.extend(np.array(frame, order="C", copy=True) for frame in recorded_prefix)
        live_base, live_wrist, initial_retries = _current_images(shell, env, ep_args, wrist_camera)
        render_retries = initial_retries
        render_mae = float(np.mean(np.abs(live_base.astype(np.float32) - recorded_prefix[-1].astype(np.float32))))

        votes = {cup: 0 for cup in shell.CUP_NAMES}
        distance_sums = {cup: 0.0 for cup in shell.CUP_NAMES}
        max_target_lift = 0.0
        min_target_xy = float("inf")
        target_initial_z = float(scene["settle_cup_pos"][scene["target_cup"]][2])
        target_initial_xy = np.asarray(
            scene["settle_cup_pos"][scene["target_cup"]][:2], dtype=np.float32
        )
        first_close_step = None
        first_close_xy = None
        first_close_eef_z = None
        close_xy_values = []
        max_close_cup_xy_displacement = 0.0
        gripper_contacted_cups: set[str] = set()
        rollout_steps = 0
        success = False
        final_stats = None
        trace = []
        while rollout_steps < args.max_policy_steps and not success:
            state = _policy_state(shell, env)
            prediction = policy.infer(
                {
                    "state_raw": state,
                    "semantic_memory_raw": memory,
                    "base_rgb": live_base,
                    "wrist_rgb": live_wrist,
                    "prompt": args.prompt,
                    "episode_index": episode,
                    "frame_index": 59 + rollout_steps,
                    "episode_T": 155,
                }
            )
            actions = np.asarray(prediction["actions"], dtype=np.float32)
            if actions.shape != (16, 7) or not np.all(np.isfinite(actions)):
                raise ValueError(f"Invalid action chunk: {actions.shape}")
            inference_ms.append(float(prediction["policy_timing"]["infer_ms"]))
            for chunk_index, command in enumerate(actions[: args.replan_steps]):
                action_low, action_high = env.action_spec
                command = np.clip(command, action_low, action_high)
                env.step(command)
                gripper_contacted_cups.update(
                    shell_main._current_gripper_contacted_cups(env)  # noqa: SLF001
                )
                rollout_steps += 1
                live_base, live_wrist, retries = _current_images(shell, env, ep_args, wrist_camera)
                render_retries += retries
                replay.append(np.array(live_base, order="C", copy=True))
                cups = shell_main._cup_positions(shell, env)  # noqa: SLF001
                eef = np.asarray(shell.get_eef_pos(env), dtype=np.float32)
                distances = {cup: float(np.linalg.norm(eef[:2] - position[:2])) for cup, position in cups.items()}
                nearest = min(distances, key=distances.get)
                min_target_xy = min(min_target_xy, distances[scene["target_cup"]])
                target_lift = float(cups[scene["target_cup"]][2] - target_initial_z)
                max_target_lift = max(max_target_lift, target_lift)
                if float(command[6]) > 0.0:
                    close_xy = distances[scene["target_cup"]]
                    close_xy_values.append(close_xy)
                    cup_displacement = float(
                        np.linalg.norm(cups[scene["target_cup"]][:2] - target_initial_xy)
                    )
                    max_close_cup_xy_displacement = max(
                        max_close_cup_xy_displacement, cup_displacement
                    )
                    if first_close_step is None:
                        first_close_step = rollout_steps
                        first_close_xy = close_xy
                        first_close_eef_z = float(eef[2])
                selection_end = args.selection_skip + args.selection_window
                if args.selection_skip <= rollout_steps - 1 < selection_end and distances[nearest] <= args.selection_radius:
                    votes[nearest] += 1
                    distance_sums[nearest] += distances[nearest]
                success, final_stats = shell_main._success(  # noqa: SLF001
                    shell,
                    env,
                    scene["target_cup"],
                    scene["settle_cup_pos"],
                    args.lift_success_height,
                )
                if rollout_steps % 5 == 0 or success:
                    trace.append(
                        {
                            "step": rollout_steps,
                            "chunk_index": chunk_index,
                            "eef": eef.astype(float).tolist(),
                            "command": np.asarray(command, dtype=float).tolist(),
                            "target_xy_m": distances[scene["target_cup"]],
                            "target_lift_m": target_lift,
                        }
                    )
                if success or rollout_steps >= args.max_policy_steps:
                    break
        selected = _selected_cup(votes, distance_sums)
        video_path = None
        if not args.no_videos:
            video_path = args.video_dir / f"episode_{episode:06d}_{'success' if success else 'failure'}.mp4"
            _write_video(video_path, replay, args.video_fps)
        return {
            "episode": episode,
            "qwen_initial_slot": qwen["initial_slot"],
            "qwen_pairs": qwen["pairs"],
            "qwen_event_sequence": qwen["event_sequence"],
            "qwen_predicted_final_slot": _apply_sequence(qwen["initial_slot"], qwen["pairs"]),
            "gt_final_slot_scoring_only": str(metadata["final_ball_cup"]),
            "qwen_final_slot_correct": _apply_sequence(qwen["initial_slot"], qwen["pairs"])
            == str(metadata["final_ball_cup"]),
            "target_cup_identity_scoring_only": scene["target_cup"],
            "selected_cup_identity": selected,
            "cup_selection_correct": selected == scene["target_cup"],
            "gripper_contacted_cups": sorted(gripper_contacted_cups),
            "any_cup_contact": bool(gripper_contacted_cups),
            "target_cup_contact": scene["target_cup"] in gripper_contacted_cups,
            "selected_cup_contact": selected in gripper_contacted_cups,
            "correct_selection_and_contact": bool(
                selected == scene["target_cup"]
                and scene["target_cup"] in gripper_contacted_cups
            ),
            "success": bool(success),
            "rollout_steps": rollout_steps,
            "selection_votes": votes,
            "min_target_xy_m": min_target_xy,
            "max_target_lift_m": max_target_lift,
            "first_close_step": first_close_step,
            "first_close_target_xy_m": first_close_xy,
            "first_close_eef_z_m": first_close_eef_z,
            "mean_close_target_xy_m": (
                float(np.mean(close_xy_values)) if close_xy_values else None
            ),
            "max_close_target_xy_m": max(close_xy_values) if close_xy_values else None,
            "max_close_cup_xy_displacement_m": max_close_cup_xy_displacement,
            "final_stats": final_stats,
            "mean_inference_ms": float(np.mean(inference_ms)) if inference_ms else None,
            "render_frame59_mae_uint8": render_mae,
            "egl_corrupt_read_retries": render_retries,
            "video": str(video_path.resolve()) if video_path is not None else None,
            "trace": trace,
        }
    finally:
        env.close()


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    return {
        "episodes": count,
        "qwen_final_slot_correct": sum(bool(row["qwen_final_slot_correct"]) for row in records),
        "qwen_final_slot_accuracy": float(np.mean([row["qwen_final_slot_correct"] for row in records])) if count else None,
        "cup_selection_correct": sum(bool(row["cup_selection_correct"]) for row in records),
        "cup_selection_accuracy": float(np.mean([row["cup_selection_correct"] for row in records])) if count else None,
        "lift_successes": sum(bool(row["success"]) for row in records),
        "lift_success_rate": float(np.mean([row["success"] for row in records])) if count else None,
        "mean_min_target_xy_m": float(np.mean([row["min_target_xy_m"] for row in records])) if count else None,
        "mean_max_target_lift_m": float(np.mean([row["max_target_lift_m"] for row in records])) if count else None,
        "mean_policy_inference_ms": float(np.mean([row["mean_inference_ms"] for row in records])) if count else None,
    }


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.raw_root = args.raw_root.expanduser().resolve()
    args.initial_cache = args.initial_cache.expanduser().resolve()
    args.trigger_summary = args.trigger_summary.expanduser().resolve()
    args.memory_bank = args.memory_bank.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.video_dir = args.video_dir.expanduser().resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite")
    if args.replan_steps < 1 or args.replan_steps > 16 or args.max_policy_steps < 1:
        raise ValueError("Invalid replanning/rollout settings")
    episodes = [int(value.strip()) for value in args.episodes.split(",") if value.strip()]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    qwen_by_episode = _qwen_sequences(args.initial_cache, args.trigger_summary)
    missing = [episode for episode in episodes if episode not in qwen_by_episode]
    if missing:
        raise ValueError(f"Missing complete Qwen event sequences for episodes: {missing}")
    memory_lookup_mode, memory_lookup = _memory_lookup(args.memory_bank)
    if memory_lookup_mode == "event_sequence":
        missing_memories = [
            (episode, qwen_by_episode[episode]["event_sequence"])
            for episode in episodes
            if tuple(qwen_by_episode[episode]["event_sequence"]) not in memory_lookup
        ]
    else:
        missing_memories = [episode for episode in episodes if episode not in memory_lookup]
    if missing_memories:
        raise ValueError(f"Missing recurrent memories: {missing_memories}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if args.remote_policy:
        from openpi_client import websocket_client_policy

        logging.info("Connecting to remote action policy at %s:%d", args.host, args.port)
        policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    else:
        logging.info("Loading action policy from %s", args.checkpoint)
        policy = _load_policy(args.checkpoint, args.num_sampling_steps)
    shell = shell_main._import_shellgame_tools(args.robosuite_root)  # noqa: SLF001
    payload = {
        "schema_version": 1,
        "experiment": "Qwen causal sliding events -> recurrent memory -> Pi absolute-EEF7 closed loop",
        "checkpoint": str(args.checkpoint),
        "qwen_initial_cache": str(args.initial_cache),
        "qwen_trigger_summary": str(args.trigger_summary),
        "memory_bank": str(args.memory_bank),
        "memory_lookup_mode": memory_lookup_mode,
        "uses_gt_for_policy_memory": memory_lookup_mode == "episode",
        "memory_causality_note": (
            "Direct-visual cache used simulator initial-slot labels as an exact proxy for the "
            "audited Qwen reveal output; swap and final-slot GT were not model inputs."
            if memory_lookup_mode == "episode"
            else "Qwen reveal and event sequence select a recurrent memory template."
        ),
        "metadata_usage": "episode reconstruction and post-hoc scoring only",
        "control": {
            "action_mode": "absolute_eef7 raw controller command",
            "replan_steps": args.replan_steps,
            "num_sampling_steps": args.num_sampling_steps,
            "max_policy_steps": args.max_policy_steps,
            "remote_policy": bool(args.remote_policy),
            "policy_endpoint": f"{args.host}:{args.port}" if args.remote_policy else None,
        },
        "episode_ids": episodes,
        "episodes": [],
    }
    started = time.monotonic()
    try:
        for ordinal, episode in enumerate(episodes, start=1):
            sequence = tuple(qwen_by_episode[episode]["event_sequence"])
            memory_key = episode if memory_lookup_mode == "episode" else sequence
            record = _run_episode(
                episode,
                policy,
                memory_lookup[memory_key],
                qwen_by_episode[episode],
                shell,
                args,
            )
            payload["episodes"].append(record)
            payload["summary"] = _summary(payload["episodes"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(
                f"[{ordinal}/{len(episodes)}] ep={episode} qwen={record['qwen_final_slot_correct']} "
                f"select={record['cup_selection_correct']} lift={record['success']} "
                f"min_xy={record['min_target_xy_m'] * 1000:.1f}mm "
                f"elapsed={(time.monotonic() - started) / 60:.1f}m",
                flush=True,
            )
    finally:
        if args.remote_policy and hasattr(policy, "_ws"):
            policy._ws.close()  # noqa: SLF001
        del policy
        if not args.remote_policy:
            jax.clear_caches()
        gc.collect()
    payload["summary"] = _summary(payload["episodes"])
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    print(f"output={args.output}", flush=True)


if __name__ == "__main__":
    main()

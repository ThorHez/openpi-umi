#!/usr/bin/env python3
"""Paired PickXtimes closed loop for action-only and predicted-memory chunks."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from typing import Any

WORKSPACE = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / "robomme" / "src"))
sys.path.append(str(WORKSPACE / "robomme" / ".venv" / "lib" / "python3.11" / "site-packages"))

import flax
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip_mem_semantic as memory_core
from openpi.models import tokenizer as _tokenizer
from openpi.tasks.robomme.pickxtimes import eef_action_adapter
from openpi.tasks.robomme.pickxtimes import eef_action_chunk_adapter
from openpi.tasks.robomme.pickxtimes import semantic_memory_event
from openpi.training.mem import robomme_pickxtimes_action_chunk_dataset as chunk_data
from openpi.training.mem import robomme_pickxtimes_action_dataset as action_data
from openpi.training.mem.recipes import shellgame_semantic_action

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.mem import cache_pickxtimes_siglip_features as feature_cache
from scripts.mem import eval_pickxtimes_causal_event_memory as memory_eval
from scripts.mem import robomme_fixed_chunk_inference as fixed_memory

DEFAULT_BACKBONE = feature_cache.DEFAULT_CHECKPOINT
DEFAULT_MEMORY_RUN = pathlib.Path(
    "evaluation/robomme/pickxtimes_event_memory_split260827/"
    "goal_gated_press_residual_lr1e3_seed260902_200"
)
DEFAULT_MEMORY_CHECKPOINT = DEFAULT_MEMORY_RUN / "checkpoints/step_25.msgpack"
DEFAULT_MEMORY_EVAL = pathlib.Path(
    "evaluation/robomme/pickxtimes_event_memory_split260827/"
    "goal_gated_press_step25_train70_dev15_calibrated.json"
)
DEFAULT_SPLIT = pathlib.Path(
    "data/robomme_extracted/pickxtimes_split_seed260827_train70_dev15_test15.json"
)
CONTROLS = ("action_only", "initial_memory", "predicted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-only-run", type=pathlib.Path, required=True)
    parser.add_argument("--predicted-run", type=pathlib.Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=pathlib.Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--memory-run", type=pathlib.Path, default=DEFAULT_MEMORY_RUN)
    parser.add_argument("--memory-checkpoint", type=pathlib.Path, default=DEFAULT_MEMORY_CHECKPOINT)
    parser.add_argument("--memory-eval", type=pathlib.Path, default=DEFAULT_MEMORY_EVAL)
    parser.add_argument(
        "--fixed-chunk-memory-training",
        type=pathlib.Path,
        help="Use a trigger-free fixed-chunk student instead of the legacy event detector/tracker.",
    )
    parser.add_argument(
        "--controls",
        default=",".join(CONTROLS),
        help=f"Comma-separated subset of {CONTROLS}",
    )
    parser.add_argument("--split", type=pathlib.Path, default=DEFAULT_SPLIT)
    parser.add_argument("--episodes")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--execute-steps", type=int, default=2)
    parser.add_argument("--max-position-step", type=float, default=0.02)
    parser.add_argument("--trace-stride", type=int, default=10)
    parser.add_argument(
        "--fixed-token-cache-dir",
        type=pathlib.Path,
        help="Optionally save online pooled 4x4 tokens for no-progress MEM hard negatives.",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _as_bool(value) -> bool:
    array = _as_numpy(value)
    return bool(array.reshape(-1)[0])


def _latest(observation: dict[str, Any], key: str) -> np.ndarray:
    value = observation[key]
    if isinstance(value, list):
        value = value[-1]
    return _as_numpy(value)


def _parse_goal(prompt: str) -> tuple[str, int]:
    color_match = re.search(r"\b(red|green|blue)\b", prompt.lower())
    count_match = re.search(r"\b(one|two|three|four|five|[1-5])\s+times?\b", prompt.lower())
    if color_match is None:
        raise ValueError(f"Could not parse PickXtimes goal: {prompt!r}")
    # RoboMME omits the repetition phrase for the one-cycle form.
    if count_match is None:
        return color_match.group(1), 1
    counts = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    raw_count = count_match.group(1)
    return color_match.group(1), counts.get(raw_count, int(raw_count) if raw_count.isdigit() else 0)


def _parse_episode_list(value: str | None, split_path: pathlib.Path, max_episodes: int | None) -> list[int]:
    if value:
        episodes = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        episodes = [int(item) for item in payload["val_episode_indices"]]
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise ValueError("No rollout episodes selected")
    return episodes


def _load_action(run_dir: pathlib.Path):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    checkpoint = run_dir / "checkpoints/best.msgpack"
    stats = action_data.ActionNormalization.from_json(config["normalization"])
    spatial_visual_tokens = int(config.get("spatial_visual_tokens", 0))
    model = eef_action_chunk_adapter.PickXtimesEEFActionChunkAdapter(
        action_horizon=int(config["action_horizon"]),
        hidden_width=int(config["hidden_width"]),
        depth=int(config["depth"]),
        memory_query_tokens=int(config["memory_query_tokens"]),
        use_memory=config["memory_mode"] != "action_only",
        spatial_visual_tokens=spatial_visual_tokens,
    )
    variables = model.init(
        jax.random.key(0),
        jnp.zeros(
            (1, spatial_visual_tokens, eef_action_adapter.VISUAL_FEATURE_DIM)
            if spatial_visual_tokens
            else (1, eef_action_adapter.VISUAL_FEATURE_DIM),
            dtype=jnp.float16,
        ),
        jnp.zeros((1, eef_action_adapter.ROBOT_GOAL_DIM), dtype=jnp.float32),
        jnp.zeros((1, eef_action_adapter.MEMORY_TOKENS, eef_action_adapter.MEMORY_WIDTH), dtype=jnp.float16),
        train=False,
    )
    params = flax.serialization.from_bytes(variables["params"], checkpoint.read_bytes())

    @jax.jit
    def predict(visual, robot_goal, memory):
        return model.apply({"params": params}, visual, robot_goal, memory, train=False)

    return config, summary, checkpoint, stats, predict


def _load_online_memory(run_dir: pathlib.Path, checkpoint: pathlib.Path, memory_eval_path: pathlib.Path):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tracker = memory_eval.make_tracker(config)
    params = memory_eval.initialize_and_restore(tracker, checkpoint)
    classifier = semantic_memory_event.PickXtimesSlidingWindowEventClassifier(
        input_width=tracker.input_width,
        width=tracker.encoder_width,
        depth=tracker.encoder_depth,
        num_heads=tracker.encoder_heads,
        dtype_mm=tracker.dtype_mm,
    )
    fusion = semantic_memory_event.PickXtimesGripperTypeFusion()
    gate = semantic_memory_event.PickXtimesGripperGateFusion()
    press = semantic_memory_event.PickXtimesPressGateFusion()

    @jax.jit
    def classify(window, gripper):
        pooled = memory_core.pool_fixed_grid(window, pool_factor=2)
        event_logits, type_logits, features = classifier.apply(
            {"params": params["window_classifier"]}, pooled, train=False, return_features=True
        )
        event_logits = gate.apply({"params": params["gripper_gate_fusion"]}, event_logits, gripper)
        type_logits = fusion.apply({"params": params["gripper_type_fusion"]}, type_logits, gripper)
        press_logits = press.apply({"params": params["press_gate_fusion"]}, event_logits, features)
        return event_logits, type_logits, press_logits

    @jax.jit
    def apply_tracker(windows, gripper, prompt_tokens, prompt_mask, event_types, sequence_mask):
        return tracker.apply(
            {"params": params},
            windows,
            gripper,
            prompt_tokens,
            prompt_mask,
            jnp.arange(semantic_memory_event.MAX_EVENTS, dtype=jnp.int32)[None],
            sequence_mask,
            causal_selection=False,
            candidate_valid_mask=sequence_mask,
            sequence_event_types=event_types,
            train=False,
        )

    calibrated = json.loads(memory_eval_path.read_text(encoding="utf-8"))["calibrated_threshold"]
    return classify, apply_tracker, calibrated["transition_config"]


def _memory_from_events(
    apply_tracker,
    patch_history: list[np.ndarray],
    gripper_history: list[bool],
    prompt_tokens: np.ndarray,
    prompt_mask: np.ndarray,
    starts: list[int],
    event_types: list[int],
) -> np.ndarray:
    count = min(len(starts), semantic_memory_event.MAX_EVENTS)
    selected_starts = starts[:count]
    selected_types = event_types[:count]
    padding_start = selected_starts[0] if selected_starts else 0
    padding_type = selected_types[0] if selected_types else semantic_memory_event.PICK_COMPLETE
    padded_starts = selected_starts + [padding_start] * (semantic_memory_event.MAX_EVENTS - count)
    padded_types = selected_types + [padding_type] * (semantic_memory_event.MAX_EVENTS - count)
    if len(patch_history) >= semantic_memory_event.WINDOW_SIZE:
        patch_array = np.asarray(patch_history)
        gripper_array = np.asarray(gripper_history, dtype=np.bool_)
        windows = np.stack(
            [patch_array[start : start + semantic_memory_event.WINDOW_SIZE] for start in padded_starts]
        )
        gripper = np.stack(
            [gripper_array[start : start + semantic_memory_event.WINDOW_SIZE] for start in padded_starts]
        )
    else:
        windows = np.zeros(
            (
                semantic_memory_event.MAX_EVENTS,
                semantic_memory_event.WINDOW_SIZE,
                256,
                eef_action_adapter.VISUAL_FEATURE_DIM,
            ),
            dtype=np.float16,
        )
        gripper = np.zeros((semantic_memory_event.MAX_EVENTS, semantic_memory_event.WINDOW_SIZE), dtype=np.bool_)
    sequence_mask = np.arange(semantic_memory_event.MAX_EVENTS) < count
    outputs = jax.device_get(
        apply_tracker(
            jnp.asarray(windows)[None],
            jnp.asarray(gripper)[None],
            jnp.asarray(prompt_tokens)[None],
            jnp.asarray(prompt_mask)[None],
            jnp.asarray(padded_types, dtype=jnp.int32)[None],
            jnp.asarray(sequence_mask)[None],
        )
    )
    return np.asarray(outputs["memory"][0], dtype=np.float16)


def _write(path: pathlib.Path, result: dict[str, Any]) -> None:
    for control_result in result["controls"].values():
        episodes = control_result["episodes"]
        successes = sum(record.get("success", False) for record in episodes)
        control_result["summary"] = {
            "attempted": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes) if episodes else None,
            "errors": sum(record.get("status") == "error" for record in episodes),
            "mean_rollout_steps": float(np.mean([record.get("rollout_steps", 0) for record in episodes]))
            if episodes
            else None,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if min(args.max_steps, args.execute_steps, args.trace_stride) < 1 or args.max_position_step <= 0:
        raise ValueError("Rollout counts and max position step must be positive")
    controls = [item.strip() for item in args.controls.split(",") if item.strip()]
    if not controls or set(controls) - set(CONTROLS):
        raise ValueError(f"--controls must be a comma-separated subset of {CONTROLS}")
    controls = list(dict.fromkeys(controls))
    episodes = _parse_episode_list(args.episodes, args.split, args.max_episodes)
    run_dirs = {
        "action_only": args.action_only_run,
        "initial_memory": args.predicted_run,
        "predicted": args.predicted_run,
    }
    action_models = {control: _load_action(run_dirs[control]) for control in controls}
    horizons = {int(value[0]["action_horizon"]) for value in action_models.values()}
    if len(horizons) != 1:
        raise ValueError("Action controls have different horizons")

    backbone = feature_cache.load_backbone(args.backbone_checkpoint)
    tokenizer = _tokenizer.PaligemmaTokenizer(shellgame_semantic_action.MODEL_CONFIG.max_token_len)
    fixed_predictor = None
    if args.fixed_chunk_memory_training is not None:
        fixed_predictor = fixed_memory.FixedChunkMemoryPredictor(
            args.fixed_chunk_memory_training
        )
        classify = apply_tracker = None
        transition_config = None
    else:
        classify, apply_tracker, transition_config = _load_online_memory(
            args.memory_run, args.memory_checkpoint, args.memory_eval
        )

    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id="PickXtimes",
        dataset="train",
        action_space="ee_pose",
        gui_render=False,
        max_steps=args.max_steps,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "train",
        "episode_indices": episodes,
        "fixed_dev_split": True,
        "frozen_test_accessed": False,
        "action_horizon": next(iter(horizons)),
        "execute_steps": args.execute_steps,
        "max_steps": args.max_steps,
        "max_position_step": args.max_position_step,
        "transition_config": transition_config,
        "memory_mode": (
            "fixed_non_overlapping_12_frame_chunks"
            if fixed_predictor is not None
            else "legacy_sliding_window_event_trigger"
        ),
        "memory_training_dir": (
            str(args.fixed_chunk_memory_training.resolve())
            if args.fixed_chunk_memory_training is not None
            else str(args.memory_run.resolve())
        ),
        "controls": {control: {"episodes": []} for control in controls},
    }
    started = time.monotonic()
    for ordinal, episode_index in enumerate(episodes, start=1):
        for control in controls:
            env = None
            trace = []
            events_trace = []
            try:
                env = builder.make_env_for_episode(episode_index)
                observation, info = env.reset()
                prompt = str(info["task_goal"][0])
                target_color, required_count = _parse_goal(prompt)
                token_ids, prompt_mask = tokenizer.tokenize(prompt)
                prompt_tokens = np.asarray(
                    feature_cache.embed_prompts(backbone, jnp.asarray(token_ids, dtype=jnp.int32)[None])
                )[0]
                patch_history: list[np.ndarray] = []
                gripper_history: list[bool] = []
                event_logits: list[float] = []
                type_logits: list[np.ndarray] = []
                press_logits: list[float] = []
                accepted_starts: list[int] = []
                accepted_types: list[int] = []
                fixed_tokens: list[np.ndarray] = []
                fixed_chunk_count = -1
                memory_state_updates: list[dict[str, Any]] = []

                def observe(
                    current_observation,
                    patch_history=patch_history,
                    gripper_history=gripper_history,
                    event_logits=event_logits,
                    type_logits=type_logits,
                    press_logits=press_logits,
                    backbone=backbone,
                    classify=classify,
                ):
                    image = _latest(current_observation, "front_rgb_list").astype(np.uint8)
                    patch = np.asarray(feature_cache.encode_images(backbone, jnp.asarray(image)[None]))[0]
                    patch_history.append(patch.astype(np.float16))
                    if fixed_predictor is not None:
                        pooled = np.asarray(
                            memory_core.pool_fixed_grid(
                                jnp.asarray(patch)[None, None], pool_factor=4
                            )
                        )[0, 0]
                        fixed_tokens.append(pooled.astype(np.float16))
                    gripper_sum = float(np.sum(_latest(current_observation, "gripper_state_list")))
                    gripper_history.append(gripper_sum < 0.058)
                    if fixed_predictor is None and len(patch_history) >= semantic_memory_event.WINDOW_SIZE:
                        window = np.asarray(patch_history[-semantic_memory_event.WINDOW_SIZE :])[None]
                        gripper_window = np.asarray(
                            gripper_history[-semantic_memory_event.WINDOW_SIZE :], dtype=np.bool_
                        )[None]
                        event, event_type, press = jax.device_get(
                            classify(jnp.asarray(window), jnp.asarray(gripper_window))
                        )
                        event_logits.append(float(event[0]))
                        type_logits.append(np.asarray(event_type[0]))
                        press_logits.append(float(press[0]))

                observe(observation)
                if fixed_predictor is None:
                    memory = _memory_from_events(
                        apply_tracker,
                        patch_history,
                        gripper_history,
                        prompt_tokens,
                        np.asarray(prompt_mask),
                        accepted_starts,
                        accepted_types,
                    )
                else:
                    initial = fixed_predictor.predict_encoded(
                        np.zeros(
                            (0, fixed_memory.CHUNK_FRAMES, fixed_memory.SPATIAL_TOKENS, fixed_memory.PATCH_WIDTH),
                            dtype=np.float16,
                        ),
                        task_id=3,
                        goal_color_ids=(fixed_memory.COLOR_IDS[target_color], 0),
                        required_count=required_count,
                        queried_ordinal=0,
                        num_regions=0,
                    )
                    memory = initial["all_memories"][-1].astype(np.float16)
                    fixed_chunk_count = 0
                status = "ongoing"
                rollout_step = 0
                final_info = info
                while rollout_step < args.max_steps and status == "ongoing":
                    if control == "predicted" and fixed_predictor is not None:
                        complete_chunks = len(fixed_tokens) // fixed_memory.CHUNK_FRAMES
                        if complete_chunks != fixed_chunk_count:
                            chunk_array = np.asarray(
                                fixed_tokens[: complete_chunks * fixed_memory.CHUNK_FRAMES],
                                dtype=np.float16,
                            ).reshape(
                                complete_chunks,
                                fixed_memory.CHUNK_FRAMES,
                                fixed_memory.SPATIAL_TOKENS,
                                fixed_memory.PATCH_WIDTH,
                            )
                            fixed_output = fixed_predictor.predict_encoded(
                                chunk_array,
                                task_id=3,
                                goal_color_ids=(fixed_memory.COLOR_IDS[target_color], 0),
                                required_count=required_count,
                                queried_ordinal=0,
                                num_regions=0,
                            )
                            memory = fixed_output["all_memories"][-1].astype(np.float16)
                            prediction = fixed_output["all_predictions"][-1]
                            decoded = {
                                "completed_count": int(
                                    prediction[fixed_memory.field_index("completed_count")]
                                ),
                                "holding": int(prediction[fixed_memory.field_index("holding")]),
                                "ready_to_press": int(
                                    prediction[fixed_memory.field_index("ready_to_press")]
                                ),
                                "done": int(prediction[fixed_memory.field_index("done")]),
                            }
                            memory_state_updates.append(
                                {
                                    "rollout_step": rollout_step,
                                    "chunk_count": complete_chunks,
                                    "write_gate": float(fixed_output["write_gates"][-1]),
                                    **decoded,
                                }
                            )
                            fixed_chunk_count = complete_chunks
                    elif control == "predicted" and event_logits:
                        starts, types = semantic_memory_event.transition_grammar_events(
                            np.asarray(gripper_history, dtype=np.bool_),
                            np.asarray(event_logits, dtype=np.float32),
                            np.asarray(type_logits, dtype=np.float32),
                            np.asarray(press_logits, dtype=np.float32),
                            required_count=required_count,
                            **transition_config,
                        )
                        if starts != accepted_starts or types != accepted_types:
                            accepted_starts, accepted_types = starts, types
                            memory = _memory_from_events(
                                apply_tracker,
                                patch_history,
                                gripper_history,
                                prompt_tokens,
                                np.asarray(prompt_mask),
                                accepted_starts,
                                accepted_types,
                            )
                            events_trace.append(
                                {"rollout_step": rollout_step, "starts": starts, "types": types}
                            )

                    action_config, _, _, stats, predict = action_models[control]
                    robot_goal = action_data.build_robot_goal(
                        _latest(observation, "eef_state_list"),
                        _latest(observation, "gripper_state_list"),
                        _latest(observation, "joint_state_list"),
                        target_color=target_color,
                        required_count=required_count,
                    )
                    robot_goal = (robot_goal - stats.robot_goal_mean) / stats.robot_goal_std
                    current_patch = np.asarray(patch_history[-1], dtype=np.float32)
                    if int(action_config.get("spatial_visual_tokens", 0)):
                        visual = chunk_data.pool_spatial_patch_tokens(current_patch[None])[0]
                    else:
                        visual = current_patch.mean(axis=0)
                    model_memory = (
                        memory if control in {"initial_memory", "predicted"} else np.zeros_like(memory)
                    )
                    outputs = jax.device_get(
                        predict(
                            jnp.asarray(visual)[None],
                            jnp.asarray(robot_goal)[None],
                            jnp.asarray(model_memory)[None],
                        )
                    )
                    poses = np.asarray(outputs["normalized_poses"][0]) * stats.pose_std + stats.pose_mean
                    close_probabilities = np.asarray(jax.nn.sigmoid(outputs["close_logits"][0]))
                    phase = int(np.argmax(np.asarray(outputs["phase_logits"][0])))
                    for chunk_index in range(min(args.execute_steps, int(action_config["action_horizon"]))):
                        current_eef = _latest(observation, "eef_state_list").astype(np.float64)
                        command_pose = poses[chunk_index].astype(np.float64).copy()
                        position_delta = command_pose[:3] - current_eef[:3]
                        position_norm = float(np.linalg.norm(position_delta))
                        if position_norm > args.max_position_step:
                            position_delta *= args.max_position_step / position_norm
                        command_pose[:3] = current_eef[:3] + position_delta
                        command_pose[:3] = np.clip(
                            command_pose[:3], [-0.45, -0.45, 0.045], [0.45, 0.45, 0.65]
                        )
                        # Dataset actions often encode the same top-down roll
                        # as +pi while online EEF state reports -pi. Choose the
                        # equivalent angle nearest the current state so the IK
                        # wrapper never sees an artificial 2*pi jump.
                        rotation_delta = (command_pose[3:] - current_eef[3:] + np.pi) % (2.0 * np.pi) - np.pi
                        command_pose[3:] = current_eef[3:] + rotation_delta
                        command = np.concatenate(
                            (command_pose, [-1.0 if close_probabilities[chunk_index] >= 0.5 else 1.0])
                        )
                        next_observation, _, terminated, truncated, step_info = env.step(command)
                        rollout_step += 1
                        final_info = step_info
                        status = str(step_info.get("status", "ongoing"))
                        if next_observation is not None and "front_rgb_list" in next_observation:
                            observation = next_observation
                            observe(observation)
                        terminated_flag = _as_bool(terminated)
                        truncated_flag = _as_bool(truncated)
                        if status == "ongoing" and truncated_flag:
                            status = "timeout"
                        elif status == "ongoing" and terminated_flag:
                            status = "terminated"
                        if rollout_step % args.trace_stride == 0 or status != "ongoing":
                            trace.append(
                                {
                                    "step": rollout_step,
                                    "status": status,
                                    "eef": _latest(observation, "eef_state_list").astype(float).tolist(),
                                    "command": command.tolist(),
                                    "close_probability": float(close_probabilities[chunk_index]),
                                    "predicted_phase": phase,
                                    "accepted_event_types": list(accepted_types),
                                    "oracle_subgoal": str(step_info.get("simple_subgoal_online", "")),
                                }
                            )
                        if status != "ongoing" or terminated_flag or truncated_flag:
                            break
                record = {
                    "episode_index": episode_index,
                    "prompt": prompt,
                    "target_color": target_color,
                    "required_count": required_count,
                    "status": status,
                    "success": status == "success",
                    "rollout_steps": rollout_step,
                    "accepted_event_starts": accepted_starts,
                    "accepted_event_types": accepted_types,
                    "event_updates": events_trace,
                    "memory_state_updates": memory_state_updates,
                    "final_memory_state": (
                        memory_state_updates[-1] if memory_state_updates else None
                    ),
                    "final_oracle_subgoal": str(final_info.get("simple_subgoal_online", "")),
                    "trace": trace,
                }
                if args.fixed_token_cache_dir is not None and fixed_predictor is not None:
                    cache_dir = args.fixed_token_cache_dir.expanduser().resolve() / control
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    token_path = cache_dir / f"episode_{episode_index:03d}.npz"
                    np.savez_compressed(
                        token_path,
                        patch_tokens=np.asarray(fixed_tokens, dtype=np.float16),
                        episode_index=np.asarray(episode_index, dtype=np.int32),
                        target_color=np.asarray(target_color),
                        required_count=np.asarray(required_count, dtype=np.int32),
                        rollout_steps=np.asarray(rollout_step, dtype=np.int32),
                    )
                    record["fixed_token_cache"] = str(token_path)
                if final_info.get("error_message"):
                    record["error"] = str(final_info["error_message"])
            except Exception as exc:
                record = {
                    "episode_index": episode_index,
                    "status": "error",
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": trace,
                }
            finally:
                if env is not None:
                    env.close()
            result["controls"][control]["episodes"].append(record)
            _write(args.output, result)
        statuses = ", ".join(
            f"{control}={result['controls'][control]['episodes'][-1]['status']}" for control in controls
        )
        print(
            f"[{ordinal}/{len(episodes)}] episode={episode_index}: {statuses} "
            f"elapsed={(time.monotonic() - started) / 60:.1f}m",
            flush=True,
        )
    _write(args.output, result)
    print(json.dumps({control: result["controls"][control]["summary"] for control in controls}, indent=2))


if __name__ == "__main__":
    main()

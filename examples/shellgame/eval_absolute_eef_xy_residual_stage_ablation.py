"""Stage-wise oracle XY residual ablation for fixed-history absolute EEF7.

This diagnostic preserves the policy history, selected cup, predicted Z,
orientation, and gripper command.  Once the policy has expressed the correct
cup selection and starts descending, only the commanded world-frame XY is
optionally replaced by the current target-cup center.

Each condition / episode pair runs in a fresh MuJoCo/EGL process to avoid the
long-rollout renderer aliasing seen in earlier evaluations.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import pathlib

import main as base
import main_absolute_eef_fixed_history as fixed_eef
import numpy as np
import tyro

CUP_NAMES = ("left", "middle", "right")
VALID_MODES = ("none", "once", "until_close", "through_close")
NOISE_SEED_KEY = "__openpi_deterministic_noise_seed__"


@dataclasses.dataclass
class Args(base.Args):
    num_trials: int = 20
    seed: int = 260813
    video_out_path: str = "evaluation/shellgame/absolute_eef_xy_residual_stage_ablation"
    physics_debug: bool = True
    physics_debug_window: int = 150
    websocket_reconnect_interval: int = 4

    episode_indices: str = "0,1,3,15,16,17"
    xy_residual_modes: str = ",".join(VALID_MODES)
    xy_residual_activation_radius: float = 0.06
    xy_residual_descent_epsilon: float = 0.0005
    xy_residual_close_threshold: float = 0.0
    xy_residual_close_hold_steps: int = 10
    deterministic_sample_salt: int = 260815


_base_policy_env_action = base._policy_env_action  # noqa: SLF001
_base_run_scripted_observation = base._run_scripted_observation  # noqa: SLF001
_base_physics_snapshot = base._physics_snapshot  # noqa: SLF001
_base_append_observation = base._append_observation  # noqa: SLF001


def _append_observation_orientation_stable(*args, **kwargs):
    """Undo the intermittent 180-degree EGL readback flip before inference.

    A corrupted readback is separated from the previous frame by ~68 mean
    pixel levels, while rotating it by 180 degrees restores sub-4 continuity.
    The large margin keeps normal robot / object motion untouched.
    """
    _base_append_observation(*args, **kwargs)
    env = args[1]
    history = args[4]
    replay = args[5]
    if len(history) < 2:
        return
    corrected_keys = []
    for key in ("base", "wrist"):
        previous = np.asarray(history[-2][key], dtype=np.int16)
        current = np.asarray(history[-1][key], dtype=np.uint8)
        rotated = np.ascontiguousarray(np.rot90(current, 2))
        direct_delta = float(np.mean(np.abs(current.astype(np.int16) - previous)))
        rotated_delta = float(np.mean(np.abs(rotated.astype(np.int16) - previous)))
        if rotated_delta + 20.0 < direct_delta:
            history[-1][key] = rotated
            corrected_keys.append(key)
            if key == "base":
                replay[-1] = np.ascontiguousarray(np.rot90(replay[-1], 2))
    if corrected_keys:
        counts = getattr(env, "_orientation_stabilization_counts", {"base": 0, "wrist": 0})
        for key in corrected_keys:
            counts[key] += 1
        env._orientation_stabilization_counts = counts  # noqa: SLF001
        if counts["base"] + counts["wrist"] <= 4:
            logging.warning("Corrected intermittent 180-degree readback flip keys=%s", corrected_keys)


def _run_scripted_observation_with_target(*args, **kwargs):
    meta = _base_run_scripted_observation(*args, **kwargs)
    env = args[1]
    env._xy_residual_target_cup = meta["target_cup"]  # noqa: SLF001
    env._xy_residual_state = {  # noqa: SLF001
        "policy_step": -1,
        "activated": False,
        "done": False,
        "correction_count": 0,
        "close_started": False,
        "close_steps_remaining": 0,
        "last_event": None,
    }
    return meta


def _xy_residual_policy_env_action(
    shell,
    env,
    history,
    start_eef_pos,
    start_eef_quat,
    action_plan,
    gripper_action,
    *,
    client,
    args: Args,
    prompt=None,
):
    env_action, next_gripper, policy_action = _base_policy_env_action(
        shell,
        env,
        history,
        start_eef_pos,
        start_eef_quat,
        action_plan,
        gripper_action,
        client=client,
        args=args,
        prompt=prompt,
    )
    target_cup = getattr(env, "_xy_residual_target_cup", None)
    state = getattr(env, "_xy_residual_state", None)
    if target_cup is None or state is None or args.action_mode != "raw7" or policy_action is None:
        return env_action, next_gripper, policy_action

    mode = getattr(args, "_active_xy_residual_mode")
    state["policy_step"] += 1
    measured = np.asarray(shell.get_eef_pos(env), dtype=np.float32)
    raw = np.asarray(env_action, dtype=np.float32).copy()
    corrected = raw.copy()
    cup_positions = base._cup_positions(shell, env)  # noqa: SLF001
    target_xy = np.asarray(cup_positions[target_cup][:2], dtype=np.float32)
    distances = {
        cup: float(np.linalg.norm(raw[:2] - np.asarray(pos[:2], dtype=np.float32)))
        for cup, pos in cup_positions.items()
    }
    predicted_cup = min(distances, key=distances.get)
    selection_expressed = (
        predicted_cup == target_cup
        and distances[predicted_cup] <= args.xy_residual_activation_radius
    )
    descending = float(raw[2]) < float(measured[2]) - args.xy_residual_descent_epsilon
    close_command = float(raw[-1]) > args.xy_residual_close_threshold

    activated_now = False
    if not state["activated"] and selection_expressed and descending:
        state["activated"] = True
        activated_now = True
        logging.info(
            "XY residual activated mode=%s step=%d target=%s raw_error=%.1fmm",
            mode,
            state["policy_step"],
            target_cup,
            distances[target_cup] * 1_000.0,
        )

    apply_correction = False
    if state["activated"] and not state["done"]:
        if mode == "none":
            state["done"] = True
        elif mode == "once":
            apply_correction = state["correction_count"] == 0
            state["done"] = True
        elif mode == "until_close":
            if close_command:
                state["done"] = True
            else:
                apply_correction = True
        elif mode == "through_close":
            if close_command and not state["close_started"]:
                state["close_started"] = True
                state["close_steps_remaining"] = args.xy_residual_close_hold_steps
            apply_correction = True
            if state["close_started"]:
                state["close_steps_remaining"] -= 1
                if state["close_steps_remaining"] <= 0:
                    state["done"] = True
        else:  # validated in main; keep child failures explicit
            raise ValueError(f"Unknown XY residual mode: {mode}")

    raw_error = float(np.linalg.norm(raw[:2] - target_xy))
    if apply_correction:
        corrected[:2] = target_xy
        state["correction_count"] += 1

    state["last_event"] = {
        "mode": mode,
        "policy_step": int(state["policy_step"]),
        "target_cup": target_cup,
        "predicted_cup": predicted_cup,
        "selection_expressed": bool(selection_expressed),
        "descending": bool(descending),
        "close_command": bool(close_command),
        "activated": bool(state["activated"]),
        "activated_now": bool(activated_now),
        "done": bool(state["done"]),
        "applied": bool(apply_correction),
        "correction_count": int(state["correction_count"]),
        "raw_command": raw.tolist(),
        "corrected_command": corrected.tolist(),
        "target_cup_pos": np.asarray(cup_positions[target_cup], dtype=np.float32).tolist(),
        "measured_eef_pos": measured.tolist(),
        "raw_xy_error_mm": raw_error * 1_000.0,
        "measured_xy_error_mm": float(np.linalg.norm(measured[:2] - target_xy)) * 1_000.0,
    }
    return corrected, next_gripper, policy_action


def _physics_snapshot_with_xy_residual(*args, **kwargs):
    payload = _base_physics_snapshot(*args, **kwargs)
    env = args[1]
    state = getattr(env, "_xy_residual_state", None)
    if state is not None and state["last_event"] is not None:
        payload["xy_residual"] = state["last_event"]
        payload["orientation_stabilization_counts"] = getattr(
            env, "_orientation_stabilization_counts", {"base": 0, "wrist": 0}
        )
    return payload


def _episode_specs(args: Args) -> list[dict]:
    requested = sorted({int(item.strip()) for item in args.episode_indices.split(",") if item.strip()})
    if not requested or requested[0] < 0 or requested[-1] >= args.num_trials:
        raise ValueError("--episode-indices must be non-empty and inside [0, num_trials)")
    rng = np.random.default_rng(args.seed)
    all_specs = []
    for episode_index in range(args.num_trials):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        initial = str(rng.choice(CUP_NAMES)) if args.initial_ball_cup == "random" else args.initial_ball_cup
        num_swaps = int(rng.integers(args.min_swaps, args.max_swaps + 1))
        all_specs.append(
            {
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "initial_ball_cup": initial,
                "num_swaps": num_swaps,
            }
        )
    return [all_specs[index] for index in requested]


def _run_episode(args: Args, *, mode: str, spec: dict) -> None:
    from openpi_client import websocket_client_policy

    episode_index = int(spec["episode_index"])
    episode_dir = pathlib.Path(args.video_out_path) / mode / f"episode_{episode_index:04d}"
    child_args = dataclasses.replace(
        args,
        num_trials=1,
        seed=0,
        initial_ball_cup=spec["initial_ball_cup"],
        min_swaps=spec["num_swaps"],
        max_swaps=spec["num_swaps"],
        video_out_path=str(episode_dir),
    )
    child_args._active_xy_residual_mode = mode

    original_episode_namespace = base._episode_namespace  # noqa: SLF001

    def episode_namespace(current_args, *, seed, initial_ball_cup, num_swaps):
        del seed
        return original_episode_namespace(
            current_args,
            seed=spec["episode_seed"],
            initial_ball_cup=initial_ball_cup,
            num_swaps=num_swaps,
        )

    base._episode_namespace = episode_namespace  # noqa: SLF001
    base._policy_input = fixed_eef._fixed_history_policy_input  # noqa: SLF001
    base._append_observation = _append_observation_orientation_stable  # noqa: SLF001
    base._run_scripted_observation = _run_scripted_observation_with_target  # noqa: SLF001
    base._policy_env_action = _xy_residual_policy_env_action  # noqa: SLF001
    base._physics_snapshot = _physics_snapshot_with_xy_residual  # noqa: SLF001

    original_client = websocket_client_policy.WebsocketClientPolicy

    class ReconnectingWebsocketClient:
        def __init__(self, host="0.0.0.0", port=None, api_key=None):
            self._host = host
            self._port = port
            self._api_key = api_key
            self._calls = 0
            self._client = original_client(host, port, api_key)

        def _reconnect(self):
            self._client._ws.close()  # noqa: SLF001
            self._client = original_client(self._host, self._port, self._api_key)

        def infer(self, observation):
            if self._calls and self._calls % args.websocket_reconnect_interval == 0:
                self._reconnect()
            # Policy inputs are converted with ``jnp.asarray`` before the
            # deterministic-noise key is consumed.  Keep the seed inside the
            # signed-int32 range so uint32 values above 2**31-1 do not fail at
            # the generic policy input boundary.  The mapping remains fully
            # deterministic and identical across paired ablation modes.
            sample_seed = int(
                np.random.SeedSequence(
                    [args.deterministic_sample_salt, int(spec["episode_seed"]), self._calls]
                ).generate_state(1, dtype=np.uint32)[0]
            ) & 0x7FFF_FFFF
            observation = {**observation, NOISE_SEED_KEY: sample_seed}
            self._calls += 1
            return self._client.infer(observation)

        def get_server_metadata(self):
            return self._client.get_server_metadata()

        def reset(self):
            self._client.reset()

    websocket_client_policy.WebsocketClientPolicy = ReconnectingWebsocketClient
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("XY residual isolated mode=%s episode=%d seed=%d", mode, episode_index, spec["episode_seed"])
    base.eval_shellgame(child_args)


def _read_result(root: pathlib.Path, mode: str, spec: dict) -> dict:
    episode_index = int(spec["episode_index"])
    episode_dir = root / mode / f"episode_{episode_index:04d}"
    trace_path = episode_dir / "physics_debug" / "trial_0000.json"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    videos = sorted(episode_dir.glob("trial_0000_*.mp4"))
    trace = payload["trace"]
    residual_events = [step["xy_residual"] for step in trace if "xy_residual" in step]
    applied = [event for event in residual_events if event["applied"]]
    activation = next((event for event in residual_events if event["activated_now"]), None)
    return {
        **spec,
        "mode": mode,
        "target_cup": payload["target_cup"],
        "final_ball_cup": payload["final_ball_cup"],
        "selected_cup": payload["selected_cup"],
        "cup_selection_correct": bool(payload["cup_selection_correct"]),
        "success": bool(payload["success"]),
        "activation_step": None if activation is None else activation["policy_step"],
        "activation_raw_xy_error_mm": None if activation is None else activation["raw_xy_error_mm"],
        "correction_steps": len(applied),
        "max_corrected_raw_xy_error_mm": None if not applied else max(event["raw_xy_error_mm"] for event in applied),
        "final_stats": payload["final_stats"],
        "video": str(videos[0]) if videos else None,
        "physics_trace": str(trace_path),
    }


def main(args: Args) -> None:
    modes = [item.strip() for item in args.xy_residual_modes.split(",") if item.strip()]
    if not modes or any(mode not in VALID_MODES for mode in modes):
        raise ValueError(f"--xy-residual-modes must contain only {VALID_MODES}")
    if args.num_frames != fixed_eef.TOTAL_FRAMES or args.frame_stride != 1:
        raise ValueError("This diagnostic requires --num-frames 61 --frame-stride 1")
    if args.action_mode != "raw7" or args.action_dim != 7 or args.osc_input_type != "absolute":
        raise ValueError("This diagnostic requires absolute raw7 actions")
    if args.physics_debug_window < args.max_policy_steps:
        raise ValueError("Use --physics-debug-window >= --max-policy-steps for full-stage analysis")
    if args.websocket_reconnect_interval <= 0 or args.xy_residual_close_hold_steps <= 0:
        raise ValueError("Reconnect interval and close hold steps must be positive")

    root = pathlib.Path(args.video_out_path)
    root.mkdir(parents=True, exist_ok=True)
    specs = _episode_specs(args)
    context = mp.get_context("spawn")
    results = []
    for mode in modes:
        for spec in specs:
            episode_dir = root / mode / f"episode_{int(spec['episode_index']):04d}"
            if episode_dir.exists():
                raise FileExistsError(f"Refusing to overwrite {episode_dir}")
            process = context.Process(target=_run_episode, kwargs={"args": args, "mode": mode, "spec": spec})
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"mode={mode} episode={spec['episode_index']} exited {process.exitcode}")
            result = _read_result(root, mode, spec)
            results.append(result)
            logging.info(
                "aggregate mode=%s completed=%d/%d success=%s selection=%s corrections=%d",
                mode,
                sum(item["mode"] == mode for item in results),
                len(specs),
                result["success"],
                result["cup_selection_correct"],
                result["correction_steps"],
            )

    by_mode = {
        mode: {
            "successes": sum(item["success"] for item in results if item["mode"] == mode),
            "selection_correct": sum(item["cup_selection_correct"] for item in results if item["mode"] == mode),
            "episodes": len(specs),
        }
        for mode in modes
    }
    summary = {
        "experiment": "stage-wise oracle target-cup XY residual; policy Z/rotation/gripper/history unchanged",
        "seed": args.seed,
        "episode_indices": [spec["episode_index"] for spec in specs],
        "modes": modes,
        "close_hold_steps": args.xy_residual_close_hold_steps,
        "deterministic_sample_salt": args.deterministic_sample_salt,
        "by_mode": by_mode,
        "results": results,
    }
    (root / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("XY residual ablation result=%s", root / "result.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

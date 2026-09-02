"""Run the fixed-history absolute-EEF XY-before-Z diagnostic safely.

Each episode gets a fresh MuJoCo/EGL process.  The WebSocket client is also
reconnected periodically inside an episode.  On this machine, retaining many
large 61-frame inference messages in one connection eventually corrupts EGL
camera readbacks; the corruption consistently appears as white noise near
frame 89.  The policy server stays alive across child processes.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import os
import pathlib

import main as base
import main_absolute_eef_fixed_history as fixed_eef
import main_absolute_eef_fixed_history_xy_before_z as guarded_eef
import numpy as np
import tyro

CUP_NAMES = ("left", "middle", "right")


@dataclasses.dataclass
class Args(guarded_eef.Args):
    num_trials: int = 20
    episode_start_index: int = 0
    seed: int = 0
    video_out_path: str = "evaluation/shellgame/absolute_eef_xy_before_z_isolated"
    physics_debug: bool = True
    physics_debug_window: int = 30
    websocket_reconnect_interval: int = 4
    xy_before_z_enabled: bool = True
    # Compatibility mode for V10 action-only evaluation. All 61 required image
    # keys contain the current observation, so no temporal history is sent.
    current_only_no_memory_input: bool = False
    # Reuse the isolated-process and aggregation machinery for policies whose
    # native main.py contract is not fixed-history absolute EEF (for example,
    # the single-frame delta-OSC pi0.5 baseline).  The default remains the
    # historical fixed-history behavior.
    native_policy_contract: bool = False


def _run_episode(
    args: Args,
    *,
    episode_index: int,
    episode_seed: int,
    initial_ball_cup: str,
    num_swaps: int,
) -> None:
    from openpi_client import websocket_client_policy

    episode_dir = pathlib.Path(args.video_out_path) / f"episode_{episode_index:04d}"
    child_args = dataclasses.replace(
        args,
        num_trials=1,
        seed=0,
        initial_ball_cup=initial_ball_cup,
        min_swaps=num_swaps,
        max_swaps=num_swaps,
        video_out_path=str(episode_dir),
    )

    original_episode_namespace = base._episode_namespace  # noqa: SLF001

    def episode_namespace(
        current_args: Args,
        *,
        seed: int,
        initial_ball_cup: str,
        num_swaps: int,
    ):
        del seed
        return original_episode_namespace(
            current_args,
            seed=episode_seed,
            initial_ball_cup=initial_ball_cup,
            num_swaps=num_swaps,
        )

    base._episode_namespace = episode_namespace  # noqa: SLF001
    if not args.native_policy_contract:
        base._policy_input = (  # noqa: SLF001
            fixed_eef._current_only_compat_policy_input  # noqa: SLF001
            if args.current_only_no_memory_input
            else fixed_eef._fixed_history_policy_input  # noqa: SLF001
        )
        base._policy_env_action = (  # noqa: SLF001
            guarded_eef._guarded_policy_env_action  # noqa: SLF001
            if args.xy_before_z_enabled
            else guarded_eef._base_policy_env_action  # noqa: SLF001
        )

    original_client = websocket_client_policy.WebsocketClientPolicy

    class ReconnectingWebsocketClient:
        def __init__(self, host: str = "0.0.0.0", port: int | None = None, api_key: str | None = None):
            self._host = host
            self._port = port
            self._api_key = api_key
            self._calls = 0
            self._client = original_client(host, port, api_key)

        def _reconnect(self) -> None:
            self._client._ws.close()  # noqa: SLF001
            self._client = original_client(self._host, self._port, self._api_key)

        def infer(self, observation: dict) -> dict:
            if self._calls and self._calls % args.websocket_reconnect_interval == 0:
                self._reconnect()
            self._calls += 1
            return self._client.infer(observation)

        def get_server_metadata(self) -> dict:
            return self._client.get_server_metadata()

        def reset(self) -> None:
            self._client.reset()

    websocket_client_policy.WebsocketClientPolicy = ReconnectingWebsocketClient

    logging.basicConfig(level=logging.INFO, force=True)
    logging.info(
        "isolated absolute-EEF episode=%d pid=%d seed=%d initial=%s swaps=%d reconnect=%d",
        episode_index,
        mp.current_process().pid,
        episode_seed,
        initial_ball_cup,
        num_swaps,
        args.websocket_reconnect_interval,
    )
    base.eval_shellgame(child_args)


def _episode_specs(args: Args) -> list[dict]:
    if args.episode_start_index < 0:
        raise ValueError("--episode-start-index must be non-negative")
    rng = np.random.default_rng(args.seed)
    specs = []
    stop = args.episode_start_index + args.num_trials
    for episode_index in range(stop):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        initial = str(rng.choice(CUP_NAMES)) if args.initial_ball_cup == "random" else args.initial_ball_cup
        num_swaps = int(rng.integers(args.min_swaps, args.max_swaps + 1))
        if episode_index < args.episode_start_index:
            continue
        specs.append(
            {
                "episode_index": episode_index,
                "episode_seed": episode_seed,
                "initial_ball_cup": initial,
                "num_swaps": num_swaps,
            }
        )
    return specs


def _read_episode_result(root: pathlib.Path, spec: dict, *, save_videos: bool) -> dict:
    episode_index = int(spec["episode_index"])
    episode_dir = root / f"episode_{episode_index:04d}"
    trace_path = episode_dir / "physics_debug" / "trial_0000.json"
    if not trace_path.is_file():
        raise RuntimeError(f"Missing physics trace for isolated episode {episode_index}: {trace_path}")
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    videos = sorted(episode_dir.glob("trial_0000_*.mp4"))
    if len(videos) != int(save_videos):
        raise RuntimeError(
            f"Expected {int(save_videos)} video for episode {episode_index}, found {len(videos)}"
        )
    return {
        **spec,
        "target_cup": payload["target_cup"],
        "selected_cup": payload["selected_cup"],
        "cup_selection_correct": bool(payload["cup_selection_correct"]),
        "gripper_contacted_cups": payload["gripper_contacted_cups"],
        "any_cup_contact": bool(payload["any_cup_contact"]),
        "target_cup_contact": bool(payload["target_cup_contact"]),
        "selected_cup_contact": bool(payload["selected_cup_contact"]),
        "correct_selection_and_contact": bool(
            payload["correct_selection_and_contact"]
        ),
        "success": bool(payload["success"]),
        "final_stats": payload["final_stats"],
        "video": str(videos[0]) if videos else None,
        "physics_trace": str(trace_path),
    }


def main(args: Args) -> None:
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if args.websocket_reconnect_interval <= 0:
        raise ValueError("--websocket-reconnect-interval must be positive")
    if not args.physics_debug:
        raise ValueError("Isolated aggregation requires --physics-debug")
    max_debug_window = int(os.environ.get("OPENPI_MAX_PHYSICS_DEBUG_WINDOW", "30"))
    if args.physics_debug_window > max_debug_window:
        raise ValueError(
            f"Use --physics-debug-window <= {max_debug_window} to keep per-episode diagnostics bounded"
        )
    if (
        not args.native_policy_contract
        and (args.num_frames != fixed_eef.TOTAL_FRAMES or args.frame_stride != 1)
    ):
        raise ValueError("Isolated fixed-history absolute-EEF evaluation requires 61 frames and stride 1")

    root = pathlib.Path(args.video_out_path)
    root.mkdir(parents=True, exist_ok=True)
    specs = _episode_specs(args)
    context = mp.get_context("spawn")
    results = []

    for spec in specs:
        episode_dir = root / f"episode_{int(spec['episode_index']):04d}"
        if episode_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing episode output: {episode_dir}")
        process = context.Process(target=_run_episode, kwargs={"args": args, **spec})
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"Isolated episode {spec['episode_index']} failed with exit code {process.exitcode}"
            )
        results.append(_read_episode_result(root, spec, save_videos=args.save_videos))
        successes = sum(item["success"] for item in results)
        selections = sum(item["cup_selection_correct"] for item in results)
        contacts = sum(item["correct_selection_and_contact"] for item in results)
        logging.info(
            "isolated aggregate: episodes=%d/%d lift=%d/%d selection=%d/%d correct_contact=%d/%d",
            len(results),
            args.num_trials,
            successes,
            len(results),
            selections,
            len(results),
            contacts,
            len(results),
        )

    summary = {
        "evaluation": "fresh MuJoCo/EGL process per episode with periodic WebSocket reconnection",
        "websocket_reconnect_interval": args.websocket_reconnect_interval,
        "physics_debug_window": args.physics_debug_window,
        "xy_before_z_enabled": args.xy_before_z_enabled,
        "current_only_no_memory_input": args.current_only_no_memory_input,
        "native_policy_contract": args.native_policy_contract,
        "seed": args.seed,
        "episode_start_index": args.episode_start_index,
        "num_trials": args.num_trials,
        "lift_successes": sum(item["success"] for item in results),
        "cup_selection_correct": sum(item["cup_selection_correct"] for item in results),
        "any_cup_contacts": sum(item["any_cup_contact"] for item in results),
        "target_cup_contacts": sum(item["target_cup_contact"] for item in results),
        "correct_selection_and_contacts": sum(
            item["correct_selection_and_contact"] for item in results
        ),
        "episodes": results,
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("isolated result=%s", result_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

"""Isolated-episode ShellGame evaluation with continuity-ranked chunks."""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import pathlib

import main as base
import main_v2_absolute_joint as joint
import main_v2_absolute_joint_fixed_history as fixed
import main_v2_absolute_joint_fixed_history_multicandidate as multicandidate
import numpy as np
import tyro

CUP_NAMES = ("left", "middle", "right")


@dataclasses.dataclass
class Args(multicandidate.Args):
    num_trials: int = 20
    seed: int = 0
    video_out_path: str = "evaluation/shellgame/fixed_history_isolated_multicandidate"
    physics_debug: bool = True
    physics_debug_window: int = 30
    websocket_reconnect_interval: int = 4


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

    def episode_namespace(
        current_args: Args,
        *,
        seed: int,
        initial_ball_cup: str,
        num_swaps: int,
    ):
        del seed
        return joint._episode_namespace(  # noqa: SLF001
            current_args,
            seed=episode_seed,
            initial_ball_cup=initial_ball_cup,
            num_swaps=num_swaps,
        )

    base._episode_namespace = episode_namespace  # noqa: SLF001
    base._append_observation = joint._append_observation  # noqa: SLF001
    base._policy_action_dim = joint._policy_action_dim  # noqa: SLF001
    base._policy_input = fixed._fixed_history_policy_input  # noqa: SLF001
    base._zero_env_action = joint._zero_env_action  # noqa: SLF001
    base._target_action_to_env_action = joint._absolute_joint_action_to_env_action  # noqa: SLF001
    base._policy_env_action = multicandidate._multicandidate_policy_env_action  # noqa: SLF001

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
    multicandidate.reset_multicandidate_state()
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info(
        "isolated multicandidate episode=%d pid=%d seed=%d initial=%s swaps=%d candidates=%d",
        episode_index,
        mp.current_process().pid,
        episode_seed,
        initial_ball_cup,
        num_swaps,
        args.candidate_count,
    )
    base.eval_shellgame(child_args)
    debug_path = episode_dir / "multicandidate_debug.json"
    debug_path.write_text(
        json.dumps(multicandidate.multicandidate_diagnostics(), indent=2),
        encoding="utf-8",
    )


def _episode_specs(args: Args) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    specs = []
    for episode_index in range(args.num_trials):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        initial = str(rng.choice(CUP_NAMES)) if args.initial_ball_cup == "random" else args.initial_ball_cup
        num_swaps = int(rng.integers(args.min_swaps, args.max_swaps + 1))
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
    candidate_path = episode_dir / "multicandidate_debug.json"
    if not trace_path.is_file() or not candidate_path.is_file():
        raise RuntimeError(f"Missing diagnostics for isolated episode {episode_index}")
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    videos = sorted(episode_dir.glob("trial_0000_*.mp4"))
    if len(videos) != int(save_videos):
        raise RuntimeError(f"Expected {int(save_videos)} video for episode {episode_index}, found {len(videos)}")
    return {
        **spec,
        "target_cup": payload["target_cup"],
        "selected_cup": payload["selected_cup"],
        "cup_selection_correct": bool(payload["cup_selection_correct"]),
        "success": bool(payload["success"]),
        "final_stats": payload["final_stats"],
        "num_replans": len(candidate_payload),
        "video": str(videos[0]) if videos else None,
        "physics_trace": str(trace_path),
        "multicandidate_trace": str(candidate_path),
    }


def main(args: Args) -> None:
    multicandidate._validate_args(args)  # noqa: SLF001
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if args.websocket_reconnect_interval <= 0:
        raise ValueError("--websocket-reconnect-interval must be positive")
    if not args.physics_debug:
        raise ValueError("Isolated aggregation requires --physics-debug")
    if args.num_frames != fixed.TOTAL_FRAMES or args.frame_stride != 1:
        raise ValueError("Isolated fixed-history evaluation requires --num-frames 61 --frame-stride 1")

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
            raise RuntimeError(f"Isolated episode {spec['episode_index']} failed with exit code {process.exitcode}")
        results.append(_read_episode_result(root, spec, save_videos=args.save_videos))
        successes = sum(item["success"] for item in results)
        selections = sum(item["cup_selection_correct"] for item in results)
        logging.info(
            "isolated multicandidate aggregate: episodes=%d/%d lift=%d/%d selection=%d/%d",
            len(results),
            args.num_trials,
            successes,
            len(results),
            selections,
            len(results),
        )

    summary = {
        "evaluation": "isolated episodes with continuity-ranked multicandidate chunks",
        "candidate_count": args.candidate_count,
        "continuity_current_weight": args.continuity_current_weight,
        "continuity_overlap_weight": args.continuity_overlap_weight,
        "replan_steps": args.replan_steps,
        "websocket_reconnect_interval": args.websocket_reconnect_interval,
        "physics_debug_window": args.physics_debug_window,
        "seed": args.seed,
        "num_trials": args.num_trials,
        "lift_successes": sum(item["success"] for item in results),
        "cup_selection_correct": sum(item["cup_selection_correct"] for item in results),
        "episodes": results,
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("isolated multicandidate result=%s", result_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

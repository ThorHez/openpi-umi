"""Isolated evaluation for arm ensemble with newest-chunk gripper targets."""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import pathlib

import eval_fixed_history_isolated_episodes_temporal_ensemble as base_eval
import main_v2_absolute_joint_fixed_history_temporal_arm_newest_gripper as variant
import main_v2_absolute_joint_fixed_history_temporal_ensemble as temporal
import tyro


@dataclasses.dataclass
class Args(base_eval.Args):
    video_out_path: str = "evaluation/shellgame/fixed_history_temporal_arm_newest_gripper"


def _run_episode(
    args: Args,
    *,
    episode_index: int,
    episode_seed: int,
    initial_ball_cup: str,
    num_swaps: int,
) -> None:
    original = temporal._temporal_ensemble_policy_env_action  # noqa: SLF001
    temporal._temporal_ensemble_policy_env_action = (  # noqa: SLF001
        variant._temporal_arm_newest_gripper_policy_env_action  # noqa: SLF001
    )
    try:
        base_eval._run_episode(  # noqa: SLF001
            args,
            episode_index=episode_index,
            episode_seed=episode_seed,
            initial_ball_cup=initial_ball_cup,
            num_swaps=num_swaps,
        )
    finally:
        temporal._temporal_ensemble_policy_env_action = original  # noqa: SLF001


def main(args: Args) -> None:
    temporal._validate_args(args)  # noqa: SLF001
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if args.websocket_reconnect_interval <= 0:
        raise ValueError("--websocket-reconnect-interval must be positive")
    if not args.physics_debug:
        raise ValueError("Isolated aggregation requires --physics-debug")
    if args.num_frames != base_eval.fixed.TOTAL_FRAMES or args.frame_stride != 1:
        raise ValueError("Isolated fixed-history evaluation requires --num-frames 61 --frame-stride 1")

    root = pathlib.Path(args.video_out_path)
    root.mkdir(parents=True, exist_ok=True)
    specs = base_eval._episode_specs(args)  # noqa: SLF001
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
        results.append(base_eval._read_episode_result(root, spec, save_videos=args.save_videos))  # noqa: SLF001
        successes = sum(item["success"] for item in results)
        selections = sum(item["cup_selection_correct"] for item in results)
        logging.info(
            "arm ensemble + newest gripper: episodes=%d/%d lift=%d/%d selection=%d/%d",
            len(results),
            args.num_trials,
            successes,
            len(results),
            selections,
            len(results),
        )

    summary = {
        "evaluation": "temporal arm-joint ensemble with newest-chunk gripper",
        "arm_ensemble_decay": args.ensemble_decay,
        "arm_ensemble_weight_order": "oldest_to_newest_exponential_decay",
        "gripper_mode": "newest_chunk",
        "ensemble_max_chunks": args.ensemble_max_chunks,
        "replan_steps": args.replan_steps,
        "action_horizon": args.action_horizon,
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
    logging.info("arm ensemble + newest gripper result=%s", result_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

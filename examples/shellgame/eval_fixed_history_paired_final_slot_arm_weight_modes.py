"""Paired final-slot evaluation for temporal arm-ensemble weighting modes."""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Literal

import eval_fixed_history_isolated_episodes_temporal_ensemble as base_eval
import main_v2_absolute_joint_fixed_history_temporal_arm_weight_modes as variant
import main_v2_absolute_joint_fixed_history_temporal_ensemble as temporal
import tyro

FinalSlot = Literal["left", "middle", "right"]


@dataclasses.dataclass
class Args(base_eval.Args):
    arm_ensemble_mode: variant.ArmEnsembleMode = "oldest_heavy"
    source_result: Path = Path(
        "evaluation/shellgame/"
        "old_tracker_full_joint_grasp_5999_temporal_arm_newest_gripper_"
        "decay025_r5_100ep_clean_260811/result.json"
    )
    final_slot: FinalSlot = "right"
    video_out_path: str = "evaluation/shellgame/paired_right_slot_arm_weight_modes"


def _source_specs(args: Args) -> list[dict]:
    payload = json.loads(args.source_result.read_text(encoding="utf-8"))
    specs = []
    for episode in payload["episodes"]:
        trace_path = Path(episode["physics_trace"])
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if trace["final_ball_cup"] != args.final_slot:
            continue
        specs.append(
            {
                "episode_index": int(episode["episode_index"]),
                "episode_seed": int(episode["episode_seed"]),
                "initial_ball_cup": str(episode["initial_ball_cup"]),
                "num_swaps": int(episode["num_swaps"]),
            }
        )
    if len(specs) < args.num_trials:
        raise ValueError(
            f"Source contains only {len(specs)} {args.final_slot!r}-slot episodes, but --num-trials={args.num_trials}"
        )
    return specs[: args.num_trials]


def _run_episode(
    args: Args,
    *,
    episode_index: int,
    episode_seed: int,
    initial_ball_cup: str,
    num_swaps: int,
) -> None:
    original = temporal._temporal_ensemble_policy_env_action  # noqa: SLF001
    temporal._temporal_ensemble_policy_env_action = variant._policy_env_action  # noqa: SLF001
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
        raise ValueError("Paired aggregation requires --physics-debug")
    if args.num_frames != base_eval.fixed.TOTAL_FRAMES or args.frame_stride != 1:
        raise ValueError("Paired fixed-history evaluation requires --num-frames 61 --frame-stride 1")

    root = Path(args.video_out_path)
    root.mkdir(parents=True, exist_ok=True)
    specs = _source_specs(args)
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
            raise RuntimeError(f"Paired episode {spec['episode_index']} failed with exit code {process.exitcode}")
        result = base_eval._read_episode_result(root, spec, save_videos=args.save_videos)  # noqa: SLF001
        trace = json.loads(Path(result["physics_trace"]).read_text(encoding="utf-8"))
        if trace["final_ball_cup"] != args.final_slot:
            raise RuntimeError(
                f"Episode {spec['episode_index']} reproduced final slot "
                f"{trace['final_ball_cup']!r}, expected {args.final_slot!r}"
            )
        results.append(result)
        logging.info(
            "paired %s %s: episodes=%d/%d lift=%d/%d selection=%d/%d",
            args.final_slot,
            args.arm_ensemble_mode,
            len(results),
            args.num_trials,
            sum(item["success"] for item in results),
            len(results),
            sum(item["cup_selection_correct"] for item in results),
            len(results),
        )

    summary = {
        "evaluation": "paired final-slot temporal arm weight modes with newest gripper",
        "arm_ensemble_mode": args.arm_ensemble_mode,
        "ensemble_decay": args.ensemble_decay,
        "gripper_mode": "newest_chunk",
        "final_slot": args.final_slot,
        "source_result": str(args.source_result),
        "replan_steps": args.replan_steps,
        "action_horizon": args.action_horizon,
        "websocket_reconnect_interval": args.websocket_reconnect_interval,
        "seed": args.seed,
        "num_trials": args.num_trials,
        "lift_successes": sum(item["success"] for item in results),
        "cup_selection_correct": sum(item["cup_selection_correct"] for item in results),
        "episodes": results,
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("paired arm weight result=%s", result_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

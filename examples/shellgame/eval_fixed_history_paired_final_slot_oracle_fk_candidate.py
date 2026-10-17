"""Paired final-slot evaluation with one oracle FK candidate replan."""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
from pathlib import Path

import eval_fixed_history_isolated_episodes_temporal_ensemble as base_eval
import eval_fixed_history_paired_final_slot_arm_weight_modes as paired
import main_v2_absolute_joint_fixed_history_oracle_fk_candidate as oracle_candidate
import main_v2_absolute_joint_fixed_history_temporal_ensemble as temporal
import tyro


@dataclasses.dataclass
class Args(paired.Args):
    candidate_count: int = 16
    oracle_replan_step: int = 48
    candidate_score_start: int = 4
    candidate_score_end: int = 13
    target_grasp_z_offset: float = 0.04
    video_out_path: str = "evaluation/shellgame/paired_right_oracle_fk_candidate"


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
        oracle_candidate._oracle_fk_candidate_policy_env_action  # noqa: SLF001
    )
    oracle_candidate.reset_oracle_candidate_state()
    try:
        base_eval._run_episode(  # noqa: SLF001
            args,
            episode_index=episode_index,
            episode_seed=episode_seed,
            initial_ball_cup=initial_ball_cup,
            num_swaps=num_swaps,
        )
        episode_dir = Path(args.video_out_path) / f"episode_{episode_index:04d}"
        diagnostic_path = episode_dir / "oracle_fk_candidate_debug.json"
        diagnostic_path.write_text(
            json.dumps(oracle_candidate.oracle_candidate_diagnostics(), indent=2),
            encoding="utf-8",
        )
    finally:
        temporal._temporal_ensemble_policy_env_action = original  # noqa: SLF001


def _read_episode_result(root: Path, spec: dict, *, save_videos: bool) -> dict:
    result = base_eval._read_episode_result(root, spec, save_videos=save_videos)  # noqa: SLF001
    diagnostic_path = root / f"episode_{int(spec['episode_index']):04d}" / "oracle_fk_candidate_debug.json"
    if not diagnostic_path.is_file():
        raise RuntimeError(f"Missing Oracle FK diagnostics for episode {spec['episode_index']}")
    result["oracle_fk_candidate_trace"] = str(diagnostic_path)
    return result


def main(args: Args) -> None:
    oracle_candidate._validate_args(args)  # noqa: SLF001
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")
    if not args.physics_debug:
        raise ValueError("Oracle FK aggregation requires --physics-debug")
    if args.num_frames != base_eval.fixed.TOTAL_FRAMES or args.frame_stride != 1:
        raise ValueError("Oracle FK fixed-history evaluation requires --num-frames 61 --frame-stride 1")

    root = Path(args.video_out_path)
    root.mkdir(parents=True, exist_ok=True)
    specs = paired._source_specs(args)  # noqa: SLF001
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
            raise RuntimeError(f"Oracle FK episode {spec['episode_index']} failed with exit code {process.exitcode}")
        result = _read_episode_result(root, spec, save_videos=args.save_videos)
        trace = json.loads(Path(result["physics_trace"]).read_text(encoding="utf-8"))
        if trace["final_ball_cup"] != args.final_slot:
            raise RuntimeError(
                f"Episode {spec['episode_index']} reproduced {trace['final_ball_cup']!r}, expected {args.final_slot!r}"
            )
        results.append(result)
        logging.info(
            "oracle FK %s: episodes=%d/%d lift=%d/%d selection=%d/%d",
            args.final_slot,
            len(results),
            args.num_trials,
            sum(item["success"] for item in results),
            len(results),
            sum(item["cup_selection_correct"] for item in results),
            len(results),
        )

    summary = {
        "evaluation": "paired final-slot oracle FK candidate selection",
        "candidate_count": args.candidate_count,
        "oracle_replan_step": args.oracle_replan_step,
        "candidate_score_start": args.candidate_score_start,
        "candidate_score_end": args.candidate_score_end,
        "target_grasp_z_offset": args.target_grasp_z_offset,
        "final_slot": args.final_slot,
        "source_result": str(args.source_result),
        "replan_steps": args.replan_steps,
        "physics_debug_window": args.physics_debug_window,
        "num_trials": args.num_trials,
        "lift_successes": sum(item["success"] for item in results),
        "cup_selection_correct": sum(item["cup_selection_correct"] for item in results),
        "episodes": results,
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("oracle FK paired result=%s", result_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

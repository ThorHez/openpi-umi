"""Run a final-slot-balanced Oracle joint replay diagnostic."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import main as base
import numpy as np
import oracle_joint_replay as oracle

SLOTS = ("left", "middle", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../robosuite/outputs/shellgame_absolute_joint_dataset"),
    )
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--episodes-per-slot", type=int, default=30)
    parser.add_argument("--sample-seed", type=int, default=260811)
    parser.add_argument("--joint-kp", type=float, default=50.0)
    parser.add_argument("--joint-damping-ratio", type=float, default=1.0)
    parser.add_argument("--cup-selection-skip-frames", type=int, default=10)
    parser.add_argument("--cup-selection-window-frames", type=int, default=30)
    parser.add_argument("--cup-selection-xy-radius", type=float, default=0.06)
    parser.add_argument("--lift-success-height", type=float, default=0.08)
    parser.add_argument("--gripper-deadband", type=float, default=0.004)
    parser.add_argument(
        "--gripper-mode",
        choices=("measured_width", "recorded_command"),
        default="measured_width",
    )
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _balanced_episode_dirs(args: argparse.Namespace) -> tuple[list[Path], dict[str, list[str]]]:
    dataset_root = args.dataset_root.expanduser().resolve()
    groups: dict[str, list[Path]] = {slot: [] for slot in SLOTS}
    for episode_dir in sorted(path for path in dataset_root.glob("episode_*") if path.is_dir()):
        metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
        slot = str(metadata["final_ball_cup"])
        if slot not in groups:
            raise ValueError(f"Unexpected final slot {slot!r} in {episode_dir}")
        groups[slot].append(episode_dir)

    rng = np.random.default_rng(args.sample_seed)
    selected: dict[str, list[Path]] = {}
    for slot, episode_dirs in groups.items():
        if args.episodes_per_slot <= 0 or args.episodes_per_slot > len(episode_dirs):
            raise ValueError(f"--episodes-per-slot must be in [1, {len(episode_dirs)}] for {slot}")
        indices = np.sort(rng.choice(len(episode_dirs), size=args.episodes_per_slot, replace=False))
        selected[slot] = [episode_dirs[int(index)] for index in indices]

    # Interleave slots so an interrupted run remains approximately balanced.
    interleaved = [selected[slot][index] for index in range(args.episodes_per_slot) for slot in SLOTS]
    selected_names = {slot: [path.name for path in paths] for slot, paths in selected.items()}
    return interleaved, selected_names


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    episode_dirs, selected_names = _balanced_episode_dirs(args)
    shell = base._import_shellgame_tools(args.robosuite_root)  # noqa: SLF001

    results = []
    for number, episode_dir in enumerate(episode_dirs, 1):
        result = oracle.replay_episode(shell, episode_dir, args)
        results.append(result)
        logging.info(
            "[%d/%d] %s slot=%s success=%s joint_rmse=%.4f eef_xy_rmse=%.4f",
            number,
            len(episode_dirs),
            episode_dir.name,
            result["final_ball_cup"],
            result["success"],
            result["joint_tracking_rmse_rad"],
            result["eef_replay_xy_rmse_m"],
        )

    by_slot = {
        slot: oracle._aggregate(  # noqa: SLF001
            [result for result in results if result["final_ball_cup"] == slot],
            args,
        )
        for slot in SLOTS
    }
    output = {
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "sample_seed": args.sample_seed,
        "episodes_per_slot": args.episodes_per_slot,
        "selected_episodes": selected_names,
        "overall": oracle._aggregate(results, args),  # noqa: SLF001
        "by_final_slot": by_slot,
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"overall": output["overall"], "by_final_slot": by_slot}, indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()

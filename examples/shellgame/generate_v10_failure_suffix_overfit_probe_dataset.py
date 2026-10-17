"""Generate an exact-state Oracle-suffix overfit probe for V10 failures.

The three source states are the failed episodes 0, 1, and 17 from the paired
seed-260813 evaluation.  For every episode, deterministic V10 controls the
normal 60-frame history plus 80 closed-loop policy steps.  None of those model
actions or intermediate policy frames enter supervision.  The stored current
observation at row 60 is followed only by the complete 95-step Oracle suffix.

Each source state is generated twice.  This deliberately allows the trainer's
episode-level split to put one duplicate in validation while retaining the
same exact state family in training; this is an overfit wiring test, not a
generalization benchmark.
"""

# ruff: noqa: SLF001

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import eval_v10_ft_oracle_handoff_paired as paired
import generate_onpolicy_eef_correction_dataset as legacy
import main as base
import main_absolute_eef_fixed_history as fixed_eef


DATASET_KIND = "v10_failure_suffix_exact_state_overfit_probe"
TARGET_SPECS = (
    {"source_episode_index": 0, "episode_seed": 991051390, "initial_ball_cup": "left"},
    {"source_episode_index": 1, "episode_seed": 1477591051, "initial_ball_cup": "right"},
    {"source_episode_index": 17, "episode_seed": 1717117365, "initial_ball_cup": "left"},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8150)
    parser.add_argument("--robosuite-root", default="../robosuite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-checkpoint-label", required=True)
    # Retained for compatibility with the validated writer; exact specs below
    # override its random episode sampler.
    parser.add_argument("--dataset-seed", type=int, default=260813)
    parser.add_argument("--repeats-per-spec", type=int, default=2)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--deterministic-sample-salt", type=int, default=260820)
    parser.add_argument("--websocket-reconnect-interval", type=int, default=4)
    parser.add_argument("--renderer-refresh-interval", type=int, default=64)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min-offset-mm", type=float, default=3.0)
    parser.add_argument("--max-offset-mm", type=float, default=35.0)
    parser.add_argument("--min-safe-height-mm", type=float, default=25.0)
    parser.add_argument("--min-open-width-m", type=float, default=0.04)
    parser.add_argument("--recenter-steps", type=int, default=10)
    parser.add_argument("--descend-steps", type=int, default=30)
    parser.add_argument("--grasp-steps", type=int, default=15)
    parser.add_argument("--lift-steps", type=int, default=40)
    parser.add_argument("--hover-height", type=float, default=0.05)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument(
        "--prefix-trace-root",
        type=Path,
        default=Path(
            "/data2/hzl_workspace_for_pi_mem/openpi-umi/evaluation/shellgame/"
            "eef7_v10_ft499_oracle_step80_paired10_seed260813_260820/v10_full"
        ),
    )
    return parser.parse_args()


class _ReplayClient:
    """Expose the recorded V10 prefix as normal replan=8 policy chunks."""

    def __init__(self, args: argparse.Namespace, source_episode_index: int):
        trace_path = (
            args.prefix_trace_root.expanduser().resolve()
            / f"episode_{source_episode_index:04d}"
            / "physics_debug"
            / "trial_0000.json"
        )
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
        trace = payload["trace"]
        prefix = [row for row in trace if int(row["step"]) < 80]
        if [int(row["step"]) for row in prefix] != list(range(80)):
            raise RuntimeError(f"{trace_path}: expected complete recorded steps 0..79")
        self._actions = [list(row["env_action"]) for row in prefix]
        self._trace_path = trace_path
        self._calls = 0

    def infer(self, observation):
        del observation
        import numpy as np

        start = self._calls * 8
        if start >= 80:
            raise RuntimeError(f"Replay client queried beyond step 79: {self._trace_path}")
        chunk = self._actions[start : start + 16]
        chunk.extend([self._actions[-1]] * (16 - len(chunk)))
        self._calls += 1
        return {"actions": np.asarray(chunk, dtype=np.float32)}

    def close(self) -> None:
        if self._calls != 10:
            raise RuntimeError(
                f"Expected ten replan=8 prefix chunks, got {self._calls}: {self._trace_path}"
            )


def _validate_args(args: argparse.Namespace) -> None:
    if args.width != args.height or args.width != 224:
        raise ValueError("Probe requires the validated 224x224 fixed-history input")
    if args.repeats_per_spec < 2:
        raise ValueError("Use at least two repeats so every exact state remains in training")
    if args.replan_steps != 8:
        raise ValueError("Probe requires replan_steps=8")
    if args.websocket_reconnect_interval <= 0 or args.renderer_refresh_interval <= 0:
        raise ValueError("Reconnect and renderer refresh intervals must be positive")
    if not args.prefix_trace_root.expanduser().resolve().is_dir():
        raise FileNotFoundError(args.prefix_trace_root)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    logging.basicConfig(level=logging.INFO, force=True)
    base._policy_input = fixed_eef._fixed_history_policy_input
    paired._RENDERER_REFRESH_INTERVAL = int(args.renderer_refresh_interval)
    base._append_observation = paired._append_observation_orientation_stable
    legacy.MODEL_APPROACH_STEPS = 80
    shell = base._import_shellgame_tools(args.robosuite_root)
    policy_args = legacy._policy_args(args)
    manifest = []

    output_index = 0
    for repeat_index in range(args.repeats_per_spec):
        for spec in TARGET_SPECS:
            episode_seed = int(spec["episode_seed"])
            initial_ball_cup = str(spec["initial_ball_cup"])
            original_randomness = legacy._episode_randomness
            legacy._episode_randomness = lambda *_args, s=episode_seed, c=initial_ball_cup: (s, c)
            client = _ReplayClient(args, int(spec["source_episode_index"]))
            original_contact_count = legacy.contact_utils._finger_contact_count
            actual_switch_contacts: list[int] = []

            def contact_count_with_switch_bypass(env, cup):
                actual = int(original_contact_count(env, cup))
                if not actual_switch_contacts:
                    # The generic writer rejects any pre-contact state.  This
                    # exact-state wiring probe intentionally admits the one
                    # diagnosed episode whose replay has a single side contact;
                    # all later Oracle/contact measurements remain untouched.
                    actual_switch_contacts.append(actual)
                    return 0
                return actual

            legacy.contact_utils._finger_contact_count = contact_count_with_switch_bypass
            try:
                audit, payload = legacy._attempt(
                    shell,
                    client,
                    args,
                    policy_args,
                    attempt_index=output_index,
                    accepted_index=output_index,
                )
            finally:
                client.close()
                legacy._episode_randomness = original_randomness
                legacy.contact_utils._finger_contact_count = original_contact_count
            if payload is None:
                raise RuntimeError(
                    f"Exact-state source episode {spec['source_episode_index']} rejected: {audit}"
                )

            (
                observations,
                actions,
                action_mask,
                phase_ids,
                supervision,
                metadata,
                initial,
                final_ball_cup,
            ) = payload
            metadata["dataset_kind"] = DATASET_KIND
            metadata["switch"]["actual_contacts_before_oracle"] = actual_switch_contacts[0]
            metadata["switch"]["generic_zero_contact_filter_bypassed"] = bool(
                actual_switch_contacts[0]
            )
            metadata["source_paired_evaluation"] = {
                "trial_seed": 260813,
                "source_episode_index": int(spec["source_episode_index"]),
                "episode_seed": episode_seed,
                "repeat_index": repeat_index,
                "known_v10_outcome": "failure",
                "known_final_spatial_slot": "right",
            }
            metadata["model_prefix"].update(
                {
                    "policy_checkpoint_label": args.policy_checkpoint_label,
                    "executed_steps": 80,
                    "replan_steps": 8,
                    "deterministic_sample_salt": args.deterministic_sample_salt,
                    "state_source": "replayed_recorded_failed_v10_commands_0_79",
                    "recorded_prefix_trace": str(client._trace_path),
                }
            )
            metadata["supervision_contract"].update(
                {
                    "model_generated_actions_supervised": False,
                    "model_generated_frames_supervised": False,
                    "supervised_action_source": "oracle_only",
                    "complete_oracle_suffix_steps": 95,
                    "full_consecutive_horizon_required": 16,
                }
            )
            episode_dir = output / f"episode_{output_index:06d}"
            legacy._save_episode(
                episode_dir,
                observations=observations,
                actions=actions,
                action_mask=action_mask,
                phase_ids=phase_ids,
                supervision_source=supervision,
                metadata=metadata,
                initial_ball_cup=initial,
                final_ball_cup=final_ball_cup,
                fps=args.fps,
            )
            row = {
                **audit,
                "output_episode_index": output_index,
                "source_episode_index": int(spec["source_episode_index"]),
                "repeat_index": repeat_index,
            }
            manifest.append(row)
            logging.info(
                "saved=%d/%d source_ep=%d repeat=%d offset=%.1fmm safe_height=%.1fmm",
                output_index + 1,
                args.repeats_per_spec * len(TARGET_SPECS),
                spec["source_episode_index"],
                repeat_index,
                metadata["switch"]["offset_m"] * 1_000.0,
                metadata["switch"]["safe_height_m"] * 1_000.0,
            )
            output_index += 1

    summary = {
        "dataset_kind": DATASET_KIND,
        "episodes": output_index,
        "source_episode_indices": [item["source_episode_index"] for item in TARGET_SPECS],
        "repeats_per_spec": args.repeats_per_spec,
        "oracle_only_supervision": True,
        "model_prefix_steps": 80,
        "replan_steps": 8,
        "deterministic_sample_salt": args.deterministic_sample_salt,
        "prefix_source": "recorded V10 env_action steps 0..79",
        "rows": manifest,
        "settings": {**vars(args), "output": str(output)},
    }
    (output / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logging.info("complete output=%s episodes=%d", output, output_index)


if __name__ == "__main__":
    main()

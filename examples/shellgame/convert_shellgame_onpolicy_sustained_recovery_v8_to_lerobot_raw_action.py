"""Convert V8 raw episodes and prove every horizon=16 Oracle target window."""

# The converter intentionally imports audited private helpers and adds the
# sibling Robosuite scripts directory before importing its script modules.
# ruff: noqa: E402, I001, SLF001

from __future__ import annotations

import json
from pathlib import Path
import sys

ROBOSUITE_SCRIPTS = Path(__file__).resolve().parents[3] / "robosuite/robosuite/scripts"
sys.path.insert(0, str(ROBOSUITE_SCRIPTS))

import audit_onpolicy_eef_sustained_recovery_dataset_v8 as audit_v8
import convert_shellgame_onpolicy_continuous_descent_v4_to_lerobot_raw_action as v4_convert
import generate_onpolicy_eef_sustained_recovery_dataset_v8 as v8
import convert_shellgame_to_lerobot_raw_action as raw
import convert_shellgame_to_openpi_umi_v2_openpi_action as common


def main() -> None:
    args = raw.parse_args()
    if not args.phase_instructions:
        raise ValueError("V8 conversion requires --phase-instructions")
    grasp_phases = {int(x.strip()) for x in args.grasp_phase_ids.split(",") if x.strip()}
    if grasp_phases != {8, 9, 10, 11}:
        raise ValueError("V8 conversion requires --grasp-phase-ids 8,9,10,11")

    input_paths = common.find_npz_paths(args.input, args.max_episodes)
    raw_audit = audit_v8.audit(Path(args.input).expanduser().resolve(), expected_episodes=len(input_paths))
    raw.main()
    output = Path(args.output).expanduser().resolve()
    converted_audit = v4_convert._audit_converted(output, input_paths, args.action_horizon)
    payload = {
        "ok": True,
        "dataset_kind": f"{v8.DATASET_KIND}_lerobot",
        "raw_audit": raw_audit,
        "converted_audit": converted_audit,
        "training_contract": {
            "sample_rows_are_unique": True,
            "rows_per_episode": {"recovery": 12, "grasp": 2, "lift": 2},
            "recovery_row_policy": "first_4_consecutive_then_8_spread_over_continuation",
            "first_aligned_pair": "observation[60] -> raw Oracle action[61]",
            "full_consecutive_horizon": args.action_horizon,
        },
    }
    path = output / "sustained_recovery_v8_oracle_supervision_audit.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

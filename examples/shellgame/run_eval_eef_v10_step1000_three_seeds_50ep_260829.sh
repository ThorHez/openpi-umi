#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
CHECKPOINT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/absolute_eef7_v10_repro_nom60_v6preserve30_v9timing10_b12_step1000_6gpu_noprealloc_260827/1000
RUN_ROOT=${OPENPI_ROOT}/evaluation/shellgame/eef7_v10_repro_step1000_replan8_3seed50_260829
SEEDS=(260829 261829 262829)
GPUS=(5 6 7)
PORTS=(8129 8130 8131)

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/uv_cache

cd "${OPENPI_ROOT}"
[[ -d "${CHECKPOINT}/params" ]] || { echo "Missing checkpoint ${CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"

pids=()
for index in 0 1 2; do
  seed=${SEEDS[$index]}
  gpu=${GPUS[$index]}
  port=${PORTS[$index]}
  label=v10_repro_step1000_replan8_seed${seed}_50ep
  output=${RUN_ROOT}/seed_${seed}
  log=${OPENPI_ROOT}/eval_${label}_260829.log
  launcher_log=${OPENPI_ROOT}/launcher_${label}_260829.log

  (
    OPENPI_WEBSOCKET_DISABLE_PING=1 \
      OPENPI_EVAL_SEED_OVERRIDE=${seed} OPENPI_EVAL_REPLAN_STEPS_OVERRIDE=8 \
      bash examples/shellgame/run_eval_eef_v6_checkpoint_260816.sh \
        "${label}" "${CHECKPOINT}" "${gpu}" "${port}" "${output}" "${log}" unguarded 50
  ) >"${launcher_log}" 2>&1 &
  pids+=("$!")
  echo "started seed=${seed} gpu=${gpu} port=${port} pid=${pids[-1]} output=${output}"
done

status=0
for index in 0 1 2; do
  if ! wait "${pids[$index]}"; then
    echo "seed=${SEEDS[$index]} failed" >&2
    status=1
  else
    echo "seed=${SEEDS[$index]} completed"
  fi
done

if (( status != 0 )); then
  exit "${status}"
fi

uv run python - "${RUN_ROOT}" "${CHECKPOINT}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
results = []
episodes = []
for path in sorted(run_root.glob("seed_*/result.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    results.append(
        {
            "seed": payload["seed"],
            "num_trials": payload["num_trials"],
            "lift_successes": payload["lift_successes"],
            "cup_selection_correct": payload["cup_selection_correct"],
            "correct_selection_and_contacts": payload["correct_selection_and_contacts"],
            "result_path": str(path.resolve()),
        }
    )
    episodes.extend(payload["episodes"])

if len(results) != 3 or len(episodes) != 150:
    raise RuntimeError(f"Expected 3 results and 150 episodes, got {len(results)} and {len(episodes)}")

n = len(episodes)
k = sum(item["success"] for item in episodes)
z = 1.959963984540054
p = k / n
denom = 1 + z * z / n
center = (p + z * z / (2 * n)) / denom
half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
summary = {
    "checkpoint": str(checkpoint.resolve()),
    "protocol": {
        "seeds": [item["seed"] for item in results],
        "episodes_per_seed": 50,
        "total_episodes": n,
        "replan_steps": 8,
        "max_policy_steps": 150,
        "num_sampling_steps": 4,
        "xy_before_z_enabled": False,
    },
    "per_seed": results,
    "aggregate": {
        "lift_successes": k,
        "lift_success_rate": p,
        "lift_success_wilson_95": [center - half, center + half],
        "cup_selection_correct": sum(item["cup_selection_correct"] for item in results),
        "correct_selection_and_contacts": sum(item["correct_selection_and_contacts"] for item in results),
    },
}
out = run_root / "summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"summary={out.resolve()}")
PY

#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
CHECKPOINT=${OPENPI_ROOT}/checkpoints/pi05_shellgame_baseline_v1/full_ft_seed42_b32_2gpu_260827_retry1/29999
RUN_ROOT=${OPENPI_ROOT}/evaluation/shellgame/pi05_baseline_v1_step29999_3seed50_260829
SEEDS=(260829 261829 262829)
GPUS=(5 6 7)
PORTS=(8229 8230 8231)

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=/data2/hzl_workspace_for_pi_mem/.codex_tmp/uv_cache

cd "${OPENPI_ROOT}"
[[ -d "${CHECKPOINT}/params" ]] || { echo "Missing checkpoint ${CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"

run_one() {
  local seed=$1
  local gpu=$2
  local port=$3
  local label=pi05_baseline_step29999_seed${seed}_50ep
  local output=${RUN_ROOT}/seed_${seed}
  local eval_log=${OPENPI_ROOT}/eval_${label}_260829.log
  local server_log=${OPENPI_ROOT}/serve_${label}_260829.log
  local cache=/data2/hzl_workspace_for_pi_mem/.codex_tmp/jax_cache_${label}
  local server_pid=""

  mkdir -p "${cache}"
  cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      kill "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
    fi
  }
  trap 'rc=$?; cleanup; trap - EXIT; exit "${rc}"' EXIT

  : >"${server_log}"
  : >"${eval_log}"
  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    JAX_COMPILATION_CACHE_DIR=${cache} \
    .venv/bin/python scripts/serve_policy.py --port "${port}" policy:checkpoint \
      --policy.config=pi05_shellgame_baseline_v1 \
      --policy.dir="${CHECKPOINT}" >"${server_log}" 2>&1 &
  server_pid=$!

  local ready=0
  for _ in $(seq 1 120); do
    if grep -q "server listening on" "${server_log}"; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "${label}: policy server exited before becoming ready" >&2
      tail -n 100 "${server_log}" >&2
      return 1
    fi
    sleep 2
  done
  if (( ready == 0 )); then
    echo "${label}: timed out waiting for policy server" >&2
    tail -n 100 "${server_log}" >&2
    return 1
  fi

  OPENPI_WEBSOCKET_DISABLE_PING=1 CUDA_VISIBLE_DEVICES=${gpu} \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    .venv/bin/python examples/shellgame/eval_absolute_eef_fixed_history_xy_before_z_isolated.py \
      --native-policy-contract \
      --host 127.0.0.1 \
      --port "${port}" \
      --robosuite-root ../robosuite \
      --num-trials 50 \
      --seed "${seed}" \
      --replan-steps 5 \
      --max-policy-steps 300 \
      --video-out-path "${output}" \
      --initial-ball-cup random \
      --min-swaps 3 \
      --max-swaps 3 \
      --phase-instructions \
      --observe-task "Observe the ball moving under a cup and remember which cup contains it." \
      --grasp-task "The shell game has ended. Grasp and lift the cup containing the ball." \
      --policy-input-mode single_frame \
      --action-mode raw7 \
      --action-dim 7 \
      --action-horizon 16 \
      --osc-input-type delta \
      --physics-debug \
      --physics-debug-window 30 \
      --websocket-reconnect-interval 4 >"${eval_log}" 2>&1
}

pids=()
for index in 0 1 2; do
  run_one "${SEEDS[$index]}" "${GPUS[$index]}" "${PORTS[$index]}" &
  pids+=("$!")
  echo "started seed=${SEEDS[$index]} gpu=${GPUS[$index]} port=${PORTS[$index]} pid=${pids[-1]}"
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

.venv/bin/python - "${RUN_ROOT}" "${CHECKPOINT}" <<'PY'
import json
import math
import statistics
import sys
from collections import defaultdict
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
            "target_cup_contacts": payload["target_cup_contacts"],
            "correct_selection_and_contacts": payload["correct_selection_and_contacts"],
            "result_path": str(path.resolve()),
        }
    )
    episodes.extend(payload["episodes"])

if len(results) != 3 or len(episodes) != 150:
    raise RuntimeError(f"Expected 3 results and 150 episodes, got {len(results)} and {len(episodes)}")

def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [center - half, center + half]

rates = [item["lift_successes"] / item["num_trials"] for item in results]
by_target = defaultdict(list)
for episode in episodes:
    by_target[episode["target_cup"]].append(episode)

n = len(episodes)
lift = sum(item["success"] for item in episodes)
selection = sum(item["cup_selection_correct"] for item in episodes)
contact = sum(item["target_cup_contact"] for item in episodes)
correct_contact = sum(item["correct_selection_and_contact"] for item in episodes)
summary = {
    "checkpoint": str(checkpoint.resolve()),
    "protocol": {
        "seeds": [item["seed"] for item in results],
        "episodes_per_seed": 50,
        "total_episodes": n,
        "policy_input_mode": "single_frame",
        "action_mode": "raw7",
        "osc_input_type": "delta",
        "action_horizon": 16,
        "replan_steps": 5,
        "max_policy_steps": 300,
    },
    "per_seed": results,
    "aggregate": {
        "lift_successes": lift,
        "lift_success_rate": lift / n,
        "lift_success_wilson_95": wilson(lift, n),
        "seed_mean_lift_rate": statistics.mean(rates),
        "seed_sample_sd_lift_rate": statistics.stdev(rates),
        "cup_selection_correct": selection,
        "cup_selection_rate": selection / n,
        "target_cup_contacts": contact,
        "target_cup_contact_rate": contact / n,
        "correct_selection_and_contacts": correct_contact,
        "correct_selection_and_contact_rate": correct_contact / n,
    },
    "by_target_cup": {
        cup: {
            "episodes": len(items),
            "lift_successes": sum(item["success"] for item in items),
            "selection_correct": sum(item["cup_selection_correct"] for item in items),
            "target_contacts": sum(item["target_cup_contact"] for item in items),
        }
        for cup, items in sorted(by_target.items())
    },
}
out = run_root / "summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"summary={out.resolve()}")
PY

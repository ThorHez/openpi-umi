#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
ROBOMME_ROOT=${WORKSPACE}/robomme_policy_learning
CHECKPOINT=${ROBOMME_ROOT}/runs/ckpts/mme_vla_shellgame_framesamp_modul_v1/framesamp_modul_pi05sg_s42_b32_4gpu_10k_260828/9999
RUN_ROOT=${OPENPI_ROOT}/evaluation/shellgame/mme_framesamp_modul_step9999_3seed50_260829
SEEDS=(260829 261829 262829)
GPUS=(5 6 7)
PORTS=(8329 8330 8331)

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache

cd "${OPENPI_ROOT}"
[[ -d "${CHECKPOINT}/params" ]] || { echo "Missing checkpoint ${CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"

run_shard() {
  local seed=$1
  local gpu=$2
  local port=$3
  local start=$4
  local count=$5
  local stop=$((start + count - 1))
  local output=${RUN_ROOT}/seed_${seed}/shard_$(printf '%02d_%02d' "${start}" "${stop}")
  local log=${OPENPI_ROOT}/eval_mme_framesamp_step9999_seed${seed}_shard${start}_${stop}_260829.log

  PYTHONPATH=${ROBOMME_ROOT}/packages/openpi-client/src \
    CUDA_VISIBLE_DEVICES=${gpu} MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    .venv/bin/python examples/shellgame/main.py \
      --host 127.0.0.1 \
      --port "${port}" \
      --robosuite-root ../robosuite \
      --num-trials "${count}" \
      --trial-start "${start}" \
      --seed "${seed}" \
      --gpu-id 0 \
      --replan-steps 8 \
      --max-policy-steps 150 \
      --video-out-path "${output}" \
      --initial-ball-cup random \
      --min-swaps 3 \
      --max-swaps 3 \
      --phase-instructions \
      --observe-task "Observe the ball moving under a cup and remember which cup contains it." \
      --grasp-task "The shell game has ended. Grasp and lift the cup containing the ball." \
      --policy-input-mode mme_framesamp \
      --num-frames 32 \
      --frame-stride 5 \
      --action-mode raw7 \
      --action-dim 7 \
      --action-horizon 16 \
      --osc-input-type delta \
      --no-control-during-scripted-observation \
      --physics-debug \
      --physics-debug-window 30 >"${log}" 2>&1
}

merge_seed() {
  local seed=$1
  .venv/bin/python - "${RUN_ROOT}/seed_${seed}" "${seed}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = int(sys.argv[2])
payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("shard_*/result.json"))]
if len(payloads) != 2:
    raise RuntimeError(f"seed={seed}: expected 2 shard results, got {len(payloads)}")
episodes = [episode for payload in payloads for episode in payload["episodes"]]
if len(episodes) != 50 or sorted(episode["trial"] for episode in episodes) != list(range(50)):
    raise RuntimeError(f"seed={seed}: incomplete or duplicated trial schedule")
if len({episode["episode_seed"] for episode in episodes}) != 50:
    raise RuntimeError(f"seed={seed}: episode seeds are not unique")
result = {
    "protocol": {
        **payloads[0]["protocol"],
        "num_trials": 50,
        "trial_start": 0,
        "video_out_path": str(root.resolve()),
        "sharded_25_plus_25": True,
    },
    "num_trials": 50,
    "strict_success": sum(episode["strict_success"] for episode in episodes),
    "cup_selection_correct": sum(episode["cup_selection_correct"] for episode in episodes),
    "cup_selection_decisions": sum(episode["selected_cup"] is not None for episode in episodes),
    "correct_selection_and_contact": sum(episode["correct_selection_and_contact"] for episode in episodes),
    "target_cup_contact": sum(episode["target_cup_contact"] for episode in episodes),
    "any_cup_contact": sum(episode["any_cup_contact"] for episode in episodes),
    "target_cup_lift_success": sum(episode["target_lift_success"] for episode in episodes),
    "any_cup_lift_success": sum(episode["any_cup_lift_success"] for episode in episodes),
    "episodes": sorted(episodes, key=lambda episode: episode["trial"]),
}
out = root / "result.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"seed={seed} target_contact={result['target_cup_contact']}/50 result={out.resolve()}")
PY
}

run_one() {
  local seed=$1
  local gpu=$2
  local port=$3
  local label=mme_framesamp_step9999_seed${seed}_50ep
  local server_log=${OPENPI_ROOT}/serve_${label}_260829.log
  local cache=${WORKSPACE}/.codex_tmp/jax_cache_${label}
  local server_pid=""

  mkdir -p "${cache}" "${RUN_ROOT}/seed_${seed}"
  cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      kill "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup INT TERM

  : >"${server_log}"
  (
    cd "${ROBOMME_ROOT}"
    exec env CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
      JAX_COMPILATION_CACHE_DIR=${cache} \
      PYTHONPATH=${ROBOMME_ROOT}/src:${ROBOMME_ROOT}/packages/openpi-client/src \
      ${OPENPI_ROOT}/.venv/bin/python scripts/serve_policy.py \
        --port "${port}" \
        --seed 42 \
        policy:checkpoint \
        --policy.config=mme_vla_shellgame_framesamp_modul_v1 \
        --policy.dir="${CHECKPOINT}"
  ) >"${server_log}" 2>&1 &
  server_pid=$!

  local ready=0
  for _ in $(seq 1 180); do
    if grep -q "server listening on" "${server_log}"; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "${label}: policy server exited before becoming ready" >&2
      tail -n 120 "${server_log}" >&2
      cleanup
      return 1
    fi
    sleep 2
  done
  if (( ready == 0 )); then
    echo "${label}: timed out waiting for policy server" >&2
    tail -n 120 "${server_log}" >&2
    cleanup
    return 1
  fi

  if ! run_shard "${seed}" "${gpu}" "${port}" 0 25; then
    cleanup
    return 1
  fi
  if ! run_shard "${seed}" "${gpu}" "${port}" 25 25; then
    cleanup
    return 1
  fi
  if ! merge_seed "${seed}"; then
    cleanup
    return 1
  fi
  cleanup
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

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
results = []
episodes = []
for path in sorted(root.glob("seed_*/result.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    results.append(
        {
            "seed": payload["protocol"]["seed"],
            "num_trials": payload["num_trials"],
            "target_cup_contacts": payload["target_cup_contact"],
            "cup_selection_correct": payload["cup_selection_correct"],
            "correct_selection_and_contacts": payload["correct_selection_and_contact"],
            "strict_successes": payload["strict_success"],
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

n = len(episodes)
target_contacts = sum(episode["target_cup_contact"] for episode in episodes)
contact_rates = [result["target_cup_contacts"] / 50 for result in results]
by_target = defaultdict(list)
for episode in episodes:
    by_target[episode["target_cup"]].append(episode)
summary = {
    "checkpoint": str(checkpoint.resolve()),
    "protocol": {
        "seeds": [result["seed"] for result in results],
        "episodes_per_seed": 50,
        "total_episodes": n,
        "policy_input_mode": "mme_framesamp",
        "replan_steps": 8,
        "max_policy_steps": 150,
        "control_during_scripted_observation": False,
        "server_sampling_seed": 42,
    },
    "per_seed": results,
    "aggregate": {
        "target_cup_contacts": target_contacts,
        "target_cup_contact_rate": target_contacts / n,
        "target_cup_contact_wilson_95": wilson(target_contacts, n),
        "seed_mean_target_contact_rate": statistics.mean(contact_rates),
        "seed_sample_sd_target_contact_rate": statistics.stdev(contact_rates),
        "cup_selection_correct": sum(episode["cup_selection_correct"] for episode in episodes),
        "correct_selection_and_contacts": sum(episode["correct_selection_and_contact"] for episode in episodes),
        "strict_successes": sum(episode["strict_success"] for episode in episodes),
    },
    "by_target_cup": {
        cup: {
            "episodes": len(items),
            "target_contacts": sum(item["target_cup_contact"] for item in items),
            "selection_correct": sum(item["cup_selection_correct"] for item in items),
            "strict_successes": sum(item["strict_success"] for item in items),
        }
        for cup, items in sorted(by_target.items())
    },
}
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"summary={out.resolve()}")
PY

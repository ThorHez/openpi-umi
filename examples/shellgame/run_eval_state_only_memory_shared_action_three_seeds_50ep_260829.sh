#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
MEMORY=${ROOT}/artifacts/shellgame_teacher_necessity_state_only_step999_val240_260829.npz
TEACHER_MEMORY=${ROOT}/artifacts/shellgame_qwen_event_final_memory_v1_260825.npz
ACTION=${ROOT}/checkpoints/pi0_shellgame_qwen_event_memory_action_eef7_260825/direct_visual_mem_step999_filtered_action250_6gpu_260825/249
RUN_ROOT=${ROOT}/evaluation/shellgame/state_only_memory_step999_shared_action_3seed50_260829
SEEDS=(260829 261829 262829)
GPUS=(1 2 3)
PORTS=(8441 8442 8443)

EPISODES_260829=31,1896,2251,1666,1082,1677,1552,2408,1843,1783,580,430,2091,1653,129,1694,1527,1131,2001,361,2346,770,2221,1489,1183,2220,1935,293,1446,359,2179,350,1716,592,1391,2247,956,1119,1097,1087,2158,724,922,854,1811,1031,1330,1498,1568,1483
EPISODES_261829=381,686,2076,2374,1720,102,466,2399,591,1217,1793,1495,1422,2443,285,456,1503,628,1391,1097,2014,2300,1116,1323,2032,455,2264,1208,1945,922,17,983,1417,890,660,1760,935,757,293,90,1087,1847,129,2427,265,1221,2220,782,2266,1236
EPISODES_262829=455,1945,1716,359,1593,285,240,2247,453,1881,102,1291,1323,621,31,195,774,769,330,272,2076,1276,2115,365,2427,519,17,1719,517,433,763,1221,782,2346,1896,72,660,1228,1633,1329,497,2177,910,779,1097,208,592,960,1999,1198

cd "${ROOT}"
[[ -f "${MEMORY}" ]] || { echo "Missing memory bank: ${MEMORY}" >&2; exit 1; }
[[ -f "${TEACHER_MEMORY}" ]] || { echo "Missing teacher bank: ${TEACHER_MEMORY}" >&2; exit 1; }
[[ -d "${ACTION}/params" ]] || { echo "Missing action checkpoint: ${ACTION}" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"

episode_list() {
  local seed=$1 name=EPISODES_${seed}
  printf '%s' "${!name}"
}

run_one() {
  local seed=$1 gpu=$2 port=$3 episodes label server_log eval_log server_pid
  episodes=$(episode_list "${seed}")
  label=state_only_memory_step999_seed${seed}
  server_log=${ROOT}/serve_${label}_260829.log
  eval_log=${ROOT}/eval_${label}_50ep_260829.log
  mkdir -p "${RUN_ROOT}/seed_${seed}/videos"
  server_pid=""

  cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      kill "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup INT TERM

  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    .venv/bin/python examples/shellgame/serve_qwen_event_memory_action_deterministic.py \
      --checkpoint "${ACTION}" \
      --port "${port}" \
      --num-sampling-steps 4 >"${server_log}" 2>&1 &
  server_pid=$!

  local ready=0
  for _ in $(seq 1 180); do
    if grep -q "server listening on" "${server_log}"; then ready=1; break; fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      tail -n 120 "${server_log}" >&2
      cleanup
      return 1
    fi
    sleep 2
  done
  if (( ready == 0 )); then
    tail -n 120 "${server_log}" >&2
    cleanup
    return 1
  fi

  .venv/bin/python scripts/mem/eval_shellgame_frozen_mem_action_paired_closed_loop.py \
    --checkpoint "${ACTION}" \
    --direct-memory "${MEMORY}" \
    --teacher-memory "${TEACHER_MEMORY}" \
    --episodes "${episodes}" \
    --conditions direct_visual \
    --host 127.0.0.1 \
    --port "${port}" \
    --noise-salt "${seed}" \
    --replan-steps 8 \
    --max-policy-steps 150 \
    --video-dir "${RUN_ROOT}/seed_${seed}/videos" \
    --output "${RUN_ROOT}/seed_${seed}/result.json" \
    --allow-incorrect-direct-memory >"${eval_log}" 2>&1
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
  if ! wait "${pids[$index]}"; then status=1; else echo "seed=${SEEDS[$index]} completed"; fi
done
(( status == 0 )) || exit "${status}"

.venv/bin/python - "${RUN_ROOT}" "${MEMORY}" "${ACTION}" <<'PY'
import json, math, statistics, sys
from collections import defaultdict
from pathlib import Path

root, memory, action = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
per_seed, records = [], []
for path in sorted(root.glob("seed_*/result.json")):
    payload = json.loads(path.read_text())
    seed = int(path.parent.name.removeprefix("seed_"))
    rows = payload["records"]
    if len(rows) != 50 or len({int(row["episode"]) for row in rows}) != 50:
        raise RuntimeError(f"seed={seed}: expected 50 unique episodes")
    current = {
        "seed": seed,
        "episodes": 50,
        "memory_semantic_correct": sum(row["direct_memory_semantic_correct"] for row in rows),
        "target_cup_contacts": sum(row["target_cup_contact"] for row in rows),
        "any_cup_contacts": sum(row["any_cup_contact"] for row in rows),
        "cup_selection_correct": sum(row["cup_selection_correct"] for row in rows),
        "correct_selection_and_contacts": sum(row["correct_selection_and_contact"] for row in rows),
        "target_lift_successes": sum(row["success"] for row in rows),
        "result_path": str(path.resolve()),
    }
    per_seed.append(current)
    records.extend({**row, "evaluation_seed": seed} for row in rows)
if len(per_seed) != 3 or len(records) != 150:
    raise RuntimeError("Expected 3 seeds and 150 rollouts")

def wilson(k, n):
    z = 1.959963984540054; p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [c-h, c+h]

contacts = sum(row["target_cup_contact"] for row in records)
rates = [row["target_cup_contacts"] / 50 for row in per_seed]
by_target = defaultdict(list)
for row in records: by_target[row["target_cup_identity_scoring_only"]].append(row)
summary = {
    "experiment": "state-only recurrent memory (teacher-memory loss=0) + frozen shared external-memory action head",
    "memory_checkpoint": "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/shellgame_qwen_distilled_direct_visual_recurrent_memory_probe/teacher_necessity_12f_state_only_seed42_260826/999",
    "memory_bank": str(memory.resolve()),
    "action_checkpoint": str(action.resolve()),
    "protocol": {"evaluation_seeds": [x["seed"] for x in per_seed], "episodes_per_seed": 50, "total_rollouts": 150, "replan_steps": 8, "max_policy_steps": 150, "success_primary": "target cup contact"},
    "caveat": "The memory was trained with discrete GT state CE and initialized in the shared canonical basis; this is not an action-loss-only training run. The initial reveal slot in the cached memory uses the established exact simulator-label proxy.",
    "per_seed": per_seed,
    "aggregate": {
        "memory_semantic_correct": sum(row["direct_memory_semantic_correct"] for row in records),
        "target_cup_contacts": contacts,
        "target_cup_contact_rate": contacts / 150,
        "target_cup_contact_wilson_95": wilson(contacts, 150),
        "seed_mean_target_contact_rate": statistics.mean(rates),
        "seed_sample_sd_target_contact_rate": statistics.stdev(rates),
        "any_cup_contacts": sum(row["any_cup_contact"] for row in records),
        "cup_selection_correct": sum(row["cup_selection_correct"] for row in records),
        "correct_selection_and_contacts": sum(row["correct_selection_and_contact"] for row in records),
        "target_lift_successes": sum(row["success"] for row in records),
    },
    "by_target_cup": {cup: {"rollouts": len(rows), "target_contacts": sum(x["target_cup_contact"] for x in rows), "selection_correct": sum(x["cup_selection_correct"] for x in rows), "target_lift_successes": sum(x["success"] for x in rows)} for cup, rows in sorted(by_target.items())},
}
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
print(f"summary={out.resolve()}")
PY

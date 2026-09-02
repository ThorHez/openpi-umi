#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
MEMORY=${ROOT}/artifacts/shellgame_teacher_necessity_state_only_step999_val240_260829.npz
TEACHER_MEMORY=${ROOT}/artifacts/shellgame_qwen_event_final_memory_v1_260825.npz
ACTION=${ROOT}/checkpoints/pi0_shellgame_qwen_event_memory_action_eef7_260825/direct_visual_mem_step999_filtered_action250_6gpu_260825/249
RUN_ROOT=${ROOT}/evaluation/shellgame/shared_action_memory_intervention_seed260829_260901_v2
SEED=260829
CONDITIONS=(direct_visual wrong_visual zero)
GPUS=(1 2 3)
PORTS=(8461 8462 8463)
EPISODES=31,1896,2251,1666,1082,1677,1552,2408,1843,1783,580,430,2091,1653,129,1694,1527,1131,2001,361,2346,770,2221,1489,1183,2220,1935,293,1446,359,2179,350,1716,592,1391,2247,956,1119,1097,1087,2158,724,922,854,1811,1031,1330,1498,1568,1483

cd "${ROOT}"
[[ -f "${MEMORY}" ]] || { echo "Missing memory bank: ${MEMORY}" >&2; exit 1; }
[[ -f "${TEACHER_MEMORY}" ]] || { echo "Missing teacher bank: ${TEACHER_MEMORY}" >&2; exit 1; }
[[ -d "${ACTION}/params" ]] || { echo "Missing action checkpoint: ${ACTION}" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"

server_pids=()
eval_pids=()

cleanup() {
  for pid in "${server_pids[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

for index in 0 1 2; do
  condition=${CONDITIONS[$index]}
  gpu=${GPUS[$index]}
  port=${PORTS[$index]}
  condition_root=${RUN_ROOT}/${condition}
  mkdir -p "${condition_root}/videos"
  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    .venv/bin/python examples/shellgame/serve_qwen_event_memory_action_deterministic.py \
      --checkpoint "${ACTION}" \
      --port "${port}" \
      --num-sampling-steps 4 >"${condition_root}/server.log" 2>&1 &
  server_pids+=("$!")
done

for index in 0 1 2; do
  condition=${CONDITIONS[$index]}
  condition_root=${RUN_ROOT}/${condition}
  ready=0
  for _ in $(seq 1 180); do
    if grep -q "server listening on" "${condition_root}/server.log"; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pids[$index]}" 2>/dev/null; then
      tail -n 120 "${condition_root}/server.log" >&2
      exit 1
    fi
    sleep 2
  done
  (( ready == 1 )) || { tail -n 120 "${condition_root}/server.log" >&2; exit 1; }
done

for index in 0 1 2; do
  condition=${CONDITIONS[$index]}
  port=${PORTS[$index]}
  condition_root=${RUN_ROOT}/${condition}
  .venv/bin/python scripts/mem/eval_shellgame_frozen_mem_action_paired_closed_loop.py \
    --checkpoint "${ACTION}" \
    --direct-memory "${MEMORY}" \
    --teacher-memory "${TEACHER_MEMORY}" \
    --episodes "${EPISODES}" \
    --conditions "${condition}" \
    --host 127.0.0.1 \
    --port "${port}" \
    --noise-salt "${SEED}" \
    --replan-steps 8 \
    --max-policy-steps 150 \
    --video-dir "${condition_root}/videos" \
    --output "${condition_root}/result.json" \
    --allow-incorrect-direct-memory >"${condition_root}/eval.log" 2>&1 &
  eval_pids+=("$!")
done

status=0
for index in 0 1 2; do
  if ! wait "${eval_pids[$index]}"; then
    status=1
    tail -n 120 "${RUN_ROOT}/${CONDITIONS[$index]}/eval.log" >&2
  else
    echo "condition=${CONDITIONS[$index]} completed"
  fi
done
(( status == 0 )) || exit "${status}"

.venv/bin/python - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
conditions = ("direct_visual", "wrong_visual", "zero")
payloads = {
    condition: json.loads((root / condition / "result.json").read_text())
    for condition in conditions
}
reference = payloads["direct_visual"]
expected_episodes = tuple(reference["episodes"])
records = []
summary = {}
for condition in conditions:
    payload = payloads[condition]
    if tuple(payload["episodes"]) != expected_episodes:
        raise RuntimeError(f"{condition}: episode list differs")
    if payload["noise_salt"] != reference["noise_salt"]:
        raise RuntimeError(f"{condition}: noise salt differs")
    rows = payload["records"]
    if len(rows) != 50 or {row["condition"] for row in rows} != {condition}:
        raise RuntimeError(f"{condition}: expected 50 condition-specific records")
    records.extend(rows)
    summary[condition] = payload["summary"][condition]

if len(records) != 150:
    raise RuntimeError("Expected exactly 150 paired records")
for episode in expected_episodes:
    present = {row["condition"] for row in records if row["episode"] == episode}
    if present != set(conditions):
        raise RuntimeError(f"episode={episode}: incomplete condition coverage {present}")

def metrics(rows):
    n = len(rows)
    return {
        "episodes": n,
        "cup_selection_correct": sum(row["cup_selection_correct"] for row in rows),
        "target_precision_reached": sum(row["target_precision_reached"] for row in rows),
        "target_cup_contacts": sum(row["target_cup_contact"] for row in rows),
        "any_cup_contacts": sum(row["any_cup_contact"] for row in rows),
        "lift_successes": sum(row["success"] for row in rows),
    }

conditioned = {}
for condition in conditions:
    rows = [
        row for row in records
        if row["condition"] == condition and row["direct_memory_semantic_correct"]
    ]
    conditioned[condition] = metrics(rows)

combined = {
    "schema_version": 1,
    "experiment": "paired memory-to-action intervention on one evaluation seed",
    "scope": "deployment-matched ShellGame state-only memory and frozen external-memory action head; not the four-task RoboMME checkpoint",
    "checkpoint": reference["checkpoint"],
    "direct_memory": reference["direct_memory"],
    "memory_parameters_updated": False,
    "action_parameters_updated": False,
    "evaluation_seed": reference["noise_salt"],
    "same_episode_and_diffusion_noise_per_condition": True,
    "episodes": list(expected_episodes),
    "conditions": list(conditions),
    "wrong_memory_donor": payloads["wrong_visual"]["wrong_memory_donor"],
    "control": reference["control"],
    "summary_all_50": summary,
    "summary_conditioned_on_direct_memory_correct": conditioned,
    "records": sorted(records, key=lambda row: (row["episode"], conditions.index(row["condition"]))),
}
(root / "result.json").write_text(json.dumps(combined, indent=2) + "\n")
print(json.dumps({
    "summary_all_50": summary,
    "summary_conditioned_on_direct_memory_correct": conditioned,
}, indent=2))
print(f"output={(root / 'result.json').resolve()}")
PY

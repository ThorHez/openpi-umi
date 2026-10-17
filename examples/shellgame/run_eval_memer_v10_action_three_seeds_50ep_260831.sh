#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
ROBOMME_PYTHON=${WORKSPACE}/robomme/.venv/bin/python
BASE_MODEL=${WORKSPACE}/Qwen3-VL-4B-Instruct
MEMER_ADAPTER=${WORKSPACE}/robomme_policy_learning/runs/ckpts/vlm_subgoal_predictor/memer/grounded_subgoal/checkpoint-1300
V10_CHECKPOINT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/absolute_eef7_v10_repro_nom60_v6preserve30_v9timing10_b12_step1000_6gpu_noprealloc_260827/1000
RUN_ROOT=${OPENPI_ROOT}/evaluation/shellgame/memer_v10_action_zero_shot_3seed50_260831
SEEDS=(260829 261829 262829)
GPUS=(5 6 7)
PORTS=(8451 8452 8453)

EPISODES_260829=4996,2518,1847,3865,3400,3183,1695,4398,1535,2703,2915,2197,3097,2553,2982,3310,4236,3658,2722,245,3915,1225,1495,1455,2273,1720,4739,2826,80,3455,2678,3378,387,4832,3809,2486,265,1138,4520,660,3105,455,3552,96,4301,3367,4894,935,293,4556
EPISODES_261829=3310,3753,2841,1138,3207,3455,4917,2100,4556,4520,3002,3097,381,4763,2197,3202,4206,387,2826,2722,2835,90,3992,1463,80,870,3050,2982,4744,1082,3460,4001,1160,4131,4996,2553,4236,3825,1225,4832,643,4178,1633,455,3378,2678,2273,1847,4060,96
EPISODES_262829=870,719,90,3753,3002,3039,1463,2273,585,3105,2826,3367,2835,265,660,2722,4236,1624,3865,3183,4173,4280,1082,80,4917,3915,2915,1325,3846,455,4060,1455,4556,1535,1720,4520,3050,3310,4763,3658,1225,2095,2553,1183,2973,3460,3378,2247,4178,3400

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "${OPENPI_ROOT}"
[[ -x "${ROBOMME_PYTHON}" ]] || { echo "Missing ${ROBOMME_PYTHON}" >&2; exit 1; }
[[ -d "${BASE_MODEL}" ]] || { echo "Missing ${BASE_MODEL}" >&2; exit 1; }
[[ -d "${MEMER_ADAPTER}" ]] || { echo "Missing ${MEMER_ADAPTER}" >&2; exit 1; }
[[ -d "${V10_CHECKPOINT}/params" ]] || { echo "Missing ${V10_CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}/logs"

ALL_EPISODES=${EPISODES_260829},${EPISODES_261829},${EPISODES_262829}
MANIFEST=${RUN_ROOT}/memer_manifest.json
CUDA_VISIBLE_DEVICES=4 \
MEMER_BASE_MODEL_PATH=${BASE_MODEL} \
MEMER_ATTN_IMPL=sdpa \
MEMER_VERBOSE=0 \
  "${ROBOMME_PYTHON}" scripts/mem/precompute_shellgame_memer_subgoals.py \
    --adapter "${MEMER_ADAPTER}" \
    --episodes "${ALL_EPISODES}" \
    --work-dir "${RUN_ROOT}/memer_cache" \
    --output "${MANIFEST}" >"${RUN_ROOT}/logs/memer_precompute.log" 2>&1

episode_list() {
  local seed=$1
  local name=EPISODES_${seed}
  printf '%s' "${!name}"
}

run_shard() {
  local seed=$1
  local gpu=$2
  local port=$3
  local ordinal=$4
  local episodes=$5
  local output_dir=${RUN_ROOT}/seed_${seed}/shard_${ordinal}
  local log=${RUN_ROOT}/logs/eval_seed${seed}_shard${ordinal}.log
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/mem/eval_shellgame_memer_v10_action.py \
      --memer-manifest "${MANIFEST}" \
      --direct-memory /tmp/memer_zero_direct.npz \
      --teacher-memory /tmp/memer_zero_teacher.npz \
      --checkpoint "${V10_CHECKPOINT}" \
      --episodes "${episodes}" \
      --conditions direct_visual \
      --host 127.0.0.1 \
      --port "${port}" \
      --noise-salt "${seed}" \
      --replan-steps 8 \
      --max-policy-steps 150 \
      --video-dir "${output_dir}/videos" \
      --output "${output_dir}/result.json" >"${log}" 2>&1
}

merge_seed() {
  local seed=$1
  .venv/bin/python - "${RUN_ROOT}/seed_${seed}" "${seed}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = int(sys.argv[2])
paths = sorted(root.glob("shard_*/result.json"))
if len(paths) != 2:
    raise RuntimeError(f"seed={seed}: expected 2 shards, got {len(paths)}")
payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
records = [record for payload in payloads for record in payload["records"]]
if len(records) != 50 or len({int(record["episode"]) for record in records}) != 50:
    raise RuntimeError(f"seed={seed}: expected 50 unique records")
records.sort(key=lambda record: int(record["episode"]))
result = {
    "schema_version": 1,
    "experiment": "MemER zero-shot subgoal -> V10 action-only/no-memory ShellGame",
    "seed": seed,
    "noise_salt": seed,
    "episodes": [int(record["episode"]) for record in records],
    "target_cup_contacts": sum(record["target_cup_contact"] for record in records),
    "any_cup_contacts": sum(record["any_cup_contact"] for record in records),
    "cup_selection_correct": sum(record["cup_selection_correct"] for record in records),
    "correct_selection_and_contacts": sum(record["correct_selection_and_contact"] for record in records),
    "target_lift_successes": sum(record["success"] for record in records),
    "memer_grounding_parseable": sum(record["memer_grounding_parseable"] for record in records),
    "memer_grounding_correct": sum(record["memer_grounding_correct"] for record in records),
    "records": records,
    "shards": [str(path.resolve()) for path in paths],
}
out = root / "result.json"
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(
    f"seed={seed} grounding={result['memer_grounding_correct']}/50 "
    f"target_contact={result['target_cup_contacts']}/50 result={out.resolve()}"
)
PY
}

run_one() {
  local seed=$1
  local gpu=$2
  local port=$3
  local list
  list=$(episode_list "${seed}")
  IFS=',' read -r -a ids <<<"${list}"
  [[ ${#ids[@]} -eq 50 ]] || { echo "seed=${seed}: expected 50 IDs" >&2; return 1; }
  local first second
  first=$(IFS=,; printf '%s' "${ids[*]:0:25}")
  second=$(IFS=,; printf '%s' "${ids[*]:25:25}")
  local server_log=${RUN_ROOT}/logs/server_seed${seed}.log
  local cache=${WORKSPACE}/.codex_tmp/jax_cache_memer_v10_seed${seed}
  local server_pid=""
  mkdir -p "${cache}" "${RUN_ROOT}/seed_${seed}"

  cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
      kill -TERM "${server_pid}" 2>/dev/null || true
      wait "${server_pid}" 2>/dev/null || true
    fi
  }
  trap cleanup INT TERM EXIT
  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    JAX_COMPILATION_CACHE_DIR=${cache} \
    .venv/bin/python examples/shellgame/serve_v10_exact_parallel_semantic_adapter_deterministic.py \
      --checkpoint-dir "${V10_CHECKPOINT}" \
      --adapter-mode v10_action_no_memory \
      --port "${port}" \
      --num-sampling-steps 4 \
      --deterministic-noise >"${server_log}" 2>&1 &
  server_pid=$!

  local ready=0
  for _ in $(seq 1 180); do
    if grep -q "server listening on" "${server_log}"; then ready=1; break; fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "seed=${seed}: server exited before ready" >&2
      tail -n 120 "${server_log}" >&2
      return 1
    fi
    sleep 2
  done
  [[ ${ready} -eq 1 ]] || { echo "seed=${seed}: server readiness timeout" >&2; return 1; }

  run_shard "${seed}" "${gpu}" "${port}" 00_24 "${first}"
  run_shard "${seed}" "${gpu}" "${port}" 25_49 "${second}"
  merge_seed "${seed}"
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
[[ ${status} -eq 0 ]] || exit "${status}"

.venv/bin/python - "${RUN_ROOT}" "${MEMER_ADAPTER}" "${V10_CHECKPOINT}" <<'PY'
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
memer_adapter = Path(sys.argv[2])
v10_checkpoint = Path(sys.argv[3])
per_seed = []
records = []
for path in sorted(root.glob("seed_*/result.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = {key: payload[key] for key in (
        "seed", "target_cup_contacts", "any_cup_contacts", "cup_selection_correct",
        "correct_selection_and_contacts", "target_lift_successes",
        "memer_grounding_parseable", "memer_grounding_correct",
    )}
    item["episodes"] = 50
    item["result_path"] = str(path.resolve())
    per_seed.append(item)
    records.extend({**record, "evaluation_seed": payload["seed"]} for record in payload["records"])
if len(per_seed) != 3 or len(records) != 150:
    raise RuntimeError(f"Expected 3 seeds and 150 records, got {len(per_seed)} and {len(records)}")

def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [center - half, center + half]

def metric(field):
    count = sum(bool(record[field]) for record in records)
    per_seed_field = {
        "memer_grounding_parseable": "memer_grounding_parseable",
        "memer_grounding_correct": "memer_grounding_correct",
        "cup_selection_correct": "cup_selection_correct",
        "any_cup_contact": "any_cup_contacts",
        "target_cup_contact": "target_cup_contacts",
        "correct_selection_and_contact": "correct_selection_and_contacts",
        "success": "target_lift_successes",
    }[field]
    rates = [item[per_seed_field] / 50 for item in per_seed]
    return {
        "count": count,
        "rate": count / 150,
        "wilson_95": wilson(count, 150),
        "seed_mean": statistics.mean(rates),
        "seed_sample_sd": statistics.stdev(rates),
    }

by_grounding = defaultdict(list)
for record in records:
    by_grounding[str(bool(record["memer_grounding_correct"]))].append(record)
summary = {
    "schema_version": 1,
    "experiment": "MemER zero-shot subgoal -> V10 action-only/no-memory ShellGame, 3-seed replication",
    "memer_adapter": str(memer_adapter.resolve()),
    "v10_checkpoint": str(v10_checkpoint.resolve()),
    "protocol": {
        "evaluation_seeds": [item["seed"] for item in per_seed],
        "episodes_per_seed": 50,
        "total_rollouts": 150,
        "paired_episode_lists": "identical to FrameSamp and V10 action baselines",
        "replan_steps": 8,
        "max_policy_steps": 150,
        "deterministic_diffusion_noise_salt_equals_evaluation_seed": True,
        "memer_shellgame_training": False,
        "memer_calls_per_episode": 1,
        "native_v10_tracker_memory": False,
        "external_semantic_memory": False,
    },
    "per_seed": per_seed,
    "aggregate": {
        "memer_grounding_parseable": metric("memer_grounding_parseable"),
        "memer_grounding_correct": metric("memer_grounding_correct"),
        "cup_selection_correct": metric("cup_selection_correct"),
        "any_cup_contact": metric("any_cup_contact"),
        "target_cup_contact": metric("target_cup_contact"),
        "correct_selection_and_contact": metric("correct_selection_and_contact"),
        "target_lift_success": metric("success"),
    },
    "by_memer_grounding_correct": {
        key: {
            "rollouts": len(items),
            "target_contacts": sum(row["target_cup_contact"] for row in items),
            "selection_correct": sum(row["cup_selection_correct"] for row in items),
            "strict_lifts": sum(row["success"] for row in items),
        }
        for key, items in sorted(by_grounding.items())
    },
}
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"summary={out.resolve()}")
PY

touch "${RUN_ROOT}/_COMPLETE"
echo "completed ${RUN_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
BANK=${WORKSPACE}/.codex_tmp/framesamp_v10_nominal5000_step9999_260829
ADAPTER=${OPENPI_ROOT}/checkpoints/pi0_shellgame_framesamp_v10_action_adapter_eef7_v1/framesamp_modul_step9999_v10_adapter_nominal5000_b8_s500_260828/499
RUN_ROOT=${OPENPI_ROOT}/evaluation/shellgame/framesamp_v10_adapter_step499_3seed50_260829
SEEDS=(260829 261829 262829)
GPUS=(5 6 7)
PORTS=(8429 8430 8431)

EPISODES_260829=4996,2518,1847,3865,3400,3183,1695,4398,1535,2703,2915,2197,3097,2553,2982,3310,4236,3658,2722,245,3915,1225,1495,1455,2273,1720,4739,2826,80,3455,2678,3378,387,4832,3809,2486,265,1138,4520,660,3105,455,3552,96,4301,3367,4894,935,293,4556
EPISODES_261829=3310,3753,2841,1138,3207,3455,4917,2100,4556,4520,3002,3097,381,4763,2197,3202,4206,387,2826,2722,2835,90,3992,1463,80,870,3050,2982,4744,1082,3460,4001,1160,4131,4996,2553,4236,3825,1225,4832,643,4178,1633,455,3378,2678,2273,1847,4060,96
EPISODES_262829=870,719,90,3753,3002,3039,1463,2273,585,3105,2826,3367,2835,265,660,2722,4236,1624,3865,3183,4173,4280,1082,80,4917,3915,2915,1325,3846,455,4060,1455,4556,1535,1720,4520,3050,3310,4763,3658,1225,2095,2553,1183,2973,3460,3378,2247,4178,3400

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache

cd "${OPENPI_ROOT}"
[[ -f "${BANK}/_COMPLETE" ]] || { echo "Incomplete FrameSamp bank ${BANK}" >&2; exit 1; }
[[ -d "${ADAPTER}/params" ]] || { echo "Missing adapter ${ADAPTER}" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite ${RUN_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"

episode_list() {
  local seed=$1
  local name=EPISODES_${seed}
  printf '%s' "${!name}"
}

run_shard() {
  local seed=$1
  local port=$2
  local ordinal=$3
  local episodes=$4
  local output_dir=${RUN_ROOT}/seed_${seed}/shard_${ordinal}
  local log=${OPENPI_ROOT}/eval_framesamp_v10_adapter_step499_seed${seed}_shard${ordinal}_260829.log
  mkdir -p "${output_dir}"

  .venv/bin/python scripts/mem/eval_shellgame_framesamp_v10_action_adapter.py \
    --direct-memory "${BANK}" \
    --checkpoint "${ADAPTER}" \
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
    "experiment": "MME FrameSamp memory -> trained interface -> frozen V10 action, 3-seed replication",
    "seed": seed,
    "noise_salt": seed,
    "checkpoint": payloads[0]["checkpoint"],
    "framesamp_bank": payloads[0]["framesamp_bank"],
    "control": payloads[0]["control"],
    "episodes": [int(record["episode"]) for record in records],
    "target_cup_contacts": sum(record["target_cup_contact"] for record in records),
    "any_cup_contacts": sum(record["any_cup_contact"] for record in records),
    "cup_selection_correct": sum(record["cup_selection_correct"] for record in records),
    "correct_selection_and_contacts": sum(record["correct_selection_and_contact"] for record in records),
    "target_lift_successes": sum(record["success"] for record in records),
    "records": records,
    "shards": [str(path.resolve()) for path in paths],
}
out = root / "result.json"
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"seed={seed} target_contact={result['target_cup_contacts']}/50 result={out.resolve()}")
PY
}

run_one() {
  local seed=$1
  local gpu=$2
  local port=$3
  local list
  list=$(episode_list "${seed}")
  IFS=',' read -r -a ids <<<"${list}"
  if [[ ${#ids[@]} -ne 50 ]]; then
    echo "seed=${seed}: expected 50 episode ids, got ${#ids[@]}" >&2
    return 1
  fi
  local first second
  first=$(IFS=,; printf '%s' "${ids[*]:0:25}")
  second=$(IFS=,; printf '%s' "${ids[*]:25:25}")
  local label=framesamp_v10_adapter_step499_seed${seed}_50ep
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
  CUDA_VISIBLE_DEVICES=${gpu} XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    JAX_COMPILATION_CACHE_DIR=${cache} \
    .venv/bin/python examples/shellgame/serve_v10_exact_parallel_semantic_adapter_deterministic.py \
      --adapter-checkpoint-dir "${ADAPTER}" \
      --adapter-mode semantic_replace \
      --semantic-memory-tokens 512 \
      --semantic-memory-width 1024 \
      --semantic-query-tokens 16 \
      --port "${port}" \
      --num-sampling-steps 4 \
      --deterministic-noise >"${server_log}" 2>&1 &
  server_pid=$!

  local ready=0
  for _ in $(seq 1 180); do
    if grep -q "server listening on" "${server_log}"; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "${label}: server exited before ready" >&2
      tail -n 120 "${server_log}" >&2
      cleanup
      return 1
    fi
    sleep 2
  done
  if (( ready == 0 )); then
    echo "${label}: timed out waiting for server" >&2
    tail -n 120 "${server_log}" >&2
    cleanup
    return 1
  fi

  if ! run_shard "${seed}" "${port}" 00_24 "${first}"; then
    cleanup
    return 1
  fi
  if ! run_shard "${seed}" "${port}" 25_49 "${second}"; then
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

.venv/bin/python - "${RUN_ROOT}" "${ADAPTER}" <<'PY'
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
results = []
records = []
for path in sorted(root.glob("seed_*/result.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    results.append(
        {
            "seed": payload["seed"],
            "episodes": 50,
            "target_cup_contacts": payload["target_cup_contacts"],
            "any_cup_contacts": payload["any_cup_contacts"],
            "cup_selection_correct": payload["cup_selection_correct"],
            "correct_selection_and_contacts": payload["correct_selection_and_contacts"],
            "target_lift_successes": payload["target_lift_successes"],
            "result_path": str(path.resolve()),
        }
    )
    records.extend({**record, "evaluation_seed": payload["seed"]} for record in payload["records"])

if len(results) != 3 or len(records) != 150:
    raise RuntimeError(f"Expected 3 results and 150 records, got {len(results)} and {len(records)}")
if len({(int(record["evaluation_seed"]), int(record["episode"])) for record in records}) != 150:
    raise RuntimeError("Duplicate seed/episode record")

def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [center - half, center + half]

n = len(records)
target_contact = sum(record["target_cup_contact"] for record in records)
rates = [result["target_cup_contacts"] / 50 for result in results]
by_target = defaultdict(list)
for record in records:
    by_target[record["target_cup_identity_scoring_only"]].append(record)
summary = {
    "checkpoint": str(checkpoint.resolve()),
    "protocol": {
        "evaluation_seeds": [result["seed"] for result in results],
        "episodes_per_seed": 50,
        "total_rollouts": n,
        "episode_pool": "balanced samples from the prior formal seed42 held-out 100-episode pool",
        "deterministic_diffusion_noise_salt_equals_evaluation_seed": True,
        "replan_steps": 8,
        "max_policy_steps": 150,
    },
    "per_seed": results,
    "aggregate": {
        "target_cup_contacts": target_contact,
        "target_cup_contact_rate": target_contact / n,
        "target_cup_contact_wilson_95": wilson(target_contact, n),
        "seed_mean_target_contact_rate": statistics.mean(rates),
        "seed_sample_sd_target_contact_rate": statistics.stdev(rates),
        "any_cup_contacts": sum(record["any_cup_contact"] for record in records),
        "cup_selection_correct": sum(record["cup_selection_correct"] for record in records),
        "correct_selection_and_contacts": sum(record["correct_selection_and_contact"] for record in records),
        "target_lift_successes": sum(record["success"] for record in records),
    },
    "by_target_cup": {
        cup: {
            "rollouts": len(items),
            "target_contacts": sum(item["target_cup_contact"] for item in items),
            "selection_correct": sum(item["cup_selection_correct"] for item in items),
            "target_lift_successes": sum(item["success"] for item in items),
        }
        for cup, items in sorted(by_target.items())
    },
}
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"summary={out.resolve()}")
PY

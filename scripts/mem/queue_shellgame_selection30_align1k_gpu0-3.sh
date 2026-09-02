#!/usr/bin/env bash
set -euo pipefail

OPENPI_ROOT=/data2/hzl_workspace_for_pi_mem/openpi-umi
WORKSPACE=/data2/hzl_workspace_for_pi_mem
EXP_NAME=direct_visual_mem_step999_v10_selection30_align1k_lr1e5_4gpu_260826
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_qwen_distilled_memory_action_selection30_align1k_eef7_260826/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_qwen_distilled_memory_action_selection30_align1k_lr1e5_4gpu_260826.log

cd "${OPENPI_ROOT}"
exec >>"${LOG}" 2>&1

echo "[$(date --iso-8601=seconds)] queued GPUs=0,1,2,3 exp=${EXP_NAME}"
if [[ -e "${OUTPUT}" ]]; then
  echo "[$(date --iso-8601=seconds)] refusing_existing_output=${OUTPUT}"
  exit 1
fi

idle_checks=0
poll_count=0
while (( idle_checks < 3 )); do
  active_pids=$(nvidia-smi -i 0,1,2,3 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | sort -u || true)
  if [[ -z "${active_pids}" ]]; then
    idle_checks=$((idle_checks + 1))
    echo "[$(date --iso-8601=seconds)] gpu_idle_check=${idle_checks}/3"
  else
    idle_checks=0
    if (( poll_count % 15 == 0 )); then
      echo "[$(date --iso-8601=seconds)] waiting active_pids=$(tr '\n' ',' <<<"${active_pids}" | sed 's/,$//')"
    fi
  fi
  poll_count=$((poll_count + 1))
  if (( idle_checks < 3 )); then
    sleep 20
  fi
done

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((50 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "[$(date --iso-8601=seconds)] insufficient_disk_kb=${available_kb}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export HF_HOME=${WORKSPACE}/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_selection30_align1k_4gpu
mkdir -p "${HF_DATASETS_CACHE}" "${JAX_COMPILATION_CACHE_DIR}"

echo "[$(date --iso-8601=seconds)] training_start steps=1000 warmup=50 peak_lr=1e-5 decay_lr=1e-6 fsdp=4"
exec .venv/bin/python \
  scripts/mem/train_shellgame_qwen_distilled_memory_action_selection30_align1k.py \
  --exp-name "${EXP_NAME}" \
  --steps 1000 \
  --batch-size 12 \
  --fsdp-devices 4 \
  --num-workers 8

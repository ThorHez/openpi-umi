#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
CONFIG=pi05_shellgame_baseline_v1
GPU_IDS=${GPU_IDS:-6,7}
FSDP_DEVICES=${FSDP_DEVICES:-2}
EXP_NAME=${EXP_NAME:-full_ft_seed42_b32_2gpu_260827}
STEPS=${STEPS:-30000}
RESUME=${RESUME:-0}
LOG=${LOG:-${OPENPI_ROOT}/train_pi05_shellgame_baseline_v1_${EXP_NAME}.log}
OUTPUT=${OPENPI_ROOT}/checkpoints/${CONFIG}/${EXP_NAME}
DATASET=${OPENPI_ROOT}/data/shellgame_static_phase_instruction_dataset2

cd "${OPENPI_ROOT}"

[[ -d "${DATASET}/data" ]] || { echo "Missing ShellGame dataset: ${DATASET}" >&2; exit 1; }
[[ -f "${DATASET}/norm_stats.json" ]] || { echo "Missing norm stats: ${DATASET}/norm_stats.json" >&2; exit 1; }
if [[ "${RESUME}" == 1 ]]; then
  [[ -d "${OUTPUT}" ]] || { echo "Cannot resume missing run: ${OUTPUT}" >&2; exit 1; }
else
  [[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite existing run: ${OUTPUT}" >&2; exit 1; }
fi

IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
if (( ${#gpu_array[@]} != FSDP_DEVICES )); then
  echo "GPU_IDS count must equal FSDP_DEVICES=${FSDP_DEVICES}; got GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi

# Saving a new full checkpoint can temporarily overlap the previous 42-GiB
# checkpoint. Resume reuses the tmpfs data cache and only needs one checkpoint
# plus margin; a fresh run keeps the more conservative original headroom.
available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
if [[ "${RESUME}" == 1 ]]; then
  required_gib=50
else
  required_gib=100
fi
required_kb=$((required_gib * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "Refusing to train with less than ${required_gib} GiB free (available_kb=${available_kb})" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
# Decoding the 4,470 local Parquet files creates a roughly 64-GiB Arrow cache.
# Keep this derived cache in tmpfs so checkpoints retain safe disk headroom.
export HF_HOME=${HF_HOME:-/dev/shm/pi05_shellgame_baseline_v1/huggingface}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export WANDB_MODE=disabled

mkdir -p "${UV_CACHE_DIR}" "${HF_DATASETS_CACHE}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] pi05 ShellGame baseline v1 start"
echo "config=${CONFIG} exp_name=${EXP_NAME} steps=${STEPS} physical_gpus=${GPU_IDS} fsdp_devices=${FSDP_DEVICES} resume=${RESUME}"
echo "dataset=${DATASET} output=${OUTPUT}"
echo "contract=current-frame two-view, raw7 action, pi05 32d padded head, no memory"

train_args=(
  "${CONFIG}"
  --exp-name "${EXP_NAME}"
  --num-train-steps "${STEPS}"
  --fsdp-devices "${FSDP_DEVICES}"
)
if [[ "${RESUME}" == 1 ]]; then
  train_args+=(--resume)
fi

exec "${OPENPI_ROOT}/.venv/bin/python" scripts/train_multi_dataset.py "${train_args[@]}"

#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
CONFIG=pi0_mem_semantic_action_shellgame_real_wrist_currentrel_eef10_stage2
DATASET=${OPENPI_ROOT}/data/shellgame_real_306_degap_state_epfirst_action_currentrel_eef10
MEMORY_CHECKPOINT=${WORKSPACE}/4999/params

GPU_IDS=${GPU_IDS:-0,1,2,3}
FSDP_DEVICES=${FSDP_DEVICES:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
STEPS=${STEPS:-21000}
EVAL_BATCHES=${EVAL_BATCHES:-64}
EXP_NAME=${EXP_NAME:-real306_currentrel_full80_interface_pi05_seed42_v1}
RESUME=${RESUME:-0}
LOG=${LOG:-${OPENPI_ROOT}/train_${CONFIG}_${EXP_NAME}.log}
OUTPUT=${OPENPI_ROOT}/checkpoints/${CONFIG}/${EXP_NAME}

cd "${OPENPI_ROOT}"

[[ -f "${DATASET}/meta/info.json" ]] || {
  echo "Missing converted LeRobot dataset: ${DATASET}" >&2
  echo "Run scripts/mem/convert_real_shellgame_stage2_epfirst.py to rebuild it." >&2
  exit 1
}
[[ -f "${DATASET}/norm_stats.json" ]] || {
  echo "Missing real-data norm stats: ${DATASET}/norm_stats.json" >&2
  exit 1
}
[[ -f "${MEMORY_CHECKPOINT}/_METADATA" ]] || {
  echo "Missing matching 306-degap MEM checkpoint: ${MEMORY_CHECKPOINT}" >&2
  exit 1
}

IFS=',' read -r -a gpu_array <<<"${GPU_IDS}"
if (( ${#gpu_array[@]} != FSDP_DEVICES )); then
  echo "GPU_IDS count must equal FSDP_DEVICES=${FSDP_DEVICES}; got ${GPU_IDS}" >&2
  exit 1
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No usable NVIDIA driver/GPU is visible; refusing to start training." >&2
  exit 1
fi

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((100 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "Refusing to train with less than 100 GiB free; available_kb=${available_kb}" >&2
  exit 1
fi

if [[ "${RESUME}" == 1 ]]; then
  [[ -d "${OUTPUT}" ]] || { echo "Cannot resume missing run: ${OUTPUT}" >&2; exit 1; }
else
  [[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite existing run: ${OUTPUT}" >&2; exit 1; }
fi

export CUDA_VISIBLE_DEVICES=${GPU_IDS}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
# The first LeRobot load materializes a large Arrow index for all 118,807
# image-bearing rows. /dev/shm and /tmp are too small on the training hosts;
# keep this cache on the checked, high-capacity workspace filesystem.
export HF_HOME=${HF_HOME:-${OPENPI_ROOT}/.cache/shellgame_real_stage2/huggingface}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export WANDB_MODE=${WANDB_MODE:-disabled}
mkdir -p "${HF_DATASETS_CACHE}"

train_args=(
  "${CONFIG}"
  --exp-name "${EXP_NAME}"
  --num-train-steps "${STEPS}"
  --batch-size "${BATCH_SIZE}"
  --fsdp-devices "${FSDP_DEVICES}"
  --eval-batches "${EVAL_BATCHES}"
)
if [[ "${RESUME}" == 1 ]]; then
  train_args+=(--resume)
fi

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] real ShellGame stage2 start"
echo "config=${CONFIG} exp_name=${EXP_NAME} steps=${STEPS} batch=${BATCH_SIZE}"
echo "gpus=${GPU_IDS} fsdp_devices=${FSDP_DEVICES} resume=${RESUME}"
echo "memory_checkpoint=${MEMORY_CHECKPOINT} dataset=${DATASET}"
echo "contract=241-frame fixed history + current wrist; state=ep-first, action=current-relative link6 EEF10"
echo "trainable=memory query resampler + memory cross-attention + Pi0.5 action expert"

exec "${OPENPI_ROOT}/.venv/bin/python" scripts/mem/train_mem.py "${train_args[@]}"

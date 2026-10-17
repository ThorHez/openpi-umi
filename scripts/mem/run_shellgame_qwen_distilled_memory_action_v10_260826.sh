#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
# GPU4 currently hosts another user's long-running policy server; GPU7 remains
# reserved as requested.  Use the six otherwise-idle physical devices.
GPU_IDS=${GPU_IDS:-0,1,2,3,5,6}
STEPS=${STEPS:-500}
EXP_NAME=${EXP_NAME:-direct_visual_mem_step999_v10_mix500_6gpu_260826}
LOG=${OPENPI_ROOT}/train_qwen_distilled_memory_action_v10_260826.log

NOMINAL_ROOT=${WORKSPACE}/robosuite/outputs/shellgame_lerobot_absolute_eef_raw7
V6_ROOT=${WORKSPACE}/robosuite/outputs/shellgame_lerobot_onpolicy_eef_low_stage_gated_v6_balanced1200_260816
V9_ROOT=${WORKSPACE}/robosuite/outputs/shellgame_lerobot_onpolicy_eef_safe_balanced_recovery_v9_balanced1200_260819
NOMINAL_MEMORY=${OPENPI_ROOT}/artifacts/shellgame_qwen_distilled_direct_visual_memory_step999_all5000_260825.npz
V6_MEMORY=${OPENPI_ROOT}/artifacts/shellgame_qwen_distilled_direct_visual_memory_step999_v6_all1200_260826.npz
V9_MEMORY=${OPENPI_ROOT}/artifacts/shellgame_qwen_distilled_direct_visual_memory_step999_v9_all1200_260826.npz
INIT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_qwen_event_memory_action_eef7_260825/direct_visual_mem_step999_filtered_action250_6gpu_260825/249/params
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_qwen_distilled_memory_action_v10_eef7_260826/${EXP_NAME}

cd "${OPENPI_ROOT}"
for path in "${NOMINAL_ROOT}" "${V6_ROOT}" "${V9_ROOT}" "${NOMINAL_MEMORY}" "${INIT}"; do
  [[ -e "${path}" ]] || { echo "Missing prerequisite: ${path}" >&2; exit 1; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((50 * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "Refusing to cache/train with less than 50 GiB free" >&2
  exit 1
fi

export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_qwen_distilled_memory_action_v10
export HF_HOME=${WORKSPACE}/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_DATASETS_CACHE}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] pipeline_start GPUs=${GPU_IDS} steps=${STEPS}"
echo "contract=frozen_visual_mem_frames0_59,current_image_state_live,source_mass=60/30/10,replan_target=8"

if [[ ! -e "${V6_MEMORY}" ]]; then
  echo "[$(date --iso-8601=seconds)] cache_v6_start"
  CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.88 \
    "${OPENPI_ROOT}/.venv/bin/python" scripts/mem/cache_shellgame_qwen_distilled_visual_memory_dataset.py \
      --dataset-root "${V6_ROOT}" \
      --output "${V6_MEMORY}" \
      --batch-size 12 \
      --fsdp-devices 6
  echo "[$(date --iso-8601=seconds)] cache_v6_done"
fi

if [[ ! -e "${V9_MEMORY}" ]]; then
  echo "[$(date --iso-8601=seconds)] cache_v9_start"
  CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.88 \
    "${OPENPI_ROOT}/.venv/bin/python" scripts/mem/cache_shellgame_qwen_distilled_visual_memory_dataset.py \
      --dataset-root "${V9_ROOT}" \
      --output "${V9_MEMORY}" \
      --batch-size 12 \
      --fsdp-devices 6
  echo "[$(date --iso-8601=seconds)] cache_v9_done"
fi

echo "[$(date --iso-8601=seconds)] action_train_start init=${INIT} output=${OUTPUT}"
CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  "${OPENPI_ROOT}/.venv/bin/python" scripts/mem/train_shellgame_qwen_distilled_memory_action_v10.py \
    --exp-name "${EXP_NAME}" \
    --nominal-memory "${NOMINAL_MEMORY}" \
    --v6-memory "${V6_MEMORY}" \
    --v9-memory "${V9_MEMORY}" \
    --init-checkpoint "${INIT}" \
    --steps "${STEPS}" \
    --batch-size 12 \
    --fsdp-devices 6 \
    --num-workers 8
echo "[$(date --iso-8601=seconds)] pipeline_done"

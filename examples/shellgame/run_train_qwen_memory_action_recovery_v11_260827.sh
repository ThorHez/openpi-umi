#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
GPU_IDS=${GPU_IDS:-0,1,2,5,6,7}
STEPS=${STEPS:-1000}
PEAK_LR=${PEAK_LR:-1e-6}
EXP_NAME=${EXP_NAME:-current_action_recovery_v11_nom50_v6_20_v11_30_step1000_lr1e-6_b12_6gpu_260827}
INIT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_qwen_distilled_memory_waypoint_grasp_v6_eef7_260826/direct_visual_waypoint_grasp_v6_60_30_5_3_2_3k_6gpu_260826/2000/params
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_qwen_memory_action_recovery_v11_eef7_260827/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_qwen_memory_action_recovery_v11_1000steps_260827.log

cd "${OPENPI_ROOT}"
[[ -d "${INIT}" ]] || { echo "Missing init checkpoint: ${INIT}" >&2; exit 1; }
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((30 * 1024 * 1024))
echo "disk_free_gib=$((available_kb / 1024 / 1024)) required_gib=30"
if (( available_kb < required_kb )); then
  echo "Refusing to train with less than 30 GiB free" >&2
  exit 1
fi

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_qwen_action_v11
export HF_HOME=${WORKSPACE}/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_DATASETS_CACHE}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Starting current-action V11 recovery adaptation"
echo "physical_gpus=${GPU_IDS} fsdp_devices=6 steps=${STEPS} peak_lr=${PEAK_LR}"
echo "source_mass=nominal:0.50,v6_preservation:0.20,v11_current_action_oracle:0.30"
echo "trainable=pi0_action_expert_and_action_time_projections"
echo "init=${INIT} output=${OUTPUT}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run python scripts/mem/train_shellgame_qwen_memory_action_recovery_v11.py \
  --exp-name "${EXP_NAME}" \
  --init-checkpoint "${INIT}" \
  --steps "${STEPS}" \
  --peak-lr "${PEAK_LR}" \
  --batch-size 12 \
  --num-workers 8 \
  --fsdp-devices 6

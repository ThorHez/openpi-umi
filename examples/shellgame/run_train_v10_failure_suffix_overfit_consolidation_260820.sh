#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5}
STEPS=${STEPS:-150}
EXP_NAME=${EXP_NAME:-absolute_eef7_v10_failure_suffix_consolidation_30_30_20_20_xy4_lr2e6_b6_${STEPS}steps_6gpu_260820}
INIT=${INIT:-${OPENPI_ROOT}/checkpoints/pi0_shellgame_v10_failure_suffix_overfit_probe_260820/absolute_eef7_v10_failure_suffix_exact3_xy4_b6_300steps_6gpu_260820/100/params}
OUTPUT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_v10_failure_suffix_overfit_consolidation_260820/${EXP_NAME}
LOG=${OPENPI_ROOT}/train_eef7_v10_failure_suffix_overfit_consolidation_${STEPS}steps_260820.log

cd "${OPENPI_ROOT}"
[[ -d "${INIT}" ]] || { echo "Missing init checkpoint: ${INIT}" >&2; exit 1; }
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing to overwrite ${OUTPUT}" >&2; exit 1; }

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((65 * 1024 * 1024))
echo "disk_free_gib=$((available_kb / 1024 / 1024)) required_gib=65"
if (( available_kb < required_kb )); then
  echo "Refusing to train with less than 65 GiB free" >&2
  exit 1
fi

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_v10_failure_suffix_probe
export HF_HOME=${WORKSPACE}/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_DATASETS_CACHE}"

exec >>"${LOG}" 2>&1
echo "[$(date --iso-8601=seconds)] Starting low-LR single-expert consolidation probe"
echo "physical_gpus=${GPU_IDS} fsdp_devices=6 steps=${STEPS}"
echo "sampler=recenter30_descend30_grasp20_lift20 xy_loss_weight=4 gripper_loss_weight=4"
echo "init=${INIT} output=${OUTPUT}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
uv run python examples/shellgame/train_v10_failure_suffix_overfit_consolidation.py \
  --exp-name "${EXP_NAME}" \
  --init-checkpoint "${INIT}" \
  --steps "${STEPS}" \
  --warmup-steps 10 \
  --peak-lr 2e-6 \
  --batch-size 6 \
  --num-workers 6 \
  --fsdp-devices 6 \
  --eval-interval 25 \
  --eval-batches 10 \
  --save-interval 25 \
  --keep-period 25 \
  --gripper-loss-weight 4.0


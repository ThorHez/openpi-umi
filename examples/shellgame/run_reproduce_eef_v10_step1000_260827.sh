#!/usr/bin/env bash
set -euo pipefail

# Reproduce the historical V10 step-1000 checkpoint without touching the
# original experiment directory.  The first stage repeats the exact 0..499
# schedule; the second stage restores the complete train state and continues
# at the terminal 3e-7 learning rate through checkpoint step 1000.

WORKSPACE=/data2/hzl_workspace_for_pi_mem
OPENPI_ROOT=${WORKSPACE}/openpi-umi
GPU_IDS=${GPU_IDS:-0,1,2,5,6,7}
XLA_MEM_FRACTION=${XLA_MEM_FRACTION:-0.90}
XLA_PREALLOCATE=${XLA_PREALLOCATE:-false}
EXP_NAME=${EXP_NAME:-absolute_eef7_v10_repro_nom60_v6preserve30_v9timing10_b12_step1000_6gpu_noprealloc_260827}
CHECKPOINT_ROOT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/${EXP_NAME}
INIT=${OPENPI_ROOT}/checkpoints/pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v6_260816/absolute_eef7_mixed_correction_v6_dynamic_phase_60_30_5_3_2_b12_3k_6gpu_260816/5999/params

LABEL=v10_repro_step1000_replan8
EVAL_OUTPUT=${OPENPI_ROOT}/evaluation/shellgame/eef7_v10_repro_step1000_replan8_isolated20_seed260813_260827
EVAL_LOG=${OPENPI_ROOT}/eval_eef7_v10_repro_step1000_replan8_isolated20_seed260813_260827.log

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export UV_CACHE_DIR=${WORKSPACE}/.codex_tmp/uv_cache
export JAX_COMPILATION_CACHE_DIR=${WORKSPACE}/.codex_tmp/jax_cache_eef_v9_safe_erroraware
export HF_HOME=${WORKSPACE}/.cache/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PREALLOCATE}

cd "${OPENPI_ROOT}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${HF_DATASETS_CACHE}"

[[ -d "${INIT}" ]] || { echo "Missing V6-5999 init: ${INIT}" >&2; exit 1; }
[[ ! -e "${CHECKPOINT_ROOT}" ]] || {
  echo "Refusing to overwrite reproduction checkpoint: ${CHECKPOINT_ROOT}" >&2
  exit 1
}
[[ ! -e "${EVAL_OUTPUT}" ]] || {
  echo "Refusing to overwrite reproduction evaluation: ${EVAL_OUTPUT}" >&2
  exit 1
}

available_kb=$(df -Pk "${WORKSPACE}" | awk 'NR==2 {print $4}')
required_kb=$((100 * 1024 * 1024))
echo "disk_free_gib=$((available_kb / 1024 / 1024)) required_gib=100"
if (( available_kb < required_kb )); then
  echo "Refusing to reproduce V10 with less than 100 GiB free" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] stage1: reproduce V10 steps 0..499"
echo "physical_gpus=${GPU_IDS} xla_mem_fraction=${XLA_MEM_FRACTION} xla_preallocate=${XLA_PREALLOCATE} source_mass=nominal:0.60,v6:0.30,v9:0.10"
CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM_FRACTION} \
  uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v10_timing_diag.py \
    --exp-name "${EXP_NAME}" \
    --init-checkpoint "${INIT}" \
    --steps 500 \
    --warmup-steps 50 \
    --peak-lr 3e-6 \
    --batch-size 12 \
    --num-workers 8 \
    --fsdp-devices 6 \
    --eval-interval 125 \
    --eval-batches 20 \
    --save-interval 250 \
    --keep-period 250 \
    --gripper-loss-weight 4.0

[[ -d "${CHECKPOINT_ROOT}/499/train_state" ]] || {
  echo "Stage 1 did not produce resumable step-499 train state" >&2
  exit 1
}

echo "[$(date --iso-8601=seconds)] stage2: resume complete state through step 1000"
CUDA_VISIBLE_DEVICES=${GPU_IDS} XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM_FRACTION} \
  uv run python examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v10_continue.py \
    --exp-name "${EXP_NAME}" \
    --steps 1001 \
    --warmup-steps 50 \
    --peak-lr 3e-6 \
    --batch-size 12 \
    --num-workers 8 \
    --fsdp-devices 6 \
    --eval-interval 250 \
    --eval-batches 20 \
    --save-interval 500 \
    --keep-period 500 \
    --gripper-loss-weight 4.0

[[ -d "${CHECKPOINT_ROOT}/1000/params" ]] || {
  echo "Stage 2 did not produce checkpoint step 1000" >&2
  exit 1
}

echo "[$(date --iso-8601=seconds)] stage3: fixed-seed 20-episode replan=8 gate"
OPENPI_EVAL_REPLAN_STEPS_OVERRIDE=8 \
  bash examples/shellgame/run_eval_eef_v6_checkpoint_260816.sh \
    "${LABEL}" "${CHECKPOINT_ROOT}/1000" 0 8062 "${EVAL_OUTPUT}" "${EVAL_LOG}" unguarded 20

echo "[$(date --iso-8601=seconds)] V10 step-1000 reproduction complete"
echo "checkpoint=${CHECKPOINT_ROOT}/1000"
echo "evaluation=${EVAL_OUTPUT}/result.json"

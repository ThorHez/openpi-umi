#!/usr/bin/env bash
set -euo pipefail

cd /data2/hzl_workspace_for_pi_mem/openpi-umi

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

result_root="evaluation/shellgame/teacher_memory_necessity_12f_3seed_260831"
mkdir -p "${result_root}/logs"

run_variant() {
  local seed="$1"
  local variant="$2"
  local distill_weight="$3"
  local exp_name="teacher_necessity_12f_${variant}_seed${seed}_260831"
  local exp_dir="checkpoints/shellgame_qwen_distilled_direct_visual_recurrent_memory_probe/${exp_name}"
  local checkpoint="checkpoints/shellgame_qwen_distilled_direct_visual_recurrent_memory_probe/${exp_name}/999/_CHECKPOINT_METADATA"
  local log_path="${result_root}/logs/${exp_name}.log"
  local resume_args=()

  if [[ -f "${checkpoint}" ]]; then
    echo "Skipping completed ${exp_name}"
    return
  fi
  if [[ -d "${exp_dir}" ]]; then
    resume_args=(--resume)
  fi

  .venv/bin/python \
    examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py \
    --exp-name "${exp_name}" \
    --seed "${seed}" \
    --split-seed 42 \
    --steps 1000 \
    --warmup-steps 50 \
    --peak-lr 3e-4 \
    --decay-lr 3e-5 \
    --memory-distill-weight "${distill_weight}" \
    --stage-slot-weight 1.0 \
    --student-segment-size 12 \
    --batch-size 12 \
    --num-workers 0 \
    --fsdp-devices 6 \
    --eval-interval 100 \
    --eval-batches 20 \
    --save-interval 500 \
    "${resume_args[@]}" \
    2>&1 | tee "${log_path}"
}

# Seed 42 is the already completed matched pair reported in the paper.
for seed in 43 44; do
  run_variant "${seed}" state_only 0.0
  run_variant "${seed}" state_plus_distill 1.0
done

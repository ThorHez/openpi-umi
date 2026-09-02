#!/usr/bin/env bash
set -euo pipefail

cd /data2/hzl_workspace_for_pi_mem/openpi-umi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

RESULT_DIR=evaluation/shellgame/causal_window6_joint_action_260821
mkdir -p "${RESULT_DIR}"

for mode in normal shuffle_batch zero; do
  .venv/bin/python3 \
    examples/shellgame/eval_causal_window6_joint_action_probe.py \
    --exp-name "causal_window6_joint_action_${mode}_260821" \
    --raw-memory-mode "${mode}" \
    --batch-size 6 \
    --fsdp-devices 6 \
    --eval-batches 5 \
    --cup-eval-episodes 30 \
    --cup-eval-batch-size 6 \
    --num-workers 0 \
    --overwrite \
    > "${RESULT_DIR}/${mode}.log" 2>&1
done

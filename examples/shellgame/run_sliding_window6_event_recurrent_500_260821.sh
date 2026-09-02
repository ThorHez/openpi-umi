#!/usr/bin/env bash
set -euo pipefail

cd /data2/hzl_workspace_for_pi_mem/openpi-umi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

exec .venv/bin/python3 \
  examples/shellgame/train_sliding_window6_event_recurrent_memory_probe.py \
  --exp-name sliding_window6_event_gate_500_260821 \
  --steps 500 \
  --warmup-steps 50 \
  --peak-lr 1e-4 \
  --batch-size 6 \
  --fsdp-devices 6 \
  --eval-interval 100 \
  --eval-batches 20 \
  --save-interval 250 \
  --num-workers 0 \
  --overwrite \
  >> evaluation/shellgame/sliding_window6_event_recurrent_260821/train.log 2>&1

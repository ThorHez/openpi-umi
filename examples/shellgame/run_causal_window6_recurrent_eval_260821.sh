#!/usr/bin/env bash
set -euo pipefail

cd /data2/hzl_workspace_for_pi_mem/openpi-umi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p evaluation/shellgame/causal_window6_recurrent_260821

exec .venv/bin/python3 \
  examples/shellgame/eval_causal_window6_recurrent_memory_probe.py \
  --exp-name causal_window6_recurrent_eval_260821 \
  --batch-size 6 \
  --fsdp-devices 6 \
  --eval-batches 20 \
  --num-workers 0 \
  --overwrite \
  >> evaluation/shellgame/causal_window6_recurrent_260821/eval.log 2>&1

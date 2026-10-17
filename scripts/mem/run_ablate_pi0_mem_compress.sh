#!/usr/bin/env bash
# Convenience wrapper around scripts/mem/ablate_pi0_mem_compress.py.
#
# Usage:
#   bash scripts/mem/run_ablate_pi0_mem_compress.sh                       # all 5 modes, default checkpoint, multi-GPU
#   bash scripts/mem/run_ablate_pi0_mem_compress.sh normal                # single mode
#   bash scripts/mem/run_ablate_pi0_mem_compress.sh all 47000 5           # all modes, step=47000, 5 episodes
#   BATCH_SIZE=16 bash scripts/mem/run_ablate_pi0_mem_compress.sh         # 16 frames per JIT call
#   CUDA_VISIBLE_DEVICES=0 NO_MULTI_GPU=1 bash scripts/...                # single-GPU fast mode
#
# Args (all optional):
#   $1: ablation mode  -> one of {normal, repeat_current, wrong_history, shuffle_history, memory_off, all}
#                         (default: all)
#   $2: checkpoint step folder name under the experiment dir (default: 47000)
#   $3: number of episodes to evaluate (default: 3)
#   $4: dataset path (default: training dataset)
#   $5: output_dir suffix; appended to /data1/.../ablation_results_<suffix> (default: empty)
#
# Env (all optional):
#   BATCH_SIZE       per-JIT-call batch (default 0 = auto = jax.device_count() in multi-GPU mode)
#   NO_MULTI_GPU=1   keep everything on jax.devices()[0] (single-device batching)
#
# Notes:
# - Memory_off runs are slightly more expensive because we re-jit the model after
#   mutating its history-memory gate logits.
# - We export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 by default so the JAX allocator does
#   not aggressively pre-reserve all 80 GB on H100s; unset/override in env if you need
#   to share GPU memory with another process.

set -euo pipefail

cd "$(dirname "$0")/../.."  # cd to openpi-umi root

MODE="${1:-all}"
STEP="${2:-47000}"
NUM_EPISODES="${3:-3}"
DATASET_PATH="${4:-/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51}"
OUTPUT_SUFFIX="${5:-}"

CONFIG_NAME="pi0_mem_compress_umi_32d_60k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322"
CHECKPOINT_DIR="/data1/hzl_workspace_for_pi/openpi-umi/checkpoints/${CONFIG_NAME}/my_experiment/${STEP}"

if [ -n "${OUTPUT_SUFFIX}" ]; then
    OUTPUT_DIR="/data1/hzl_workspace_for_pi/openpi-umi/ablation_results_${OUTPUT_SUFFIX}"
else
    OUTPUT_DIR="/data1/hzl_workspace_for_pi/openpi-umi/ablation_results_step${STEP}"
fi

BATCH_SIZE="${BATCH_SIZE:-0}"          # 0 = let the python side auto-pick
NO_MULTI_GPU="${NO_MULTI_GPU:-0}"      # set =1 to disable multi-GPU data-parallel
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

if [ "${NO_MULTI_GPU}" = "1" ]; then
    MULTI_GPU_FLAG="--no-multi-gpu"
else
    MULTI_GPU_FLAG="--multi-gpu"
fi

echo "===================================================="
echo "Pi0MemCompress ablation suite"
echo "  mode:        ${MODE}"
echo "  config:      ${CONFIG_NAME}"
echo "  checkpoint:  ${CHECKPOINT_DIR}"
echo "  dataset:     ${DATASET_PATH}"
echo "  num_eps:     ${NUM_EPISODES}"
echo "  output_dir:  ${OUTPUT_DIR}"
echo "  batch_size:  ${BATCH_SIZE} (0 = auto)"
echo "  multi_gpu:   ${MULTI_GPU_FLAG}"
echo "  visible GPU: ${CUDA_VISIBLE_DEVICES:-<all>}"
echo "  mem_frac:    ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "===================================================="

# tyro exposes dataclass fields as kebab-case CLI flags.
# AblationMode enum names are uppercase (NORMAL, REPEAT_CURRENT, ...) — uppercase
# the user's lowercase string to match.
MODE_UPPER="$(echo "${MODE}" | tr '[:lower:]' '[:upper:]')"

uv run scripts/mem/ablate_pi0_mem_compress.py \
    --config "${CONFIG_NAME}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --dataset-path "${DATASET_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --ablation-mode "${MODE_UPPER}" \
    --num-episodes "${NUM_EPISODES}" \
    --batch-size "${BATCH_SIZE}" \
    ${MULTI_GPU_FLAG}

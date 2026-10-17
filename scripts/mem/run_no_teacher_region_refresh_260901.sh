#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
log_root="${repo_root}/runs/training_logs/no_teacher_region_refresh_260901"
mkdir -p "${log_root}"

run_unmask_swap() {
    CUDA_VISIBLE_DEVICES=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${python_bin}" \
        "${repo_root}/scripts/mem/train_robomme_explicit_event_bottleneck_ablation.py" \
        --variant pooled_soft_causal \
        --supervision-mode terminal_only \
        --unmask-binding-labels native_single \
        --output-dir "${repo_root}/checkpoints/robomme_no_teacher_native_single_strict_seed260908_260901" \
        --steps 1600 \
        --operation-pretrain-steps 800 \
        --batch-size 12 \
        --eval-batch-size 3 \
        --eval-every 100 \
        --seed 260908 \
        >"${log_root}/videounmask_swap.log" 2>&1
}

run_place_order() {
    CUDA_VISIBLE_DEVICES=2 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${python_bin}" \
        "${repo_root}/scripts/mem/train_videoplaceorder_learned_event_ordinal_memory.py" \
        --supervision-mode terminal_only \
        --output-dir "${repo_root}/checkpoints/videoplaceorder_no_teacher_learned_event_ordinal_seed260831_260901" \
        --steps 3000 \
        --operation-pretrain-steps 1200 \
        --batch-size 8 \
        --eval-batch-size 5 \
        --eval-every 100 \
        --seed 260831 \
        >"${log_root}/videoplaceorder.log" 2>&1
}

pids=()
run_unmask_swap & pids+=("$!")
run_place_order & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"

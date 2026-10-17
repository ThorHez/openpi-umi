#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
log_root="${repo_root}/runs/training_logs/native_unmask_binding_260831"
mkdir -p "${log_root}"

run_condition() {
    local mode="$1"
    local output="$2"
    "${python_bin}" "${script_dir}/train_robomme_explicit_event_bottleneck_ablation.py" \
        --variant pooled_soft_causal \
        --unmask-binding-labels "${mode}" \
        --output-dir "${repo_root}/checkpoints/${output}" \
        --steps 1600 \
        --operation-pretrain-steps 800 \
        --batch-size 12 \
        --eval-batch-size 3 \
        --eval-every 100 \
        --seed 260908 \
        >"${log_root}/${mode}.log" 2>&1
}

pids=()
run_condition native_single robomme_explicit_event_native_single_seed260908_260831 & pids+=("$!")
run_condition native_full robomme_explicit_event_native_full_seed260908_260831 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"

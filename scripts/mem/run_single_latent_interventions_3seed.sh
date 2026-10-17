#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-${root}/.venv/bin/python}"
evaluator="${root}/scripts/mem/eval_robomme_four_task_fixed_chunk_distillation.py"
tag="${TAG:-260901_single_latent_confirm}"
log_dir="${root}/checkpoints/robomme_single_latent_ablation_logs_${tag}"
mkdir -p "${log_dir}"

run_seed() {
    local seed="$1"
    local gpu="$2"
    local training_dir="${root}/checkpoints/robomme_single_latent_latent_soft_seed${seed}_${tag}"
    local log="${log_dir}/interventions_latent_soft_seed${seed}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.35}" \
    PYTHONPATH="${root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python_bin}" "${evaluator}" \
            --training-dir "${training_dir}" \
            --split test \
            --modes normal zero_video reverse_chunks shuffle_episode_video \
            >"${log}" 2>&1
}

run_seed 260971 1 & pid_1=$!
run_seed 260972 2 & pid_2=$!
run_seed 260973 3 & pid_3=$!

status=0
for pid in "${pid_1}" "${pid_2}" "${pid_3}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"

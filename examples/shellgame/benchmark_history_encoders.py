"""GPU microbenchmark for original and fixed-grid Pi0 history encoders."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import siglip_mem_compress as _original
from openpi.models import siglip_mem_fixed_grid_temporal as _fixed


def _count_params(variables) -> int:
    return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(variables["params"]))


def _memory_stats(device) -> dict[str, int]:
    stats = device.memory_stats() or {}
    return {key: int(stats[key]) for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if key in stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("original", "fixed"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--history-frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    b = args.batch_size
    t = args.history_frames
    history = jnp.zeros((b, t, 256, 1152), dtype=jnp.bfloat16)
    current = jnp.zeros((b, 256, 1152), dtype=jnp.bfloat16)
    device = jax.devices()[0]
    before = _memory_stats(device)

    if args.mode == "original":
        module = _original.HistoryResampler(
            num_memory_tokens=256,
            num_heads=16,
            depth=1,
            mlp_dim=4304,
            dropout=0.0,
            dtype_mm="bfloat16",
            use_current_condition=True,
        )
        variables = module.init(jax.random.key(0), history, current, deterministic=True)

        @jax.jit
        def forward(params, hist, cur):
            return module.apply(params, hist, cur, deterministic=True)

        args_forward = (variables, history, current)
        score_elements = b * 16 * 256 * (t * 256)
    else:
        module = _fixed.FixedGridTemporalHistory(
            input_width=1152,
            temporal_width=256,
            temporal_depth=2,
            temporal_heads=8,
            spatial_pool_factor=2,
            num_memory_tokens=128,
            output_width=1152,
            dropout=0.0,
            dtype_mm="bfloat16",
        )
        variables = module.init(jax.random.key(0), history, deterministic=True)

        @jax.jit
        def forward(params, hist, cur):
            del cur
            return module.apply(params, hist, deterministic=True)

        args_forward = (variables, history, current)
        temporal_scores = b * 64 * 8 * t * t
        spatial_scores = b * t * 8 * 64 * 64
        final_scores = b * 8 * 128 * (t * 64)
        score_elements = 2 * (temporal_scores + spatial_scores) + final_scores

    compile_start = time.perf_counter()
    output = forward(*args_forward)
    jax.block_until_ready(output)
    compile_seconds = time.perf_counter() - compile_start
    for _ in range(max(args.warmup - 1, 0)):
        jax.block_until_ready(forward(*args_forward))

    samples_ms = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        jax.block_until_ready(forward(*args_forward))
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    after = _memory_stats(device)
    result = {
        "mode": args.mode,
        "device": str(device),
        "batch_size": b,
        "history_frames": t,
        "input_shape": list(history.shape),
        "output_shape": list(output.shape),
        "parameter_count": _count_params(variables),
        "attention_score_elements": int(score_elements),
        "compile_and_first_forward_seconds": compile_seconds,
        "latency_ms_mean": float(np.mean(samples_ms)),
        "latency_ms_median": float(np.median(samples_ms)),
        "latency_ms_p90": float(np.percentile(samples_ms, 90)),
        "history_frames_per_second": float(b * t / (np.mean(samples_ms) / 1000.0)),
        "device_memory_before": before,
        "device_memory_after": after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

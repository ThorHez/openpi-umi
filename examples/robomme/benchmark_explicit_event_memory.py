#!/usr/bin/env python3
"""Benchmark the latest deployed explicit-event recurrent memory on raw RGB video."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import platform
import sys
import time

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mem import robomme_explicit_event_inference as inference  # noqa: E402

DEFAULT_CHECKPOINT = ROOT / "checkpoints/robomme_explicit_event_native_single_seed260908_260831"
DEFAULT_OUTPUT = ROOT / "evaluation/robomme/efficiency/explicit_event_native_single_a100_b1_60f_gpu.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def summarize(samples_ms: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "iterations": len(values),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
        "throughput_hz": float(1000.0 / values.mean()),
    }


def memory_stats(device: jax.Device) -> dict[str, int]:
    stats = device.memory_stats() or {}
    return {key: int(stats[key]) for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if key in stats}


def count_params(tree) -> int:
    return sum(int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(tree))


def output_parity(cpu_output: dict[str, np.ndarray], gpu_output: dict[str, np.ndarray]) -> dict:
    result = {}
    for key in ("all_tables", "all_memories", "event_bottleneck"):
        difference = np.abs(np.asarray(cpu_output[key], np.float32) - np.asarray(gpu_output[key], np.float32))
        result[key] = {
            "max_abs_difference": float(difference.max(initial=0.0)),
            "mean_abs_difference": float(difference.mean()),
        }
    for key in ("all_predictions", "event_type", "chunks"):
        result[key] = {"exact_match": bool(np.array_equal(cpu_output[key], gpu_output[key]))}
    return result


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.warmup < 1 or args.iterations < 1:
        raise ValueError("frames, warmup, and iterations must be positive")

    device = jax.devices()[0]
    startup_memory = memory_stats(device)
    predictor = inference.ExplicitEventMemoryPredictor(args.training_dir)

    rng = np.random.default_rng(260831)
    frames = rng.integers(0, 256, (args.frames, 224, 224, 3), dtype=np.uint8)
    anchors_yx = np.asarray(((112.0, 56.0), (112.0, 112.0), (112.0, 168.0)), np.float32)
    normalized_anchors, anchor_mask = inference._normalized_anchors(  # noqa: SLF001
        anchors_yx, (224, 224)
    )
    chunks = inference._chunks_from_frames(frames)  # noqa: SLF001
    inputs = predictor._inputs(  # noqa: SLF001
        chunks,
        task_id=1,
        goal_color_ids=(1, 0),
        queried_ordinal=0,
        num_regions=3,
        anchor_yx=normalized_anchors,
        anchor_mask=anchor_mask,
    )

    compile_start = time.perf_counter()
    compiled_output = predictor._infer(inputs)  # noqa: SLF001
    jax.block_until_ready(compiled_output)
    compile_seconds = time.perf_counter() - compile_start
    for _ in range(args.warmup - 1):
        jax.block_until_ready(predictor._infer(inputs))  # noqa: SLF001

    model_samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        jax.block_until_ready(predictor._infer(inputs))  # noqa: SLF001
        model_samples.append((time.perf_counter() - start) * 1000.0)

    cpu_preprocess_samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        inference._chunks_from_frames(frames)  # noqa: SLF001
        cpu_preprocess_samples.append((time.perf_counter() - start) * 1000.0)

    gpu_preprocess_compile_start = time.perf_counter()
    gpu_chunks = inference._chunks_from_frames_gpu(frames)  # noqa: SLF001
    jax.block_until_ready(gpu_chunks)
    gpu_preprocess_compile_seconds = time.perf_counter() - gpu_preprocess_compile_start
    for _ in range(args.warmup - 1):
        jax.block_until_ready(inference._chunks_from_frames_gpu(frames))  # noqa: SLF001
    gpu_preprocess_samples = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        jax.block_until_ready(inference._chunks_from_frames_gpu(frames))  # noqa: SLF001
        gpu_preprocess_samples.append((time.perf_counter() - start) * 1000.0)

    gpu_chunks_host = np.asarray(jax.device_get(gpu_chunks))
    chunk_difference = np.abs(chunks.astype(np.float32) - gpu_chunks_host.astype(np.float32))
    grid_parity = {
        "allclose_atol_5e-4": bool(np.allclose(chunks, gpu_chunks_host, rtol=0.0, atol=5e-4)),
        "max_abs_difference": float(chunk_difference.max(initial=0.0)),
        "mean_abs_difference": float(chunk_difference.mean()),
    }

    fused_compile_start = time.perf_counter()
    gpu_output = predictor.predict_frames(
        frames,
        anchors_yx,
        task_id=1,
        goal_color_ids=(1, 0),
        queried_ordinal=0,
        preprocess_backend="gpu",
    )
    fused_compile_seconds = time.perf_counter() - fused_compile_start
    for _ in range(args.warmup - 1):
        predictor.predict_frames(
            frames,
            anchors_yx,
            task_id=1,
            goal_color_ids=(1, 0),
            queried_ordinal=0,
            preprocess_backend="gpu",
        )

    total_samples = []
    valid_state_shape = list(gpu_output["all_memories"].shape)
    for _ in range(args.iterations):
        start = time.perf_counter()
        output = predictor.predict_frames(
            frames,
            anchors_yx,
            task_id=1,
            goal_color_ids=(1, 0),
            queried_ordinal=0,
            preprocess_backend="gpu",
        )
        total_samples.append((time.perf_counter() - start) * 1000.0)
        valid_state_shape = list(output["all_memories"].shape)

    cpu_output = predictor.predict_frames(
        frames,
        anchors_yx,
        task_id=1,
        goal_color_ids=(1, 0),
        queried_ordinal=0,
        preprocess_backend="cpu",
    )

    final_memory = memory_stats(device)
    result = {
        "schema_version": 2,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "benchmark": "latest_explicit_event_memory_raw_rgb",
        "host": platform.node(),
        "jax_version": jax.__version__,
        "device": str(device),
        "device_kind": getattr(device, "device_kind", None),
        "checkpoint": str(args.training_dir.resolve()),
        "checkpoint_parameter_count": count_params(predictor.params),
        "batch_size": 1,
        "raw_frames": args.frames,
        "raw_frame_shape": [224, 224, 3],
        "valid_chunks": len(chunks),
        "compiled_max_chunks": inference.MAX_CHUNKS,
        "chunk_frames": inference.CHUNK_FRAMES,
        "spatial_tokens": inference.SPATIAL_TOKENS,
        "input_width": inference.PATCH_WIDTH,
        "valid_memory_trajectory_shape": valid_state_shape,
        "compile_and_first_model_forward_seconds": compile_seconds,
        "compile_and_first_gpu_preprocess_seconds": gpu_preprocess_compile_seconds,
        "compile_and_first_fused_forward_seconds": fused_compile_seconds,
        "cpu_rgb_grid_preprocess_reference": summarize(cpu_preprocess_samples),
        "rgb_grid_preprocess": summarize(gpu_preprocess_samples),
        "compiled_memory_model": summarize(model_samples),
        "raw_rgb_to_memory": summarize(total_samples),
        "numerical_parity": {
            "cpu_vs_gpu_grid_features": grid_parity,
            "cpu_vs_gpu_final_output": output_parity(cpu_output, gpu_output),
        },
        "device_memory_startup": startup_memory,
        "device_memory_after_benchmark": final_memory,
        "scope": (
            "Host-to-device transfer, GPU RGB grid preprocessing, and the explicit-event "
            "recurrent-memory model in one synchronized call. First-call JIT compilation "
            "is reported separately and excluded from warm latency. "
            "This excludes the separate MME action policy and simulator."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

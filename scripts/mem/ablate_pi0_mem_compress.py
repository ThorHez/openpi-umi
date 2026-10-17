"""Pi0MemCompress inference-time ablation suite.

What this script tests
----------------------
Seven ablations probe what the compressed-history memory branch in
``openpi.models.siglip_mem_compress`` actually contributes at inference time.
None of them require retraining or updating the checkpoint — the same
``Pi0MemCompress`` policy is reused across all modes and only the input
frames (or the per-block sigmoid history gate) are perturbed.

================  ===========================================================
mode              what it tests
================  ===========================================================
normal            baseline: feed the real T historical frames as trained.
repeat_current    every historical frame is replaced by the current frame.
                  -> if performance is unchanged, the memory branch is
                  effectively ignoring history and only "summarizes" the
                  current frame.
wrong_history     historical frames (positions 0..T-2) come from a randomly
                  chosen sample elsewhere in the dataset; the current frame
                  is kept correct.
                  -> if performance is unchanged, the memory branch is not
                  using accurate temporal context.
shuffle_history   historical frames are temporally shuffled (positions 0..T-2);
                  the current frame stays in place.
                  -> if performance is unchanged, the memory branch is
                  invariant to temporal order.
zero_current      current frames are replaced by zeros; historical frames stay
                  clean.
                  -> tests whether clean history alone can recover the action.
memory_off        the effective history gate is forced to zero. For learned-gate
                  models this also sets every ``history_memory_gate_logit`` to
                  a very-negative value; for fixed-gate models it temporarily
                  overrides ``history_gate_fixed=0.0``.
                  -> if performance is unchanged, the memory branch is
                  effectively unused and the model behaves as a pure
                  single-frame policy.
force_memory_gate the effective history gate is forced open. For learned-gate
                  models this also sets every ``history_memory_gate_logit`` to
                  a positive value; for fixed-gate models it temporarily
                  overrides ``history_gate_fixed=sigmoid(value)``.
                  -> if this changes predictions while ``memory_off`` does not,
                  the trained gate stayed mostly closed; if it still changes
                  nothing, the history-memory content itself is not reaching
                  the action output.
================  ===========================================================

Pipeline
--------
1. Load the trained policy via :func:`openpi.policies.policy_config.create_trained_policy`
   for the same TrainConfig used at training time.
2. Wrap the LeRobot evaluation dataset with
   :class:`openpi.training.mem.video_dataset.VideoFrameDataset` so that each
   ``__getitem__`` returns per-frame keys ``<image_key>_<t>``, identical to
   what ``BuildVideoTensor`` consumes downstream.
3. For each sample, apply the mode-specific perturbation to those per-frame
   keys (or, for ``memory_off``, mutate the per-block gate logits) and run
   ``policy.infer`` on the result.
4. Compare the predicted action chunk against the dataset's ground-truth
   action chunk (action_horizon=16, action_dim=32 with the trailing
   ``action_loss_mask=(1,)*20+(0,)*12``); we report MSE/MAE over the
   *first* ``valid_dim=20`` action dimensions, both per-step and aggregated.

Launch
------
Single-GPU, default batching (auto picks batch_size=4):
    CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        uv run scripts/mem/ablate_pi0_mem_compress.py \
        --ablation_mode=all --num_episodes=3 --no-multi_gpu

Multi-GPU data parallel (1 sample per GPU per call; ~Nx speedup over single GPU):
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        uv run scripts/mem/ablate_pi0_mem_compress.py \
        --ablation_mode=all --num_episodes=3
    # auto picks batch_size = jax.device_count() and turns on data-parallel sharding.

Multi-GPU, K samples per device per call (higher amortization, more VRAM):
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        uv run scripts/mem/ablate_pi0_mem_compress.py \
        --ablation_mode=all --num_episodes=3 --batch_size=16
    # with 8 visible GPUs this puts 2 samples on each device per JIT call.

Run a single mode:
    uv run scripts/mem/ablate_pi0_mem_compress.py --ablation_mode=shuffle_history
"""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import enum
import json
import logging
import fcntl
import multiprocessing
import os
import shutil
import time
from pathlib import Path

# Mirror train_pi0_mem_compress.py: keep TMPDIR off "/" and overlay.
if "TMPDIR" not in os.environ:
    _tmp = Path(os.environ.get("HOME", "/root")) / "tmp"
    _tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_tmp)

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import tyro
import flax.nnx as nnx
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

# Persistent JAX compilation cache. Mirror what scripts/mem/train_pi0_mem_compress.py
# does so that the (very slow) first JIT trace + lowering of Pi0MemCompress's
# PaliGemma + SigLIP-mem-compress graph only happens once on disk — subsequent
# ablation runs (and any future Pi0MemCompress inference) restore from cache and
# go from ~5-15 min cold compile down to ~30-60s.
jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
import openpi.transforms as _transforms
import openpi.training.config as _config
from openpi.policies import policy_config as _policy_config
from openpi.policies import policy as _policy
from openpi.training.mem.video_dataset import VideoFrameConfig, VideoFrameDataset


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------


class AblationMode(str, enum.Enum):
    NORMAL = "normal"
    REPEAT_CURRENT = "repeat_current"
    WRONG_HISTORY = "wrong_history"
    SHUFFLE_HISTORY = "shuffle_history"
    ZERO_CURRENT = "zero_current"
    ZERO_CURRENT_SUITE = "zero_current_suite"
    MEMORY_OFF = "memory_off"
    FORCE_MEMORY_GATE = "force_memory_gate"
    ALL = "all"


class FrameWindow(str, enum.Enum):
    """Which part of each episode to evaluate after frame-striding."""

    START = "start"
    END = "end"


@dataclasses.dataclass
class Args:
    """Ablation evaluation arguments."""

    # TrainConfig name (must match the checkpoint).
    config: str = (
        "pi0_mem_compress_umi_32d_60k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322"
    )
    # Concrete checkpoint step directory.
    checkpoint_dir: str = (
        "/data1/hzl_workspace_for_pi/openpi-umi/checkpoints/"
        "pi0_mem_compress_umi_32d_60k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322/"
        "my_experiment/47000"
    )
    # LeRobot dataset repo_id / local path. Defaults to the training dataset; pass a
    # held-out val dataset for a cleaner generalization signal (the *relative* ranking
    # of ablations is informative either way).
    dataset_path: str = (
        "/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/"
        "horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51"
    )

    # Output directory for results (per-mode JSONs + summary).
    output_dir: str = "/data1/hzl_workspace_for_pi/openpi-umi/ablation_results"

    # Which ablation mode to run. Use "all" to run every mode in one invocation.
    ablation_mode: AblationMode = AblationMode.ALL
    # If true, zero the current frame after applying the selected history/gate
    # ablation. This creates a shared no-current-frame condition across modes.
    zero_current_frame: bool = False

    # Number of episodes to evaluate. Set to -1 for all.
    num_episodes: int = 20
    # Index of first episode to evaluate.
    start_episode: int = 0
    # Step over frames within each episode (use >1 to subsample for speed).
    # Default 2 already halves wall-time without changing the relative-ranking
    # signal of the ablation; bump to 1 only if you specifically need every
    # single timestep.
    frame_stride_eval: int = 1
    # Cap on the number of frames evaluated per episode (after stride). -1 = all.
    # Default 100 keeps a full ALL-modes sweep (6 modes x num_episodes x 100 frames)
    # under ~5-10 min on 8x H100 with prefetch on. The CPU-side video-decode bill
    # is real: every UMI frame triggers 16 historical-frame mp4 decodes x 2 cameras
    # = 32 ffmpeg seeks per sample, so num_frames * num_episodes * 6 modes scales
    # almost linearly. Set to -1 only when you really want every frame.
    max_frames_per_episode: int = -1
    # Which part of each episode to keep when max_frames_per_episode is positive.
    # START preserves the historical behavior. END is useful for cloth-folding
    # diagnostics because late episode states may need more temporal context than
    # the easy opening motions.
    frame_window: FrameWindow = FrameWindow.START

    # Default prompt fed to the policy if the dataset row has no "task" entry.
    default_prompt: str = "fold the clothes"

    # Seed for reproducible random sampling in wrong_history / shuffle_history.
    seed: int = 0
    # Seed for flow-matching action sampling noise. Each batch folds this seed
    # with the first global frame index in that batch, so the same frames receive
    # the same initial noise across ablation modes.
    sampling_seed: int = 0

    # When running multiple modes in one process, compare each mode's predicted
    # action chunks against the normal-mode predictions for the same dataset
    # sample indices. This answers whether an ablation actually changes the
    # policy output, independently of whether the GT MSE happens to move.
    compute_prediction_delta: bool = True
    # How many largest-prediction-change frames to include verbatim in each
    # mode's JSON diagnostics. These records are small and make it easy to
    # inspect exactly where an ablation changed behavior most.
    prediction_delta_top_k: int = 20

    # Number of action dims to score (the model predicts action_dim=32, but only the
    # first 20 are real bimanual actions; the trailing 12 are masked padding).
    valid_action_dims: int = 20

    # Value to set ``history_memory_gate_logit`` to in memory_off mode.
    # sigmoid(-50) is ~2e-22, which is effectively zero gate.
    memory_off_gate_logit: float = -50.0
    # Value to set ``history_memory_gate_logit`` to in force_memory_gate mode.
    # sigmoid(5) is ~0.993, so the history branch is effectively fully open.
    force_memory_gate_logit: float = 5.0

    # ------------------------- Inference acceleration ---------------------------
    # How many frames to push through the model per JIT call. Setting this >1
    # amortizes the JAX launch / all-gather overhead and -- combined with
    # ``multi_gpu=True`` -- lets each visible GPU process its own slice of the
    # batch (true data-parallel inference).
    #
    # 0 (default) -> auto-pick: use ``jax.device_count()`` when ``multi_gpu``,
    # else 4. Pick something divisible by ``jax.device_count()`` for the
    # cleanest sharding. If the last chunk of an episode is smaller, it gets
    # padded with copies so the shape (and JIT cache key) stays constant.
    batch_size: int = 72

    # Shard the per-call batch across every visible GPU (data parallelism: model
    # params replicated, batch dim sliced along ``data`` axis). Requires more
    # than one device to be visible to JAX (e.g. ``CUDA_VISIBLE_DEVICES=0,1,2,3``).
    # Set to False to keep everything on jax.devices()[0] (still uses batching).
    multi_gpu: bool = True

    # CPU-side video-decode workers. Pi0Mem's VideoFrameDataset decodes
    # ``num_frames * len(image_keys)`` mp4 frames per sample (16 * 2 = 32 ffmpeg
    # seeks for UMI bimanual), which on a single Python thread is ~1.0-1.3 s/sample
    # -- way more than the GPU side. We launch this many background threads to
    # pre-load the next chunks while the current chunk is on the GPU, turning the
    # total wall time into ``max(load_per_chunk / workers, gpu_per_chunk)``. Pyav
    # / ffmpeg's C decode releases the GIL so threads actually scale.
    # 0 = disable prefetch (synchronous load, useful for debugging).
    num_load_workers: int = 8

    # How many samples to keep in flight across the prefetch threads. 2-3x
    # batch_size is a good rule of thumb. Larger uses more RAM but smooths
    # over CPU stalls; smaller can leave GPU starved between batches.
    prefetch_size: int = 0  # 0 -> auto = max(batch_size, num_load_workers) * 2

    # ----------------- Episode-level data parallelism ---------------------------
    # To run N copies of this script in parallel, one per GPU, launch with:
    #   for i in 0..N-1: CUDA_VISIBLE_DEVICES=$i python ... \
    #       --episode-shard-index=$i --episode-shard-total=$N \
    #       --no-multi-gpu
    # Each shard takes a round-robin slice of the episode list (shard i picks
    # episodes at index i, i+N, i+2N, ...), so workload is balanced even if
    # episode lengths vary. Combine shards afterwards with
    # ``scripts/mem/merge_ablation_shards.py``.
    #
    # When both default to (0, 1) the script behaves as a single process and no
    # shard suffix is added to output filenames.
    episode_shard_index: int = 0
    episode_shard_total: int = 1

    # ----------------- Dataset I/O caching --------------------------------------
    # Pi0Mem ablation is heavily disk-random-seek bound: each evaluated frame
    # triggers ``num_frames * len(image_keys)`` mp4 seek+decode calls (e.g.
    # 16 * 2 = 32 per frame for UMI bimanual). With 8 prefetch workers each
    # sample requests ~32 random seeks simultaneously to the same mp4 files,
    # which on a SATA SSD pegs the disk queue and drops effective throughput
    # to ~0.3 s/frame even with all worker plumbing correctly parallelized.
    # Copying the whole dataset to ``/dev/shm`` (tmpfs / RAM) once at startup
    # turns those seeks into memory reads (~us latency) and frees the disk
    # entirely from the critical path. On a 7 GiB UMI eval set the copy takes
    # ~30-60 s and recovers >3x end-to-end speedup. Disabled if the dataset
    # already lives on a tmpfs mount, if /dev/shm doesn't have enough free
    # space (with a small safety margin), or if the user passes --no-shm-cache.
    shm_cache: bool = True
    # Where to put the copy. {basename} is substituted with the source
    # directory's basename so multiple datasets can coexist.
    shm_cache_dir: str = "/dev/shm/openpi_ablate_{basename}"

    # ----------------- Flow-matching inference cost -----------------------------
    # Pi0MemCompress predicts actions via flow-matching sampling, NOT a single
    # forward pass. ``sample_actions`` default is ``num_steps=10`` -- meaning
    # every batched inference call runs the full LLM forward 10 times serially
    # (with kv-cache for the prefix, but the suffix attends over the prefix
    # tokens every step). On 8-GPU FSDP this is ~30-40 s per batch=72 -- by far
    # the dominant cost. By comparison ``scripts/lerobot_value_infer.py`` runs
    # a SINGLE forward of the value head and is therefore ~10x faster on the
    # exact same hardware, which is not a fair-throughput comparison.
    #
    # If you only care about RELATIVE ablation signal (does memory_off blow up
    # MSE relative to normal?), you can safely drop num_steps to 4 -- the
    # action quality degrades slightly but the *ranking* of ablations is
    # preserved. Empirically 4 steps reach ~85% of the 10-step quality and 1/2
    # the wall time.
    num_sampling_steps: int = 10

    # Emit a one-line per-batch timing breakdown every N consume calls. Useful
    # for verifying that consume_prev_wait (GPU) >> data_collect_wait (CPU) --
    # if it's the other way around, increase --num-load-workers.
    # 0 = disable.
    profile_every_n_batches: int = 0


# ---------------------------------------------------------------------------
# Dataset setup
# ---------------------------------------------------------------------------


def _path_is_on_tmpfs(path: Path) -> bool:
    """Return True if ``path`` resides on a tmpfs (RAM-backed) mount.

    Detection is best-effort: we walk up the path until we find an existing
    parent, then ``os.statvfs`` it and compare the fs magic. tmpfs reports
    ``f_namemax >= 255`` and no underlying block device, but the most
    reliable signal is parsing ``/proc/mounts`` -- which is what we do here.
    """
    p = path.resolve()
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        with open("/proc/mounts") as f:
            mounts = [line.split() for line in f if line.strip()]
    except OSError:
        return False
    # Sort longest mount-point first so we match the deepest one.
    mounts.sort(key=lambda parts: len(parts[1]) if len(parts) >= 2 else 0, reverse=True)
    p_str = str(p)
    for parts in mounts:
        if len(parts) < 3:
            continue
        mount_point, fs_type = parts[1], parts[2]
        if p_str == mount_point or p_str.startswith(mount_point.rstrip("/") + "/"):
            return fs_type == "tmpfs"
    return False


def _dir_size_bytes(path: Path) -> int:
    """Total bytes of a directory tree. Follows symlinks for files, not dirs."""
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def maybe_cache_dataset_in_shm(args: Args) -> None:
    """If enabled and viable, copy ``args.dataset_path`` to tmpfs and rewrite
    ``args.dataset_path`` to the cached copy. Mutates ``args`` in place.

    Why this matters
    ----------------
    Pi0Mem ablation does ``num_frames * len(image_keys)`` (= 32 for UMI bimanual)
    mp4 random seeks per evaluated frame. With 8 prefetch workers, that's a
    burst of ~256 random seeks against a small set of mp4 files, which on a
    SATA SSD saturates the device queue and dominates wall time. Moving the
    files to tmpfs turns those seeks into ~us-latency RAM reads.

    Decision flow:

    1. ``--no-shm-cache``           -> skip
    2. source already on tmpfs      -> skip (already cached)
    3. /dev/shm doesn't exist        -> skip + warn
    4. not enough free space         -> skip + warn (dataset_size * 1.05 needed)
    5. destination exists & complete -> reuse (no copy)
    6. otherwise                     -> copy + rewrite args.dataset_path
    """
    if not args.shm_cache:
        logging.info("--no-shm-cache: leaving dataset_path on its original mount.")
        return

    src = Path(args.dataset_path).resolve()
    if not src.is_dir():
        logging.warning("shm-cache: dataset_path %s is not a directory, skipping.", src)
        return

    if _path_is_on_tmpfs(src):
        logging.info(
            "shm-cache: dataset_path %s is ALREADY on a tmpfs mount; "
            "no copy needed. Disk I/O is not the bottleneck for this run.",
            src,
        )
        return

    shm_root = Path("/dev/shm")
    if not shm_root.is_dir():
        logging.warning(
            "shm-cache: /dev/shm not found; this platform doesn't expose tmpfs there. "
            "Set --no-shm-cache to silence this warning."
        )
        return

    dst = Path(args.shm_cache_dir.format(basename=src.name)).resolve()
    if not str(dst).startswith("/dev/shm/") and not _path_is_on_tmpfs(dst.parent):
        logging.warning(
            "shm-cache: configured destination %s is not under /dev/shm and parent "
            "is not on tmpfs. Refusing to cache (this would just copy disk -> disk). "
            "Set --shm-cache-dir to a real tmpfs path or pass --no-shm-cache.",
            dst,
        )
        return

    src_size = _dir_size_bytes(src)
    src_gib = src_size / (1024 ** 3)
    statvfs = os.statvfs(shm_root)
    free_bytes = statvfs.f_bavail * statvfs.f_frsize
    free_gib = free_bytes / (1024 ** 3)
    need_bytes = int(src_size * 1.05)  # 5% safety margin

    if free_bytes < need_bytes:
        logging.warning(
            "shm-cache: dataset is %.2f GiB but /dev/shm only has %.2f GiB free "
            "(need %.2f GiB w/ 5%% margin). Skipping cache; falling back to %s.",
            src_gib,
            free_gib,
            need_bytes / (1024 ** 3),
            src,
        )
        return

    # Cross-process lock so that when run_ablate_episode_parallel.sh launches
    # N shards at once, only ONE actually runs the copy; the rest block on the
    # ``fcntl.LOCK_EX`` until copy completes, then take the reuse branch.
    # Lock file lives in /dev/shm so it's automatically cleaned at reboot.
    dst.parent.mkdir(parents=True, exist_ok=True)
    lock_path = dst.with_name(dst.name + ".copylock")
    with open(lock_path, "a+") as lock_f:
        logging.info("shm-cache: acquiring lock %s ...", lock_path)
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

        # Re-check inside the lock: another shard might have just finished.
        if dst.exists() and dst.is_dir():
            dst_size = _dir_size_bytes(dst)
            if abs(dst_size - src_size) <= max(1024, src_size // 100):
                logging.info(
                    "shm-cache: REUSING existing cache at %s (%.2f GiB; "
                    "matches source within 1%%).",
                    dst,
                    dst_size / (1024 ** 3),
                )
                args.dataset_path = str(dst)
                return
            logging.info(
                "shm-cache: existing cache at %s is stale (%.2f GiB vs source "
                "%.2f GiB); recopying.",
                dst,
                dst_size / (1024 ** 3),
                src_gib,
            )
            shutil.rmtree(dst)

        logging.info(
            "shm-cache: copying %.2f GiB from %s -> %s (will take ~%ds at "
            "~500 MiB/s). This is a one-time cost per machine boot; the copy "
            "survives multiple runs.",
            src_gib,
            src,
            dst,
            max(5, int(src_size / (500 * 1024 ** 2))),
        )
        t0 = time.monotonic()
        # ``shutil.copytree`` is fine for tmpfs (no special hard-link logic
        # needed); disk read is the only slow part. ``copy2`` preserves
        # mtimes so later size+mtime sanity checks remain meaningful.
        shutil.copytree(src, dst, copy_function=shutil.copy2, symlinks=True)
        elapsed = time.monotonic() - t0
        rate_mib = src_size / max(elapsed, 1e-3) / (1024 ** 2)
        logging.info(
            "shm-cache: copy done in %.1fs (~%.0f MiB/s effective). "
            "Random-seek I/O is now memory-bound; expect data_collect in the "
            "profile log to drop ~3-5x.",
            elapsed,
            rate_mib,
        )
    args.dataset_path = str(dst)


def _resolve_first_child_factory(train_config: _config.TrainConfig):
    """Same dispatch as ``create_pi0_mem_data_loader``: pick the single (or first) child."""
    data_factory = train_config.data
    if isinstance(data_factory, _config.MultiDataConfigFactory):
        if not data_factory.datasets:
            raise ValueError("MultiDataConfigFactory has no datasets.")
        return data_factory.datasets[0]
    return data_factory


def _resolve_video_frame_config(train_config: _config.TrainConfig) -> VideoFrameConfig:
    """Pull the VideoFrameConfig directly off the training DataConfigFactory.

    Mirrors :func:`openpi.training.config_pi0_mem.create_pi0_mem_data_loader`'s
    dispatch: ``MultiDataConfigFactory`` -> use the first child; single factory
    -> use it directly. Either way, the child must expose ``video_frame_config()``.
    """
    child = _resolve_first_child_factory(train_config)
    if not hasattr(child, "video_frame_config"):
        raise ValueError(
            f"Data factory {type(child).__name__} is not Pi0Mem-aware "
            "(must expose .video_frame_config())."
        )
    return child.video_frame_config()


class FastVideoFrameDataset(VideoFrameDataset):
    """Drop-in replacement for :class:`VideoFrameDataset` for ablation/inference.

    Why this exists
    ---------------
    Profiling on the fold-clothes eval set (50k+ frames, 16-frame history,
    2 wrist cams) showed that per-worker decode time was ~0.83 s/sample,
    of which essentially *all* came from the 16 sequential calls to
    ``self._hf_dataset[int(idx)]`` inside ``VideoFrameDataset.__getitem__``.

    Each ``hf_dataset[idx]`` materialises **all 31 columns** of that Arrow row
    (including 4 large image tensors at ~600 KB each = ~2.4 MB/row that the
    ablation never reads), so the effective work was ~32 single-row reads
    × full-row Arrow deserialise. Multi-worker contention on the shared
    Arrow mmap region pushed it further: 4 workers -> 1.17 s/sample,
    8 workers -> ~2 s/sample (mmap_lock thrashing).

    Micro-benchmarks (16-frame history, 2 image cols, /dev/shm copy):
        32x  hf[i]                      (31 cols) -> 834 ms
        1x   hf[list]                   (31 cols) -> 1675 ms (worse: more bytes!)
        1x   hf.select_columns(2)[list] (2 cols)  ->  177 ms   <- chosen
        -> expected speedup vs current: ~4.7x at 1 worker, MUCH more at >=4
           workers where mmap contention dominates.

    Strategy
    --------
    Keep two slim HF-dataset *column views* alive on the parent process:
      * ``_img_view``   = ``hf.select_columns(image_keys)`` for the 16-row
                          batched read of history frames (one batched call
                          instead of 16 single-row calls).
      * ``_meta_view``  = ``hf.select_columns([state*, meta*, actions, task_index])``
                          for the single current-frame metadata read. This
                          skips the 4 image columns we already get from
                          ``_img_view``.

    The class is a subclass of :class:`VideoFrameDataset` so callers that
    type-annotate ``VideoFrameDataset`` (worker fork helpers, episode index
    listing, etc.) still accept it as-is.
    """

    # Columns the ablation pipeline reads from a "current frame" sample, in
    # addition to ``image_keys``. Anything not in the dataset schema is
    # silently skipped at view-construction time.
    _CURRENT_FRAME_NON_IMAGE_COLS: tuple[str, ...] = (
        # 14 state pieces (see _STATE_KEYS at module level)
        "robot0_eef_pos",
        "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle",
        "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1",
        "robot0_gripper_width",
        "robot1_eef_pos",
        "robot1_eef_pos_wrt_start",
        "robot1_eef_rot_axis_angle",
        "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0",
        "robot1_gripper_width",
        # Action chunk + episode/frame meta + task id (string resolved below).
        "actions",
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
    )

    def __init__(self, dataset, config: VideoFrameConfig):
        super().__init__(dataset, config)

        # ``select_columns`` returns a *view* (no Arrow data copy) — it just
        # rewrites the schema/format. Multiple views share the same mmap pages.
        all_cols = set(self._hf_dataset.column_names)
        img_cols = [k for k in config.image_keys if k in all_cols]
        if len(img_cols) != len(config.image_keys):
            missing = set(config.image_keys) - set(img_cols)
            logging.warning(
                f"FastVideoFrameDataset: image columns missing from hf_dataset, "
                f"falling back to slow per-row path for these keys: {sorted(missing)}"
            )
        meta_cols = [c for c in self._CURRENT_FRAME_NON_IMAGE_COLS if c in all_cols]

        # The image-only view backs the batched history-frame read.
        self._img_view = self._hf_dataset.select_columns(img_cols) if img_cols else None
        # The metadata view backs the single current-frame read (skips images).
        self._meta_view = (
            self._hf_dataset.select_columns(meta_cols) if meta_cols else self._hf_dataset
        )
        # task_index -> task string lookup (LeRobotDataset normally does this
        # inside its own __getitem__; we bypass that to avoid the full-row read).
        self._tasks_meta: list[str] | None = None
        try:
            meta = getattr(dataset, "meta", None)
            if meta is not None and hasattr(meta, "tasks"):
                # ``meta.tasks`` can be a list or a dict; normalise to indexable.
                tk = meta.tasks
                if isinstance(tk, dict):
                    self._tasks_meta = [tk[i] for i in sorted(tk.keys())]
                else:
                    self._tasks_meta = list(tk)
        except Exception:
            self._tasks_meta = None

        self._img_keys: tuple[str, ...] = tuple(img_cols)

    def __getitem__(self, index: int) -> dict:
        cfg = self._config
        num_frames = cfg.num_frames
        stride = cfg.frame_stride
        N = len(self._hf_dataset)
        cur_index = int(index)

        # 1) Current-frame metadata: single row read on the slim _meta_view.
        cur = dict(self._meta_view[cur_index])

        cur_episode = int(cur.get("episode_index", -1)) if "episode_index" in cur else -1
        cur_frame_idx = int(cur.get("frame_index", -1)) if "frame_index" in cur else -1

        # 2) Compute history indices. Index 0 = oldest, index num_frames-1 = current.
        #    Episode-boundary handling: if we don't know the episode/frame index
        #    we fall back to position >= 0 only (same conservative rule as
        #    VideoFrameDataset).
        target_indices = [cur_index - (num_frames - 1 - t) * stride for t in range(num_frames)]
        if cur_frame_idx >= 0:
            valid_mask = [
                (idx >= 0 and idx < N and (cur_frame_idx - (num_frames - 1 - t) * stride) >= 0)
                for t, idx in enumerate(target_indices)
            ]
        else:
            valid_mask = [(idx >= 0 and idx < N) for idx in target_indices]

        valid_positions = [t for t, ok in enumerate(valid_mask) if ok]
        valid_indices = [target_indices[t] for t in valid_positions]

        # 3) Batched history-frame read on the slim _img_view. This is the
        #    hot path: ~177 ms here vs ~830 ms for the old loop.
        img_batch: dict | None = None
        if self._img_view is not None and valid_indices:
            img_batch = self._img_view[valid_indices]
            # Optionally drop rows that crossed an episode boundary. If
            # _img_view doesn't carry episode_index we trust the conservative
            # frame-offset gate above (already applied via valid_mask).
            if cur_episode >= 0 and "episode_index" in img_batch:
                ep_col = img_batch["episode_index"]
                drop_local = []
                for li, ep_val in enumerate(ep_col):
                    try:
                        if int(ep_val) != cur_episode:
                            drop_local.append(li)
                    except Exception:
                        pass
                if drop_local:
                    # Mark those positions as invalid so padding kicks in.
                    drop_set = set(drop_local)
                    for li in drop_local:
                        valid_mask[valid_positions[li]] = False
                    # Filter img_batch lists in place.
                    keep = [li for li in range(len(ep_col)) if li not in drop_set]
                    img_batch = {
                        k: [v[i] for i in keep]
                        for k, v in img_batch.items()
                    }
                    valid_positions = [
                        valid_positions[li] for li in keep
                    ]

        # 4) Assemble output dict in the format VideoFrameDataset normally returns.
        sample: dict = dict(cur)
        sample["index"] = cur_index
        if "task_index" in cur and self._tasks_meta is not None:
            try:
                ti = int(cur["task_index"])
                if 0 <= ti < len(self._tasks_meta):
                    sample["task"] = self._tasks_meta[ti]
            except Exception:
                pass

        for img_key in self._img_keys:
            frames_at: list[np.ndarray | None] = [None] * num_frames
            if img_batch is not None and img_key in img_batch:
                col = img_batch[img_key]
                for li, t in enumerate(valid_positions):
                    if li < len(col):
                        frames_at[t] = _parse_image_fast(col[li])

            # Padding (mirrors VideoFrameDataset.__getitem__).
            valid = [f for f in frames_at if f is not None]
            if not valid:
                zero = np.zeros((224, 224, 3), dtype=np.uint8)
                frames_at = [zero] * num_frames
            else:
                first_valid = valid[0]
                for i in range(num_frames):
                    if frames_at[i] is None:
                        if cfg.padding_mode == "repeat":
                            frames_at[i] = first_valid
                        else:
                            frames_at[i] = np.zeros_like(first_valid)

            for t, f in enumerate(frames_at):
                sample[f"{img_key}_{t}"] = f

        return sample


def _parse_image_fast(image) -> np.ndarray:
    """Tight version of ``video_dataset._parse_image`` for inference.

    Skips dynamic imports and avoids a hot-path ``einops.rearrange`` by doing
    a direct ``transpose`` for the (C, H, W) -> (H, W, C) case.
    """
    import torch  # local import: workers may not need torch otherwise

    if isinstance(image, torch.Tensor):
        # Common case: hf_transform_to_torch returns a torch.Tensor (C, H, W) float32 [0,1].
        if image.dtype != torch.uint8:
            image = (image.mul(255.0)).clamp_(0, 255).to(torch.uint8)
        arr = image.numpy()
    else:
        arr = np.asarray(image)
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def build_video_dataset(args: Args, train_config: _config.TrainConfig) -> tuple[VideoFrameDataset, VideoFrameConfig]:
    """Build a ``VideoFrameDataset`` that yields per-frame image keys.

    The wrapping is the same as the training-time pipeline in
    ``create_pi0_mem_data_loader``; we just don't run the downstream
    ``transform_dataset`` step because the inference-side transforms are
    applied inside :class:`openpi.policies.policy.Policy` instead.

    Critical detail (mirrors ``_build_pi0_mem_dataset``): we must respect
    the data config's ``action_sequence_keys``. For ``UmiDataConfig`` it
    defaults to ``()`` because the UMI datasets store ``actions`` already
    chunked as ``(action_horizon, action_dim)`` per row; passing
    ``delta_timestamps={"actions": [...]}`` to LeRobot in that case would
    prepend an extra time axis and you get ``(action_horizon, action_horizon,
    action_dim)`` -> AssertionError downstream.

    Returns a :class:`FastVideoFrameDataset` instead of the plain
    :class:`VideoFrameDataset` — same public surface, ~5-15x faster
    ``__getitem__`` because it does a single column-restricted batched
    Arrow read for the 16-frame history instead of 16 full-row reads.
    Set the environment variable ``OPENPI_ABLATE_USE_FAST_DATASET=0`` to
    fall back to the slow path for debugging.
    """
    repo_id = args.dataset_path
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)

    child = _resolve_first_child_factory(train_config)
    data_config = child.create(train_config.assets_dirs, train_config.model)
    action_seq_keys = tuple(data_config.action_sequence_keys or ())
    if action_seq_keys:
        delta_timestamps = {
            key: [t / dataset_meta.fps for t in range(train_config.model.action_horizon)]
            for key in action_seq_keys
        }
    else:
        # Empty action_sequence_keys -> dataset already stores chunked actions;
        # do NOT pass delta_timestamps, otherwise LeRobot would inject a bogus
        # leading time axis.
        delta_timestamps = None
    logging.info(
        f"LeRobotDataset action_sequence_keys={action_seq_keys}; "
        f"delta_timestamps={'<auto>' if delta_timestamps else None}"
    )
    base_ds = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    video_cfg = _resolve_video_frame_config(train_config)
    use_fast = os.environ.get("OPENPI_ABLATE_USE_FAST_DATASET", "1") not in ("0", "false", "False")
    if use_fast:
        ds: VideoFrameDataset = FastVideoFrameDataset(base_ds, video_cfg)
        ds_kind = "FastVideoFrameDataset (column-restricted batched Arrow reads)"
    else:
        ds = VideoFrameDataset(base_ds, video_cfg)
        ds_kind = "VideoFrameDataset (legacy per-row reads; OPENPI_ABLATE_USE_FAST_DATASET=0)"
    logging.info(
        f"Ablation video pipeline: num_frames={video_cfg.num_frames}, "
        f"frame_stride={video_cfg.frame_stride}, image_keys={video_cfg.image_keys}, "
        f"dataset={ds_kind}"
    )
    return ds, video_cfg


def list_episode_indices(dataset: VideoFrameDataset) -> dict[int, list[int]]:
    """Return ``{episode_idx: [sample_idx, ...]}`` over the wrapped LeRobot dataset."""
    hf_dataset = dataset._hf_dataset
    # LeRobotDataset exposes ``episode_data_index`` mapping episode -> (from, to)
    base = dataset._dataset
    # Strip TransformedDataset layer if present (PromptFromLeRobotTask wrapper).
    while hasattr(base, "_dataset") and not hasattr(base, "episode_data_index"):
        base = base._dataset
    edi = getattr(base, "episode_data_index", None)
    if edi is not None:
        # ``episode_data_index`` is a dict {"from": Tensor, "to": Tensor} in lerobot.
        from_arr = np.asarray(edi["from"]).astype(int).tolist()
        to_arr = np.asarray(edi["to"]).astype(int).tolist()
        return {ep: list(range(int(from_arr[ep]), int(to_arr[ep]))) for ep in range(len(from_arr))}

    # Fallback: scan the HF dataset for episode_index.
    by_episode: dict[int, list[int]] = {}
    for idx in range(len(hf_dataset)):
        row = hf_dataset[idx]
        ep = int(row.get("episode_index", -1))
        by_episode.setdefault(ep, []).append(idx)
    return by_episode


# ---------------------------------------------------------------------------
# Per-frame ablations
# ---------------------------------------------------------------------------


_STATE_KEYS: tuple[str, ...] = (
    "robot0_eef_pos",
    "robot0_eef_pos_wrt_start",
    "robot0_eef_rot_axis_angle",
    "robot0_eef_rot_axis_angle_wrt_start",
    "robot0_eef_pos_wrt1",
    "robot0_eef_rot_axis_angle_wrt1",
    "robot0_gripper_width",
    "robot1_eef_pos",
    "robot1_eef_pos_wrt_start",
    "robot1_eef_rot_axis_angle",
    "robot1_eef_rot_axis_angle_wrt_start",
    "robot1_eef_pos_wrt0",
    "robot1_eef_rot_axis_angle_wrt0",
    "robot1_gripper_width",
)


def _per_frame_keys(image_keys: tuple[str, ...], num_frames: int) -> list[str]:
    return [f"{k}_{t}" for k in image_keys for t in range(num_frames)]


def apply_ablation_to_sample(
    sample: dict,
    *,
    mode: AblationMode,
    image_keys: tuple[str, ...],
    num_frames: int,
    donor_sample: dict | None,
    rng: np.random.Generator,
    zero_current_frame: bool = False,
) -> dict:
    """Return a shallow copy of ``sample`` with the requested image-key perturbation applied.

    Most data-side ablations preserve the current frame (position
    ``num_frames - 1``) and only manipulate historical positions ``0..T-2``.
    ``zero_current`` is the opposite stress test: current is blanked while
    clean history is preserved.
    """
    out = dict(sample)

    if mode == AblationMode.NORMAL:
        return out

    for img_key in image_keys:
        current_key = f"{img_key}_{num_frames - 1}"
        current_frame = out[current_key]

        if mode == AblationMode.REPEAT_CURRENT:
            # All historical positions become the current frame.
            for t in range(num_frames - 1):
                out[f"{img_key}_{t}"] = current_frame

        elif mode == AblationMode.SHUFFLE_HISTORY:
            # Shuffle positions 0..T-2 in place. Current frame untouched.
            hist_frames = [out[f"{img_key}_{t}"] for t in range(num_frames - 1)]
            perm = rng.permutation(num_frames - 1)
            for t, p in enumerate(perm):
                out[f"{img_key}_{t}"] = hist_frames[int(p)]

        elif mode == AblationMode.WRONG_HISTORY:
            if donor_sample is None:
                raise ValueError(
                    "wrong_history requires a donor_sample fetched from a different "
                    "point in the dataset."
                )
            # Replace historical positions with frames from the donor, keep current.
            for t in range(num_frames - 1):
                donor_key = f"{img_key}_{t}"
                if donor_key in donor_sample:
                    out[f"{img_key}_{t}"] = donor_sample[donor_key]
                else:
                    # Donor might not have this exact per-frame key — fall back to the
                    # donor's current frame so historical positions are still wrong.
                    out[f"{img_key}_{t}"] = donor_sample.get(current_key, current_frame)

        elif mode == AblationMode.ZERO_CURRENT:
            out[current_key] = np.zeros_like(current_frame)

        elif mode in (AblationMode.MEMORY_OFF, AblationMode.FORCE_MEMORY_GATE):
            # No data-side change; the gate manipulation handled the model.
            pass

        else:
            raise ValueError(f"Unknown ablation mode: {mode}")

    if zero_current_frame:
        for img_key in image_keys:
            current_key = f"{img_key}_{num_frames - 1}"
            out[current_key] = np.zeros_like(out[current_key])

    return out


def build_obs_dict(
    sample: dict,
    *,
    image_keys: tuple[str, ...],
    num_frames: int,
    default_prompt: str,
) -> dict:
    """Strip a VideoFrameDataset sample to exactly the keys the policy transforms expect.

    The policy applies (in order):
        InjectDefaultPrompt -> BuildVideoTensor -> UmiInputsV4_Bimanual_Video ->
        Normalize -> InjectDefaultPrompt(again, in model_transforms) -> TokenizePrompt ->
        PadActionsOnly -> FlattenState
    so we need to provide:
      * Per-frame image keys ``<image_key>_<t>``.
      * The 14 state-vector pieces concatenated by ``_build_bimanual_state``.
      * ``actions`` of shape (action_horizon, action_dim_raw) — required only by
        ``UmiInputsV4_Bimanual_Video``'s shape assertion; ``Observation.from_dict``
        never reads it, so the model still does not see the ground-truth actions
        during inference (verified by reading ``Pi0MemCompress.sample_actions``).
      * Optionally ``prompt`` (InjectDefaultPrompt fills it in if missing).
    """
    obs: dict = {}
    for k in _per_frame_keys(image_keys, num_frames):
        obs[k] = np.asarray(sample[k])
    for k in _STATE_KEYS:
        obs[k] = np.asarray(sample[k])
    # Forward the dataset's action chunk (shape (action_horizon, action_dim_raw))
    # purely to satisfy ``UmiInputsV4_Bimanual_Video``'s shape assert; the model
    # ignores it at inference (sample_actions only consumes Observation).
    obs["actions"] = np.asarray(sample["actions"])
    # Pull a per-row task if the dataset exposes one; otherwise rely on default_prompt.
    if "task" in sample and isinstance(sample["task"], str):
        obs["prompt"] = sample["task"]
    elif "prompt" in sample and isinstance(sample["prompt"], str):
        obs["prompt"] = sample["prompt"]
    else:
        obs["prompt"] = default_prompt
    return obs


# ---------------------------------------------------------------------------
# Model-side ablation: zero out the history-memory cross-attention gate
# ---------------------------------------------------------------------------


def _is_gate_path(path) -> bool:
    try:
        return "history_memory_gate_logit" in jax.tree_util.keystr(path)
    except Exception:
        return any("history_memory_gate_logit" in str(p) for p in path)


def snapshot_gate_values(model) -> dict[str, jnp.ndarray]:
    """Return a path-string -> array snapshot of every history_memory_gate_logit."""
    state = nnx.state(model)
    snap: dict[str, jnp.ndarray] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        if _is_gate_path(path):
            snap[jax.tree_util.keystr(path)] = jnp.asarray(leaf)
    return snap


def summarize_gate_values(snapshot: dict[str, jnp.ndarray]) -> dict[str, float]:
    """Small numeric summary of gate logits and their sigmoid values."""
    if not snapshot:
        return {
            "count": 0,
            "logit_min": float("nan"),
            "logit_mean": float("nan"),
            "logit_max": float("nan"),
            "sigmoid_min": float("nan"),
            "sigmoid_mean": float("nan"),
            "sigmoid_max": float("nan"),
        }
    flat = np.concatenate(
        [np.asarray(jax.device_get(v), dtype=np.float32).reshape(-1) for v in snapshot.values()]
    )
    gate = 1.0 / (1.0 + np.exp(-flat))
    return {
        "count": int(flat.size),
        "logit_min": float(flat.min()),
        "logit_mean": float(flat.mean()),
        "logit_max": float(flat.max()),
        "sigmoid_min": float(gate.min()),
        "sigmoid_mean": float(gate.mean()),
        "sigmoid_max": float(gate.max()),
    }


def log_gate_summary(label: str, snapshot: dict[str, jnp.ndarray]) -> dict[str, float]:
    """Log and return a compact history gate diagnostic."""
    stats = summarize_gate_values(snapshot)
    logging.info(
        "%s history_memory_gate_logit: count=%d "
        "logit[min/mean/max]=%.4f/%.4f/%.4f "
        "sigmoid[min/mean/max]=%.6f/%.6f/%.6f",
        label,
        stats["count"],
        stats["logit_min"],
        stats["logit_mean"],
        stats["logit_max"],
        stats["sigmoid_min"],
        stats["sigmoid_mean"],
        stats["sigmoid_max"],
    )
    return stats


def set_gate_values(model, value: float) -> None:
    """Force every history_memory_gate_logit to ``value`` (in-place on model state)."""
    state = nnx.state(model)
    new_state = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.full_like(leaf, value) if _is_gate_path(path) else leaf,
        state,
    )
    nnx.update(model, new_state)


def restore_gate_values(model, snapshot: dict[str, jnp.ndarray]) -> None:
    """Restore gate logits captured by :func:`snapshot_gate_values`."""
    state = nnx.state(model)
    new_state = jax.tree_util.tree_map_with_path(
        lambda path, leaf: snapshot[jax.tree_util.keystr(path)]
        if _is_gate_path(path) and jax.tree_util.keystr(path) in snapshot
        else leaf,
        state,
    )
    nnx.update(model, new_state)


def _fixed_gate_module(model):
    """Return the wrapped SigLIP Linen module that owns ``history_gate_fixed``."""
    try:
        module = model.PaliGemma.img.module
    except AttributeError:
        return None
    return module if hasattr(module, "history_gate_fixed") else None


def snapshot_fixed_gate_value(model) -> float | None:
    """Return the current static ``history_gate_fixed`` value, if supported."""
    module = _fixed_gate_module(model)
    if module is None:
        return None
    value = module.history_gate_fixed
    return None if value is None else float(value)


def set_fixed_gate_value(model, value: float | None) -> bool:
    """Temporarily override the static fixed-gate probability on the image encoder.

    Returns True if the model supports the override. The wrapped Linen module is
    immutable/dataclass-like, so replace the module object rather than mutating
    the field in place.
    """
    module = _fixed_gate_module(model)
    if module is None:
        return False
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"fixed gate override must be in [0, 1], got {value}")
    model.PaliGemma.img.module = dataclasses.replace(module, history_gate_fixed=value)
    return True


def effective_fixed_gate_value(mode: AblationMode, args: Args) -> float:
    if mode == AblationMode.MEMORY_OFF:
        return 0.0
    if mode == AblationMode.FORCE_MEMORY_GATE:
        return float(1.0 / (1.0 + np.exp(-float(args.force_memory_gate_logit))))
    raise ValueError(f"No fixed-gate override for mode {mode}")


def reset_policy_jit(policy: _policy.Policy) -> None:
    """Re-capture the model's (possibly mutated) state into a fresh module_jit wrapper.

    ``nnx_utils.module_jit`` snapshots ``nnx.split(model)`` at construction time, so
    after mutating ``policy._model``'s params we need to rebuild the JIT closure for
    those changes to take effect at the next ``policy.infer`` call.
    """
    policy._sample_actions = nnx_utils.module_jit(policy._model.sample_actions)


@dataclasses.dataclass(frozen=True)
class _DropKeys(_transforms.DataTransformFn):
    """Pop the named keys out of the data dict (silent if a key is missing)."""

    keys: tuple[str, ...]

    def __call__(self, data: dict) -> dict:
        out = dict(data)
        for k in self.keys:
            out.pop(k, None)
        return out


def patch_output_transform_for_chunked_actions(
    policy: _policy.Policy, valid_action_dims: int, model_action_dim: int
) -> None:
    """Prepend pre-Unnormalize fixups to ``policy._output_transform``.

    Why this is necessary
    ---------------------
    ``LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem`` (the data config used by
    every Pi0Mem / Pi0MemCompress UMI training entry) declares its
    ``model_transforms`` Group with ``inputs=[...]`` only — there is no
    ``outputs=[ChunkActions(target_dim=20)]`` like its non-Pi0Mem siblings have.
    On top of that, ``MultiDataConfigFactory.state_pad_dim=128`` pads the
    input state from its raw 38-d shape up to 128, but no symmetric un-pad
    runs on the output side either.

    On the forward path the policy stuffs both ``actions`` (shape
    ``[H, action_dim=32]``) and ``state`` (shape ``[128]``) into the output
    dict, and ``Unnormalize`` tries to walk it with the *training-time*
    ``normalize_masks`` (``actions`` mask is 20-d, ``state`` mask is 38-d).
    Without the patch you get
    ``assert len(norm_mask) == y.shape[-1]`` failing on whichever key
    happens to come first.

    We fix this at inference time inside our script (without touching the
    shared data config) by prepending two cheap transforms:

      1. ``ChunkActions(target_dim=valid_action_dims)`` — slices the trailing
         padding columns off actions so its last dim matches the action mask.
      2. ``_DropKeys(keys=("state",))`` — drops state from the output dict
         entirely. The ablation script only needs predicted actions; keeping
         state would force us to also reverse-engineer the (2, 19) ->
         flatten -> pad(128) transform chain just to make Unnormalize happy.
    """
    existing = policy._output_transform
    existing_transforms = list(getattr(existing, "transforms", [existing]))

    prepend: list[_transforms.DataTransformFn] = []
    if valid_action_dims < model_action_dim:
        prepend.append(_transforms.ChunkActions(target_dim=valid_action_dims))
    prepend.append(_DropKeys(keys=("state",)))

    if not prepend:
        return
    policy._output_transform = _transforms.compose([*prepend, *existing_transforms])
    logging.info(
        "Patched policy._output_transform: prepended "
        f"{[type(t).__name__ for t in prepend]} so Unnormalize sees "
        f"actions with last dim={valid_action_dims} and no state key."
    )


# ---------------------------------------------------------------------------
# Multi-GPU + batched inference
# ---------------------------------------------------------------------------


def _resolve_batch_size(requested: int, *, multi_gpu: bool) -> int:
    """Choose a sensible default batch size if the user passed ``0``.

    With multi_gpu we want each device to hold ~1 sample (so the cost reduces
    to the per-device latency, not the all-gather), which means
    ``batch_size = num_devices``. Without multi_gpu we default to 4 (good amortization
    on a single H100/A100 without blowing VRAM on the PaliGemma KV cache).
    """
    if requested > 0:
        return requested
    n = max(1, jax.device_count())
    return n if multi_gpu else min(4, n if n > 1 else 4)


def _build_data_sharding(
    *, batch_size: int, multi_gpu: bool
) -> tuple[jax.sharding.Mesh | None, jax.sharding.NamedSharding | None]:
    """Return ``(mesh, batch_sharding)`` for data-parallel inference.

    Reuses the SAME mesh axis name (``"x"``) that ``openpi.models.model.restore_params``
    uses to load the policy weights, so the replicated params and our
    batch-sharded inputs live on a single coherent mesh -- no cross-mesh
    re-transfer when we call ``policy._sample_actions``.

    Falls back to ``(None, None)`` when there's only one visible device or the
    user disabled multi_gpu; in that case ``batched_infer`` skips the device_put
    and JAX places everything on ``jax.devices()[0]``.
    """
    if not multi_gpu:
        return None, None
    n = jax.device_count()
    if n <= 1:
        return None, None
    if batch_size % n != 0:
        logging.warning(
            f"batch_size={batch_size} is not divisible by jax.device_count()={n}; "
            "data-parallel sharding requires divisibility. Falling back to single-device "
            "batching. Pass --batch-size <multiple of %d> to enable multi-GPU.",
            n,
        )
        return None, None
    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
    return mesh, sharding


def _stack_dicts(dicts: list[dict]) -> dict:
    """Stack a list of structurally-identical pytree dicts along a new axis-0.

    Drops keys that are non-array (e.g. leftover Python strings) so they don't
    end up in the model's ``Observation`` -- ``policy._input_transform`` already
    consumed the ``prompt`` string and produced ``tokenized_prompt`` arrays, so
    nothing downstream needs the raw string.
    """
    if not dicts:
        return {}

    def _walk(values: list):
        first = values[0]
        if isinstance(first, dict):
            keys = list(first.keys())
            return {k: _walk([d[k] for d in values]) for k in keys}
        if isinstance(first, str):
            return None
        arrs = [np.asarray(v) for v in values]
        return np.stack(arrs, axis=0)

    stacked = _walk(dicts)

    def _prune(d):
        if not isinstance(d, dict):
            return d
        return {k: _prune(v) for k, v in d.items() if v is not None}

    return _prune(stacked)


def batched_infer(
    policy: _policy.Policy,
    obs_list: list[dict],
    *,
    data_sharding: jax.sharding.NamedSharding | None,
    pad_to: int | None = None,
    sample_rng: jax.Array | None = None,
) -> list[dict]:
    """Run ``policy.infer`` over a list of observations in one fused JIT call.

    Why this exists
    ---------------
    ``policy.infer`` is per-sample (batch=1). Calling it in a Python loop pays
    the JAX launch + (with FSDP-style sharding) all-gather cost N times. The
    speedup from batching even on a single GPU is typically 3-6x; combined
    with ``data_sharding`` it scales near-linearly with the number of visible
    GPUs because each device only sees ``batch_size / num_devices`` samples.

    Why we still loop the transforms
    --------------------------------
    The input transform chain contains per-sample ops that don't vectorize
    cleanly (``InjectDefaultPrompt`` reads a Python str, ``TokenizePrompt``
    runs the SentencePiece tokenizer host-side, etc.), and the output
    transform chain runs ``Unnormalize`` over a NormStats mask whose length
    must match ``last_dim`` (assertion enforced upstream). So we run them
    per-sample and only batch the *model call* itself, which is where >99%
    of the wall time lives anyway.

    Padding
    -------
    If ``pad_to`` is supplied and ``len(obs_list) < pad_to``, we duplicate the
    last sample to reach the target size, so every batched call traces with
    the same shape and reuses the cached XLA executable. The padded outputs
    are discarded by the caller (we still return them so indices line up;
    pass ``[:original_len]``).
    """
    pending = submit_batched_infer(
        policy,
        obs_list,
        data_sharding=data_sharding,
        pad_to=pad_to,
        sample_rng=sample_rng,
    )
    if pending is None:
        return []
    return materialize_batch_outputs(policy, pending)


@dataclasses.dataclass
class _PendingBatch:
    """Carries a not-yet-materialized GPU output back to the main loop.

    The whole point of splitting submit/materialize is to enable JAX's async
    dispatch: the call to ``policy._sample_actions(rng, obs)`` returns a
    not-yet-realized ``jax.Array`` immediately (the kernel is queued on the
    GPU), so the main loop can submit the *next* batch, do CPU bookkeeping,
    or trigger the next-batch prefetch in the loader pool -- all while the
    GPU is still chewing on this batch. We only block on the GPU when we
    actually need the numbers via ``np.asarray(batched_actions)`` in
    ``materialize_batch_outputs``.

    This pattern is the same one ``scripts/lerobot_value_infer.py`` uses to
    keep the GPU near 100% utilization regardless of how fast or slow the
    data loader is.
    """

    transformed: list[dict]          # per-sample transformed obs dicts (CPU)
    batched_actions: jax.Array       # not-yet-realized; np.asarray() blocks
    valid_size: int                  # how many of ``transformed`` are real (rest = padding)
    # Fine-grained timing for profiling. Lets us tell apart:
    #   - input_transform_s: main-thread serial per-sample preprocessing
    #     (Normalize, ResizeImages, TokenizePrompt, BuildVideoTensor, ...).
    #     GIL-bound and NOT overlapped with GPU. Often the hidden bottleneck
    #     when nvidia-smi shows idle GPU despite long batch wall time.
    #   - stack_put_s: stack dicts + jnp.asarray + device_put (host->device
    #     transfer + sharding). Usually small but jumps with big batches.
    #   - dispatch_s: the actual ``policy._sample_actions(...)`` call. With
    #     async dispatch this should be microseconds (just queueing the
    #     kernels), NOT the GPU forward time. If it's >50ms either we're
    #     missing async dispatch or something forces a host sync.
    input_transform_s: float = 0.0
    stack_put_s: float = 0.0
    dispatch_s: float = 0.0


def submit_batched_infer(
    policy: _policy.Policy,
    obs_list: list[dict],
    *,
    data_sharding: jax.sharding.NamedSharding | None,
    pad_to: int | None = None,
    sample_rng: jax.Array | None = None,
) -> _PendingBatch | None:
    """Run per-sample input transform + stacking and **queue** the model call.

    Returns immediately with a ``_PendingBatch`` whose ``batched_actions`` is a
    *future* ``jax.Array`` -- the actual GPU kernel hasn't necessarily completed
    yet. Pass the returned ``_PendingBatch`` to ``materialize_batch_outputs``
    later (typically on the next loop iteration) to retrieve the per-sample
    output dicts. While the caller is doing other work between submit and
    materialize, the GPU is busy.
    """
    if not obs_list:
        return None

    t0 = time.perf_counter()
    transformed: list[dict] = []
    for obs in obs_list:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = policy._input_transform(inputs)
        transformed.append(inputs)

    valid_size = len(transformed)
    if pad_to is not None and valid_size < pad_to:
        pad_n = pad_to - valid_size
        transformed.extend(transformed[-1] for _ in range(pad_n))
    t_input_done = time.perf_counter()

    stacked = _stack_dicts(transformed)
    stacked = jax.tree.map(jnp.asarray, stacked)

    if data_sharding is not None:
        stacked = jax.device_put(stacked, data_sharding)
    t_stack_done = time.perf_counter()

    observation = _model.Observation.from_dict(stacked)
    if sample_rng is None:
        policy._rng, sample_rng = jax.random.split(policy._rng)
    sample_kwargs = dict(policy._sample_kwargs)
    batched_actions = policy._sample_actions(sample_rng, observation, **sample_kwargs)
    t_dispatch_done = time.perf_counter()
    # IMPORTANT: do NOT call np.asarray() here -- that would block on the GPU
    # and kill the whole point of async dispatch. The caller does that later
    # in materialize_batch_outputs(), ideally after submitting the next batch.
    return _PendingBatch(
        transformed=transformed,
        batched_actions=batched_actions,
        valid_size=valid_size,
        input_transform_s=t_input_done - t0,
        stack_put_s=t_stack_done - t_input_done,
        dispatch_s=t_dispatch_done - t_stack_done,
    )


def materialize_batch_outputs(
    policy: _policy.Policy,
    pending: _PendingBatch,
) -> list[dict]:
    """Block on the GPU result of ``pending`` and run per-sample output transforms.

    Only call this once you no longer need the GPU to be racing ahead on this
    batch -- i.e. after the *next* batch has been submitted. The first
    ``np.asarray()`` here is the only synchronization point in the pipeline.
    """
    batched_actions = np.asarray(pending.batched_actions)
    results: list[dict] = []
    for i in range(pending.valid_size):
        t = pending.transformed[i]
        out = {
            "state": np.asarray(t["state"]),
            "actions": batched_actions[i],
        }
        out = policy._output_transform(out)
        results.append(out)
    return results


# ---------------------------------------------------------------------------
# Episode loop
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EpisodeStats:
    episode_idx: int
    num_frames: int
    sample_indices: np.ndarray  # [N]
    preds: np.ndarray  # [N, action_horizon, valid_action_dims]
    gts: np.ndarray  # [N, action_horizon, valid_action_dims]
    mse_per_dim: np.ndarray  # [valid_action_dims]
    mae_per_dim: np.ndarray
    overall_mse: float
    overall_mae: float
    # First action step only (single-step lookahead): often a more sensitive signal
    # than the chunk mean because errors compound less.
    first_step_mse: float
    first_step_mae: float


def _coerce_actions(predicted: np.ndarray, valid_dims: int) -> np.ndarray:
    """Force the policy output into shape ``[action_horizon, valid_dims]``.

    Pi0MemCompress predicts ``[action_horizon, action_dim=32]``. Only the first
    ``valid_dims`` (e.g. 20 for bimanual UMI) carry signal; the rest are
    zero-loss padding.
    """
    predicted = np.asarray(predicted)
    if predicted.ndim == 1:
        predicted = predicted[None, :]
    return predicted[:, :valid_dims]


# Module-level worker state used by the fork-based prefetch process pool.
# ``fork()`` on Linux makes child processes inherit the parent's memory
# (copy-on-write), so we set ``_WORKER_DATASET`` once in main() *before*
# spawning the executor and every child sees it without us paying the
# ~5 MB-per-task pickle cost of sending VideoFrameDataset over the pipe.
_WORKER_DATASET: VideoFrameDataset | None = None


def _set_worker_dataset(d: VideoFrameDataset) -> None:
    """Stash the per-process VideoFrameDataset. Must be called BEFORE forking
    the loader workers; once they're forked the assignment in the parent does
    not propagate to children."""
    global _WORKER_DATASET
    _WORKER_DATASET = d


def _worker_warmup_sleep(sleep_s: float) -> int:
    """Tiny task that just busy-waits and returns ``os.getpid()``.

    We submit ``num_load_workers`` of these and block on **all** of them right
    after creating the ``ProcessPoolExecutor``. The sleep guarantees the first
    task can't complete and free its worker for reuse before the executor has
    spawned all peers -- forcing ``ProcessPoolExecutor._adjust_process_count``
    to fork the full ``max_workers`` count *right now*, while we're still
    single-threaded and JAX has not yet touched CUDA.

    Background: ``ProcessPoolExecutor`` is lazy -- it doesn't fork worker
    processes at construction time, only when ``submit`` calls actually need
    them. Without this barrier, the first prefetch call inside an episode
    fires off all the forks **after** ``create_trained_policy`` has already
    initialized JAX, CUDA, and several background threads. Those workers
    inherit JAX's CUDA context + threads and either crash or hang -- you see
    the kernel warning::

        RuntimeWarning: os.fork() was called. os.fork() is incompatible
        with multithreaded code, and JAX is multithreaded, so this will
        likely lead to a deadlock.
    """
    import time as _time
    _time.sleep(sleep_s)
    return os.getpid()


def _worker_load_one(
    idx: int,
    mode_value: str,
    image_keys: tuple[str, ...],
    num_frames: int,
    default_prompt: str,
    donor_indices_pool: tuple[int, ...] | None,
    valid_dims: int,
    base_seed: int,
    zero_current_frame: bool,
) -> tuple[dict, np.ndarray, float]:
    """Module-level entry point invoked inside fork-spawned loader processes.

    Each child has its own private PyAV / ffmpeg decoder state (opened lazily
    on the first ``_WORKER_DATASET[idx]`` call), and crucially its own GIL --
    so 8 workers actually run in parallel during mp4 decode, unlike the
    earlier ThreadPoolExecutor version which was effectively serialized by
    HF datasets' formatter calls.

    Returns ``(obs, gt, worker_wall_s)`` where ``worker_wall_s`` is the wall
    time spent inside this worker call (excluding pickle / IPC). Summed over
    all calls in a profile window and divided by the parent's wall window,
    we get the *effective parallelism* the pool is achieving -- which lets
    us diagnose whether per-sample decode is slow or whether IPC / shared-
    resource contention is artificially serializing workers.

    The function is positional-only so ``executor.submit`` can serialize the
    args quickly. ``donor_indices_pool`` is a tuple (not a list) because
    tuples pickle slightly smaller; we materialize it back to a list inside
    ``_load_one_obs`` if the mode actually consumes it.
    """
    assert _WORKER_DATASET is not None, (
        "_WORKER_DATASET was not set in the parent before forking the worker "
        "pool. Call _set_worker_dataset(dataset) BEFORE creating the executor."
    )
    t0 = time.perf_counter()
    mode = AblationMode(mode_value)
    local_rng = np.random.default_rng(int(base_seed) + int(idx))
    obs, gt = _load_one_obs(
        _WORKER_DATASET,
        idx,
        mode=mode,
        image_keys=image_keys,
        num_frames=num_frames,
        default_prompt=default_prompt,
        rng=local_rng,
        donor_indices_pool=list(donor_indices_pool) if donor_indices_pool else None,
        valid_dims=valid_dims,
        zero_current_frame=bool(zero_current_frame),
    )
    return obs, gt, time.perf_counter() - t0


def _load_one_obs(
    dataset: VideoFrameDataset,
    idx: int,
    *,
    mode: AblationMode,
    image_keys: tuple[str, ...],
    num_frames: int,
    default_prompt: str,
    rng: np.random.Generator,
    donor_indices_pool: list[int] | None,
    valid_dims: int,
    zero_current_frame: bool,
) -> tuple[dict, np.ndarray]:
    """Load + perturb + obs-dict one frame index. Returns ``(obs, gt_actions)``.

    ``rng`` must be a per-call ``np.random.Generator``; the prefetch worker pool
    creates a fresh seeded RNG per idx so concurrent calls are race-free and
    perturbation outputs are reproducible from ``(args.seed, idx, mode)``.
    """
    sample = dataset[idx]

    donor_sample: dict | None = None
    if mode == AblationMode.WRONG_HISTORY:
        assert donor_indices_pool, "wrong_history needs a non-empty donor pool"
        donor_idx = int(rng.choice(donor_indices_pool))
        tries = 0
        while donor_idx == idx and tries < 5:
            donor_idx = int(rng.choice(donor_indices_pool))
            tries += 1
        donor_sample = dataset[donor_idx]

    perturbed = apply_ablation_to_sample(
        sample,
        mode=mode,
        image_keys=image_keys,
        num_frames=num_frames,
        donor_sample=donor_sample,
        rng=rng,
        zero_current_frame=zero_current_frame,
    )
    obs = build_obs_dict(
        perturbed,
        image_keys=image_keys,
        num_frames=num_frames,
        default_prompt=default_prompt,
    )
    true_actions = np.asarray(sample["actions"])
    if true_actions.ndim == 1:
        true_actions = true_actions[None, :]
    return obs, true_actions[:, :valid_dims]


def _prefetched_obs_iter(
    sample_indices: list[int],
    *,
    dataset: VideoFrameDataset,
    mode: AblationMode,
    image_keys: tuple[str, ...],
    num_frames: int,
    default_prompt: str,
    donor_indices_pool: list[int] | None,
    valid_dims: int,
    base_seed: int,
    zero_current_frame: bool,
    executor: concurrent.futures.ProcessPoolExecutor | None,
    prefetch_size: int,
):
    """Yield ``(obs, gt)`` pairs in ``sample_indices`` order with up to
    ``prefetch_size`` items being decoded concurrently in background **processes**.

    Why process pool, not thread pool
    ---------------------------------
    HF datasets' formatter chain does numpy->torch conversion in a Python
    ``list`` comprehension after the PyAV decode releases the GIL. That
    Python loop is GIL-bound and serializes ThreadPoolExecutor workers down
    to roughly 1 effective worker (measured: 8-thread prefetch only sped this
    up from 1.25 s/frame to 1.18 s/frame, i.e. ~5%). A fork-based
    ``ProcessPoolExecutor`` sidesteps the GIL entirely -- each child process
    decodes mp4s with its own Python interpreter -- so 8 workers actually
    parallelize the decode side.

    The executor is created once at ``main()`` start (before any
    inference-side state mutation) and reused across every mode/episode. The
    dataset is shared via the inherited ``_WORKER_DATASET`` module global
    (cheap CoW after fork), and only small per-call args (idx, mode, image
    keys, donor pool) are pickled into the worker queue.

    If ``executor is None`` we fall back to a fully synchronous generator so
    this function still works for single-process debug runs.
    """
    donor_tuple: tuple[int, ...] | None = (
        tuple(int(x) for x in donor_indices_pool) if donor_indices_pool else None
    )

    def args_for(idx: int) -> tuple:
        return (
            int(idx),
            mode.value,
            image_keys,
            int(num_frames),
            default_prompt,
            donor_tuple,
            int(valid_dims),
            int(base_seed),
            bool(zero_current_frame),
        )

    if executor is None:
        # Synchronous fallback: run every load on the main process. Used for
        # debugging or when args.num_load_workers == 0.
        for idx in sample_indices:
            yield _worker_load_one(*args_for(idx))
        return

    pending: collections.deque = collections.deque()
    idx_iter = iter(sample_indices)
    for _ in range(max(1, prefetch_size)):
        try:
            pending.append(executor.submit(_worker_load_one, *args_for(next(idx_iter))))
        except StopIteration:
            break
    while pending:
        fut = pending.popleft()
        try:
            obs, gt, worker_s = fut.result()
        except Exception as exc:
            logging.exception("Loader worker failed: %s", exc)
            raise
        try:
            pending.append(executor.submit(_worker_load_one, *args_for(next(idx_iter))))
        except StopIteration:
            pass
        yield obs, gt, worker_s


def evaluate_episode(
    *,
    dataset: VideoFrameDataset,
    sample_indices: list[int],
    policy: _policy.Policy,
    image_keys: tuple[str, ...],
    num_frames: int,
    mode: AblationMode,
    valid_dims: int,
    default_prompt: str,
    rng: np.random.Generator,
    frame_stride_eval: int,
    max_frames: int,
    frame_window: FrameWindow,
    donor_indices_pool: list[int] | None,
    episode_label: str = "",
    batch_size: int = 1,
    data_sharding: jax.sharding.NamedSharding | None = None,
    is_first_batch_ref: list[bool] | None = None,
    load_executor: concurrent.futures.ProcessPoolExecutor | None = None,
    prefetch_size: int = 0,
    base_seed: int = 0,
    zero_current_frame: bool = False,
    sampling_seed: int = 0,
    profile_every_n_batches: int = 0,
) -> EpisodeStats | None:
    """Run inference over (a stride of) an episode and aggregate errors.

    The pipeline is **streamed**: as soon as we accumulate ``batch_size`` frames
    of CPU-side prep work (image decode + obs-dict build), we hand them off to
    the GPU. Doing it this way (instead of "load everything first, then infer")
    means GPU utilization rises within the first few seconds and stays at 0%
    only during the genuine XLA compilation window of the first batch, not
    during data loading.
    """
    n = len(sample_indices)
    if n == 0:
        return None

    if frame_stride_eval > 1:
        sample_indices = sample_indices[::frame_stride_eval]
    if max_frames > 0 and len(sample_indices) > max_frames:
        if frame_window == FrameWindow.START:
            sample_indices = sample_indices[:max_frames]
        elif frame_window == FrameWindow.END:
            sample_indices = sample_indices[-max_frames:]
        else:
            raise ValueError(f"Unsupported frame_window: {frame_window}")

    bs = max(1, batch_size)
    n_total = len(sample_indices)

    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    pbar = tqdm.tqdm(
        total=n_total,
        desc=(episode_label or f"{mode.value}") + f" [bs={bs}]",
        leave=False,
        dynamic_ncols=True,
    )
    running_sq_sum = 0.0
    running_count = 0

    obs_buf: list[dict] = []
    gt_buf: list[np.ndarray] = []
    idx_buf: list[int] = []

    # Async pipeline state: when we ``submit_batched_infer(...)`` for chunk N,
    # the GPU starts work immediately (jax.Array future), and we hold onto
    # the future + the chunk's ground-truths in (prev_pending, prev_gts).
    # On chunk N+1's submit, *then* we materialize chunk N's results -- so
    # while we're blocking on ``np.asarray(prev.batched_actions)``, chunk
    # N+1 is already running on the GPU (with chunk N+2's data already being
    # decoded by the loader workers in the background). Same pattern as
    # ``scripts/lerobot_value_infer.py``.
    prev_pending: _PendingBatch | None = None
    prev_gts: list[np.ndarray] = []

    # Per-batch timing accumulators (only used if profile_every_n_batches > 0).
    # Breakdown of where each batch's wall time goes:
    #   data:      ProcessPool / loader workers prepping the next ``bs`` obs
    #              (mp4 decode etc.). Hidden behind GPU work in an ideal pipe.
    #   input_tf:  MAIN-thread serial per-sample input transforms in
    #              submit_batched_infer (Normalize, ResizeImages,
    #              TokenizePrompt, BuildVideoTensor, ...). GIL-bound, NOT
    #              overlapped with GPU. Often the hidden killer when GPU
    #              looks idle in nvidia-smi.
    #   stack_put: stack + jax.device_put (host->device transfer).
    #   dispatch:  the ``policy._sample_actions(...)`` call itself. With
    #              JAX async dispatch this should be microseconds. If it's
    #              not, something forces a host-side sync.
    #   consume:   np.asarray(prev_future) (THIS is when GPU work actually
    #              shows up as wall time) + per-sample output transforms.
    timing_acc = {
        "data": 0.0,
        "input_tf": 0.0,
        "stack_put": 0.0,
        "dispatch": 0.0,
        "consume": 0.0,
        # Sum of per-worker self-reported wall time within this batch's
        # data_collect window. If pool truly has N workers running in
        # parallel, ``worker_sum / data_collect`` -> N. If it's ~1, workers
        # are effectively serialized (IPC / shared lock / etc).
        "worker_sum": 0.0,
        "n": 0,
    }
    # Per-batch worker-wall accumulator: cleared at each _flush so we can
    # compare against that batch's data_collect window.
    cur_batch_worker_sum = [0.0]

    def _consume_prev() -> float:
        """Block on the GPU result for ``prev_pending`` and update metrics.
        Returns wall time in seconds."""
        nonlocal prev_pending, prev_gts, running_sq_sum, running_count
        if prev_pending is None:
            return 0.0
        t0 = time.perf_counter()
        outs = materialize_batch_outputs(policy, prev_pending)
        for j, out in enumerate(outs):
            predicted_actions = _coerce_actions(out["actions"], valid_dims)
            gt_chunked = prev_gts[j][: predicted_actions.shape[0], :valid_dims]
            preds.append(predicted_actions)
            gts.append(gt_chunked)
            running_sq_sum += float(((predicted_actions - gt_chunked) ** 2).mean())
            running_count += 1
        pbar.update(len(outs))
        if running_count > 0:
            pbar.set_postfix({"running_mse": running_sq_sum / running_count})
        prev_pending = None
        prev_gts = []
        return time.perf_counter() - t0

    def _flush(
        obs_chunk: list[dict],
        gt_chunk: list[np.ndarray],
        idx_chunk: list[int],
        data_collect_s: float,
    ) -> None:
        """Submit ``obs_chunk`` to the GPU (non-blocking) and materialize the
        previous chunk's results. This is the heart of the async pipeline:
        between the line that queues this chunk's kernels and the line that
        blocks on the previous chunk's results, the GPU has *both* a queued
        and an in-flight batch -- there's never a moment where it has nothing
        to do."""
        nonlocal prev_pending, prev_gts
        if not obs_chunk:
            return
        if not idx_chunk:
            raise ValueError("idx_chunk must be non-empty when obs_chunk is non-empty")

        sample_rng = jax.random.fold_in(
            jax.random.PRNGKey(int(sampling_seed)),
            int(idx_chunk[0]),
        )

        first_call = bool(is_first_batch_ref and is_first_batch_ref[0])
        if first_call:
            logging.info(
                "First submit_batched_infer call: XLA is about to (re)compile "
                "sample_actions for shape=(%d, ...) on %s. GPU-Util will stay at 0%% "
                "for the next 5-15 min while the XLA compiler runs on CPU. "
                "Persistent cache at ~/.cache/jax will make subsequent runs near-instant.",
                bs,
                "all visible GPUs (data-parallel)" if data_sharding is not None else "1 GPU",
            )

        # Submit the new chunk first (non-blocking; returns a future).
        cur_pending = submit_batched_infer(
            policy,
            obs_chunk,
            data_sharding=data_sharding,
            pad_to=bs,
            sample_rng=sample_rng,
        )

        if first_call:
            assert is_first_batch_ref is not None
            is_first_batch_ref[0] = False
            logging.info(
                "Submitted first batch; XLA tracing should be in flight. "
                "Subsequent submits are O(microseconds) until first materialize."
            )

        # NOW consume the previous pending batch -- GPU is already busy on the
        # new submit, so this is a cheap sync point.
        consume_s = _consume_prev()

        # Buffer the new pending future for the NEXT iteration's consume.
        prev_pending = cur_pending
        prev_gts = list(gt_chunk)

        # Optional per-batch profile log.
        if profile_every_n_batches > 0 and cur_pending is not None:
            timing_acc["data"] += data_collect_s
            timing_acc["input_tf"] += cur_pending.input_transform_s
            timing_acc["stack_put"] += cur_pending.stack_put_s
            timing_acc["dispatch"] += cur_pending.dispatch_s
            timing_acc["consume"] += consume_s
            timing_acc["worker_sum"] += cur_batch_worker_sum[0]
            cur_batch_worker_sum[0] = 0.0
            timing_acc["n"] += 1
            if timing_acc["n"] % profile_every_n_batches == 0:
                n = timing_acc["n"]
                avg_d = timing_acc["data"] / n
                avg_i = timing_acc["input_tf"] / n
                avg_sp = timing_acc["stack_put"] / n
                avg_dp = timing_acc["dispatch"] / n
                avg_c = timing_acc["consume"] / n
                avg_ws = timing_acc["worker_sum"] / n
                # ``avg_ws`` is the sum of all per-worker self-timed wall
                # seconds across one batch. If the pool achieves perfect
                # N-way parallelism, this should equal ``avg_d * N``.
                # ``effective_parallelism`` is the actual parallelism factor
                # we get -- anywhere below the worker count means workers
                # are contending on something (IPC pickle on main thread,
                # shared HF datasets lock, disk I/O, etc).
                eff_parallel = (avg_ws / avg_d) if avg_d > 0 else float("nan")
                # Async pipeline wall = max(data_collect, input_tf+stack+dispatch+consume).
                # data_collect runs in the loader pool concurrently with the
                # main thread, so it overlaps with everything else.
                main_thread = avg_i + avg_sp + avg_dp + avg_c
                total = max(avg_d, main_thread)
                candidates = (
                    ("data_collect", avg_d),
                    ("input_transform", avg_i),
                    ("stack+device_put", avg_sp),
                    ("dispatch", avg_dp),
                    ("consume (GPU sync + output_tf)", avg_c),
                )
                bottleneck = max(candidates, key=lambda t: t[1])
                logging.info(
                    "[profile bs=%d, last %d batches avg, frames/s=%.2f]\n"
                    "    data_collect      = %.3fs  (loader workers; overlaps w/ main thread)\n"
                    "      worker_self_sum = %.3fs  -> effective_parallelism = %.2fx of pool\n"
                    "    input_transform   = %.3fs  (MAIN, serial per-sample; GIL-bound)\n"
                    "    stack+device_put  = %.3fs  (host->device + sharding)\n"
                    "    dispatch          = %.3fs  (should be ~0 with async; if not, host sync)\n"
                    "    consume(GPU sync) = %.3fs  (np.asarray(prev_future) + output_tf)\n"
                    "    -> per-batch wall = %.2fs ; bottleneck = %s (%.2fs)",
                    bs,
                    n,
                    bs / total if total > 0 else float("inf"),
                    avg_d,
                    avg_ws,
                    eff_parallel,
                    avg_i,
                    avg_sp,
                    avg_dp,
                    avg_c,
                    total,
                    bottleneck[0],
                    bottleneck[1],
                )

    data_t0 = time.perf_counter()
    for idx, (obs, gt, worker_s) in zip(
        sample_indices,
        _prefetched_obs_iter(
            sample_indices,
            dataset=dataset,
            mode=mode,
            image_keys=image_keys,
            num_frames=num_frames,
            default_prompt=default_prompt,
            donor_indices_pool=donor_indices_pool,
            valid_dims=valid_dims,
            base_seed=base_seed,
            zero_current_frame=zero_current_frame,
            executor=load_executor,
            prefetch_size=prefetch_size,
        ),
    ):
        obs_buf.append(obs)
        gt_buf.append(gt)
        idx_buf.append(int(idx))
        cur_batch_worker_sum[0] += worker_s
        if len(obs_buf) >= bs:
            data_collect_s = time.perf_counter() - data_t0
            _flush(obs_buf, gt_buf, idx_buf, data_collect_s)
            obs_buf = []
            gt_buf = []
            idx_buf = []
            data_t0 = time.perf_counter()

    # Flush the trailing partial chunk (padded up to bs to keep XLA cache hot).
    data_collect_s = time.perf_counter() - data_t0
    _flush(obs_buf, gt_buf, idx_buf, data_collect_s)
    # Final drain: nothing's been submitted after this, so the last in-flight
    # batch still needs to be materialized.
    _consume_prev()

    pbar.close()
    preds_arr = np.stack(preds, axis=0)  # [N, H, D]
    gts_arr = np.stack(gts, axis=0)

    sq = (preds_arr - gts_arr) ** 2
    ae = np.abs(preds_arr - gts_arr)
    mse_per_dim = sq.mean(axis=(0, 1))
    mae_per_dim = ae.mean(axis=(0, 1))

    return EpisodeStats(
        episode_idx=int(sample_indices[0]),  # informational only; lookup is via list ordering
        num_frames=len(sample_indices),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        preds=preds_arr,
        gts=gts_arr,
        mse_per_dim=mse_per_dim,
        mae_per_dim=mae_per_dim,
        overall_mse=float(sq.mean()),
        overall_mae=float(ae.mean()),
        first_step_mse=float(((preds_arr[:, 0, :] - gts_arr[:, 0, :]) ** 2).mean()),
        first_step_mae=float(np.abs(preds_arr[:, 0, :] - gts_arr[:, 0, :]).mean()),
    )


def _sample_record_map(per_episode_stats: list[EpisodeStats]) -> dict[int, dict[str, np.ndarray | float]]:
    """Build per-frame records for delta/error diagnostics.

    The returned dict is kept in memory only. We intentionally avoid writing
    full action predictions to JSON; summaries below include only aggregate
    metrics plus a small top-k list.
    """
    records: dict[int, dict[str, np.ndarray | float]] = {}
    for stats in per_episode_stats:
        if len(stats.sample_indices) != len(stats.preds) or len(stats.preds) != len(stats.gts):
            raise ValueError(
                f"Prediction bookkeeping mismatch for episode {stats.episode_idx}: "
                f"{len(stats.sample_indices)} indices vs {len(stats.preds)} predictions "
                f"vs {len(stats.gts)} GT chunks"
            )
        for idx, pred, gt in zip(stats.sample_indices, stats.preds, stats.gts):
            pred_arr = np.asarray(pred)
            gt_arr = np.asarray(gt)
            records[int(idx)] = {
                "pred": pred_arr,
                "gt": gt_arr,
                "mse": float(np.mean(np.square(pred_arr - gt_arr))),
                "mae": float(np.mean(np.abs(pred_arr - gt_arr))),
                "first_step_mse": float(np.mean(np.square(pred_arr[0] - gt_arr[0]))),
                "first_step_mae": float(np.mean(np.abs(pred_arr[0] - gt_arr[0]))),
            }
    return records


def _prediction_delta_metrics(
    reference_records: dict[int, dict[str, np.ndarray | float]],
    candidate_records: dict[int, dict[str, np.ndarray | float]],
    *,
    reference_mode: str = "normal",
    top_k: int = 20,
) -> dict[str, object]:
    """Compare two modes' predicted action chunks on shared dataset indices.

    This is deliberately independent of ground-truth actions. If GT MSE barely
    changes but these deltas are also near zero, the ablation did not materially
    change the policy output; if these deltas are large while GT MSE is flat,
    the task metric is masking behavior changes.
    """
    common_indices = sorted(set(reference_records) & set(candidate_records))
    if not common_indices:
        return {
            "reference_mode": reference_mode,
            "num_common_frames": 0,
            "pred_delta_mse": float("nan"),
            "pred_delta_rmse": float("nan"),
            "pred_delta_mae": float("nan"),
            "pred_delta_max_abs": float("nan"),
            "pred_delta_first_step_mse": float("nan"),
            "pred_delta_first_step_mae": float("nan"),
            "gt_mse_delta_mean": float("nan"),
            "gt_mse_delta_on_changed_corr": float("nan"),
            "fraction_improved": float("nan"),
            "top_delta_frames": [],
            "delta_bins": [],
        }

    flat_diffs: list[np.ndarray] = []
    first_step_diffs: list[np.ndarray] = []
    frame_rows: list[dict[str, float | int]] = []
    for idx in common_indices:
        ref_record = reference_records[idx]
        cand_record = candidate_records[idx]
        ref = np.asarray(ref_record["pred"])
        cand = np.asarray(cand_record["pred"])
        horizon = min(ref.shape[0], cand.shape[0])
        dims = min(ref.shape[1], cand.shape[1])
        diff = cand[:horizon, :dims] - ref[:horizon, :dims]
        flat_diffs.append(diff.reshape(-1, dims))
        first_step_diffs.append(diff[0])
        pred_delta_mse = float(np.mean(np.square(diff)))
        pred_delta_mae = float(np.mean(np.abs(diff)))
        normal_mse = float(ref_record["mse"])
        candidate_mse = float(cand_record["mse"])
        normal_first_mse = float(ref_record["first_step_mse"])
        candidate_first_mse = float(cand_record["first_step_mse"])
        frame_rows.append(
            {
                "dataset_index": int(idx),
                "pred_delta_rmse": float(np.sqrt(pred_delta_mse)),
                "pred_delta_mae": pred_delta_mae,
                "normal_mse": normal_mse,
                "candidate_mse": candidate_mse,
                "gt_mse_delta": candidate_mse - normal_mse,
                "normal_first_step_mse": normal_first_mse,
                "candidate_first_step_mse": candidate_first_mse,
                "gt_first_step_mse_delta": candidate_first_mse - normal_first_mse,
            }
        )

    flat = np.concatenate(flat_diffs, axis=0)
    first = np.stack(first_step_diffs, axis=0)
    mse = float(np.mean(np.square(flat)))
    first_mse = float(np.mean(np.square(first)))
    pred_delta_by_frame = np.asarray([float(r["pred_delta_rmse"]) for r in frame_rows], dtype=np.float64)
    gt_mse_delta_by_frame = np.asarray([float(r["gt_mse_delta"]) for r in frame_rows], dtype=np.float64)
    if len(frame_rows) > 1 and np.std(pred_delta_by_frame) > 0 and np.std(gt_mse_delta_by_frame) > 0:
        corr = float(np.corrcoef(pred_delta_by_frame, gt_mse_delta_by_frame)[0, 1])
    else:
        corr = float("nan")

    sorted_rows = sorted(frame_rows, key=lambda r: float(r["pred_delta_rmse"]), reverse=True)

    def summarize_rows(rows: list[dict[str, float | int]], label: str) -> dict[str, object]:
        if not rows:
            return {
                "label": label,
                "num_frames": 0,
                "pred_delta_rmse_mean": float("nan"),
                "gt_mse_delta_mean": float("nan"),
                "gt_mse_delta_median": float("nan"),
                "fraction_improved": float("nan"),
            }
        deltas = np.asarray([float(r["gt_mse_delta"]) for r in rows], dtype=np.float64)
        pred_deltas = np.asarray([float(r["pred_delta_rmse"]) for r in rows], dtype=np.float64)
        return {
            "label": label,
            "num_frames": len(rows),
            "pred_delta_rmse_mean": float(np.mean(pred_deltas)),
            "gt_mse_delta_mean": float(np.mean(deltas)),
            "gt_mse_delta_median": float(np.median(deltas)),
            "fraction_improved": float(np.mean(deltas < 0.0)),
        }

    delta_bins = [
        summarize_rows(frame_rows, "all_common_frames"),
    ]
    for frac in (0.01, 0.05, 0.10, 0.20):
        n = max(1, int(np.ceil(len(sorted_rows) * frac)))
        delta_bins.append(summarize_rows(sorted_rows[:n], f"top_{int(frac * 100)}pct_pred_delta"))

    return {
        "reference_mode": reference_mode,
        "num_common_frames": len(common_indices),
        "pred_delta_mse": mse,
        "pred_delta_rmse": float(np.sqrt(mse)),
        "pred_delta_mae": float(np.mean(np.abs(flat))),
        "pred_delta_max_abs": float(np.max(np.abs(flat))),
        "pred_delta_first_step_mse": first_mse,
        "pred_delta_first_step_mae": float(np.mean(np.abs(first))),
        "gt_mse_delta_mean": float(np.mean(gt_mse_delta_by_frame)),
        "gt_mse_delta_median": float(np.median(gt_mse_delta_by_frame)),
        "gt_mse_delta_on_changed_corr": corr,
        "fraction_improved": float(np.mean(gt_mse_delta_by_frame < 0.0)),
        "top_delta_frames": sorted_rows[: max(0, int(top_k))],
        "delta_bins": delta_bins,
    }


def run_mode(
    *,
    mode: AblationMode,
    args: Args,
    policy: _policy.Policy,
    dataset: VideoFrameDataset,
    episodes: dict[int, list[int]],
    image_keys: tuple[str, ...],
    num_frames: int,
    batch_size: int,
    data_sharding: jax.sharding.NamedSharding | None,
    is_first_batch_ref: list[bool],
    load_executor: concurrent.futures.ProcessPoolExecutor | None,
    prefetch_size: int,
) -> tuple[dict, dict[int, dict[str, np.ndarray | float]]]:
    """Evaluate a single ablation mode across the requested episodes."""
    logging.info(f"=== Ablation mode: {mode.value} ===")

    original_gate_snapshot = snapshot_gate_values(policy._model)
    original_gate_stats = log_gate_summary(
        f"[{mode.value}] original", original_gate_snapshot
    )
    fixed_gate_snapshot = snapshot_fixed_gate_value(policy._model)
    logging.info(
        "[%s] original history_gate_fixed=%s",
        mode.value,
        fixed_gate_snapshot if fixed_gate_snapshot is not None else "None",
    )

    # Gate ablations need a model mutation; the data-side modes do not.
    gate_snapshot: dict[str, jnp.ndarray] | None = None
    fixed_gate_changed = False
    active_gate_stats: dict[str, float] | None = original_gate_stats
    active_fixed_gate = fixed_gate_snapshot
    if mode in (AblationMode.MEMORY_OFF, AblationMode.FORCE_MEMORY_GATE):
        gate_snapshot = original_gate_snapshot
        forced_logit = (
            args.memory_off_gate_logit
            if mode == AblationMode.MEMORY_OFF
            else args.force_memory_gate_logit
        )
        logging.info(
            f"Snapshotted {len(gate_snapshot)} history_memory_gate_logit param(s); "
            f"forcing all to {forced_logit:.1f} for {mode.value} run."
        )
        set_gate_values(policy._model, forced_logit)
        forced_fixed_gate = effective_fixed_gate_value(mode, args)
        fixed_gate_changed = set_fixed_gate_value(policy._model, forced_fixed_gate)
        if fixed_gate_changed:
            active_fixed_gate = forced_fixed_gate
            logging.info(
                "[%s] forcing effective history_gate_fixed to %.6f",
                mode.value,
                forced_fixed_gate,
            )
        else:
            logging.info(
                "[%s] model does not expose history_gate_fixed; using logit-only gate override.",
                mode.value,
            )
        reset_policy_jit(policy)
        active_gate_stats = log_gate_summary(
            f"[{mode.value}] active", snapshot_gate_values(policy._model)
        )
        logging.info(
            "[%s] active history_gate_fixed=%s",
            mode.value,
            snapshot_fixed_gate_value(policy._model)
            if snapshot_fixed_gate_value(policy._model) is not None
            else "None",
        )

    rng = np.random.default_rng(args.seed)

    sorted_eps = sorted(episodes.keys())
    if args.num_episodes >= 0:
        sorted_eps = sorted_eps[args.start_episode : args.start_episode + args.num_episodes]
    else:
        sorted_eps = sorted_eps[args.start_episode :]

    # ---- Episode-level shard split ---------------------------------------------
    # When running N parallel copies of the script (one per GPU), pick every
    # Nth episode round-robin so each shard handles roughly the same total
    # number of frames even if episode lengths are non-uniform.
    if args.episode_shard_total > 1:
        full_count = len(sorted_eps)
        sorted_eps = sorted_eps[args.episode_shard_index :: args.episode_shard_total]
        logging.info(
            "[%s] Shard %d/%d -> handling %d/%d episodes: %s",
            mode.value,
            args.episode_shard_index,
            args.episode_shard_total,
            len(sorted_eps),
            full_count,
            sorted_eps if len(sorted_eps) <= 12 else f"first 12: {sorted_eps[:12]} ...",
        )

    # For wrong_history, pre-build a pool of donor indices that excludes the *current*
    # episode (so the perturbed history really comes from a different trajectory).
    donor_pool_per_ep: dict[int, list[int]] = {}
    if mode == AblationMode.WRONG_HISTORY:
        all_other_eps = [ep for ep in episodes if ep not in set(sorted_eps)]
        for ep in sorted_eps:
            pool = []
            if all_other_eps:
                # Prefer indices outside the evaluated set first.
                for other in all_other_eps:
                    pool.extend(episodes[other])
            else:
                # Fallback: use indices from sibling evaluated episodes (still better
                # than reusing the same episode's frames).
                for other in sorted_eps:
                    if other != ep:
                        pool.extend(episodes[other])
            if not pool:
                # Last resort: same episode pool (will be filtered against ``idx`` itself).
                pool = list(episodes[ep])
            donor_pool_per_ep[ep] = pool

    per_episode_stats: list[EpisodeStats] = []
    try:
        for i, ep in enumerate(sorted_eps):
            stats = evaluate_episode(
                dataset=dataset,
                sample_indices=episodes[ep],
                policy=policy,
                image_keys=image_keys,
                num_frames=num_frames,
                mode=mode,
                valid_dims=args.valid_action_dims,
                default_prompt=args.default_prompt,
                rng=rng,
                frame_stride_eval=args.frame_stride_eval,
                max_frames=args.max_frames_per_episode,
                frame_window=args.frame_window,
                donor_indices_pool=donor_pool_per_ep.get(ep),
                episode_label=f"{mode.value} ep={ep} ({i + 1}/{len(sorted_eps)})",
                batch_size=batch_size,
                data_sharding=data_sharding,
                is_first_batch_ref=is_first_batch_ref,
                load_executor=load_executor,
                prefetch_size=prefetch_size,
                base_seed=int(args.seed),
                zero_current_frame=bool(args.zero_current_frame),
                sampling_seed=int(args.sampling_seed),
                profile_every_n_batches=int(args.profile_every_n_batches),
            )
            if stats is None:
                continue
            logging.info(
                f"[{mode.value}] ep={ep} N={stats.num_frames} "
                f"MSE={stats.overall_mse:.6f} MAE={stats.overall_mae:.6f} "
                f"first_step_MSE={stats.first_step_mse:.6f}"
            )
            per_episode_stats.append(dataclasses.replace(stats, episode_idx=ep))
    finally:
        if gate_snapshot is not None:
            restore_gate_values(policy._model, gate_snapshot)
            if fixed_gate_changed:
                set_fixed_gate_value(policy._model, fixed_gate_snapshot)
            reset_policy_jit(policy)
            logging.info("Restored original gate logits/fixed-gate config and re-jitted policy.")

    if not per_episode_stats:
        return {"mode": mode.value, "num_episodes": 0}, {}

    overall_mse = float(np.mean([s.overall_mse for s in per_episode_stats]))
    overall_mae = float(np.mean([s.overall_mae for s in per_episode_stats]))
    first_step_mse = float(np.mean([s.first_step_mse for s in per_episode_stats]))
    first_step_mae = float(np.mean([s.first_step_mae for s in per_episode_stats]))
    mse_per_dim = np.stack([s.mse_per_dim for s in per_episode_stats], axis=0).mean(axis=0)
    mae_per_dim = np.stack([s.mae_per_dim for s in per_episode_stats], axis=0).mean(axis=0)

    result = {
        "mode": mode.value,
        "num_episodes": len(per_episode_stats),
        "overall_mse": overall_mse,
        "overall_mae": overall_mae,
        "first_step_mse": first_step_mse,
        "first_step_mae": first_step_mae,
        "gate_stats": active_gate_stats,
        "original_gate_stats": original_gate_stats,
        "fixed_gate": active_fixed_gate,
        "original_fixed_gate": fixed_gate_snapshot,
        "mse_per_dim": mse_per_dim.tolist(),
        "mae_per_dim": mae_per_dim.tolist(),
        "per_episode": [
            {
                "episode_idx": s.episode_idx,
                "num_frames": s.num_frames,
                "overall_mse": s.overall_mse,
                "overall_mae": s.overall_mae,
                "first_step_mse": s.first_step_mse,
                "first_step_mae": s.first_step_mae,
            }
            for s in per_episode_stats
        ],
    }
    return result, _sample_record_map(per_episode_stats)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _print_summary(summary: list[dict]) -> str:
    has_prediction_delta = any("prediction_delta_vs_normal" in entry for entry in summary)
    width = 106 if has_prediction_delta else 80
    header = (
        f"{'Ablation Mode':<20} {'#Episodes':>10} {'MSE':>14} {'MAE':>14} "
        f"{'FirstStepMSE':>14}"
    )
    if has_prediction_delta:
        header += f" {'PredDeltaRMSE':>14} {'PredDeltaMAE':>14}"
    rows = [
        "",
        "=" * width,
        header,
        "-" * width,
    ]
    for entry in summary:
        if "overall_mse" not in entry:
            continue
        row = (
            f"{entry['mode']:<20} {entry['num_episodes']:>10d} "
            f"{entry['overall_mse']:>14.6f} {entry['overall_mae']:>14.6f} "
            f"{entry['first_step_mse']:>14.6f}"
        )
        if has_prediction_delta:
            delta = entry.get("prediction_delta_vs_normal") or {}
            if "pred_delta_rmse" in delta:
                row += f" {delta['pred_delta_rmse']:>14.6f} {delta['pred_delta_mae']:>14.6f}"
            else:
                row += f" {'-':>14} {'-':>14}"
        rows.append(row)
    rows.append("=" * width)
    text = "\n".join(rows)
    print(text)
    return text


def _plot_summary(summary: list[dict], output_dir: Path) -> None:
    """Bar chart comparing MSE / MAE / first-step MSE across ablation modes.

    Skipped silently when matplotlib is unavailable or only one mode was run
    (a comparison chart needs at least two bars to be useful).
    """
    rows = [s for s in summary if "overall_mse" in s]
    if len(rows) < 2:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib not available; skipping comparison chart.")
        return

    modes = [r["mode"] for r in rows]
    mse_vals = [r["overall_mse"] for r in rows]
    mae_vals = [r["overall_mae"] for r in rows]
    fs_mse_vals = [r["first_step_mse"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ("Overall MSE", "Overall MAE", "First-Step MSE")
    values = (mse_vals, mae_vals, fs_mse_vals)
    for ax, title, vals in zip(axes, titles, values):
        bars = ax.bar(modes, vals, edgecolor="black", alpha=0.8)
        # Highlight baseline ("normal") in green so eye-catching deltas pop.
        for mode, bar in zip(modes, bars):
            if mode == AblationMode.NORMAL.value:
                bar.set_color("#3a9b3a")
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)
        # Annotate baseline-relative deltas above each bar.
        baseline_idx = modes.index(AblationMode.NORMAL.value) if AblationMode.NORMAL.value in modes else None
        baseline = vals[baseline_idx] if baseline_idx is not None else None
        for i, (bar, v) in enumerate(zip(bars, vals)):
            label = f"{v:.4f}"
            if baseline is not None and i != baseline_idx:
                delta = (v - baseline) / max(abs(baseline), 1e-9) * 100
                label = f"{v:.4f}\n({delta:+.1f}%)"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Pi0MemCompress Ablation Comparison")
    plt.tight_layout()
    out_file = output_dir / "ablation_comparison.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved comparison chart to {out_file}")


def _init_logging() -> None:
    """Force root + openpi loggers to INFO and (re)install a stdout handler.

    We need ``force=True`` because by the time ``main()`` runs, several upstream
    imports (``openpi.training.config`` -> a chain of openpi submodules,
    absl, JAX, tyro, ...) have already attached handlers / set levels on the
    root logger. Without ``force=True``, ``basicConfig`` silently no-ops and
    every ``logging.info(...)`` in this script gets dropped at root's default
    WARNING level -- that's why the "Ablation video pipeline: ..." line never
    showed up in the user's stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    # Pull the openpi-specific logger up too -- some openpi modules call
    # ``logging.getLogger("openpi")`` directly and inherit only the level we set
    # on root, but a few set a per-logger level explicitly during import.
    for name in ("", "openpi", "root"):
        logging.getLogger(name).setLevel(logging.INFO)


def main(args: Args) -> None:
    _init_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = _config.get_config(args.config)

    # Optionally hoist the dataset into /dev/shm BEFORE we touch it. This
    # mutates ``args.dataset_path`` to the cached copy, so all downstream code
    # (build_video_dataset, _set_worker_dataset, ...) transparently uses the
    # tmpfs-backed location. See ``maybe_cache_dataset_in_shm`` for why this
    # exists and when it's a no-op (already on tmpfs / not enough RAM / etc).
    maybe_cache_dataset_in_shm(args)

    # =====================================================================
    # IMPORTANT ORDERING: build the dataset & fork the loader-worker pool
    # BEFORE ``create_trained_policy`` runs. ``create_trained_policy`` is the
    # first thing in this script that touches JAX/CUDA and (transitively)
    # starts several background threads. If we fork loader workers AFTER that
    # point, the children inherit those threads & the CUDA context, which
    # gives us the warning seen in the wild::
    #
    #   RuntimeWarning: os.fork() was called. os.fork() is incompatible
    #   with multithreaded code, and JAX is multithreaded, so this will
    #   likely lead to a deadlock.
    #
    # In practice the workers don't fully deadlock but they become extremely
    # slow / unreliable -- which is why GPU-Util stays at 0% even though the
    # async dispatch pipeline is set up correctly. Fixing the fork ordering
    # is the difference between ~0.5 frames/s and ~30+ frames/s.
    # =====================================================================
    dataset, video_cfg = build_video_dataset(args, train_config)
    episodes = list_episode_indices(dataset)
    logging.info(
        f"Dataset has {len(episodes)} episodes / {sum(len(v) for v in episodes.values())} frames."
    )

    load_executor: concurrent.futures.ProcessPoolExecutor | None = None
    # Mutable container so the signal/atexit handlers (defined inside this
    # block) can read the worker PID set after it gets populated below.
    _worker_pids_ref: list[set[int]] = [set()]
    if args.num_load_workers > 0:
        # Set the module-level dataset pointer BEFORE forking; children inherit
        # it via copy-on-write so we don't pay the ~5 MB-per-call pickle cost
        # of sending VideoFrameDataset through the executor's queue.
        _set_worker_dataset(dataset)
        mp_ctx = multiprocessing.get_context("fork")
        load_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_load_workers,
            mp_context=mp_ctx,
        )
        # Register a best-effort cleanup so that on SIGINT/SIGTERM (Ctrl-C,
        # `kill`, OOM-killer-when-it-uses-SIGTERM) the loader workers don't
        # outlive the parent. Without this, orphaned fork workers can keep
        # the inherited /dev/nvidia* file descriptors alive in the kernel's
        # process table and prevent the CUDA driver from releasing the JAX
        # pre-allocated GPU memory until they too die. Note: nothing helps
        # against `kill -9` (SIGKILL bypasses Python entirely) -- in that
        # case use `pkill -9 -f ablate_pi0_mem_compress` and wait ~10s.
        import atexit as _atexit
        import signal as _signal

        def _shutdown_loader_workers(*_unused) -> None:  # signal handler signature
            ex = load_executor
            if ex is not None:
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            # Belt-and-braces: directly signal known worker PIDs so the OS
            # reaps them even if the executor's pipe-based wakeup gets stuck.
            for pid in _worker_pids_ref[0]:
                try:
                    os.kill(pid, _signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass

        _atexit.register(_shutdown_loader_workers)
        # Install handlers for the two signals that we can intercept. Re-raise
        # the default behaviour after cleanup so the shell still gets the right
        # exit code (130 for SIGINT, 143 for SIGTERM).
        def _signal_then_default(signum, _frame):
            _shutdown_loader_workers()
            _signal.signal(signum, _signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for _sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                _signal.signal(_sig, _signal_then_default)
            except (ValueError, OSError):
                pass  # not main thread or signal not supported on this platform
        # Force EAGER fork of all workers right now, while we're still
        # single-threaded. ProcessPoolExecutor is lazy by default -- without
        # this barrier the workers wouldn't fork until the first prefetch call,
        # which is well after ``create_trained_policy`` has init'd JAX/CUDA.
        warmup_t0 = time.monotonic()
        warmup_futures = [
            load_executor.submit(_worker_warmup_sleep, 0.5)
            for _ in range(args.num_load_workers)
        ]
        worker_pids = {f.result(timeout=60) for f in warmup_futures}
        # Publish the PID set to the signal/atexit handlers above so they
        # can kill leftover workers on Ctrl-C / SIGTERM.
        _worker_pids_ref[0] = set(worker_pids)
        warmup_elapsed = time.monotonic() - warmup_t0
        if len(worker_pids) != args.num_load_workers:
            logging.warning(
                "Expected %d distinct loader workers but only got %d (pids=%s). "
                "ProcessPoolExecutor may have reused workers; some forks may still "
                "happen post-JAX-init.",
                args.num_load_workers,
                len(worker_pids),
                sorted(worker_pids),
            )
        else:
            logging.info(
                "Spawned %d fork-based loader workers in %.2fs BEFORE JAX init "
                "(pids=%s). Decode side should now run truly in parallel; expect "
                "per-frame wall to drop from ~1.2s to ~0.15-0.25s once XLA finishes "
                "compiling on the first batch.",
                args.num_load_workers,
                warmup_elapsed,
                sorted(worker_pids),
            )

    logging.info(f"Loading policy: config={args.config}, dir={args.checkpoint_dir}")
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
    )

    # Override the flow-matching sampling step count. Each step is a full LLM
    # forward (suffix tokens over kv-cached prefix); fewer steps -> lower
    # per-batch wall time at a small cost to absolute action quality. Relative
    # ablation rankings are preserved as long as the same num_steps is used
    # across all modes (which we do).
    if args.num_sampling_steps != 10:
        policy._sample_kwargs = {
            **(policy._sample_kwargs or {}),
            "num_steps": int(args.num_sampling_steps),
        }
        logging.info(
            "Overriding flow-matching num_steps -> %d (default is 10). "
            "Per-batch GPU wall should scale ~linearly with this.",
            args.num_sampling_steps,
        )

    # LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem's model_transforms forgets to
    # add ChunkActions(target_dim=20) on the output side, so the policy's
    # Unnormalize asserts on the trailing padding. Inject it here, in-script.
    patch_output_transform_for_chunked_actions(
        policy,
        valid_action_dims=args.valid_action_dims,
        model_action_dim=int(train_config.model.action_dim),
    )

    # Inference acceleration: figure out the effective batch size and (optional)
    # multi-GPU data sharding. Logged loudly so the user can sanity-check it.
    n_devices = jax.device_count()
    batch_size = _resolve_batch_size(args.batch_size, multi_gpu=args.multi_gpu)
    mesh, data_sharding = _build_data_sharding(batch_size=batch_size, multi_gpu=args.multi_gpu)
    prefetch_size = args.prefetch_size
    if prefetch_size <= 0:
        prefetch_size = max(batch_size, max(1, args.num_load_workers)) * 2
    logging.info(
        "Inference plan: batch_size=%d, jax.device_count()=%d, multi_gpu=%s, "
        "data_sharding=%s, num_load_workers=%d, prefetch_size=%d",
        batch_size,
        n_devices,
        args.multi_gpu,
        "ON (data-parallel across all visible GPUs)" if data_sharding is not None
        else "OFF (single-device batching)",
        args.num_load_workers,
        prefetch_size,
    )
    if data_sharding is not None:
        logging.info(
            "Per-device batch = %d (= batch_size / n_devices). First chunk will trigger a "
            "one-time XLA recompile for this sharding; subsequent chunks hit the JIT cache.",
            batch_size // n_devices,
        )

    if args.ablation_mode == AblationMode.ZERO_CURRENT_SUITE:
        args.zero_current_frame = True
        modes_to_run = [
            AblationMode.NORMAL,
            AblationMode.MEMORY_OFF,
            AblationMode.WRONG_HISTORY,
            AblationMode.REPEAT_CURRENT,
            AblationMode.SHUFFLE_HISTORY,
        ]
    elif args.ablation_mode == AblationMode.ALL:
        modes_to_run = [
            AblationMode.NORMAL,
            AblationMode.REPEAT_CURRENT,
            AblationMode.WRONG_HISTORY,
            AblationMode.SHUFFLE_HISTORY,
            AblationMode.ZERO_CURRENT,
            AblationMode.MEMORY_OFF,
            AblationMode.FORCE_MEMORY_GATE,
        ]
    else:
        modes_to_run = [args.ablation_mode]

    # Heads-up estimate: warn loudly if the user is about to wait > 1 h.
    # Empirically (16 frames * 2 cams * UMI ~1280-token decode):
    #   - GPU steady state: ~0.1 s/frame with prefetch+8GPU+bs=8, or ~0.05 s/frame with bs=32
    #   - CPU steady state: ~0.15 s/frame with 8 prefetch workers, ~1.2 s/frame without
    # In aggregate ~0.2 s/frame is a defensible "with everything on" estimate.
    n_modes = len(modes_to_run)
    # We can't cheaply know episode lengths without iterating, but for UMI the
    # post-stride per-episode frame count is ~ min(max_frames, ~500/stride).
    if args.max_frames_per_episode > 0:
        frames_per_ep = args.max_frames_per_episode
    else:
        frames_per_ep = 500 // max(1, args.frame_stride_eval)
    total_frames = n_modes * args.num_episodes * frames_per_ep
    sec_per_frame = 0.2 if args.num_load_workers > 0 else 1.3
    est_min = total_frames * sec_per_frame / 60.0
    logging.info(
        "Estimated wall time: ~%.0f min for %d modes x %d episodes x %d frames/ep "
        "(window=%s, roughly %.2f s/frame at this configuration). If this is too long, lower "
        "--num-episodes, --max-frames-per-episode, or raise --frame-stride-eval.",
        est_min,
        n_modes,
        args.num_episodes,
        frames_per_ep,
        args.frame_window.value,
        sec_per_frame,
    )

    # Shared flag across modes: True until the first batched_infer call returns
    # (so we only log the "XLA is compiling, expect 5-15 min" hint once per run,
    # not once per ablation mode).
    is_first_batch_ref: list[bool] = [True]

    # NOTE: ``load_executor`` was already created above, BEFORE policy load,
    # so worker processes were forked while the parent was still single-
    # threaded. Do NOT re-create it here.

    summary: list[dict] = []
    normal_records: dict[int, dict[str, np.ndarray | float]] | None = None
    try:
        for mode in modes_to_run:
            result, records_by_index = run_mode(
                mode=mode,
                args=args,
                policy=policy,
                dataset=dataset,
                episodes=episodes,
                image_keys=tuple(video_cfg.image_keys),
                num_frames=video_cfg.num_frames,
                batch_size=batch_size,
                data_sharding=data_sharding,
                is_first_batch_ref=is_first_batch_ref,
                load_executor=load_executor,
                prefetch_size=prefetch_size,
            )
            if args.compute_prediction_delta:
                if mode == AblationMode.NORMAL:
                    normal_records = records_by_index
                    result["prediction_delta_vs_normal"] = _prediction_delta_metrics(
                        normal_records,
                        records_by_index,
                        reference_mode=AblationMode.NORMAL.value,
                        top_k=args.prediction_delta_top_k,
                    )
                elif normal_records:
                    result["prediction_delta_vs_normal"] = _prediction_delta_metrics(
                        normal_records,
                        records_by_index,
                        reference_mode=AblationMode.NORMAL.value,
                        top_k=args.prediction_delta_top_k,
                    )
                    delta = result["prediction_delta_vs_normal"]
                    logging.info(
                        "[%s] prediction delta vs normal: common=%d "
                        "rmse=%.6f mae=%.6f max_abs=%.6f first_step_mae=%.6f "
                        "gt_mse_delta=%.6g improved=%.1f%% corr=%.3f",
                        mode.value,
                        delta["num_common_frames"],
                        delta["pred_delta_rmse"],
                        delta["pred_delta_mae"],
                        delta["pred_delta_max_abs"],
                        delta["pred_delta_first_step_mae"],
                        delta["gt_mse_delta_mean"],
                        100.0 * delta["fraction_improved"],
                        delta["gt_mse_delta_on_changed_corr"],
                    )
            summary.append(result)
            # Add a per-shard suffix so parallel shards don't clobber each other.
            # ``scripts/mem/merge_ablation_shards.py`` reads {mode}_shard_*.json
            # back and re-aggregates into the canonical {mode}.json afterwards.
            shard_suffix = (
                f"_shard_{args.episode_shard_index}"
                if args.episode_shard_total > 1
                else ""
            )
            out_file = output_dir / f"{mode.value}{shard_suffix}.json"
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)
            logging.info(f"Saved {mode.value} results to {out_file}")
    finally:
        if load_executor is not None:
            # wait=True so the parent does not exit while worker processes
            # still hold inherited /dev/nvidia* fds; that would defer the
            # CUDA driver's release of JAX's pre-allocated GPU memory until
            # the OS reaps the orphans.
            load_executor.shutdown(wait=True, cancel_futures=True)

    summary_text = _print_summary(summary)
    shard_suffix = (
        f"_shard_{args.episode_shard_index}" if args.episode_shard_total > 1 else ""
    )
    summary_file = output_dir / f"summary{shard_suffix}.txt"
    summary_file.write_text(summary_text + "\n")
    with open(output_dir / f"summary{shard_suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Wrote summary to {summary_file}")
    # Skip the cross-mode comparison chart in shard mode -- the merged data is
    # only complete after all shards finish, so plotting per-shard is misleading.
    if args.episode_shard_total <= 1:
        _plot_summary(summary, output_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))

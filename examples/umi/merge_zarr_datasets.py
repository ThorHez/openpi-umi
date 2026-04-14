#!/usr/bin/env python3
"""
Merge multiple UMI / HITL zarr datasets (same schema) into one zarr.

Each input must have ``data/<feature>`` arrays with identical dtypes and trailing
dimensions, and ``meta/episode_ends`` as cumulative end indices per episode (same
semantics as the rest of this repo). Arrays are concatenated along time (axis 0);
episode boundaries are shifted and concatenated.

Supports ``.zarr`` directories and ``.zarr.zip`` archives (same as
``convert_zarr_to_unified_frame.py``).

Example:
    python examples/umi/merge_zarr_datasets.py \\
        --input-dir /root/openpi-umi/data/fold_clothes_value_training \\
        --output /root/openpi-umi/data/fold_clothes_value_training_merged.zarr.zip

    # Exclude unified copies and sort by filename:
    python examples/umi/merge_zarr_datasets.py -i ./data/fold_clothes_value_training \\
        -o ./merged.zarr.zip --exclude '*_unified*'

Large RGB/depth keys are slow (minutes per key); logs and tqdm show progress per key and
per source file. Set MERGE_ZARR_NO_TQDM=1 to disable tqdm bars.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
import numcodecs
import zarr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    **({"force": True} if sys.version_info >= (3, 8) else {}),
)
logger = logging.getLogger(__name__)


def open_zarr(path: str, mode: str = "r"):
    if path.endswith(".zarr.zip"):
        ZipStore = None
        if hasattr(zarr, "storage") and hasattr(zarr.storage, "ZipStore"):
            ZipStore = zarr.storage.ZipStore
        elif hasattr(zarr, "ZipStore"):
            ZipStore = zarr.ZipStore
        else:
            raise RuntimeError("Could not find ZipStore in zarr.")
        store = ZipStore(path, mode=mode)
        return zarr.open_group(store, mode=mode), store
    return zarr.open_group(path, mode=mode), None


def _pack_zarr_directory_to_zip(zarr_dir: Path, zip_path: Path) -> None:
    """
    Pack a filesystem zarr root directory into a single .zarr.zip file.

    Each file is added once (no duplicate zip members). Chunk bytes are usually
    already Blosc-compressed, so we use ZIP_STORED at the archive level.
    """
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    zarr_dir = zarr_dir.resolve()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for fpath in sorted(zarr_dir.rglob("*")):
            if fpath.is_file():
                arcname = fpath.relative_to(zarr_dir).as_posix()
                zf.write(fpath, arcname)


def _create_array_compat(group, name: str, *, data, chunks, compressor, overwrite: bool = True):
    """Create an array on a zarr Group, compatible with zarr v2/v3."""
    if hasattr(group, "create_array"):
        if compressor is None:
            compressors = None
        else:
            if hasattr(zarr, "codecs") and hasattr(zarr.codecs, "BloscCodec"):
                compressors = zarr.codecs.BloscCodec(
                    cname="lz4",
                    clevel=5,
                    shuffle=zarr.codecs.BloscShuffle.noshuffle,
                )
            else:
                compressors = compressor
        try:
            return group.create_array(
                name,
                data=data,
                chunks=chunks,
                compressors=compressors,
                overwrite=overwrite,
            )
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument 'data'" not in msg:
                raise
            arr = group.create_array(
                name,
                shape=data.shape,
                dtype=data.dtype,
                chunks=chunks,
                compressors=compressors,
                overwrite=overwrite,
            )
            arr[...] = data
            return arr
    return group.array(
        name,
        data=data,
        chunks=chunks,
        compressor=compressor,
        overwrite=overwrite,
    )


def _create_empty_array_compat(
    group,
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    chunks: tuple[int, ...],
    compressor,
    overwrite: bool = True,
):
    """Create an uninitialized array of given shape (zarr v2/v3)."""
    dtype = np.dtype(dtype)
    if hasattr(group, "create_array"):
        if compressor is None:
            compressors = None
        else:
            if hasattr(zarr, "codecs") and hasattr(zarr.codecs, "BloscCodec"):
                compressors = zarr.codecs.BloscCodec(
                    cname="lz4",
                    clevel=5,
                    shuffle=zarr.codecs.BloscShuffle.noshuffle,
                )
            else:
                compressors = compressor
        try:
            return group.create_array(
                name,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                compressors=compressors,
                overwrite=overwrite,
            )
        except TypeError:
            arr = group.create_array(
                name,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                compressors=compressors,
                overwrite=overwrite,
            )
            return arr
    if hasattr(group, "zeros"):
        return group.zeros(
            name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compressor=compressor,
            overwrite=overwrite,
        )
    return group.array(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=compressor,
        overwrite=overwrite,
    )


def _list_zarr_inputs(input_dir: Path, pattern: str, excludes: Sequence[str]) -> list[Path]:
    globs = sorted(input_dir.glob(pattern))
    out: list[Path] = []
    for p in globs:
        if not p.is_file() and not (p.is_dir() and p.name.endswith(".zarr")):
            continue
        rel = p.name
        skip = False
        for ex in excludes:
            if fnmatch.fnmatch(rel, ex) or fnmatch.fnmatch(str(p), ex):
                skip = True
                break
        if not skip:
            out.append(p)
    return out


def _validate_and_describe(
    paths: Sequence[Path],
) -> tuple[list[str], dict[str, tuple[tuple[int, ...], np.dtype]], list[tuple[str, int, int]]]:
    """
    Returns:
        keys: sorted data keys
        schema: key -> (shape_without_t0, dtype)
        file_stats: (path_str, n_frames, n_episodes) per file
    """
    if not paths:
        raise ValueError("No input zarr files matched.")

    ref_keys: set[str] | None = None
    ref_schema: dict[str, tuple[tuple[int, ...], np.dtype]] | None = None
    stats: list[tuple[str, int, int]] = []

    for p in paths:
        root, store = open_zarr(str(p), mode="r")
        try:
            if "data" not in root or "meta" not in root:
                raise ValueError(f"Missing data/ or meta/ group: {p}")
            data = root["data"]
            meta = root["meta"]
            if "episode_ends" not in meta:
                raise ValueError(f"Missing meta/episode_ends: {p}")

            keys = sorted(data.keys())
            if ref_keys is None:
                ref_keys = set(keys)
            elif set(keys) != ref_keys:
                raise ValueError(
                    f"Key mismatch in {p}: expected {sorted(ref_keys)}, got {keys}"
                )

            ee = np.asarray(meta["episode_ends"][:])
            n_ep = int(ee.size)
            n_frames = int(ee[-1]) if n_ep > 0 else 0

            sch: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
            for k in keys:
                arr = data[k]
                shape = tuple(arr.shape)
                dt = np.dtype(arr.dtype)
                if shape[0] != n_frames:
                    raise ValueError(
                        f"{p} data/{k} length {shape[0]} != episode_ends last {n_frames}"
                    )
                rest = shape[1:]
                sch[k] = (rest, dt)

            if ref_schema is None:
                ref_schema = sch
            else:
                for k in keys:
                    if sch[k] != ref_schema[k]:
                        raise ValueError(
                            f"Schema mismatch for {k} in {p}: {sch[k]} vs {ref_schema[k]}"
                        )

            stats.append((str(p), n_frames, n_ep))
        finally:
            if store:
                store.close()

    assert ref_keys is not None and ref_schema is not None
    return sorted(ref_keys), ref_schema, stats


def _copy_array_chunked(
    dst_arr,
    src_arr,
    dst_start: int,
    row_chunk: int,
) -> int:
    """Copy src_arr into dst_arr[dst_start:dst_start+T] in row chunks. Returns T."""
    n = int(src_arr.shape[0])
    i = 0
    while i < n:
        j = min(i + row_chunk, n)
        dst_arr[dst_start + i : dst_start + j] = np.asarray(src_arr[i:j])
        i = j
    return n


def merge(
    paths: Sequence[Path],
    output_path: str,
    *,
    row_chunk: int,
    compressor,
) -> None:
    keys, schema, stats = _validate_and_describe(paths)
    total_frames = sum(s[1] for s in stats)
    total_episodes = sum(s[2] for s in stats)
    logger.info(
        "Merging %d files -> %d frames, %d episodes",
        len(paths),
        total_frames,
        total_episodes,
    )

    # Build merged episode_ends
    parts: list[np.ndarray] = []
    offset = 0
    for p in paths:
        root, store = open_zarr(str(p), mode="r")
        try:
            ee = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
            parts.append(offset + ee)
            offset += int(ee[-1]) if ee.size > 0 else 0
        finally:
            if store:
                store.close()
    merged_ends = np.concatenate(parts, axis=0)

    if merged_ends.size and int(merged_ends[-1]) != total_frames:
        raise RuntimeError(
            f"episode_ends last {merged_ends[-1]} != total_frames {total_frames}"
        )

    CHUNK_DEFAULT = 2000
    out_is_zip = output_path.endswith(".zarr.zip")
    if out_is_zip and os.path.exists(output_path):
        os.remove(output_path)

    root0, st0 = open_zarr(str(paths[0]), mode="r")
    try:
        base_attrs = {k: root0.attrs[k] for k in root0.attrs}
    finally:
        if st0:
            st0.close()

    def _write_all(dst_root: zarr.Group) -> None:
        dst_data = dst_root.require_group("data")
        dst_meta = dst_root.require_group("meta")

        logger.info(
            "Writing %d data keys (large image keys can take tens of minutes each; "
            "progress bars show per-input-file copy).",
            len(keys),
        )
        for ki, key in enumerate(keys, 1):
            rest, dt = schema[key]
            out_shape = (total_frames,) + rest
            logger.info(
                "[%d/%d] merging data/%s shape=%s ...",
                ki,
                len(keys),
                key,
                out_shape,
            )
            cks = (min(CHUNK_DEFAULT, max(1, total_frames)),) + rest
            arr = _create_empty_array_compat(
                dst_data,
                key,
                shape=out_shape,
                dtype=dt,
                chunks=cks,
                compressor=compressor,
                overwrite=True,
            )
            pos = 0
            for p in tqdm(
                paths,
                desc=f"data/{key}",
                unit="src",
                leave=False,
                disable=os.environ.get("MERGE_ZARR_NO_TQDM", "").lower()
                in ("1", "true", "yes"),
            ):
                root, store = open_zarr(str(p), mode="r")
                try:
                    src = root["data"][key]
                    n = _copy_array_chunked(arr, src, pos, row_chunk)
                    pos += n
                finally:
                    if store:
                        store.close()
            assert pos == total_frames, (key, pos, total_frames)
            logger.info("  done data/%s", key)

        _create_array_compat(
            dst_meta,
            "episode_ends",
            data=merged_ends,
            chunks=merged_ends.shape,
            compressor=None,
            overwrite=True,
        )
        logger.info("  wrote meta/episode_ends shape=%s", merged_ends.shape)

        for ak, av in base_attrs.items():
            dst_root.attrs[ak] = av
        dst_root.attrs["merged_from"] = [str(p) for p in paths]
        dst_root.attrs["merge_total_frames"] = int(total_frames)
        dst_root.attrs["merge_total_episodes"] = int(total_episodes)

    if out_is_zip:
        # Avoid zarr ZipStore incremental writes: they can emit zipfile "Duplicate name"
        # warnings for the same chunk path. Write a directory store, then pack once.
        tmp_root = Path(tempfile.mkdtemp(prefix="merge_zarr_"))
        zarr_dir = tmp_root / "merged.zarr"
        try:
            zarr_dir.mkdir(parents=True, exist_ok=True)
            dst_root = zarr.open_group(str(zarr_dir), mode="w")
            try:
                _write_all(dst_root)
            finally:
                if hasattr(dst_root, "close"):
                    dst_root.close()
            _pack_zarr_directory_to_zip(zarr_dir, Path(output_path))
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
    else:
        os.makedirs(output_path, exist_ok=True)
        dst_root = zarr.open_group(output_path, mode="w")
        try:
            _write_all(dst_root)
        finally:
            if hasattr(dst_root, "close"):
                dst_root.close()

    size_mb = os.path.getsize(output_path) / 1024**2 if out_is_zip else 0
    if out_is_zip:
        logger.info("Done. Output size: %.1f MB", size_mb)
    else:
        logger.info("Done. Output directory: %s", output_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge zarr datasets with identical schema.")
    p.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=None,
        help="Directory containing .zarr.zip and/or .zarr stores (used with --glob unless --inputs is set)",
    )
    p.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Explicit zarr paths to merge in order (if set, --input-dir/--glob are ignored)",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path: .zarr.zip or .zarr directory",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.zarr.zip",
        help="Glob pattern under input-dir (default: *.zarr.zip)",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern to exclude (repeatable), e.g. '*_unified*'",
    )
    p.add_argument(
        "--row-chunk",
        type=int,
        default=512,
        help="Rows to read/write per step for large arrays (default 512)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.inputs:
        paths = [Path(p).resolve() for p in args.inputs]
        for p in paths:
            if not p.exists():
                raise SystemExit(f"Missing input: {p}")
    else:
        if not args.input_dir:
            raise SystemExit("Either --inputs or --input-dir is required.")
        input_dir = Path(args.input_dir).resolve()
        if not input_dir.is_dir():
            raise SystemExit(f"Not a directory: {input_dir}")
        paths = _list_zarr_inputs(input_dir, args.glob, args.exclude)
        if not paths:
            raise SystemExit(
                f"No inputs under {input_dir} with glob {args.glob!r} (after excludes)."
            )

    logger.info("Inputs (%d):", len(paths))
    for p in paths:
        logger.info("  %s", p.name)

    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE)
    merge(paths, args.output, row_chunk=max(1, args.row_chunk), compressor=compressor)


if __name__ == "__main__":
    main()

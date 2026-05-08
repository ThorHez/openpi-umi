"""Fix LeRobot dataset parquet files where image columns were stored as raw
``struct<bytes, path>`` without HuggingFace ``datasets.Image()`` typing.

Symptom this fixes
------------------
When loading the dataset, LeRobot's ``hf_transform_to_torch`` raises::

    RuntimeError: Could not infer dtype of dict

because HF ``load_dataset("parquet", ...)`` returns the rgb columns as plain
``{"bytes": ..., "path": ...}`` dicts (no Image feature attached), and then
``torch.tensor(some_dict)`` blows up.

What this script does
---------------------
For every ``data/chunk-*/episode_*.parquet`` under each given dataset root:

1. Read the parquet (no row reformat).
2. Build the proper HF ``Features`` from ``meta/info.json`` via LeRobot's
   ``get_hf_features_from_features`` (image columns become ``datasets.Image()``).
3. Embed those features into the parquet's schema-level ``huggingface``
   metadata so ``Features.from_arrow_schema`` recovers them on load.
4. Atomically rewrite the file in place.

The actual column data is untouched -- only the schema metadata changes. The
underlying ``struct<bytes, path>`` storage is exactly what ``datasets.Image``
expects internally, so no re-encoding is needed.

After running, clear any stale parquet cache::

    rm -rf ~/.cache/huggingface/datasets/parquet/

Usage
-----
    python scripts/fix_lerobot_image_features.py <dataset_root> [<dataset_root> ...] [--dry-run] [-j N]

Example::

    python scripts/fix_lerobot_image_features.py \\
        /root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260323_20hz
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import datasets
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_PATH = REPO_ROOT / "third_party" / "lerobot"
if str(LEROBOT_PATH) not in sys.path:
    sys.path.insert(0, str(LEROBOT_PATH))

from lerobot.common.datasets.utils import get_hf_features_from_features  # noqa: E402

HF_META_KEY = b"huggingface"


def _load_meta_features(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"info.json not found at {info_path}")
    with info_path.open() as f:
        info = json.load(f)
    if "features" not in info:
        raise KeyError(f"'features' key missing from {info_path}")
    return info["features"]


def _build_hf_metadata(hf_features: datasets.Features) -> bytes:
    """Build the bytes payload that goes under the ``huggingface`` schema key.

    Mirrors what ``datasets.Dataset.to_parquet`` writes, so that
    ``Features.from_arrow_schema`` recovers the typed features on load.
    """
    info = datasets.DatasetInfo(features=hf_features)
    payload = {"info": dataclasses.asdict(info)}
    return json.dumps(payload, default=str).encode("utf-8")


def _needs_fix(parquet_path: Path, image_keys: set[str]) -> bool:
    """Return True iff the parquet schema metadata is missing Image typing
    for any of the declared image columns.
    """
    schema = pq.read_schema(str(parquet_path))
    md = schema.metadata or {}
    blob = md.get(HF_META_KEY)
    if blob is None:
        return True
    try:
        meta = json.loads(blob.decode("utf-8"))
        feats = meta.get("info", {}).get("features", {})
    except Exception:
        return True
    for key in image_keys:
        spec = feats.get(key)
        if not isinstance(spec, dict):
            return True
        if spec.get("_type") != "Image":
            return True
    return False


def _patch_parquet(parquet_path: Path, hf_meta_bytes: bytes) -> None:
    """Rewrite ``parquet_path`` with updated schema metadata, atomically."""
    table = pq.read_table(str(parquet_path))
    existing = dict(table.schema.metadata or {})
    existing[HF_META_KEY] = hf_meta_bytes
    new_table = table.replace_schema_metadata(existing)

    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    try:
        pq.write_table(new_table, str(tmp_path))
        tmp_path.replace(parquet_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _process_one(args: tuple[str, str, list[str]]) -> tuple[str, str]:
    """Worker entry: returns (path, status) where status is one of
    'rewrote', 'skipped', or 'error: <msg>'.
    """
    parquet_path_str, hf_meta_b64, image_keys = args
    parquet_path = Path(parquet_path_str)
    try:
        if not _needs_fix(parquet_path, set(image_keys)):
            return (parquet_path_str, "skipped")
        import base64

        hf_meta_bytes = base64.b64decode(hf_meta_b64)
        _patch_parquet(parquet_path, hf_meta_bytes)
        return (parquet_path_str, "rewrote")
    except Exception as exc:  # pragma: no cover - defensive
        return (parquet_path_str, f"error: {exc!r}")


def fix_dataset_root(root: Path, *, dry_run: bool, workers: int) -> tuple[int, int, int]:
    print(f"\n[fix] dataset root: {root}")
    meta_features = _load_meta_features(root)
    hf_features = get_hf_features_from_features(meta_features)
    image_keys = sorted(k for k, ft in meta_features.items() if ft["dtype"] == "image")

    if not image_keys:
        print("  no 'image' dtype columns in info.json; nothing to do")
        return (0, 0, 0)

    print(f"  image columns: {image_keys}")

    parquet_files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        print(f"  no parquet files under {root / 'data'}")
        return (0, 0, 0)

    hf_meta_bytes = _build_hf_metadata(hf_features)
    if dry_run:
        n_need = sum(1 for p in parquet_files if _needs_fix(p, set(image_keys)))
        print(f"  dry-run: {n_need}/{len(parquet_files)} files would be rewritten")
        return (n_need, len(parquet_files) - n_need, 0)

    import base64

    hf_meta_b64 = base64.b64encode(hf_meta_bytes).decode("ascii")
    work = [(str(p), hf_meta_b64, image_keys) for p in parquet_files]

    rewrote = skipped = errored = 0
    if workers <= 1:
        for i, item in enumerate(work, 1):
            path_str, status = _process_one(item)
            _print_progress(i, len(work), Path(path_str).name, status)
            rewrote, skipped, errored = _bump(status, rewrote, skipped, errored)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_one, item): item for item in work}
            done = 0
            for fut in as_completed(futures):
                path_str, status = fut.result()
                done += 1
                _print_progress(done, len(work), Path(path_str).name, status)
                rewrote, skipped, errored = _bump(status, rewrote, skipped, errored)

    print(
        f"  done: rewrote={rewrote}, skipped={skipped}, errored={errored} "
        f"(total {len(parquet_files)})"
    )
    return rewrote, skipped, errored


def _print_progress(i: int, n: int, name: str, status: str) -> None:
    print(f"  [{i:>4}/{n}] {name}: {status}")


def _bump(status: str, rewrote: int, skipped: int, errored: int) -> tuple[int, int, int]:
    if status == "rewrote":
        rewrote += 1
    elif status == "skipped":
        skipped += 1
    else:
        errored += 1
    return rewrote, skipped, errored


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="One or more LeRobot dataset roots (each containing meta/info.json and data/chunk-*).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which parquet files would be rewritten; do not modify anything.",
    )
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=8,
        help="Parallel workers (default: 8). Use 1 for sequential.",
    )
    p.add_argument(
        "--clear-hf-cache",
        action="store_true",
        help="After patching, also clear ~/.cache/huggingface/datasets/parquet/ "
             "to force HF to re-generate the arrow cache from the patched files.",
    )
    return p.parse_args()


def _clear_hf_parquet_cache() -> None:
    cache = Path.home() / ".cache" / "huggingface" / "datasets" / "parquet"
    if cache.is_dir():
        print(f"\n[fix] clearing HF parquet cache: {cache}")
        shutil.rmtree(cache)
    else:
        print(f"\n[fix] no HF parquet cache at {cache}; skipping")


def main() -> None:
    args = parse_args()

    total_rewrote = total_skipped = total_errored = 0
    for root in args.roots:
        r, s, e = fix_dataset_root(root.resolve(), dry_run=args.dry_run, workers=args.workers)
        total_rewrote += r
        total_skipped += s
        total_errored += e

    print(
        f"\n[fix] grand total: rewrote={total_rewrote}, "
        f"skipped={total_skipped}, errored={total_errored}"
    )

    if args.clear_hf_cache and not args.dry_run:
        _clear_hf_parquet_cache()

    if total_errored:
        sys.exit(1)


if __name__ == "__main__":
    main()

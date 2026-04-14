#!/usr/bin/env python3
"""
Clean old cache files to free up disk space.

Targets:
- ~/.cache/huggingface/datasets - HuggingFace dataset cache (parquet files)
- ~/.cache/openpi - OpenPI cache
- ~/.cache/uv - UV package manager cache

Usage:
    python scripts/clean_cache.py --dry-run          # Preview what will be deleted
    python scripts/clean_cache.py --days 7           # Delete caches older than 7 days
    python scripts/clean_cache.py --keep-latest 3    # Keep only 3 most recent caches
    python scripts/clean_cache.py --all              # Clean all caches except the latest
"""

import argparse
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple


class CacheEntry(NamedTuple):
    path: Path
    size_bytes: int
    mtime: datetime


def get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def get_cache_entries(cache_dir: Path) -> list[CacheEntry]:
    """Get all cache entries sorted by modification time (newest first)."""
    entries = []
    
    if not cache_dir.exists():
        return entries
    
    for item in cache_dir.iterdir():
        if item.is_dir():
            mtime = datetime.fromtimestamp(item.stat().st_mtime)
            size = get_dir_size(item)
            entries.append(CacheEntry(item, size, mtime))
    
    # Sort by modification time, newest first
    entries.sort(key=lambda x: x.mtime, reverse=True)
    return entries


def clean_huggingface_cache(
    days: int | None = None,
    keep_latest: int | None = None,
    dry_run: bool = True,
) -> tuple[int, int]:
    """Clean HuggingFace datasets cache.
    
    Returns:
        Tuple of (files_deleted, bytes_freed)
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "parquet"
    
    if not cache_dir.exists():
        print(f"  Cache directory not found: {cache_dir}")
        return 0, 0
    
    entries = get_cache_entries(cache_dir)
    if not entries:
        print("  No cache entries found")
        return 0, 0
    
    print(f"\n  Found {len(entries)} cache entries in {cache_dir}")
    
    # Determine which entries to delete
    to_delete = []
    cutoff_time = datetime.now() - timedelta(days=days) if days else None
    
    for i, entry in enumerate(entries):
        should_delete = False
        
        if keep_latest is not None and i >= keep_latest:
            should_delete = True
        elif cutoff_time and entry.mtime < cutoff_time:
            should_delete = True
        
        if should_delete:
            to_delete.append(entry)
    
    if not to_delete:
        print("  No cache entries to delete")
        return 0, 0
    
    total_size = sum(e.size_bytes for e in to_delete)
    print(f"\n  Will delete {len(to_delete)} entries ({format_size(total_size)}):")
    
    for entry in to_delete:
        age_days = (datetime.now() - entry.mtime).days
        print(f"    - {entry.path.name} ({format_size(entry.size_bytes)}, {age_days} days old)")
    
    if dry_run:
        print("\n  [DRY RUN] No files deleted")
        return 0, 0
    
    # Actually delete
    deleted_count = 0
    deleted_size = 0
    for entry in to_delete:
        try:
            shutil.rmtree(entry.path)
            deleted_count += 1
            deleted_size += entry.size_bytes
            print(f"    Deleted: {entry.path.name}")
        except Exception as e:
            print(f"    Error deleting {entry.path.name}: {e}")
    
    # Also clean up lock files
    lock_dir = cache_dir.parent
    for lock_file in lock_dir.glob("*.lock"):
        try:
            lock_file.unlink()
        except Exception:
            pass
    
    return deleted_count, deleted_size


def clean_openpi_cache(
    days: int | None = None,
    dry_run: bool = True,
) -> tuple[int, int]:
    """Clean OpenPI cache."""
    cache_dir = Path.home() / ".cache" / "openpi"
    
    if not cache_dir.exists():
        print(f"  Cache directory not found: {cache_dir}")
        return 0, 0
    
    size = get_dir_size(cache_dir)
    print(f"\n  OpenPI cache size: {format_size(size)}")
    
    if dry_run:
        print("  [DRY RUN] Would delete entire cache")
        return 0, 0
    
    try:
        shutil.rmtree(cache_dir)
        print(f"  Deleted OpenPI cache ({format_size(size)})")
        return 1, size
    except Exception as e:
        print(f"  Error: {e}")
        return 0, 0


def clean_uv_cache(
    days: int | None = None,
    dry_run: bool = True,
) -> tuple[int, int]:
    """Clean UV package manager cache."""
    cache_dir = Path.home() / ".cache" / "uv"
    
    if not cache_dir.exists():
        print(f"  Cache directory not found: {cache_dir}")
        return 0, 0
    
    size = get_dir_size(cache_dir)
    print(f"\n  UV cache size: {format_size(size)}")
    
    if dry_run:
        print("  [DRY RUN] Would delete entire cache")
        return 0, 0
    
    try:
        shutil.rmtree(cache_dir)
        print(f"  Deleted UV cache ({format_size(size)})")
        return 1, size
    except Exception as e:
        print(f"  Error: {e}")
        return 0, 0


def clean_pip_cache(dry_run: bool = True) -> tuple[int, int]:
    """Clean pip cache."""
    cache_dir = Path.home() / ".cache" / "pip"
    
    if not cache_dir.exists():
        print(f"  Cache directory not found: {cache_dir}")
        return 0, 0
    
    size = get_dir_size(cache_dir)
    print(f"\n  Pip cache size: {format_size(size)}")
    
    if dry_run:
        print("  [DRY RUN] Would delete entire cache")
        return 0, 0
    
    try:
        shutil.rmtree(cache_dir)
        print(f"  Deleted pip cache ({format_size(size)})")
        return 1, size
    except Exception as e:
        print(f"  Error: {e}")
        return 0, 0


def show_cache_summary():
    """Show summary of all cache directories."""
    cache_dirs = [
        ("HuggingFace Datasets", Path.home() / ".cache" / "huggingface" / "datasets"),
        ("OpenPI", Path.home() / ".cache" / "openpi"),
        ("UV", Path.home() / ".cache" / "uv"),
        ("Pip", Path.home() / ".cache" / "pip"),
        ("JAX", Path.home() / ".cache" / "jax"),
    ]
    
    print("\n=== Cache Summary ===\n")
    total_size = 0
    
    for name, path in cache_dirs:
        if path.exists():
            size = get_dir_size(path)
            total_size += size
            print(f"  {name:25} {format_size(size):>12}  ({path})")
        else:
            print(f"  {name:25} {'N/A':>12}  ({path})")
    
    print(f"\n  {'Total':25} {format_size(total_size):>12}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean old cache files to free up disk space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/clean_cache.py --summary           # Show cache sizes
  python scripts/clean_cache.py --dry-run           # Preview what would be deleted  
  python scripts/clean_cache.py --days 7            # Delete caches older than 7 days
  python scripts/clean_cache.py --keep-latest 3     # Keep only 3 most recent HF caches
  python scripts/clean_cache.py --all               # Delete all caches except latest
  python scripts/clean_cache.py --hf --days 14      # Only clean HuggingFace cache
  python scripts/clean_cache.py --uv                # Only clean UV cache
        """
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        help="Delete caches older than N days"
    )
    parser.add_argument(
        "--keep-latest", "-k",
        type=int,
        help="Keep only the N most recent HuggingFace dataset caches"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Delete all caches except the most recent one"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Only show cache summary, don't delete anything"
    )
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Only clean HuggingFace datasets cache"
    )
    parser.add_argument(
        "--openpi",
        action="store_true",
        help="Only clean OpenPI cache"
    )
    parser.add_argument(
        "--uv",
        action="store_true",
        help="Only clean UV cache"
    )
    parser.add_argument(
        "--pip",
        action="store_true",
        help="Only clean pip cache"
    )
    
    args = parser.parse_args()
    
    # Show summary
    show_cache_summary()
    
    if args.summary:
        return
    
    # Determine what to clean
    clean_all = not (args.hf or args.openpi or args.uv or args.pip)
    
    # Set defaults
    if args.all:
        args.keep_latest = 1
    elif args.days is None and args.keep_latest is None:
        # Default: dry run with keep_latest=1
        args.dry_run = True
        args.keep_latest = 1
    
    print("\n=== Cleaning Caches ===")
    
    total_deleted = 0
    total_freed = 0
    
    # Clean HuggingFace cache
    if clean_all or args.hf:
        print("\n[HuggingFace Datasets Cache]")
        count, size = clean_huggingface_cache(
            days=args.days,
            keep_latest=args.keep_latest,
            dry_run=args.dry_run,
        )
        total_deleted += count
        total_freed += size
    
    # Clean OpenPI cache
    if clean_all or args.openpi:
        print("\n[OpenPI Cache]")
        count, size = clean_openpi_cache(
            days=args.days,
            dry_run=args.dry_run,
        )
        total_deleted += count
        total_freed += size
    
    # Clean UV cache
    if clean_all or args.uv:
        print("\n[UV Cache]")
        count, size = clean_uv_cache(
            days=args.days,
            dry_run=args.dry_run,
        )
        total_deleted += count
        total_freed += size
    
    # Clean pip cache
    if clean_all or args.pip:
        print("\n[Pip Cache]")
        count, size = clean_pip_cache(dry_run=args.dry_run)
        total_deleted += count
        total_freed += size
    
    # Summary
    print("\n" + "=" * 40)
    if args.dry_run:
        print(f"[DRY RUN] Would free: {format_size(total_freed)}")
        print("\nRun without --dry-run to actually delete files.")
    else:
        print(f"Total freed: {format_size(total_freed)}")


if __name__ == "__main__":
    main()


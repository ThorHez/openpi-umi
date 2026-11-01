#!/usr/bin/env python3
"""Reject tracked files over 5 MiB without reading file contents or fetching blobs.

The default checks the current working tree. ``--staged`` checks the index,
including the original blob size when a staged file is subsequently changed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import subprocess
import sys


MAX_BYTES = 5 * 1024 * 1024


def git(*args: str, cwd: Path | None = None, data: bytes | None = None) -> bytes:
    env = os.environ.copy()
    env["GIT_NO_LAZY_FETCH"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", *args], cwd=cwd, input=data, capture_output=True, env=env, check=False
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def working_tree_sizes(root: Path) -> list[tuple[str, int]]:
    sizes = []
    for raw_path in git("ls-files", "--cached", "-z", cwd=root).split(b"\0"):
        if not raw_path:
            continue
        path = os.fsdecode(raw_path)
        try:
            info = (root / path).lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            sizes.append((path, info.st_size))
    return sizes


def staged_sizes(root: Path) -> list[tuple[str, int]]:
    # Disable rename detection: a rename becomes a deletion plus an addition,
    # allowing this query to inspect names without reading existing blob data.
    changed = set(
        git(
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            "-z",
            "--",
            cwd=root,
        ).split(b"\0")
    )
    entries = []
    for entry in git("ls-files", "--stage", "-z", cwd=root).split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, oid, stage = metadata.split()
        # Gitlinks are commit objects, not files stored in this repository.
        if path in changed and stage == b"0" and mode != b"160000":
            entries.append((path, oid))
    if not entries:
        return []

    object_ids = sorted({oid for _, oid in entries})
    output = git(
        "cat-file",
        "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        cwd=root,
        data=b"\n".join(object_ids) + b"\n",
    )
    sizes = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 3 or parts[1] != b"blob":
            raise RuntimeError(
                "Cannot inspect a staged blob locally; no objects were fetched. "
                f"git cat-file returned: {line.decode(errors='replace')}"
            )
        sizes[parts[0]] = int(parts[2])
    return [(os.fsdecode(path), sizes[oid]) for path, oid in entries]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Check staged index blobs instead of working files.")
    args = parser.parse_args()
    try:
        root = Path(os.fsdecode(git("rev-parse", "--show-toplevel").rstrip(b"\n")))
        sizes = staged_sizes(root) if args.staged else working_tree_sizes(root)
        oversized = sorted((path, size) for path, size in sizes if size > MAX_BYTES)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Repository size check failed: {exc}", file=sys.stderr)
        return 2

    scope = "staged index" if args.staged else "tracked working tree"
    if oversized:
        print(f"Repository size check rejected {len(oversized)} file(s) in the {scope}.", file=sys.stderr)
        print(f"Limit: 5 MiB ({MAX_BYTES:,} bytes) per file.", file=sys.stderr)
        for path, size in oversized:
            print(f"  {path!r}: {size / (1024 * 1024):.2f} MiB ({size:,} bytes)", file=sys.stderr)
        print(
            "Store generated outputs and large assets outside Git; remove them from the index before committing.",
            file=sys.stderr,
        )
        return 1
    print(f"Repository size check passed: {len(sizes)} file(s) in the {scope}, each at most 5 MiB.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

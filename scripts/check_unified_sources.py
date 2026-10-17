#!/usr/bin/env python3
"""Validate the unified source manifest without importing the training runtime.

Run with Python >= 3.10. The default check is portable: it reads the committed
manifest and selected sources, without requiring Git history, local snapshots,
JAX, Torch, datasets, checkpoints, or a GPU. New recipe names are allowed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_ALGORITHM = "sha256(ast.dump(TrainConfig call, include_attributes=False).encode('utf-8'))"
CONFLICT_MARKER = re.compile(r"^(?:<{7}(?: .*)?|={7}|>{7}(?: .*)?)$", re.MULTILINE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe_fingerprints(source: str, filename: str = "<config>") -> dict[str, str]:
    """Hash complete recipe expressions, including dataset, weights and steps."""
    recipes = {}
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "TrainConfig":
            continue
        name_node = next((item.value for item in node.keywords if item.arg == "name"), None)
        if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
            raise ValueError(f"{filename}:{node.lineno}: TrainConfig requires a literal name for static auditing")
        name = name_node.value
        if name in recipes:
            raise ValueError(f"{filename}:{node.lineno}: duplicate recipe name: {name}")
        recipes[name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest()
    return recipes


def missing_umi_definitions(config_source: str, policy_source: str) -> list[str]:
    policy_tree = ast.parse(policy_source)
    definitions = {
        node.name for node in policy_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    references = {
        node.attr for node in ast.walk(ast.parse(config_source))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "umi_policy"
    }
    return sorted(references - definitions)


def validate(root: Path, manifest: dict, verify_local_snapshots: bool = False) -> dict:
    errors = []
    checked_paths = []
    recipe_count = 0
    production_count = 0
    if manifest.get("schema_version") != 1:
        errors.append("unsupported source_unification manifest schema_version")
    baseline = manifest.get("production_recipe_baseline", {})
    if baseline.get("fingerprint_algorithm") != FINGERPRINT_ALGORITHM:
        errors.append("unsupported production recipe fingerprint algorithm")
    expected = baseline.get("recipes", {})
    production_count = len(expected)
    if baseline.get("recipe_count") != production_count:
        errors.append("production baseline recipe_count does not match recorded fingerprints")
    config_path = root / baseline.get("config_path", "src/openpi/training/config.py")
    policy_path = root / "src/openpi/policies/umi_policy.py"
    try:
        config_source = config_path.read_text(encoding="utf-8")
        current = recipe_fingerprints(config_source, str(config_path.relative_to(root)))
        recipe_count = len(current)
        for name, digest in sorted(expected.items()):
            if name not in current:
                errors.append(f"missing production recipe: {name}")
            elif current[name] != digest:
                errors.append(f"production recipe drift: {name}")
        policy_source = policy_path.read_text(encoding="utf-8")
        for name in missing_umi_definitions(config_source, policy_source):
            errors.append(f"undefined umi_policy reference: {name}")
    except (OSError, SyntaxError, ValueError) as exc:
        errors.append(str(exc))

    # Scope is explicit: legacy experiment scripts outside this list are not
    # reclassified as regressions introduced by source unification.
    scope = manifest.get("validation", {}).get("scoped_paths", [])
    syntax_paths = set(manifest.get("validation", {}).get("syntax_paths", []))
    scope = set(scope)
    for pattern in manifest.get("validation", {}).get("conflict_marker_globs", []):
        scope.update(str(path.relative_to(root)) for path in root.glob(pattern) if path.is_file())
    for relative in sorted(set(scope) | syntax_paths):
        path = root / relative
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            errors.append(f"validation path escapes repository: {relative}")
            continue
        try:
            source = path.read_text(encoding="utf-8")
            checked_paths.append(relative)
            if CONFLICT_MARKER.search(source):
                errors.append(f"unresolved merge conflict marker: {relative}")
            if relative in syntax_paths:
                ast.parse(source, filename=relative)
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors.append(f"{relative}: {exc}")

    if verify_local_snapshots:
        # Paths are relative to the containing platform repository. These
        # snapshots are intentionally absent from the published submodule.
        platform_root = root.parent.parent
        for snapshot in manifest.get("local_source_snapshots", []):
            patch = platform_root / snapshot["patch_path_relative_to_platform"]
            try:
                if sha256_file(patch) != snapshot["patch_sha256"]:
                    errors.append(f"snapshot patch digest mismatch: {snapshot['name']}")
            except OSError as exc:
                errors.append(f"snapshot unavailable: {snapshot['name']}: {exc}")
    return {
        "ok": not errors,
        "production_recipes_checked": production_count,
        "current_recipes_found": recipe_count,
        "additional_recipes_allowed": recipe_count - production_count,
        "scoped_paths_checked": checked_paths,
        "local_snapshots_checked": verify_local_snapshots,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, help="Defaults to ROOT/source_unification.json")
    parser.add_argument("--verify-local-snapshots", action="store_true", help="Also verify optional platform-local patch backups")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest or root / "source_unification.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = validate(root, manifest, args.verify_local_snapshots)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {"ok": False, "errors": [str(exc)]}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["ok"]:
        print(
            f"PASS: {result['production_recipes_checked']} production recipes preserved; "
            f"{result['current_recipes_found']} current recipes; "
            f"{len(result['scoped_paths_checked'])} scoped sources checked."
        )
    else:
        for error in result["errors"]:
            print(f"FAIL: {error}", file=sys.stderr)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

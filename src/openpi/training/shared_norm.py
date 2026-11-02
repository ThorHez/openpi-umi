"""Local provenance for statistics computed from weighted raw numeric samples.

This module intentionally has no model, tensor, or training-library imports.
Checking provenance reads metadata and file manifests, never the sample data.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import stat


_IGNORED_RESOURCES = {
    "repo_id", "assets", "asset_id", "norm_stats", "norm_stats_metadata",
    # These formulas run after the numeric vectors measured by the artifact.
    "normalization_method", "use_quantile_norm",
}
_MAX_FILES = 100_000
_MAX_JSON = 8 * 1024 * 1024
_MAX_SOURCE = 16 * 1024 * 1024


def _error(message):
    return ValueError("SHARED_NORM_INVALID: " + message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _type_name(value):
    cls = value if isinstance(value, type) else type(value)
    return cls.__module__ + "." + cls.__qualname__


def _canonical(value, depth=0, *, ignore_resources=True):
    if depth > 24:
        raise _error("processor configuration is too deeply nested")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _error("processor configuration contains a non-finite number")
        return value
    if isinstance(value, enum.Enum):
        return {"enum": _type_name(value), "value": _canonical(value.value, depth + 1, ignore_resources=ignore_resources)}
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": _type_name(value), "fields": {
            field.name: _canonical(getattr(value, field.name), depth + 1, ignore_resources=ignore_resources)
            for field in dataclasses.fields(value) if not ignore_resources or field.name not in _IGNORED_RESOURCES
        }}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise _error("processor configuration must use string mapping keys")
        return {key: _canonical(item, depth + 1, ignore_resources=ignore_resources) for key, item in value.items()
                if not ignore_resources or key not in _IGNORED_RESOURCES}
    if isinstance(value, (tuple, list)):
        return [_canonical(item, depth + 1, ignore_resources=ignore_resources) for item in value]
    if isinstance(value, functools.partial):
        if type(value) is not functools.partial:
            raise _error("stateful callable partial subclasses require an explicit dataclass configuration")
        return {"partial": _canonical(value.func, depth + 1),
                "args": _canonical(value.args, depth + 1, ignore_resources=False),
                "keywords": _canonical(value.keywords or {}, depth + 1, ignore_resources=False)}
    if callable(value):
        owner = getattr(value, "__self__", None)
        if inspect.ismethod(value) or owner is not None and not inspect.ismodule(owner):
            raise _error("bound processor methods cannot establish immutable numeric semantics")
        if not (inspect.isfunction(value) or inspect.isbuiltin(value) or inspect.isclass(value)):
            raise _error("stateful callable processor instances require an explicit dataclass configuration")
        module, qualified = getattr(value, "__module__", None), getattr(value, "__qualname__", None)
        if (not isinstance(module, str) or not module or not isinstance(qualified, str) or not qualified
                or "<lambda>" in qualified or "<locals>" in qualified):
            raise _error("anonymous processor functions cannot establish shared numeric semantics")
        name = module + "." + qualified
        try:
            source = inspect.getsource(value).encode()
        except (OSError, TypeError) as exc:
            if not inspect.isbuiltin(value) and not (inspect.isclass(value) and module == "builtins"):
                raise _error("named processor callable implementation cannot be verified from source") from exc
            source = name.encode()
        return {"callable": name, "source_sha256": hashlib.sha256(source).hexdigest()}
    raise _error("unsupported processor configuration value: " + _type_name(value))


def _implementation(factory):
    """Freeze actual source contents without making checkout locations semantic."""
    try:
        filename = inspect.getsourcefile(type(factory))
        if not filename:
            raise _error("factory implementation has no inspectable Python source file")
        source = Path(filename).resolve(strict=True)
        # A module path alone is insufficient for dynamically constructed types:
        # inspect must also locate this class definition in its real source.
        inspect.getsource(type(factory))
    except (OSError, TypeError) as exc:
        raise _error("factory implementation cannot be verified from its Python source") from exc
    package = next((parent for parent in source.parents
                    if parent.name == "openpi" and parent.parent.name == "src"), None)
    files = {source.relative_to(package).as_posix() if package else source.name: source}
    if package is not None:
        for relative in ("training/composed_processing.py", "transforms.py", "policies/umi_policy.py"):
            path = package / relative
            if path.exists():
                files[relative] = path
    evidence = {}
    for name, path in files.items():
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_SOURCE:
            raise _error("processor implementation must be a bounded regular Python source file")
        contents = path.read_bytes()
        if len(contents) > _MAX_SOURCE:
            raise _error("processor implementation exceeds the source size limit")
        evidence[name] = hashlib.sha256(contents).hexdigest()
    return evidence


def _processor(factory, *, compatibility=False):
    if not dataclasses.is_dataclass(factory) or isinstance(factory, type):
        raise _error("each source must use a dataclass data factory")
    value = _canonical(factory)
    value["implementation"] = _implementation(factory)
    if type(factory).__name__ == "ComposableDataConfig":
        composition = value["fields"].get("composition")
        if not isinstance(composition, dict):
            raise _error("ComposableDataConfig requires an explicit numeric composition")
        # Formula choice is downstream of raw numeric statistics. Camera
        # routing and language may differ across compatible source layouts.
        omitted = {"normalization"}
        if compatibility:
            omitted.update({"cameras", "prompt_key"})
        value["fields"]["composition"] = {
            key: item for key, item in composition.items() if key not in omitted
        }
    return value


def _numeric_source(factory, info):
    """Retain raw coordinate identities as well as declared transform semantics."""
    selected = None
    if type(factory).__name__ == "ComposableDataConfig":
        composition = getattr(factory, "composition", None) or {}
        selected = {row.get("key") for row in composition.get("state_fields", []) if isinstance(row, dict)}
        selected.add(composition.get("action_key"))
    features = info.get("features") or {}
    if not isinstance(features, dict):
        raise _error("source feature inventory must be a mapping")
    numeric = {key: value for key, value in features.items()
               if isinstance(value, dict) and (selected is None or key in selected)
               and value.get("dtype") not in {"image", "video", "string", "str"}}
    annotations = info.get("field_annotations") or []
    if not isinstance(annotations, list):
        raise _error("source field annotations must be a list")
    declarations = {key: info[key] for key in (
        "action_representation", "action_mode", "rotation_representation", "position_unit", "joint_unit",
        "coordinate_frame", "gripper_range", "arm_mode", "robot_type",
    ) if key in info}
    return {"features": numeric, "declarations": declarations,
            "field_annotations": sorted(
                [row for row in annotations if isinstance(row, dict) and row.get("key") in numeric],
                key=lambda row: str(row.get("key")),
            )}


def _read_document(path):
    path = Path(path).resolve(strict=True)
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_JSON:
            raise _error("metadata must be a bounded regular JSON file: " + str(path))
        content = stream.read(_MAX_JSON + 1)
    if len(content) > _MAX_JSON:
        raise _error("metadata exceeds the JSON size limit")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise _error("duplicate metadata key: " + key)
            result[key] = value
        return result
    document = json.loads(content, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(_error("non-finite JSON: " + value)))
    if not isinstance(document, dict):
        raise _error("metadata must be a JSON object")
    return document, hashlib.sha256(content).hexdigest()


def _source(factory, weight):
    value = getattr(factory, "repo_id", None)
    if not isinstance(value, (str, Path)) or "://" in str(value) or not Path(value).is_absolute():
        raise _error("source repo_id must be an absolute local LeRobot directory")
    root = Path(value).resolve(strict=True)
    if not root.is_dir():
        raise _error("source repo_id must identify a directory")
    document, info_sha = _read_document(root / "meta/info.json")
    if type(document.get("total_frames")) is not int or document["total_frames"] <= 0:
        raise _error("source meta/info.json needs a positive total_frames")
    rows = []
    for folder in ("meta", "data"):
        directory = root / folder
        if not directory.is_dir() or directory.is_symlink():
            raise _error("source requires regular meta and data directories")
        for current, directories, files in os.walk(directory, followlinks=False):
            directories.sort()
            if any((Path(current) / name).is_symlink() for name in directories):
                raise _error("source manifests do not follow symbolic-link directories")
            for name in sorted(files):
                path = Path(current) / name
                if folder == "data" and path.suffix != ".parquet":
                    continue
                if path.is_symlink():
                    raise _error("source manifests require regular metadata and parquet files")
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise _error("source manifest contains a non-regular file")
                rows.append([path.relative_to(root).as_posix(), info.st_size, info.st_mtime_ns])
                if len(rows) > _MAX_FILES:
                    raise _error("source file manifest exceeds the bounded inventory")
    if not any(row[0].startswith("data/") and row[1] > 0 for row in rows):
        raise _error("source contains no nonempty parquet files")
    return {
        "repo_id": str(root), "weight": float(weight), "info_sha256": info_sha,
        "data_manifest_digest": _digest(sorted(rows)),
        "processor_sha256": _digest(_processor(factory)),
        "compatibility_sha256": _digest({"processor": _processor(factory, compatibility=True),
                                          "numeric_source": _numeric_source(factory, document)}),
    }


def source_descriptors(factories, weights):
    """Describe ordered full local sources and reject incompatible numeric layouts."""
    if (not isinstance(factories, (tuple, list)) or not 1 <= len(factories) <= 64
            or not isinstance(weights, (tuple, list)) or len(weights) != len(factories)):
        raise _error("source factories and weights must have the same bounded nonzero length")
    if any(isinstance(weight, bool) or not isinstance(weight, (float, int))
           or not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise _error("source weights must be positive finite numbers")
    descriptors = [_source(factory, weight) for factory, weight in zip(factories, weights)]
    if len({item["compatibility_sha256"] for item in descriptors}) != 1:
        raise _error("source state/action layouts, ordering, encoding, units or numeric processing differ")
    return descriptors


def _validate_statistics(document):
    stats = document.get("norm_stats")
    if not isinstance(stats, dict) or not all(key in stats for key in ("state", "actions")):
        raise _error("shared statistics must contain state and actions in one norm_stats.json")
    for key in ("state", "actions"):
        values = stats[key]
        if not isinstance(values, dict) or values.get("quantile_method") != "q01_q99":
            raise _error(key + " statistics must declare quantile_method=q01_q99")
        lengths = []
        for field in ("mean", "std", "q01", "q99"):
            array = values.get(field)
            if (not isinstance(array, list) or not 1 <= len(array) <= 4096
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) for value in array)):
                raise _error(key + "." + field + " must be a finite numeric vector")
            lengths.append(len(array))
        if len(set(lengths)) != 1 or any(value < 0 for value in values["std"]):
            raise _error(key + " statistics have inconsistent widths or negative standard deviations")
        if any(low > high for low, high in zip(values["q01"], values["q99"])):
            raise _error(key + " q01 must not exceed q99")


def validate_shared_metadata(document, factories, weights):
    """Validate a full raw-sample artifact against current ordered inputs, read-only."""
    if not isinstance(document, dict):
        raise _error("shared normalization document must be an object")
    _validate_statistics(document)
    metadata = document.get("shared_normalization")
    required = {
        "schema_version": 1, "method": "weighted_raw_samples", "weight_semantics": "per_sample_repeat",
        "coverage": "full", "quantile_method": "weighted_histogram", "quantile_bins": 5000,
    }
    if (not isinstance(metadata, dict) or set(metadata) != {*required, "sources"}
            or type(metadata.get("schema_version")) is not int
            or type(metadata.get("quantile_bins")) is not int
            or any(metadata.get(key) != value for key, value in required.items())):
        raise _error("missing or unsupported weighted raw-sample shared_normalization provenance; recompute statistics")
    expected = source_descriptors(factories, weights)
    recorded = metadata.get("sources")
    if not isinstance(recorded, list) or len(recorded) != len(expected):
        raise _error("shared normalization source count differs; recompute statistics")
    for current, saved in zip(expected, recorded):
        if (not isinstance(saved, dict) or set(saved) != {*current, "rows", "state_vectors", "action_vectors"}
                or any(saved.get(key) != value for key, value in current.items())
                or any(type(saved.get(key)) is not int or saved[key] <= 0
                       for key in ("rows", "state_vectors", "action_vectors"))):
            raise _error("shared normalization source, weight or processor changed; recompute statistics")
        info, _ = _read_document(Path(current["repo_id"]) / "meta/info.json")
        if (saved["rows"] != info["total_frames"] or saved["state_vectors"] < saved["rows"]
                or saved["action_vectors"] < saved["rows"]):
            raise _error("shared normalization does not cover every source row; recompute statistics")
    return metadata


def load_shared_norm_document(path, factories, weights):
    """Validate one bounded read and return those exact bytes' decoded document."""
    document, _ = _read_document(path)
    validate_shared_metadata(document, factories, weights)
    return document


def load_shared_norm_metadata(path, factories, weights):
    """Read a bounded local artifact and return its validated provenance."""
    return load_shared_norm_document(path, factories, weights)["shared_normalization"]

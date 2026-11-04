"""Prepare one task-local, weighted raw-sample statistic before model startup.

Planning is metadata-only and imports no tensor/numeric backends. Execution
projects numeric Parquet columns, performs two finite passes, then publishes a
new immutable artifact. Existing dataset statistics are never merged or edited.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

from openpi.training.shared_norm import load_shared_norm_document, source_descriptors, validate_shared_metadata


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_files():
    root = Path(__file__).resolve().parents[1]
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in (
        "training/training_normalization.py", "training/numeric_stats.py",
        "training/shared_norm.py", "training/composed_processing.py", "shared/weighted_stats.py",
    )}


def _weights(factory):
    weights = list(factory.weights) if factory.weights is not None else [1.] * len(factory.datasets)
    alternate = getattr(factory, "norm_weights", None)
    if alternate is not None and list(alternate) != weights:
        raise ValueError("SHARED_NORM_INVALID: shared statistics must use the actual training row weights")
    return weights


def _context(config, sources, numeric_configs):
    return {"schema_version": 1, "sources": sources,
        "action_horizon": config.model.action_horizon, "model_action_dim": config.model.action_dim,
        "effective_fps": [item.effective_fps for item in numeric_configs],
        "implementation": _source_files()}


def _protect_output(output, config):
    output = Path(output).resolve()
    for factory in config.data.datasets:
        dataset = Path(factory.repo_id).resolve(strict=True)
        if output.is_relative_to(dataset):
            raise ValueError("SHARED_NORM_INVALID: automatic statistics must be outside input datasets")
        assets = getattr(factory, "assets", None)
        if assets is not None:
            base = getattr(assets, "assets_dir", None) or config.assets_dirs
            identity = getattr(assets, "asset_id", None) or factory.repo_id
            if "://" not in str(base):
                original = (Path(base) / identity).resolve()
                if output.is_relative_to(original):
                    raise ValueError("SHARED_NORM_INVALID: automatic statistics must be outside input norm assets")
    return output


def planned_training_normalization(config):
    """Freeze automatic normalization inputs without reading Parquet payloads.

    Registered source statistics remain historical references; automatic mode
    computes the actual training statistic from every raw row. Merely finding
    an existing output never changes this frozen plan.
    """
    factory = config.data
    children = getattr(factory, "datasets", None)
    if not isinstance(children, (list, tuple)) or len(children) <= 1:
        return {"mode": "per_dataset"}
    path = getattr(factory, "shared_norm_stats_path", None)
    if path is not None and getattr(factory, "use_merged_norm_stats", False):
        path = Path(path).expanduser().resolve()
        if path.name != "norm_stats.json":
            path = path / "norm_stats.json"
        return {"mode": "shared", "path": str(path)}
    if not (getattr(factory, "auto_shared_norm", False) or getattr(factory, "use_merged_norm_stats", False)):
        return {"mode": "per_dataset"}
    from openpi.training.numeric_stats import build_numeric_statistics_config, numeric_io_evidence

    weights = _weights(factory)
    sources = source_descriptors(children, weights)
    numeric = [build_numeric_statistics_config(child, config.model,
        effective_fps=getattr(child, "_norm_effective_fps", None), validate_model=False) for child in children]
    context = _context(config, sources, numeric)
    fingerprint = _digest(context)
    checkpoint = Path(config.checkpoint_dir).resolve()
    output = _protect_output(checkpoint.parent / (checkpoint.name + "-normalization") / fingerprint, config)
    return {"mode": "auto_shared", "output_path": str(output / "norm_stats.json"),
        "fingerprint": fingerprint, "method": "weighted_raw_samples", "weight_semantics": "per_sample_repeat",
        "coverage": "full", "quantile_method": "weighted_histogram", "quantile_bins": 5000,
        "action_horizon": context["action_horizon"], "model_action_dim": context["model_action_dim"],
        "sources": sources, "effective_fps": context["effective_fps"],
        "implementation": context["implementation"], "numeric_io": [numeric_io_evidence(item) for item in numeric]}


def _batches(numeric_config, batch_size):
    import numpy as np
    from openpi.training.numeric_stats import NumericParquetDataset

    dataset = NumericParquetDataset(numeric_config)
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        yield {key: np.stack([row[key] for row in rows]) for key in ("state", "actions")}


def _publish(path, document, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".normalization-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _method(config):
    method = getattr(config, "normalization_method", None)
    if method is None:
        values = {(getattr(child, "composition", None) or {}).get("normalization") for child in config.data.datasets}
        if len(values) == 1:
            method = values.pop()
    return method if method in {"quantile", "minmax", "zscore"} else None


def _write_receipt(config, path, document, *, fingerprint=None):
    if fingerprint is None:
        fingerprint = document.get("training_normalization", {}).get("fingerprint")
    receipt = {"schema_version": 1, "mode": "shared", "norm_path": str(path),
        "norm_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "checkpoint_asset_id": "shared",
        "normalization_method": _method(config), "shared_normalization": document["shared_normalization"],
        "statistics_widths": {key: len(values["mean"]) for key, values in document["norm_stats"].items()},
        "fingerprint": fingerprint}
    checkpoint = Path(config.checkpoint_dir).resolve()
    receipt_path = checkpoint.parent / (checkpoint.name + "-normalization.json")
    if getattr(config, "resume", False) and receipt_path.is_file():
        previous = json.loads(receipt_path.read_text())
        if any(previous.get(key) != receipt[key] for key in ("norm_sha256", "normalization_method")):
            raise ValueError("SHARED_NORM_INVALID: resuming cannot change the model normalization statistic or formula")
    _publish(receipt_path, receipt, replace=True)


def prepare_training_normalization(config):
    """Return a config sharing one verified statistic, before GPU/model startup."""
    plan = planned_training_normalization(config)
    if plan["mode"] == "per_dataset":
        return config
    factory = config.data
    weights = _weights(factory)
    if plan["mode"] == "shared":
        path = Path(plan["path"])
        document = load_shared_norm_document(path, factory.datasets, weights)
        if getattr(factory, "auto_shared_norm", False) and "training_normalization" in document:
            automatic = planned_training_normalization(dataclasses.replace(config,
                data=dataclasses.replace(factory, shared_norm_stats_path=None)))
            if document["training_normalization"] != {"schema_version": 1, "fingerprint": automatic["fingerprint"]}:
                raise ValueError("SHARED_NORM_INVALID: automatic statistics do not match the current horizon, FPS or execution context")
        _write_receipt(config, path, document)
        logging.info("Normalization: using registered shared statistic %s", path)
        return config

    from openpi.shared.weighted_stats import compute_weighted_statistics
    from openpi.training.numeric_stats import build_numeric_statistics_config

    path = Path(plan["output_path"])
    if getattr(config, "resume", False):
        checkpoint = Path(config.checkpoint_dir).resolve()
        receipt_path = checkpoint.parent / (checkpoint.name + "-normalization.json")
        if receipt_path.is_file():
            previous = json.loads(receipt_path.read_text())
            if (previous.get("fingerprint") != plan["fingerprint"]
                    or previous.get("normalization_method") != _method(config)):
                raise ValueError("SHARED_NORM_INVALID: resuming cannot change data, weights, processing or normalization formula")
    if path.is_file():
        document = load_shared_norm_document(path, factory.datasets, weights)
        if document.get("training_normalization") != {"schema_version": 1, "fingerprint": plan["fingerprint"]}:
            raise ValueError("SHARED_NORM_INVALID: existing automatic statistics have a conflicting execution context")
        logging.info("Normalization: reusing complete task-local weighted statistic %s", path)
    else:
        if path.exists() or path.is_symlink():
            raise ValueError("SHARED_NORM_INVALID: statistics destination is not a regular file")
        batch_size = config.batch_size
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("SHARED_NORM_INVALID: statistics batch size must be a positive integer")
        numeric = [build_numeric_statistics_config(child, config.model,
            effective_fps=getattr(child, "_norm_effective_fps", None), validate_model=False) for child in factory.datasets]
        logging.info("Normalization: computing one shared statistic from all %d datasets before training; no image decoding", len(numeric))
        values, coverage = compute_weighted_statistics(
            [lambda item=item: _batches(item, batch_size) for item in numeric], weights)
        if planned_training_normalization(config) != plan:
            raise ValueError("SHARED_NORM_INVALID: data or numeric processing changed during automatic statistics")
        document = {"norm_stats": {key: {**{name: array.tolist() for name, array in stats.items()},
            "quantile_method": "q01_q99"} for key, stats in values.items()}, "shared_normalization": {
                "schema_version": 1, "method": "weighted_raw_samples", "weight_semantics": "per_sample_repeat",
                "coverage": "full", "sources": [{**source, **count} for source, count in
                    zip(plan["sources"], coverage, strict=True)],
                "quantile_method": "weighted_histogram", "quantile_bins": 5000},
            "training_normalization": {"schema_version": 1, "fingerprint": plan["fingerprint"]}}
        validate_shared_metadata(document, factory.datasets, weights)
        try:
            _publish(path, document)
        except FileExistsError:
            existing = load_shared_norm_document(path, factory.datasets, weights)
            if existing != document:
                raise ValueError("SHARED_NORM_INVALID: another statistics writer published conflicting values") from None
        logging.info("Normalization: shared weighted statistic ready at %s", path)
    _write_receipt(config, path, document, fingerprint=plan["fingerprint"])
    return dataclasses.replace(config, data=dataclasses.replace(factory,
        use_merged_norm_stats=True, auto_shared_norm=True, shared_norm_stats_path=str(path)))

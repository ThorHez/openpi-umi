"""Generate Kai0-style advantage labels in a derived LeRobot dataset."""

import json
import logging
from pathlib import Path
import shutil
import tempfile

import numpy as np


DATASET_NAME = "stage_advantage_dataset"
RECEIPT_NAME = "stage_advantage.json"


def _predict_batches(predict, reference, current, batch_size):
    scores = np.empty(len(current), dtype=np.float32)
    for start in range(0, len(current), batch_size):
        stop = min(start + batch_size, len(current))
        batch = np.asarray(predict(reference[start:stop], current[start:stop]), dtype=np.float32)
        if batch.shape != (stop - start,) or not np.all(np.isfinite(batch)):
            raise ValueError("Stage progress estimator returned invalid pair scores")
        scores[start:stop] = batch
    return scores


def score_episodes(episodes, frame_count, predict, *, n_step, batch_size):
    """Score pairs and first-frame-anchored progress on complete episodes."""
    if n_step < 1 or batch_size < 1 or frame_count < 1:
        raise ValueError("Stage advantage requires positive frame, window and batch counts")
    absolute_value = np.zeros(frame_count, dtype=np.float32)
    absolute_advantage = np.zeros_like(absolute_value)
    relative_advantage = np.zeros_like(absolute_value)
    seen = np.zeros(frame_count, dtype=bool)
    for episode in episodes:
        if (episode.ndim != 1 or not len(episode) or np.any(episode < 0) or np.any(episode >= frame_count)
                or np.any(seen[episode]) or len(np.unique(episode)) != len(episode)):
            raise ValueError("Stage progress scoring requires distinct, complete episode frame positions")
        seen[episode] = True
        current = episode[:-1]
        steps = np.arange(len(current))
        future = np.minimum(steps + n_step, len(episode) - 1)
        scale = n_step / (future - steps)
        reference = np.full(len(current), episode[0], dtype=episode.dtype)
        absolute_value[episode[1:]] = _predict_batches(predict, reference, episode[1:], batch_size)
        absolute_advantage[current] = np.clip(
            (absolute_value[episode[future]] - absolute_value[current]) * scale, -1.0, 1.0
        )
        relative_advantage[current] = np.clip(
            _predict_batches(predict, current, episode[future], batch_size) * scale, -1.0, 1.0
        )
    if not seen.all():
        raise ValueError("Stage progress scoring did not cover every source frame")
    return {"absolute_value": absolute_value, "absolute_advantage": absolute_advantage,
            "relative_advantage": relative_advantage}


def discretize_stages(progress, scores, *, stage_count, positive_ratio):
    """Rank by absolute advantage within the human-labeled progress intervals."""
    if (progress.shape != scores.shape or progress.ndim != 1 or not np.all(np.isfinite(progress))
            or not np.all(np.isfinite(scores)) or np.any((progress < 0) | (progress > 1))
            or stage_count < 1 or not 0 < positive_ratio <= 1):
        raise ValueError("Invalid progress values, scores, stage count or positive ratio")
    stage_indices = np.minimum((progress * stage_count).astype(np.int64), stage_count - 1)
    labels = np.zeros(len(scores), dtype=np.int64)
    stages = []
    for stage in range(stage_count):
        mask = stage_indices == stage
        values = scores[mask]
        threshold = float(np.quantile(values, 1.0 - positive_ratio)) if values.size else None
        if threshold is not None:
            labels[mask] = values >= threshold
        stages.append({"stage": stage, "frame_count": int(values.size),
                       "positive_count": int(labels[mask].sum()), "threshold": threshold})
    return labels, stages


def _copy_assets(root, target):
    def ignore(directory, names):
        parent = Path(directory)
        for name in names:
            if (parent / name).is_symlink():
                raise ValueError(f"Derived LeRobot datasets require local files, found a link at {parent / name}")
        if parent.is_relative_to(root / "data"):
            return [name for name in names if name.endswith(".parquet") and (parent / name).is_file()]
        return []

    shutil.copytree(root, target, ignore=ignore)


def _update_huggingface_schema(table, features):
    metadata = table.schema.metadata
    if not metadata or b"huggingface" not in metadata:
        return table
    document = json.loads(metadata[b"huggingface"])
    stored_features = document.get("info", {}).get("features")
    if isinstance(stored_features, dict):
        for key, feature in features.items():
            stored_features[key] = {"dtype": feature["dtype"], "_type": "Value"}
        return table.replace_schema_metadata({**metadata, b"huggingface": json.dumps(document).encode("utf-8")})
    return table


def _write_parquets(root, target, indices, columns, features):
    import pyarrow as pa
    import pyarrow.parquet as pq

    order = np.argsort(indices)
    ordered_indices = indices[order]
    if np.any(ordered_indices[1:] == ordered_indices[:-1]):
        raise ValueError("Source dataset contains duplicate LeRobot indices")
    seen = np.zeros(len(indices), dtype=bool)
    for parquet in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(parquet)
        row_indices = table["index"].to_numpy()
        if row_indices.dtype.kind not in "iu":
            raise ValueError(f"Source parquet indices are not integers in {parquet.name}")
        positions = np.searchsorted(ordered_indices, row_indices)
        if (np.any(positions >= len(indices)) or np.any(ordered_indices[positions] != row_indices)
                or len(np.unique(positions)) != len(positions) or np.any(seen[positions])):
            raise ValueError(f"Source parquet indices do not match scored frames in {parquet.name}")
        seen[positions] = True
        rows = order[positions]
        for key, values in columns.items():
            array = pa.array(values[rows])
            column = table.schema.get_field_index(key)
            table = table.set_column(column, key, array) if column >= 0 else table.append_column(key, array)
        destination = target / parquet.relative_to(root)
        pq.write_table(_update_huggingface_schema(table, features), destination)
    if not seen.all():
        raise ValueError("Source parquet files did not contain every scored frame")


def _prepare_destination(root, target):
    if target.parent.resolve().is_relative_to(root) or target == root:
        raise ValueError("Advantage dataset output must be outside the source dataset")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Advantage dataset output already exists at {target}")


def publish_labeled_copy(root, target, indices, columns, *, receipt, receipt_name):
    """Write all labels by LeRobot index and publish a complete dataset atomically."""
    root = Path(root).resolve(strict=True)
    target = Path(target).absolute()
    _prepare_destination(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    features = {key: {"dtype": str(values.dtype), "shape": [1], "names": None} for key, values in columns.items()}
    logging.info("Writing derived LeRobot dataset to %s", target)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".advantage-") as temporary:
        staged = Path(temporary) / "dataset"
        _copy_assets(root, staged)
        _write_parquets(root, staged, indices, columns, features)
        info_path = staged / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["features"].update(features)
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (staged / "meta" / receipt_name).write_text(
            json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
        )
        staged.rename(target)
    return target


def annotate_dataset(config, source, predict, output_dir):
    """Publish a complete copy with stage labels after scoring succeeds."""
    root = source.root.resolve(strict=True)
    target = Path(output_dir).resolve() / DATASET_NAME
    _prepare_destination(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)

    n_step = config.model.n_step
    positive_ratio = config.model.positive_ratio
    stage_count = config.model.stage_count
    indices = np.asarray(source.indices, dtype=np.int64)
    logging.info("Scoring %d frames for stage advantage with a %d-frame window", len(indices), n_step)
    columns = score_episodes(source.episodes, len(indices), predict, n_step=n_step, batch_size=config.batch_size)
    columns["is_positive"], stages = discretize_stages(
        source.progress, columns["absolute_advantage"], stage_count=stage_count, positive_ratio=positive_ratio
    )
    receipt = {"schema_version": 1, "method": "kai0_stage_progress", "score": "absolute_advantage",
               "n_step": n_step, "positive_ratio": positive_ratio, "stage_count": stage_count,
               "frame_count": len(indices), "positive_count": int(columns["is_positive"].sum()), "stages": stages}

    publish_labeled_copy(root, target, indices, columns, receipt=receipt, receipt_name=RECEIPT_NAME)
    logging.info("Stage advantage labeled %d of %d frames positive and published %s",
                 receipt["positive_count"], receipt["frame_count"], target)
    return target

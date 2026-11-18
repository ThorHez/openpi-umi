import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training.stage_progress_labels import annotate_dataset, discretize_stages, score_episodes


def test_anchored_and_direct_advantages_include_the_short_tail():
    progress = np.array([0.25, 0.0, 1.0, 0.5])
    batches = []

    def predict(reference, current):
        batches.append(len(current))
        return progress[current] - progress[reference] + 0.1 * progress[reference]

    scores = score_episodes([np.array([1, 0, 3, 2])], 4, predict, n_step=2, batch_size=2)
    np.testing.assert_allclose(scores["absolute_value"], progress)
    np.testing.assert_allclose(scores["absolute_advantage"], [0.75, 0.5, 0.0, 1.0])
    np.testing.assert_allclose(scores["relative_advantage"], [0.775, 0.5, 0.0, 1.0])
    assert batches == [2, 1, 2, 1]
    labels, stages = discretize_stages(progress, scores["absolute_advantage"], stage_count=2, positive_ratio=0.5)
    assert labels.tolist() == [1, 0, 0, 1]
    assert [stage["positive_count"] for stage in stages] == [1, 1]


@pytest.mark.parametrize("invalid_index", [False, True])
def test_derived_dataset_joins_indices_and_preserves_task_metadata(tmp_path, invalid_index):
    root = tmp_path / "source"
    parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    progress_key = "annotation.progress"
    table = pa.table({"index": [101, 502, 700, 900 if invalid_index else 901], "task_index": [12] * 4,
                      progress_key: [0.0, 0.25, 0.5, 1.0]})
    pq.write_table(table, parquet)
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text('{"features": {"task_index": {"dtype": "int64"}}}')
    tasks = '{"task_index": 12, "task": "fold the cloth"}\n'
    (root / "meta" / "tasks.jsonl").write_text(tasks)
    (root / "videos").mkdir()
    (root / "videos" / "frame.mp4").write_bytes(b"source video")
    source = SimpleNamespace(root=root, indices=np.array([502, 101, 901, 700]),
                             episodes=(np.array([1, 0, 3, 2]),), progress=np.array([0.25, 0, 1, 0.5]))
    config = SimpleNamespace(batch_size=2, model=SimpleNamespace(n_step=2, positive_ratio=0.5, stage_count=2))
    output = tmp_path / "run"

    def predict(reference, current):
        return source.progress[current] - source.progress[reference]

    if invalid_index:
        with pytest.raises(ValueError, match="indices do not match"):
            annotate_dataset(config, source, predict, output)
        assert list(output.iterdir()) == []
    else:
        derived = annotate_dataset(config, source, predict, output)
        labeled = pq.read_table(derived / parquet.relative_to(root))
        assert labeled["is_positive"].to_pylist() == [0, 1, 1, 0]
        assert labeled["task_index"].to_pylist() == [12] * 4
        assert labeled[progress_key].equals(table[progress_key])
        assert (derived / "meta" / "tasks.jsonl").read_text() == tasks
        assert (derived / "videos" / "frame.mp4").read_bytes() == b"source video"
        receipt = json.loads((derived / "meta" / "stage_advantage.json").read_text())
        assert receipt["frame_count"] == 4
        assert receipt["positive_count"] == 2
    assert pq.read_table(parquet).equals(table)
    assert (root / "meta" / "tasks.jsonl").read_text() == tasks

"""Exercise native timestamp initialization with CPU Torch and controlled HF views.

The actual LeRobot class and validators are loaded without importing its camera,
robot, Hub, or video dependencies. No model, network, dataset cache, or GPU is used.
"""
from __future__ import annotations

import ast
from itertools import accumulate
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
LEROBOT = ROOT / "third_party/lerobot/lerobot/common/datasets"


def definitions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


class HFView:
    """Formatting a view affects reads, while with_format returns a separate view."""
    def __init__(self, columns, owner=None):
        self.columns = columns
        self.owner = owner if owner is not None else self
        self.transform = None
        self.format = {"type": None}
        if owner is None:
            self.formatted_reads = []
            self.raw_reads = []
            self.requested_formats = []

    def set_transform(self, transform):
        self.transform = transform
        self.format = {"type": "custom", "format_kwargs": {"transform": transform}}

    def with_format(self, kind):
        self.owner.requested_formats.append(kind)
        assert kind is None
        return HFView(self.columns, self.owner)

    def __getitem__(self, key):
        if self.transform is not None:
            self.owner.formatted_reads.append(key)
            return self.transform({key: list(self.columns[key])})[key]
        self.owner.raw_reads.append(key)
        return list(self.columns[key])


@pytest.fixture
def native(tmp_path):
    namespace = {"torch": torch, "np": np, "accumulate": accumulate, "pformat": pformat,
                 "Path": Path, "CODEBASE_VERSION": "v2.1",
                 "PILImage": SimpleNamespace(Image=type("UnusedImage", (), {}))}
    definitions(LEROBOT / "utils.py", {
        "hf_transform_to_torch", "check_timestamps_sync", "check_delta_timestamps",
        "get_episode_data_index", "get_delta_indices",
    }, namespace)
    definitions(LEROBOT / "lerobot_dataset.py", {"LeRobotDataset"}, namespace)
    validator = namespace["check_timestamps_sync"]
    calls = []
    state = {"columns": {"timestamp": [0.0, 0.05, 0.0, 0.05], "episode_index": [0, 0, 1, 1]},
             "episodes": {0: {"length": 2}, 1: {"length": 2}}, "fps": 20}

    class Metadata:
        def __init__(self, repo_id, root, revision, force_cache_sync=False):
            self.episodes = state["episodes"]
            self.fps = state["fps"]

    def load_dataset(*args, **kwargs):
        assert args == ("parquet",)
        assert kwargs == {"data_dir": str(tmp_path / "data"), "split": "train"}
        state["view"] = HFView(state["columns"])
        return state["view"]

    def checked(*args):
        calls.append(args)
        return validator(*args)

    namespace.update(LeRobotDatasetMetadata=Metadata, load_dataset=load_dataset,
                     get_safe_default_codec=lambda: "cpu-fixture", check_timestamps_sync=checked)
    cls = namespace["LeRobotDataset"]
    # The constructor's file-presence branch needs no on-disk parquet in this
    # controlled view; all timestamp loading and validation remain native code.
    cls.get_episodes_file_paths = lambda self: []
    return SimpleNamespace(cls=cls, namespace=namespace, state=state, calls=calls,
                           validator=validator, root=tmp_path)


def construct(native, **kwargs):
    return native.cls("local-fixture", root=native.root, **kwargs)


def legacy_column(native, values):
    formatted = native.namespace["hf_transform_to_torch"]({"column": list(values)})["column"]
    return torch.stack(formatted).numpy()


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scalar", [float, np.float32, np.float64])
def test_batched_columns_match_legacy_scalar_tensor_precision_and_default_dtype(native, default_dtype, scalar):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        native.state["columns"]["timestamp"] = [scalar(value) for value in [0.0, 0.05, 0.0, 0.05]]
        expected_timestamps = legacy_column(native, native.state["columns"]["timestamp"])
        expected_indices = legacy_column(native, native.state["columns"]["episode_index"])
        dataset = construct(native)
        timestamps, indices, episodes, fps, tolerance = native.calls[0]
        np.testing.assert_array_equal(timestamps, expected_timestamps, strict=True)
        np.testing.assert_array_equal(indices, expected_indices, strict=True)
        assert fps == 20 and tolerance == 1e-4
        assert episodes.keys() == dataset.episode_data_index.keys()
        for key, value in dataset.episode_data_index.items():
            np.testing.assert_array_equal(episodes[key], value.numpy(), strict=True)
        if scalar is float:
            assert timestamps.dtype == (np.float32 if default_dtype is torch.float32 else np.float64)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("frames", [4, 40000])
def test_initialization_converts_two_columns_once_without_per_row_sample_formatting(native, monkeypatch, frames):
    native.state["columns"] = {"timestamp": [index / 20 for index in range(frames)],
                               "episode_index": [0] * frames}
    native.state["episodes"] = {0: {"length": frames}}
    original_tensor = torch.tensor
    tensor_calls = []

    def tensor(values, *args, **kwargs):
        tensor_calls.append(len(values))
        return original_tensor(values, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "tensor", tensor)
        dataset = construct(native)
    view = native.state["view"]
    assert tensor_calls == [frames, frames]
    assert view.formatted_reads == []
    assert view.raw_reads == ["timestamp", "episode_index"]
    assert view.requested_formats == [None]
    assert view.transform is native.namespace["hf_transform_to_torch"]
    assert dataset.hf_dataset is view and view.format["type"] == "custom"
    # Ordinary sample reads still use the original configured Torch transform.
    assert torch.is_tensor(dataset.hf_dataset["timestamp"][0])
    assert view.formatted_reads == ["timestamp"]


def test_bad_timestamp_near_the_end_still_fails_full_native_validation(native):
    length = 10000
    timestamps = [index / 20 for _ in range(2) for index in range(length)]
    timestamps[-1] += 0.01
    native.state["columns"] = {"timestamp": timestamps,
                               "episode_index": [0] * length + [1] * length}
    native.state["episodes"] = {0: {"length": length}, 1: {"length": length}}
    with pytest.raises(ValueError, match="timestamps unexpectedly violate"):
        construct(native)
    assert len(native.calls) == 1
    assert len(native.calls[0][0]) == length * 2
    assert native.state["view"].formatted_reads == []


def test_episode_boundaries_delta_indices_and_explicit_tolerance_are_unchanged(native):
    native.state["columns"] = {"timestamp": [0.0, 0.05, 10.0, 10.05], "episode_index": [0, 0, 1, 1]}
    dataset = construct(native, delta_timestamps={"actions": [0.0, 0.05]}, tolerance_s=1e-3)
    assert dataset.delta_indices == {"actions": [0, 1]}
    assert native.calls[0][3:] == (20, 1e-3)
    np.testing.assert_array_equal(dataset.episode_data_index["from"].numpy(), [0, 2])
    np.testing.assert_array_equal(dataset.episode_data_index["to"].numpy(), [2, 4])
    assert native.validator(*native.calls[0]) is True


@pytest.mark.parametrize("fps,tolerance", [(60, 1e-4), (20, 1e-6)])
def test_incorrect_fps_or_out_of_tolerance_interval_remains_rejected(native, fps, tolerance):
    native.state["fps"] = fps
    native.state["columns"]["timestamp"] = [0.0, 0.05002, 0.0, 0.05002]
    with pytest.raises(ValueError, match="timestamps unexpectedly violate"):
        construct(native, tolerance_s=tolerance)
    assert native.calls[0][3:] == (fps, tolerance)


@pytest.mark.parametrize("empty_columns", [("timestamp",), ("episode_index",), ("timestamp", "episode_index")])
def test_empty_columns_cannot_silently_pass_after_replacing_scalar_stack(native, empty_columns):
    for name in empty_columns:
        native.state["columns"][name] = []
    with pytest.raises(ValueError, match="columns must not be empty"):
        construct(native)
    assert native.calls == []

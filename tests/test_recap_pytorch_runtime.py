import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch

from openpi.training import optimizer, recap_pytorch
from openpi.training.recap_advantages import label_value_advantages


class _Batches(list):
    sampler = None


class _TinyEstimatorOrPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))

    def gradient_checkpointing_enable(self):
        pass

    def forward(self, observation, target_or_current, target=None, actions=None):
        if actions is not None:
            assert actions.shape == (2, 1, 1)
            state = target_or_current.state - observation.state
        else:
            state = observation.state
            target = target_or_current
        return (self.weight * state.reshape(target.shape) - target).square()


@pytest.mark.parametrize(
    ("stage", "val_ratio"), [("value", 0.2), ("stage_progress", 0.0), ("stage_progress", 0.2), ("offline_rl", 0.0)],
)
def test_native_cpu_step_saves_safetensors_and_checkpoint_marker(stage, val_ratio, tmp_path, monkeypatch, capsys):
    base = tmp_path / "base"
    base.mkdir()
    safetensors.torch.save_model(_TinyEstimatorOrPolicy(), str(base / "model.safetensors"))
    path = tmp_path / "run"
    path.mkdir()
    config = SimpleNamespace(model=SimpleNamespace(), recap_training_stage=stage, checkpoint_dir=path,
        pytorch_weight_path=str(base),
        pytorch_training_precision="float32", optimizer=optimizer.AdamW(), lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=0, peak_lr=0.1, decay_steps=10, decay_lr=0.01), num_train_steps=1, log_interval=1,
        save_interval=1, eval_interval=1, eval_batches=1, val_ratio=val_ratio, seed=42,
        keep_period=None, wandb_enabled=False)
    sample = {"image": {}, "image_mask": {}, "state": torch.ones(2, 1),
              "actions": torch.zeros(2, 1, 1), "value_target": torch.zeros(2)}
    pair = {"reference": {**sample, "state": torch.zeros(2, 1)}, "current": sample,
            "target": torch.zeros(2), "actions": sample["actions"]}
    batches = _Batches([pair if stage == "stage_progress" else sample])
    monkeypatch.setattr(recap_pytorch, "_training_loaders", lambda *_: (batches, batches if val_ratio else None, []))
    monkeypatch.setattr(recap_pytorch.recap, "create_recap_model", lambda *_: _TinyEstimatorOrPolicy())
    monkeypatch.setattr(recap_pytorch.recap, "configure_recap_trainable", lambda *_: None)

    recap_pytorch._train(config, 0, 1, torch.device("cpu"))

    checkpoint = path / "0"
    assert safetensors.torch.load_file(str(checkpoint / "model.safetensors"))["weight"].item() < 0.5
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "assets").is_dir()
    assert json.loads((checkpoint / "recap_checkpoint.json").read_text()) == {
        "schema_version": 1, "backend": "pytorch", "stage": stage, "step": 0,
    }
    metrics = json.loads(capsys.readouterr().out.split("VLA_EVENT:", 1)[1])["data"]
    assert metrics["step"] == 0 and metrics["loss"] > 0
    assert ("val_loss" in metrics) == bool(val_ratio)


def test_stage_progress_validation_holds_out_complete_episodes(monkeypatch):
    from openpi.training import stage_progress

    episodes = (np.arange(2), np.arange(2, 5), np.arange(5, 9), np.arange(9, 10))
    source = SimpleNamespace(episodes=episodes, data_config=object(), progress=np.linspace(0., 1., 10),
        observation=int, observation_with_actions=lambda row: (int(row), [int(row)]))

    def open_source(config, *, skip_norm_stats):
        assert skip_norm_stats is True
        return source

    monkeypatch.setattr(stage_progress, "StageProgressSource", open_source)
    config = SimpleNamespace(recap_training_stage="stage_progress", val_ratio=0.34, seed=42)
    train, validation, data, weights = recap_pytorch._training_datasets(config, 0)
    other_train, other_validation, _, _ = recap_pytorch._training_datasets(config, 1)
    assert data == [source.data_config] and weights is None
    assert other_validation is None
    assert np.array_equal(other_train._rows, train._rows)
    assert set(train._rows).isdisjoint(validation._rows)
    assert set(train._rows) | set(validation._rows) == set(range(9))
    assert sum(set(episode) == set(validation._rows) for episode in episodes) == 1
    for index in range(len(validation)):
        pair = validation[index]
        assert pair == validation[index]
        assert pair["reference"] != pair["current"]
        assert pair["reference"] in validation._rows and pair["current"] in validation._rows
        assert pair["target"] == source.progress[pair["current"]] - source.progress[pair["reference"]]

    train, validation, _, _ = recap_pytorch._training_datasets(
        SimpleNamespace(recap_training_stage="stage_progress", val_ratio=0.0), 0
    )
    assert set(train._rows) == set(range(9)) and validation is None
    source.episodes = (episodes[0], episodes[-1])
    with pytest.raises(ValueError, match="至少需要两条各含两帧"):
        recap_pytorch._training_datasets(config, 0)


def test_stage_progress_training_is_independent_of_annotation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "existing_progress" / "model.safetensors"
    checkpoint.parent.mkdir()
    safetensors.torch.save_model(_TinyEstimatorOrPolicy(), str(checkpoint))
    initial_bytes = checkpoint.read_bytes()
    config = SimpleNamespace(recap_training_stage="stage_progress", checkpoint_dir=tmp_path / "new_run",
                             pytorch_weight_path=checkpoint, resume=False, resume_checkpoint=None, overwrite=False,
                             ema_decay=None, batch_size=1, num_train_steps=1)
    trained = []
    monkeypatch.setattr(recap_pytorch.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(recap_pytorch, "prepare_training_normalization", lambda cfg: cfg)
    monkeypatch.setattr(recap_pytorch, "_train_rank", lambda rank, world_size, cfg: trained.append(cfg))

    recap_pytorch.run(config)

    assert trained == [config]
    assert config.checkpoint_dir.is_dir()
    assert checkpoint.read_bytes() == initial_bytes
    assert not (tmp_path / "stage_advantage_dataset").exists()


def test_standalone_annotation_requires_a_trained_progress_checkpoint(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/acp/lerobot_stage_progress_infer_pytorch.py"
    resolve_checkpoint = runpy.run_path(str(script))["_stage_progress_checkpoint"]
    checkpoint = tmp_path / "checkpoint" / "model.safetensors"
    checkpoint.parent.mkdir()
    safetensors.torch.save_model(_TinyEstimatorOrPolicy(), str(checkpoint))
    with pytest.raises(ValueError, match="recap_checkpoint.json"):
        resolve_checkpoint(checkpoint)
    marker = checkpoint.parent / "recap_checkpoint.json"
    receipt = {"schema_version": 1, "backend": "pytorch", "stage": "value", "step": 42}
    marker.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="有效的 PyTorch Kai0"):
        resolve_checkpoint(checkpoint.parent)
    marker.write_text(json.dumps({**receipt, "stage": "stage_progress"}))
    assert resolve_checkpoint(checkpoint) == checkpoint.parent


def test_standalone_annotation_publishes_using_the_selected_checkpoint(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from openpi import training

    checkpoint = tmp_path / "progress_model"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    safetensors.torch.save_model(_TinyEstimatorOrPolicy(), str(weights))
    receipt = {"schema_version": 1, "backend": "pytorch", "stage": "stage_progress", "step": 42}
    (checkpoint / "recap_checkpoint.json").write_text(json.dumps(receipt))
    source = tmp_path / "source"
    parquet = source / "data/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    (source / "meta").mkdir()
    (source / "meta/info.json").write_text('{"features": {}}')
    pq.write_table(pa.table({"index": [17, 42], "task_index": [0, 0], "stage_progress_gt": [0.0, 1.0]}), parquet)
    config = SimpleNamespace(pytorch_weight_path=weights, recap_training_stage="stage_annotation", batch_size=2,
        seed=1, data=SimpleNamespace(repo_id=source), model=SimpleNamespace(n_step=1, stage_count=1, positive_ratio=0.5,
                                                                           image_keys=(), depth_keys=()))
    dataset = SimpleNamespace(root=source, indices=np.array([17, 42]), episodes=(np.arange(2),),
                              progress=np.array([0.0, 1.0]), observation=torch.tensor)
    monkeypatch.setattr(training, "stage_progress", SimpleNamespace(StageProgressSource=lambda cfg, **_: dataset), raising=False)
    monkeypatch.setattr(training, "data_loader", SimpleNamespace(_collate_fn=lambda rows, **_: torch.stack(rows)), raising=False)
    monkeypatch.setattr(recap_pytorch, "torch_observation", lambda rows, device: rows.to(device))

    def load_scorer(cfg, path, stage, device):
        assert cfg is config and path == checkpoint and stage == "stage_progress" and device.type == "cpu"
        return SimpleNamespace(predict_progress=lambda reference, current: (current - reference).float())

    monkeypatch.setattr(recap_pytorch, "load_scoring_model", load_scorer)
    output = tmp_path / "annotation"
    script = Path(__file__).resolve().parents[1] / "scripts/acp/lerobot_stage_progress_infer_pytorch.py"
    published = runpy.run_path(str(script))["main"](config, output_dir=output, device="cpu")

    assert published == output / "stage_advantage_dataset"
    assert pq.read_table(published / parquet.relative_to(source))["is_positive"].to_pylist() == [1, 0]
    assert "is_positive" not in pq.read_schema(parquet).names


@pytest.mark.parametrize("multi_factory", [False, True])
def test_value_annotation_restores_selected_scorer_and_publishes_derived_parquet(tmp_path, monkeypatch, multi_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from openpi import training

    class ValueScorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value_fc1 = torch.nn.Linear(1, 1, bias=False)

        def predict_value(self, observation):
            return self.value_fc1(observation.state)

    class RawFrames(dict):
        @property
        def column_names(self):
            return self.keys()

        def with_format(self, kind):
            assert kind is None
            return self

    checkpoint = tmp_path / "value_checkpoint"
    checkpoint.mkdir()
    scorer = ValueScorer()
    scorer.value_fc1.weight.data.fill_(0.25)
    safetensors.torch.save_model(scorer, str(checkpoint / "model.safetensors"))
    marker = checkpoint / "recap_checkpoint.json"
    receipt = {"schema_version": 1, "backend": "pytorch", "stage": "stage_progress", "step": 42}
    marker.write_text(json.dumps(receipt))
    source = tmp_path / "source"
    parquet = source / "data/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    (source / "meta").mkdir()
    (source / "meta/info.json").write_text('{"features": {}}')
    raw = RawFrames(index=[17, 24, 42], episode_index=[0] * 3, frame_index=[0, 1, 2], task_index=[0] * 3,
                    value_target=[-1.0, -0.5, 0.0])
    pq.write_table(pa.table(raw), parquet)
    data = SimpleNamespace(repo_id=str(source), norm_stats={"bound": True})
    created = []

    def create_all(*_):
        created.append("multi")
        return [data]

    def create(*_):
        created.append("single")
        return data

    def apply_global_normalization(resolved, _):
        assert resolved is data
        created.append("norm")
        return resolved

    factory = (SimpleNamespace(datasets=[object()], create_all=create_all) if multi_factory
               else SimpleNamespace(create=create))
    config = SimpleNamespace(pytorch_weight_path=checkpoint, recap_training_stage="value_annotation", batch_size=2,
        num_workers=0, seed=1, pytorch_training_precision="bfloat16", assets_dirs=tmp_path,
        data=factory, model=SimpleNamespace(action_horizon=1, image_keys=(), depth_keys=()))
    transformed = [{"state": torch.tensor([state]), "image": {}, "image_mask": {}} for state in (-2.0, -1.0, 0.0)]
    loader = SimpleNamespace(
        _apply_global_normalization=apply_global_normalization, TransformedDataset=type("TransformedDataset", (), {}),
        create_torch_dataset=lambda *_: SimpleNamespace(hf_dataset=raw), transform_dataset=lambda *_: transformed,
        _collate_fn=lambda rows, **_: torch.utils.data.default_collate(rows))
    monkeypatch.setattr(training, "data_loader", loader, raising=False)
    monkeypatch.setattr(recap_pytorch.recap, "create_recap_model", lambda *_: ValueScorer())
    monkeypatch.setattr(recap_pytorch.torch.cuda, "is_available", lambda: False)
    script = Path(__file__).resolve().parents[1] / "scripts/acp/lerobot_value_infer_pytorch.py"
    run_annotation = runpy.run_path(str(script))["main"]
    output = tmp_path / "annotation"
    with pytest.raises(ValueError, match="有效的 PyTorch ReCap 价值模型"):
        run_annotation(config, output_dir=output)
    marker.write_text(json.dumps({**receipt, "stage": "value"}))
    if multi_factory:
        factory.datasets.append(object())
        with pytest.raises(ValueError, match="单个 LeRobot 数据集"):
            run_annotation(config, output_dir=output)
        factory.datasets.pop()

    published = run_annotation(config, output_dir=output, n_step=1, positive_ratio=0.34)

    assert created == ["multi" if multi_factory else "single", "norm"]
    assert published == output / "value_advantage_dataset"
    labeled = pq.read_table(published / parquet.relative_to(source))
    assert labeled["predicted_value"].to_pylist() == [-0.5, -0.25, 0.0]
    assert labeled["is_positive"].to_pylist() == [0, 0, 1]
    assert "is_positive" not in pq.read_schema(parquet).names
    evidence = json.loads((published / "meta/value_advantage.json").read_text())
    assert evidence["checkpoint"] == str(checkpoint) and evidence["frame_count"] == 3
    assert evidence["n_step"] == 1 and evidence["positive_ratio"] == 0.34 and evidence["positive_count"] == 1


def test_value_advantages_preserve_original_row_order_and_stop_at_episode_boundaries():
    values = np.array([-0.6, -0.8, -0.1, -0.7], dtype=np.float32)
    targets = np.array([-0.5, -0.5, 0.0, -0.9], dtype=np.float32)
    episodes = np.array([1, 0, 0, 1])
    frames = np.array([1, 0, 1, 0])
    tasks = np.array([4, 3, 3, 4])

    advantages, labels, thresholds = label_value_advantages(
        values, targets, episodes, frames, tasks, n_step=1, positive_ratio=0.5
    )

    np.testing.assert_allclose(advantages, [0.1, 0.2, 0.1, -0.3], atol=1e-7)
    np.testing.assert_array_equal(labels, [1, 1, 0, 0])
    assert set(thresholds) == {3, 4}


def test_rank_uses_rebound_worker_configuration(monkeypatch):
    original = SimpleNamespace(seed=42)
    rebound = SimpleNamespace(seed=123)
    seen = []
    monkeypatch.setattr(recap_pytorch, "_device", lambda rank: torch.device("cpu"))
    monkeypatch.setattr(recap_pytorch, "_train", lambda config, *_: seen.append(config))

    def worker_setup(config):
        assert config is original
        return rebound

    recap_pytorch._train_rank(0, 1, original, worker_setup=worker_setup)

    assert seen == [rebound]

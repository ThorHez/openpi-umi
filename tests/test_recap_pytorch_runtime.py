import json
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


@pytest.mark.parametrize("stage", ["value", "stage_progress", "offline_rl"])
def test_native_cpu_step_saves_safetensors_and_checkpoint_marker(stage, tmp_path, monkeypatch, capsys):
    base = tmp_path / "base"
    base.mkdir()
    safetensors.torch.save_model(_TinyEstimatorOrPolicy(), str(base / "model.safetensors"))
    path = tmp_path / "run"
    path.mkdir()
    config = SimpleNamespace(model=SimpleNamespace(), recap_training_stage=stage, checkpoint_dir=path,
        pytorch_weight_path=str(base),
        pytorch_training_precision="float32", optimizer=optimizer.AdamW(), lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=0, peak_lr=0.1, decay_steps=10, decay_lr=0.01), num_train_steps=1, log_interval=1,
        save_interval=1, eval_interval=1, eval_batches=1, val_ratio=0.2 if stage == "value" else 0,
        keep_period=None, wandb_enabled=False)
    sample = {"image": {}, "image_mask": {}, "state": torch.ones(2, 1),
              "actions": torch.zeros(2, 1, 1), "value_target": torch.zeros(2)}
    pair = {"reference": {**sample, "state": torch.zeros(2, 1)}, "current": sample,
            "target": torch.zeros(2), "actions": sample["actions"]}
    batches = _Batches([pair if stage == "stage_progress" else sample])
    monkeypatch.setattr(recap_pytorch, "_training_loaders", lambda *_: (batches, batches, []))
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
    assert ("val_loss" in metrics) == (stage == "value")


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

from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models_pytorch import recap


def test_two_hot_return_loss_interpolates_adjacent_bins():
    config = SimpleNamespace(num_value_bins=3, value_min=-1.0, value_max=0.0, soft_value_targets=True)
    logits = torch.tensor([[1.0, 2.0, 3.0]] * 3, requires_grad=True)
    probabilities = logits.log_softmax(dim=-1)
    loss = recap.distributional_value_loss(logits, torch.tensor([-1.0, -0.75, 0.0]), config)
    expected = torch.stack([-probabilities[0, 0], -probabilities[1, :2].mean(), -probabilities[2, 2]])
    torch.testing.assert_close(loss, expected)
    loss.mean().backward()
    assert torch.isfinite(logits.grad).all()


def test_acp_per_sample_mask_does_not_average_over_padding():
    errors = torch.tensor([[[1.0, 100.0], [3.0, 100.0]], [[100.0, 5.0], [100.0, 7.0]]])
    masks = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(recap._action_mean(errors, masks), torch.tensor([[1.0, 3.0], [5.0, 7.0]]))


def test_stage_progress_uses_recorded_actions_in_training_and_noise_in_inference(monkeypatch):
    def preprocess(self, observation, *, train):
        return [observation], [], None, None, torch.zeros((1, 1))

    def suffix(self, state, actions, time):
        return actions

    monkeypatch.setattr(recap.StageProgressEstimator, "_preprocess_observation", preprocess)
    monkeypatch.setattr(recap.StageProgressEstimator, "embed_prefix", lambda self, images, *_: images)
    monkeypatch.setattr(recap.StageProgressEstimator, "embed_suffix", suffix)
    monkeypatch.setattr(recap.StageProgressEstimator, "sample_noise", lambda self, shape, device: torch.ones(shape))
    monkeypatch.setattr(
        recap.StageProgressEstimator, "sample_time", lambda self, batch, device: torch.full((batch,), 0.25),
    )
    monkeypatch.setattr(recap.StageProgressEstimator, "_joint_suffix", lambda self, prefix, suffix: suffix)
    estimator = recap.StageProgressEstimator.__new__(recap.StageProgressEstimator)
    nn.Module.__init__(estimator)
    estimator.config = SimpleNamespace(action_horizon=1, action_dim=1)
    estimator.value_head = nn.Identity()

    target = torch.zeros(1)
    loss = estimator("reference", "current", target, torch.zeros((1, 1, 1)))
    torch.testing.assert_close(loss, torch.tensor([0.25**2]))
    torch.testing.assert_close(estimator.predict_progress("reference", "current"), torch.ones(1))


def test_safetensors_can_initialize_new_head_but_scoring_requires_a_trained_head(tmp_path):
    class TinyModel(nn.Module):
        def __init__(self, *, head):
            super().__init__()
            self.action_in_proj = nn.Linear(2, 2)
            if head:
                self.value_head = nn.Linear(2, 1)

    base = TinyModel(head=False)
    base_path = tmp_path / "base.safetensors"
    safetensors.torch.save_model(base, base_path)
    estimator = TinyModel(head=True)
    recap.load_recap_weights(estimator, base_path, "stage_progress")
    torch.testing.assert_close(estimator.action_in_proj.weight, base.action_in_proj.weight)
    with pytest.raises(ValueError, match="Missing keys"):
        recap.load_recap_weights(estimator, base_path, "stage_progress", require_trained_head=True)

    trained_path = tmp_path / "model.safetensors"
    safetensors.torch.save_model(estimator, trained_path)
    recap.load_recap_weights(TinyModel(head=True), tmp_path, "stage_progress", require_trained_head=True)
    with pytest.raises(ValueError, match="Unexpected keys"):
        recap.load_recap_weights(base, trained_path, "offline_rl")

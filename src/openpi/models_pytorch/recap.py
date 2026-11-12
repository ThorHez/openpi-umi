"""PyTorch estimators and advantage-conditioned policies for ReCap."""

import dataclasses
import math
from pathlib import Path

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models import gemma
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


class ReturnValueEstimator(PI0Pytorch):
    """Predict the return distribution from a single visual-language observation."""

    def __init__(self, config):
        super().__init__(config)
        width = gemma.get_config(config.paligemma_variant).width
        self.final_norm = nn.LayerNorm(width, eps=1e-6)
        self.value_fc1 = nn.Linear(width, width)
        self.value_dropout = nn.Dropout(config.value_head_dropout)
        self.value_fc2 = nn.Linear(width, config.num_value_bins)
        self.register_buffer(
            "value_centers", torch.linspace(config.value_min, config.value_max, config.num_value_bins), persistent=False,
        )

    def value_logits(self, observation):
        images, masks, tokens, token_mask, _ = self._preprocess_observation(observation, train=self.training)
        prefix, valid, attention = self.embed_prefix(images, masks, tokens, token_mask)
        dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        (hidden, _), _ = self.paligemma_with_expert(
            attention_mask=self._prepare_attention_masks_4d(make_att_2d_masks(valid, attention)),
            position_ids=torch.cumsum(valid, dim=1) - 1,
            inputs_embeds=[prefix.to(dtype=dtype), None],
            use_cache=False,
        )
        valid = valid.unsqueeze(-1)
        pooled = (hidden.float() * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        hidden = F.gelu(self.value_fc1(self.final_norm(pooled)), approximate="tanh")
        return self.value_fc2(self.value_dropout(hidden))

    def predict_value(self, observation):
        probs = self.value_logits(observation).float().softmax(dim=-1)
        return (probs * self.value_centers).sum(dim=-1)

    def forward(self, observation, targets):
        logits = self.value_logits(observation).float()
        targets = _scalar_targets(targets, logits.shape[0]).float()
        return distributional_value_loss(logits, targets, self.config)


def distributional_value_loss(logits, targets, config):
    steps = config.num_value_bins - 1
    coordinates = ((targets - config.value_min) * (steps / (config.value_max - config.value_min))).clamp(0, steps)
    if not config.soft_value_targets:
        return F.cross_entropy(logits, coordinates.round().long(), reduction="none")
    lower = coordinates.floor().long()
    upper = (lower + 1).clamp_max(steps)
    upper_weight = coordinates - lower
    log_probs = logits.log_softmax(dim=-1)
    lower_log_prob = log_probs.gather(1, lower[:, None]).squeeze(1)
    upper_log_prob = log_probs.gather(1, upper[:, None]).squeeze(1)
    return -(1 - upper_weight) * lower_log_prob - upper_weight * upper_log_prob


class StageProgressEstimator(PI0Pytorch):
    """Kai0's paired stage-progress estimator with its native value-head names."""

    def __init__(self, config):
        super().__init__(config)
        width = gemma.get_config(config.action_expert_variant).width
        self.value_head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width),
                                        nn.SiLU(), nn.Linear(width, 1), nn.Tanh())

    def _predict(self, reference, current, actions):
        ref_images, ref_masks, _, _, _ = self._preprocess_observation(reference, train=False)
        cur_images, cur_masks, tokens, token_mask, state = self._preprocess_observation(current, train=False)
        prefix = self.embed_prefix(ref_images + cur_images, ref_masks + cur_masks, tokens, token_mask)
        shape = (state.shape[0], self.config.action_horizon, self.config.action_dim)
        noise = self.sample_noise(shape, state.device)
        time = self.sample_time(state.shape[0], state.device)
        if actions is not None:
            if actions.shape != shape:
                raise ValueError(f"Kai0 current actions must have shape {shape}, got {tuple(actions.shape)}")
            weight = time[:, None, None]
            noise = weight * noise + (1 - weight) * actions
        suffix = self.embed_suffix(state, noise, time)
        hidden = self._joint_suffix(prefix, suffix)[:, 0].float()
        return self.value_head(hidden).squeeze(-1)

    def predict_progress(self, reference, current):
        return self._predict(reference, current, None)

    def forward(self, reference, current, targets, actions):
        prediction = self._predict(reference, current, actions)
        targets = _scalar_targets(targets, prediction.shape[0]).float().clamp(-1, 1)
        return F.mse_loss(prediction, targets, reduction="none")


def _scalar_targets(targets, batch_size):
    if targets.shape not in {(batch_size,), (batch_size, 1)}:
        raise ValueError(f"Expected {batch_size} scalar supervision targets, got {tuple(targets.shape)}")
    return targets.reshape(batch_size)


def _action_mean(loss, mask):
    if mask is None:
        return loss.mean(dim=-1)
    mask = torch.as_tensor(mask, dtype=loss.dtype, device=loss.device)
    if mask.ndim not in (1, 2) or mask.shape[-1] != loss.shape[-1] or (
            mask.ndim == 2 and mask.shape[0] != loss.shape[0]):
        raise ValueError("ACP action loss mask must match the action dimension and optional batch dimension")
    mask = mask.unsqueeze(-2)
    return (loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1e-8)


class RecapPolicy(PI0Pytorch):
    """Flow matching with optional per-dataset action and binary-gripper losses."""

    def __init__(self, config):
        from openpi.models.pi0_gripper import infer_gripper_indices
        from openpi.models.pi0_gripper import Pi0GripperConfig

        super().__init__(config)
        self.gripper_binary_indices = ()
        self.gripper_binary_head = None
        if isinstance(config, Pi0GripperConfig):
            self.gripper_binary_indices = infer_gripper_indices(config.action_dim, config.gripper_binary_indices)
            if len(set(self.gripper_binary_indices)) != len(self.gripper_binary_indices) or any(
                    index < 0 or index >= config.action_dim for index in self.gripper_binary_indices):
                raise ValueError("ACP gripper indices must be unique and inside the action dimension")
            if self.gripper_binary_indices:
                self.gripper_binary_head = nn.Linear(self.action_out_proj.in_features, len(self.gripper_binary_indices))
            midpoint = (config.gripper_binary_close_value + config.gripper_binary_open_value) / 2
            threshold = config.gripper_binary_threshold
            self.gripper_threshold = midpoint if threshold is None else threshold

    def forward(self, observation, actions, noise=None, time=None):
        hidden, velocity = self._flow_matching_features(observation, actions, noise, time, train=self.training)
        predicted = self._apply_checkpoint(self.action_out_proj, hidden)
        mask = observation.action_loss_mask
        if mask is None:
            mask = self.config.action_loss_mask
        loss = _action_mean(F.mse_loss(predicted, velocity, reduction="none"), mask)
        if self.gripper_binary_head is not None:
            indices = list(self.gripper_binary_indices)
            logits = self.gripper_binary_head(hidden)
            labels = (actions[..., indices] <= self.gripper_threshold).to(dtype=logits.dtype)
            gripper_mask = None if mask is None else torch.as_tensor(mask, device=actions.device)[..., indices]
            gripper_loss = _action_mean(F.binary_cross_entropy_with_logits(logits, labels, reduction="none"),
                                        gripper_mask)
            loss = loss + self.config.gripper_binary_loss_weight * gripper_loss
        return loss

    @torch.no_grad()
    def _decode_actions(self, state, prefix_pad_masks, past_key_values, actions):
        if self.gripper_binary_head is None:
            return actions
        time = actions.new_full((actions.shape[0],), 1e-3)
        hidden = self._denoise_features(state, prefix_pad_masks, past_key_values, actions, time)
        logits = self.gripper_binary_head(hidden)
        actions = actions.clone()
        actions[..., list(self.gripper_binary_indices)] = torch.where(
            logits > 0, self.config.gripper_binary_close_value, self.config.gripper_binary_open_value,
        ).to(actions.dtype)
        return actions


def create_recap_model(model_config, stage, precision):
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.pi0_gripper import Pi0GripperConfig
    from openpi.models.pi0_stage_progress import Pi0StageProgressConfig
    from openpi.models.pi0_value import Pi0ValueConfig

    models = {
        "value": ((Pi0ValueConfig,), ReturnValueEstimator),
        "stage_progress": ((Pi0StageProgressConfig,), StageProgressEstimator),
        "offline_rl": ((Pi0Config, Pi0GripperConfig), RecapPolicy),
    }
    if stage not in models or type(model_config) not in models[stage][0]:
        raise ValueError(f"PyTorch ReCap stage {stage!r} cannot use {type(model_config).__name__}")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError("PyTorch ReCap precision must be float32 or bfloat16")
    if any("lora" in variant for variant in (model_config.paligemma_variant, model_config.action_expert_variant)):
        raise ValueError("PyTorch ReCap requires standard Gemma weights")
    if stage == "value" and (model_config.num_value_bins < 2
                             or not all(map(math.isfinite, (model_config.value_min, model_config.value_max)))
                             or model_config.value_min >= model_config.value_max):
        raise ValueError("The PyTorch return model requires at least two ordered value bins")
    return models[stage][1](dataclasses.replace(model_config, dtype=precision))


def configure_recap_trainable(model, train_config):
    """Map the supported recipe freeze rules onto native Torch parameters."""
    from flax import nnx

    freeze = train_config.freeze_filter
    if freeze != nnx.Nothing():
        if (not isinstance(model, ReturnValueEstimator)
                or freeze != train_config.model.get_freeze_filter_value_head_only()):
            raise ValueError("PyTorch ReCap supports full fine-tuning and the standard return value head-only filter")
        model.requires_grad_(False)
        for layer in (model.final_norm, model.value_fc1, model.value_fc2):
            layer.requires_grad_(True)
    elif isinstance(model, ReturnValueEstimator):
        model.paligemma_with_expert.gemma_expert.requires_grad_(False)
        for name in ("action_in_proj", "action_out_proj", "state_proj", "time_mlp_in", "time_mlp_out",
                     "action_time_mlp_in", "action_time_mlp_out"):
            layer = getattr(model, name, None)
            if layer is not None:
                layer.requires_grad_(False)
    elif isinstance(model, StageProgressEstimator):
        model.action_out_proj.requires_grad_(False)


def load_recap_weights(model, path, stage, *, require_trained_head=False):
    """Load a compatible policy base or a complete native scorer checkpoint."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    head_prefixes = {
        "value": ("final_norm.", "value_fc1.", "value_fc2."),
        "stage_progress": ("value_head.",),
        "offline_rl": ("gripper_binary_head.",),
    }[stage]
    optional = {key for key in model.state_dict() if key.startswith(head_prefixes)}
    missing, unexpected = safetensors.torch.load_model(model, str(path), strict=False)
    missing = set(missing)
    if unexpected or (missing and (require_trained_head or missing != optional)):
        raise ValueError(f"Incompatible PyTorch ReCap {stage} checkpoint. "
                         f"Missing keys: {sorted(missing)[:12]}. Unexpected keys: {sorted(unexpected)[:12]}.")

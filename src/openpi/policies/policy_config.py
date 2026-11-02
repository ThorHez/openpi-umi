import dataclasses
import logging
import os
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config
import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    client_rtc: bool = False,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        client_rtc: Use kai0's client-managed RTC sampler with an ordinary Pi0/Pi05 JAX checkpoint.

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    if client_rtc:
        # Reject derived architectures such as MEM/value rather than dropping their fields.
        if type(train_config.model) not in (pi0_config.Pi0Config, pi0_config.Pi0RTCConfig):
            raise ValueError("Client-managed kai0 RTC supports only ordinary Pi0/Pi05 JAX models")
        rtc_model_config = pi0_config.Pi0RTCConfig(
            **{
                field.name: getattr(train_config.model, field.name)
                for field in dataclasses.fields(pi0_config.Pi0Config)
                if field.init
            }
        )
        train_config = dataclasses.replace(train_config, model=rtc_model_config)
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)
    if is_pytorch and isinstance(train_config.model, pi0_config.Pi0RTCConfig):
        raise ValueError("Client-managed kai0 RTC requires a JAX checkpoint with params/, not model.safetensors")

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    # Keep inference tokenization identical to the training data loader.  The
    # loader injects ``robot_type`` into tokenization transforms at runtime;
    # without doing the same here, policies configured with a robot tag see a
    # different prompt at serving / evaluation time than they saw in training.
    if data_config.robot_type is not None:
        data_config = _config._set_robot_type(data_config, data_config.robot_type)  # noqa: SLF001
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _normalize.load(checkpoint_dir / "assets" / data_config.asset_id)
    if validator := getattr(train_config.data, "validate_norm_stats", None):
        validator(norm_stats)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    action_normalize = transforms.Normalize(
        norm_stats,
        use_quantiles=data_config.use_quantile_norm,
        key_masks=getattr(data_config, "normalize_masks", None),
    )
    # Guide in the *current* observation frame, using the same action path as
    # training (including delta/robot conversions), before model-only padding.
    action_inputs = transforms.compose(
        [
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            action_normalize,
        ]
    )
    repack_inputs = transforms.compose(repack_transforms.inputs)

    def rtc_action_transform(obs: dict, actions: np.ndarray) -> np.ndarray:
        inputs = repack_inputs(jax.tree.map(lambda x: x.copy() if isinstance(x, np.ndarray) else x, obs))
        valid_horizon = len(actions)
        # Fixed-horizon training adapters may assert H rows. Extend with the
        # final valid row only for preprocessing, then remove those rows again.
        inputs["actions"] = np.concatenate(
            [
                actions,
                np.repeat(actions[-1:], model.action_horizon - valid_horizon, axis=0),
            ]
        )
        return np.asarray(action_inputs(inputs)["actions"])[:valid_horizon]

    metadata = dict(train_config.policy_metadata or {})
    if isinstance(train_config.model, pi0_config.Pi0RTCConfig):
        metadata["rtc"] = {
            **metadata.get("rtc", {}),
            "enabled": True,
            "mode": "client",
            "model_actions_normalized": True,
        }
    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            action_normalize,
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                key_masks=getattr(data_config, "normalize_masks", None),
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=metadata,
        rtc_action_transform=rtc_action_transform,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )

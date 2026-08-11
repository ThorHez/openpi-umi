from collections.abc import Sequence
import inspect
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)
        # Inspect the class method because torch.compile may replace the bound
        # method with a generic (*args, **kwargs) wrapper.
        self._supports_rtc = "rtc_actions" in inspect.signature(type(model).sample_actions).parameters
        self._last_model_actions: np.ndarray | None = None

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        rtc_actions: np.ndarray | None = None,
        rtc_mask: np.ndarray | None = None,
        rtc_guidance_weight: float = 5.0,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if (rtc_actions is None) != (rtc_mask is None):
            raise ValueError("rtc_actions and rtc_mask must be provided together")
        if rtc_actions is not None:
            if not self._supports_rtc:
                raise ValueError("This model does not support RTC guided inference")
            rtc_actions_array = np.asarray(rtc_actions)
            rtc_mask_array = np.asarray(rtc_mask)
            if rtc_actions_array.ndim == 2:
                rtc_actions_array = rtc_actions_array[None, ...]
            if rtc_mask_array.ndim == 1:
                rtc_mask_array = rtc_mask_array[None, :, None]
            if self._is_pytorch_model:
                sample_kwargs["rtc_actions"] = torch.from_numpy(rtc_actions_array).to(self._pytorch_device)
                sample_kwargs["rtc_mask"] = torch.from_numpy(rtc_mask_array).to(self._pytorch_device)
            else:
                sample_kwargs["rtc_actions"] = jnp.asarray(rtc_actions_array)
                sample_kwargs["rtc_mask"] = jnp.asarray(rtc_mask_array)
            sample_kwargs["rtc_guidance_weight"] = rtc_guidance_weight

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        # 保留原有信息
        outputs = inputs.copy()
        outputs["actions"] = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        # outputs = {
        #     "state": inputs["state"],
        #     "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        # }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        self._last_model_actions = np.asarray(outputs["actions"]).copy()
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def supports_rtc(self) -> bool:
        return self._supports_rtc

    @property
    def action_horizon(self) -> int:
        return self._model.action_horizon if not self._is_pytorch_model else self._model.config.action_horizon

    @property
    def last_model_actions(self) -> np.ndarray | None:
        return self._last_model_actions

    def warmup_rtc(
        self,
        obs: dict,
        *,
        rtc_actions: np.ndarray,
        rtc_mask: np.ndarray,
        rtc_guidance_weight: float,
    ) -> None:
        """Compile/warm the guided sampling path without replacing A_init."""
        previous_actions = self._last_model_actions
        try:
            self.infer(
                obs,
                rtc_actions=rtc_actions,
                rtc_mask=rtc_mask,
                rtc_guidance_weight=rtc_guidance_weight,
            )
        finally:
            self._last_model_actions = previous_actions

    @override
    def reset(self) -> None:
        self._last_model_actions = None


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()

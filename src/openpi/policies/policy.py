from collections.abc import Callable, Sequence
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

_CLIENT_RTC_FIELDS = frozenset(
    {
        "prev_action_chunk",
        "inference_delay",
        "execute_horizon",
        "enable_rtc",
        "mask_prefix_delay",
        "prefix_attention_schedule",
        "max_guidance_weight",
        "prev_action_chunk_normalized",
    }
)


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
        rtc_action_transform: Callable[[dict, np.ndarray], np.ndarray] | None = None,
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
            rtc_action_transform: Convert physical previous actions using the checkpoint's input action transforms.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._rtc_action_transform = rtc_action_transform
        # Inspect the class method because torch.compile may wrap the bound method.
        sample_parameters = inspect.signature(type(model).sample_actions).parameters
        self._supports_rtc = {"rtc_actions", "rtc_mask", "rtc_guidance_weight"}.issubset(sample_parameters)
        self._supports_client_rtc = {"prev_action_chunk", "inference_delay", "execute_horizon"}.issubset(
            sample_parameters
        )
        self._fixed_shape_client_rtc = "prev_action_dim" in sample_parameters

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = (
                nnx_utils.module_jit(model.sample_actions, static_argnames=("prefix_attention_schedule",))
                if self._supports_client_rtc
                else nnx_utils.module_jit(model.sample_actions)
            )
            compute_history_classification = getattr(model, "compute_history_classification", None)
            self._compute_history_classification = (
                nnx_utils.module_jit(compute_history_classification)
                if compute_history_classification is not None
                else None
            )
            self._rng = rng or jax.random.key(0)
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
        client_rtc = {key: obs[key] for key in _CLIENT_RTC_FIELDS if key in obs}
        if client_rtc and (rtc_actions is not None or rtc_mask is not None):
            raise ValueError("Client-managed and server-managed RTC guidance cannot be combined")
        if client_rtc and not self._supports_client_rtc:
            raise ValueError("This model does not support client-managed RTC; enable --rtc.enabled --rtc.mode client")
        obs = {key: value for key, value in obs.items() if key not in _CLIENT_RTC_FIELDS}
        client_kwargs = (
            self._prepare_client_rtc(obs, client_rtc)
            if client_rtc or (self._supports_client_rtc and self._fixed_shape_client_rtc)
            else {}
        )
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
        sample_kwargs.update(client_kwargs)
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
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        self._last_model_actions = np.asarray(outputs["actions"]).copy()
        outputs = self._output_transform(outputs)
        if self._supports_client_rtc:
            # Feed this buffer back with prev_action_chunk_normalized=True.
            # Keep it separate from physical output transforms and caller mutation.
            outputs["model_actions"] = self._last_model_actions.copy()
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _prepare_client_rtc(self, obs: dict, fields: dict) -> dict:
        """Validate kai0's wire fields outside JIT and keep action spaces explicit."""
        fields = {
            **{
                key: value
                for key, value in self._sample_kwargs.items()
                if key in _CLIENT_RTC_FIELDS - {"prev_action_chunk", "prev_action_chunk_normalized"}
            },
            **fields,
        }
        kwargs = {key: value for key, value in fields.items() if key != "prev_action_chunk_normalized"}
        for key in ("enable_rtc", "mask_prefix_delay", "prev_action_chunk_normalized"):
            if key in fields and not isinstance(fields[key], (bool, np.bool_)):
                raise ValueError(f"{key} must be a boolean")
        schedule = fields.get("prefix_attention_schedule", "exp")
        if not isinstance(schedule, str) or schedule not in ("ones", "zeros", "linear", "exp"):
            raise ValueError("prefix_attention_schedule must be ones, zeros, linear, or exp")
        weight = fields.get("max_guidance_weight", 0.5)
        if (
            isinstance(weight, (bool, np.bool_))
            or not isinstance(weight, (int, float, np.integer, np.floating))
            or not np.isfinite(weight)
            or weight < 0
        ):
            raise ValueError("max_guidance_weight must be finite and nonnegative")
        kwargs["max_guidance_weight"] = float(weight)

        previous = fields.get("prev_action_chunk")
        end_limit = self.action_horizon
        if previous is not None:
            previous = np.asarray(previous, dtype=np.float32)
            if previous.ndim == 3 and previous.shape[0] == 1:
                previous = previous[0]
            if (
                previous.ndim != 2
                or not 0 < previous.shape[0] <= self.action_horizon
                or not 0 < previous.shape[1] <= self._model.action_dim
                or not np.all(np.isfinite(previous))
            ):
                raise ValueError("prev_action_chunk must be finite with shape (prefix_horizon <= H, action_dim <= D)")
            end_limit = previous.shape[0]
            if not fields.get("prev_action_chunk_normalized", False):
                if self._rtc_action_transform is None:
                    raise ValueError(
                        "Physical prev_action_chunk requires an action transform; use create_trained_policy or "
                        "feed model_actions back with prev_action_chunk_normalized=True"
                    )
                previous = np.asarray(self._rtc_action_transform(obs, previous.copy()), dtype=np.float32)
                if (
                    previous.ndim != 2
                    or previous.shape[0] != end_limit
                    or not 0 < previous.shape[1] <= self._model.action_dim
                    or not np.all(np.isfinite(previous))
                ):
                    raise ValueError(
                        "RTC action transform must retain the prefix horizon and produce finite model actions"
                    )
            if not self._fixed_shape_client_rtc:
                kwargs["prev_action_chunk"] = jnp.asarray(previous[None])

        end = fields.get("execute_horizon", end_limit)
        delay = fields.get("inference_delay", 0)
        for key, value in (("execute_horizon", end), ("inference_delay", delay)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{key} must be an integer")
        if not 0 <= delay <= end <= end_limit or end == 0:
            raise ValueError("RTC requires 0 <= inference_delay <= execute_horizon <= available prefix horizon")
        kwargs.update(inference_delay=int(delay), execute_horizon=int(end))
        if self._fixed_shape_client_rtc:
            # A controller's remaining prefix shrinks on every request. Pad on
            # the host to avoid recompilation for each length or action width.
            # Zero valid dimensions on the first request also compiles the
            # guided branch before the controller starts executing any plan.
            valid_dim = previous.shape[-1] if previous is not None else 0
            if previous is None:
                previous = np.zeros((self.action_horizon, self._model.action_dim), dtype=np.float32)
            else:
                previous = np.pad(
                    previous,
                    ((0, self.action_horizon - len(previous)), (0, self._model.action_dim - valid_dim)),
                )
            kwargs.update(
                prev_action_chunk=jnp.asarray(previous[None]),
                prev_action_dim=valid_dim,
                enable_rtc=fields.get("enable_rtc", True),
                mask_prefix_delay=fields.get("mask_prefix_delay", False),
                prefix_attention_schedule=schedule,
            )
        return kwargs

    def infer_memory(self, obs: dict) -> dict:
        """Run an optional model-native history classifier on a policy observation.

        This deliberately uses the same input transforms as :meth:`infer`, so
        deployment diagnostics see exactly the images and preprocessing used by
        action inference.  Action-only models do not expose this method and
        fail explicitly instead of returning a proxy derived from trajectories.
        """
        if self._is_pytorch_model or self._compute_history_classification is None:
            raise NotImplementedError("This policy model does not expose a history classifier")

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, classifier_rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        logits, aux = self._compute_history_classification(
            classifier_rng,
            observation,
        )
        model_time = time.monotonic() - start_time
        outputs = {"joint_logits": logits, **aux}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs["policy_timing"] = {"memory_infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def supports_rtc(self) -> bool:
        return self._supports_rtc

    @property
    def supports_client_rtc(self) -> bool:
        return self._supports_client_rtc

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
        """Warm the guided sampling path while preserving the initial chunk."""
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

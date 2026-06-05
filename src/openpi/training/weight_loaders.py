import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoaderWithDiscreteHead(WeightLoader):
    """Loads weights from a checkpoint, gracefully handling missing discrete head weights.
    
    When loading from a checkpoint (e.g., pi05_base) into a model with discrete action head,
    the discrete head weights will be kept as randomly initialized.
    
    Compatible with:
      - pi05_base checkpoint -> Pi0Discrete model
      - Any trained checkpoint
    """
    
    params_path: str
    
    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Pattern matches: LoRA weights, discrete_head weights
        return _merge_params(loaded_params, params, missing_regex=r".*(lora|discrete_head).*")


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoaderIgnoreDiscreteHead(WeightLoader):
    """Loads weights from a hybrid-trained checkpoint for INFERENCE with standard Pi0 model.
    
    Use this loader when:
    - You trained with Pi0Discrete (hybrid training with discrete_head/FAST)
    - You want to run inference using standard Pi0 model (flow matching only)
    
    This loader will:
    - Load all weights needed by Pi0 model
    - Automatically ignore discrete_head weights (not needed for inference)
    - Ignore any LoRA weights if present
    
    Example usage:
        weight_loader = CheckpointWeightLoaderIgnoreDiscreteHead(
            params_path="./checkpoints/my_hybrid_experiment/30000/params"
        )
    """
    
    params_path: str
    
    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params_ignore_extra(loaded_params, params)


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoaderWithValueHead(WeightLoader):
    """Loads weights from a checkpoint, gracefully handling missing value head weights.

    When loading from a pretrained Pi0/Pi0.5 checkpoint into a Pi0Value model,
    the value head weights (final_norm, value_fc1, value_fc2, value_dropout)
    will be kept as randomly initialized.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex=r".*(lora|final_norm|value_fc1|value_fc2|value_dropout).*")


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoaderWithGripperHead(WeightLoader):
    """Loads weights from a checkpoint, gracefully handling missing binary gripper head weights."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex=r".*(lora|gripper_binary_head).*")


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoaderWithMemoryCompress(WeightLoader):
    """Loads weights from a checkpoint, gracefully handling Pi0MemCompress params.

    When loading from a pretrained Pi0/Pi0.5 SigLIP checkpoint (e.g.
    ``pi05_base``) into a Pi0MemCompress model that uses the
    :mod:`openpi.models.siglip_mem_compress` visual encoder, the following
    newly-introduced parameters do not exist in the checkpoint and must fall
    back to the model's random initialization rather than failing the
    pytree-structure check:

    - ``Transformer/HistoryResampler_0/...``: the learned history compressor
      (memory queries, cross-attention, mlp refinement layers, optional
      current-frame conditioning projection).
    - ``Transformer/encoderblock*/history_memory_gate_logit``: per-block
      sigmoid gate logit that controls how strongly current-frame tokens
      attend to compressed history.

    LoRA-style adapters are also allowed-missing for parity with
    :class:`CheckpointWeightLoader`.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(
            loaded_params,
            params,
            missing_regex=r".*(lora|HistoryResampler|history_memory_gate_logit).*",
        )


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoaderIgnoreGripperHead(WeightLoader):
    """Loads a gripper-head checkpoint into a standard Pi0/Pi0.5 model, ignoring the extra head."""

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params_ignore_extra(loaded_params, params)


def _merge_params_ignore_extra(loaded_params: at.Params, params: at.Params) -> at.Params:
    """Merges loaded parameters, ignoring extra weights not in the target model.
    
    This is useful for loading hybrid-trained checkpoints (with discrete_head)
    into standard Pi0 model for inference.
    
    Args:
        loaded_params: The parameters from checkpoint (may contain extra weights like discrete_head).
        params: The reference parameters (target model structure).
        
    Returns:
        Merged parameters matching the target model structure.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    
    result = {}
    ignored_keys = []
    missing_keys = []
    
    # Take all weights from checkpoint that exist in target model
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
        else:
            ignored_keys.append(k)
    
    # Check for missing weights (exist in target but not in checkpoint)
    for k in flat_ref:
        if k not in result:
            missing_keys.append(k)
            # Use random init from reference
            result[k] = flat_ref[k]
    
    # Log ignored weights (e.g., discrete_head)
    if ignored_keys:
        logger.info("Ignored %d weights from checkpoint (not needed for inference):", len(ignored_keys))
        discrete_head_keys = [k for k in ignored_keys if "discrete_head" in k]
        gripper_head_keys = [k for k in ignored_keys if "gripper_binary_head" in k]
        lora_keys = [k for k in ignored_keys if "lora" in k]
        other_keys = [k for k in ignored_keys if k not in discrete_head_keys and k not in gripper_head_keys and k not in lora_keys]
        
        if discrete_head_keys:
            logger.info("  - discrete_head weights: %d", len(discrete_head_keys))
        if gripper_head_keys:
            logger.info("  - gripper_binary_head weights: %d", len(gripper_head_keys))
        if lora_keys:
            logger.info("  - lora weights: %d", len(lora_keys))
        if other_keys:
            logger.info("  - other weights: %d", len(other_keys))
            for k in other_keys[:5]:
                logger.info("    - %s", k)
    
    # Warn about missing weights
    if missing_keys:
        logger.warning("Missing %d weights in checkpoint (using random init):", len(missing_keys))
        for k in missing_keys[:10]:
            logger.warning("  - %s", k)
        if len(missing_keys) > 10:
            logger.warning("  ... and %d more", len(missing_keys) - 10)
    
    return flax.traverse_util.unflatten_dict(result, sep="/")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")

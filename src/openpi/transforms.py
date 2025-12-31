from collections.abc import Callable, Mapping, Sequence
import dataclasses
from functools import partial
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

from openpi.utils.cv_util import generate_image_pipeline

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")

def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization for all keys (default behavior).
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False
    # Dict mapping key names to their norm_mask tuples.
    # Use make_bool_mask() to create masks easily.
    # Example: key_masks={"actions": make_bool_mask(3, -7), "state": make_bool_mask(3, -13)}
    # Keys not in this dict will be normalized without mask (all dimensions normalized).
    key_masks: dict[str, tuple[bool, ...]] | None = None
    # Internal field, will be populated in __post_init__
    key_methods: dict[str, Callable] | None = None

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)
        
        # Build key_methods from key_masks configuration
        methods = {}
        if self.key_masks:
            for key, mask in self.key_masks.items():
                if self.use_quantiles:
                    methods[key] = partial(self._normalize_quantile, norm_mask=mask)
                else:
                    methods[key] = partial(self._normalize, norm_mask=mask)
        
        # Use object.__setattr__ to bypass frozen dataclass restriction
        object.__setattr__(self, 'key_methods', methods if methods else None)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        default_fn = self._normalize_quantile if self.use_quantiles else self._normalize
        return apply_tree(
            data,
            self.norm_stats,
            self.key_methods,
            strict=self.strict,
            default_fn=default_fn,
        )

    def _normalize(self, x, stats: NormStats, norm_mask: tuple[bool, ...] | None = None):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        result = (x - mean) / (std + 1e-6)
        
        # Handle mask: True = normalize, False = keep original
        if norm_mask is not None:
            assert len(norm_mask) == x.shape[-1]
            result = result.copy() if isinstance(result, np.ndarray) else np.array(result)
            for dim in range(min(x.shape[-1], len(norm_mask))):
                if not norm_mask[dim]:  # False means skip normalization
                    result[..., dim] = x[..., dim]
        
        return result

    

    # def _normalize_quantile(self, x, stats: NormStats):
    #     assert stats.q01 is not None
    #     assert stats.q99 is not None
    #     q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
    #     return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def _normalize_quantile(self, x, stats: NormStats, output_max: float = 1.0,
                        output_min: float = -1.0,
                        range_eps: float = 1e-7, norm_mask: tuple[bool, ...] | None = None):
        assert stats.q01 is not None
        assert stats.q99 is not None

        # 截断到当前 x 真实的维度数
        input_min = stats.min[..., : x.shape[-1]]
        input_max = stats.max[..., : x.shape[-1]]

        # 像 get_range_normalizer_from_stat 一样，先算"范围"
        input_range = input_max - input_min                     # 对应 input_max - input_min
        ignore_dim = input_range < range_eps        # 哪些维度几乎是常数

        # 避免除以很小的数：这些维度的范围直接设为 output_max - output_min（=2）
        input_range_safe = input_range.copy() if isinstance(input_range, np.ndarray) else input_range
        input_range_safe[ignore_dim] = (output_max - output_min)

        # 跟 range_normalizer 一样算 scale / offset
        scale = (output_max - output_min) / input_range_safe
        offset = output_min - scale * input_min

        # 对"常数维"，把中心对齐到 0：
        # offset[ignore_dim] = (output_max + output_min) / 2 - q01[ignore_dim] = 0 - q01[ignore_dim]
        offset[ignore_dim] = (output_max + output_min) / 2.0 - input_min[ignore_dim]

        # Handle mask: True = normalize, False = pass through (scale=1, offset=0)
        if norm_mask is not None:
            if len(norm_mask) != x.shape[-1]:
                raise ValueError(f"Mask length {len(norm_mask)} does not match input shape {x.shape[-1]}")
            for dim in range(min(x.shape[-1], len(norm_mask))):
                if not norm_mask[dim]:  # False means skip normalization
                    scale[..., dim] = 1.0
                    offset[..., dim] = 0.0

        # Final normalize: x * scale + offset
        return x * scale + offset


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile unnormalization for all keys (default behavior).
    use_quantiles: bool = False
    # Dict mapping key names to their norm_mask tuples. Must match key_masks in Normalize.
    # Use make_bool_mask() to create masks easily.
    # Example: key_masks={"actions": make_bool_mask(3, -7), "state": make_bool_mask(3, -13)}
    # Keys not in this dict will be unnormalized without mask.
    key_masks: dict[str, tuple[bool, ...]] | None = None
    # Internal field, will be populated in __post_init__
    key_methods: dict[str, Callable] | None = None

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)
        
        # Build key_methods from key_masks configuration
        methods = {}
        if self.key_masks:
            for key, mask in self.key_masks.items():
                if self.use_quantiles:
                    methods[key] = partial(self._unnormalize_quantile, norm_mask=mask)
                else:
                    methods[key] = partial(self._unnormalize, norm_mask=mask)
        
        # Use object.__setattr__ to bypass frozen dataclass restriction
        object.__setattr__(self, 'key_methods', methods if methods else None)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        default_fn = self._unnormalize_quantile if self.use_quantiles else self._unnormalize
        
        return apply_tree(
            data,
            self.norm_stats,
            self.key_methods,
            strict=False,
            default_fn=default_fn,
        )

    def _unnormalize(self, x, stats: NormStats, norm_mask: tuple[bool, ...] | None = None):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        result = x * (std + 1e-6) + mean
        
        # Handle mask: True = was normalized, False = pass through
        if norm_mask is not None:
            assert len(norm_mask) == x.shape[-1]
            result = result.copy() if isinstance(result, np.ndarray) else np.array(result)
            for dim in range(min(x.shape[-1], len(norm_mask))):
                if not norm_mask[dim]:  # False means was not normalized
                    result[..., dim] = x[..., dim]
        
        return result

    # def _unnormalize_quantile(self, x, stats: NormStats):
    #     assert stats.q01 is not None
    #     assert stats.q99 is not None
    #     q01, q99 = stats.q01, stats.q99
    #     if (dim := q01.shape[-1]) < x.shape[-1]:
    #         return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
    #     return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    def _unnormalize_quantile(self, y, stats: NormStats,
                          output_max: float = 1.0,
                          output_min: float = -1.0,
                          range_eps: float = 1e-7,
                          norm_mask: tuple[bool, ...] | None = None):
        """
        Unnormalize quantile-normalized values from [-1, 1] back to original scale.
        Requires stats with q01/q99, matching the stats used in _normalize_quantile.
        """
        assert stats.q01 is not None
        assert stats.q99 is not None

        # Truncate to current y's actual dimension count
        input_min = stats.min[..., : y.shape[-1]]
        input_max = stats.max[..., : y.shape[-1]]

        # Same range and ignore_dim logic as in normalize
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps

        input_range_safe = input_range.copy() if isinstance(input_range, np.ndarray) else input_range
        input_range_safe[ignore_dim] = (output_max - output_min)  # = 2

        scale = (output_max - output_min) / input_range_safe
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2.0 - input_min[ignore_dim]

        # Handle mask: True = was normalized, False = pass through (scale=1, offset=0)
        if norm_mask is not None:
            assert len(norm_mask) == y.shape[-1]
            for dim in range(min(y.shape[-1], len(norm_mask))):
                if not norm_mask[dim]:  # False means was not normalized
                    scale[..., dim] = 1.0
                    offset[..., dim] = 0.0

        # Inverse affine transform: x = (y - offset) / scale
        x = (y - offset) / scale
        return x



@dataclasses.dataclass(frozen=True)
class UmiImageTransform(DataTransformFn):
    """Transform the UMI image to the desired resolution."""
    out_res: tuple[int, int] = (224, 224)

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: generate_image_pipeline(v, out_res=self.out_res) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class RelativeState(DataTransformFn):
    base_state_mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "base_state" not in data or self.base_state_mask is None:
            return data

        base_state = data["base_state"]
        base_state_mask = np.asarray(self.base_state_mask)
        dims = base_state_mask.shape[-1]
        base_state = pad_to_dim(base_state, dims, axis=-1)
    
        copy_state = data["state"].copy()
        data["raw_state"] = data["state"].copy()
        copy_state[..., :dims] -= np.where(base_state_mask, base_state[..., :dims], 0)
        data["state"] = copy_state
        # base = data["base_state"]
        # print(f"base_state: {base}")
        # print(f"state: {data['state']}")
        # print(f"raw_state: {data['raw_state']}")
        return data



@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None
    

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class PadActionsOnly(DataTransformFn):
    """Zero-pads actions only (not states) to the model action dimension.
    
    Use this when:
    - Pi0.5 with discrete_state_input=False (state is not used by model)
    - You want to pad actions to match pretrained model but keep original state dim
    """

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class FlattenState(DataTransformFn):
    """Platten the state to the model action dimension."""

    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data or len(data["state"].shape) <= 1:
            return data
        if len(data["state"].shape) > 2:
            raise ValueError("State must be 2D or 3D")
        data["state"] = data["state"].flatten()
        return data



def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], 
    selector: at.PyTree[S], 
    fn: Callable[[T, S], T] | dict[str, Callable[[T, S], T]], 
    *, 
    strict: bool = False,
    default_fn: Callable[[T, S], T] | None = None,
) -> at.PyTree[T]:
    """Apply function(s) to tree based on selector.
    
    Args:
        tree: The data tree to transform.
        selector: The selector tree (e.g., norm_stats).
        fn: Either a single function to apply to all keys, or a dict mapping 
            key names to their specific functions.
            Example: {"actions": normalize_quantile, "state": normalize_zscore}
        strict: If True, raise error if selector keys are not in tree.
        default_fn: When fn is a dict, this function is used for keys not in fn.
                   If None and fn is a dict, keys not in fn are left unchanged.
    
    Returns:
        Transformed tree.
    """
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)
    
    # Check if fn is a dict of functions or a single function
    fn_is_dict = isinstance(fn, dict)

    def transform(k: str, v: T) -> T:
        if k in selector:
            if fn_is_dict:
                # Use specific function for this key, or default_fn, or skip
                if k in fn:
                    return fn[k](v, selector[k])
                elif default_fn is not None:
                    return default_fn(v, selector[k])
                else:
                    return v  # No function specified for this key, keep original
            else:
                return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )


@dataclasses.dataclass(frozen=True)
class ChunkActions(DataTransformFn):
    """Truncate actions to a specified dimension.
    
    This is useful when:
    - Model outputs more dimensions than the robot needs
    - You want to extract only the first N dimensions from actions
    
    Example:
        ChunkActions(target_dim=20)  # Keep only first 20 dims from 32-dim output
    """
    
    # Target dimension to truncate to
    target_dim: int
    
    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        
        actions = data["actions"]
        # actions shape: (horizon, action_dim) or (action_dim,)
        
        current_dim = actions.shape[-1]
        if current_dim <= self.target_dim:
            # return data  # No truncation needed
            return {
                "actions": actions,
            }
        
        # Truncate to target dimension
        # data["actions"] = actions[..., :self.target_dim]
        return {
            "actions": actions[..., :self.target_dim],
        }



@dataclasses.dataclass(frozen=True)
class ExpandBimanualActions(DataTransformFn):
    """Expand bimanual robot actions from compact format to padded format.
    
    Original format: [left_arm (10), right_arm (10)] = 20 dims
    Expanded format: [left_arm (10) + padding (6), right_arm (10) + padding (6)] = 32 dims
    
    This is used during training to match Pi0.5's 32-dim action space.
    """
    
    # Number of dimensions per arm in original data
    arm_dim: int = 10
    # Target dimension per arm (with padding)
    target_arm_dim: int = 16
    # Padding value
    pad_value: float = 0.0
    
    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        
        actions = data["actions"]
        # actions shape: (horizon, 20) or (20,)
        
        has_horizon = len(actions.shape) == 2
        if not has_horizon:
            actions = actions[np.newaxis, :]  # Add horizon dim
        
        original_dim = actions.shape[-1]
        
        # Validate input dimension
        expected_dim = self.arm_dim * 2
        if original_dim != expected_dim:
            raise ValueError(f"Expected action dim {expected_dim}, got {original_dim}")
        
        # Split into left and right arms
        left_arm = actions[..., :self.arm_dim]  # (horizon, 10)
        right_arm = actions[..., self.arm_dim:]  # (horizon, 10)
        
        # Pad each arm to target dimension
        pad_size = self.target_arm_dim - self.arm_dim
        left_padded = np.pad(
            left_arm, 
            [(0, 0), (0, pad_size)], 
            mode='constant', 
            constant_values=self.pad_value
        )  # (horizon, 16)
        right_padded = np.pad(
            right_arm, 
            [(0, 0), (0, pad_size)], 
            mode='constant', 
            constant_values=self.pad_value
        )  # (horizon, 16)
        
        # Concatenate
        expanded_actions = np.concatenate([left_padded, right_padded], axis=-1)  # (horizon, 32)
        
        if not has_horizon:
            expanded_actions = expanded_actions[0]  # Remove horizon dim
        
        data["actions"] = expanded_actions
        return data


@dataclasses.dataclass(frozen=True)
class CompactBimanualActions(DataTransformFn):
    """Compact bimanual robot actions from padded format back to original format.
    
    Padded format: [left_arm (10) + padding (6), right_arm (10) + padding (6)] = 32 dims
    Original format: [left_arm (10), right_arm (10)] = 20 dims
    
    This is used during inference to extract actual robot commands from Pi0.5's output.
    """
    
    # Number of actual dimensions per arm (excluding padding)
    arm_dim: int = 10
    # Padded dimension per arm
    padded_arm_dim: int = 16
    
    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        
        actions = data["actions"]
        # actions shape: (horizon, 32) or (32,)
        
        has_horizon = len(actions.shape) == 2
        if not has_horizon:
            actions = actions[np.newaxis, :]  # Add horizon dim
        
        expected_dim = self.padded_arm_dim * 2
        if actions.shape[-1] != expected_dim:
            raise ValueError(f"Expected action dim {expected_dim}, got {actions.shape[-1]}")
        
        # Extract left and right arms (ignoring padding)
        left_arm = actions[..., :self.arm_dim]  # (horizon, 10)
        right_arm = actions[..., self.padded_arm_dim:self.padded_arm_dim + self.arm_dim]  # (horizon, 10)
        
        # Concatenate back to original format
        compact_actions = np.concatenate([left_arm, right_arm], axis=-1)  # (horizon, 20)
        
        if not has_horizon:
            compact_actions = compact_actions[0]  # Remove horizon dim
        
        data["actions"] = compact_actions
        return data


@dataclasses.dataclass(frozen=True)
class ExpandBimanualState(DataTransformFn):
    """Expand bimanual robot state from compact format to padded format.
    
    Original format: [left_arm (10), right_arm (10)] = 20 dims (or with additional dims)
    Expanded format: [left_arm (10) + padding (6), right_arm (10) + padding (6)] = 32 dims
    
    This is used during training/inference to match Pi0.5's expected state format.
    """
    
    # Number of dimensions per arm in original data
    arm_dim: int = 10
    # Target dimension per arm (with padding)
    target_arm_dim: int = 16
    # Padding value
    pad_value: float = 0.0
    
    def __call__(self, data: DataDict) -> DataDict:
        if "state" not in data:
            return data
        
        state = data["state"]
        # state shape: (20,) or (20 + extra,)
        
        original_dim = state.shape[-1]
        
        # Validate minimum input dimension
        min_expected_dim = self.arm_dim * 2
        if original_dim < min_expected_dim:
            raise ValueError(f"Expected state dim >= {min_expected_dim}, got {original_dim}")
        
        # Split into left and right arms
        left_arm = state[..., :self.arm_dim]  # (10,)
        right_arm = state[..., self.arm_dim:self.arm_dim * 2]  # (10,)
        
        # Pad each arm to target dimension
        pad_size = self.target_arm_dim - self.arm_dim
        left_padded = np.pad(
            left_arm, 
            [(0, pad_size)], 
            mode='constant', 
            constant_values=self.pad_value
        )  # (16,)
        right_padded = np.pad(
            right_arm, 
            [(0, pad_size)], 
            mode='constant', 
            constant_values=self.pad_value
        )  # (16,)
        
        # Concatenate
        expanded_state = np.concatenate([left_padded, right_padded], axis=-1)  # (32,)
        
        data["state"] = expanded_state
        return data

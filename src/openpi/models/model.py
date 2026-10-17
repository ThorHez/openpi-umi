import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# Defaults used only to initialize legacy model specifications. Runtime preprocessing
# preserves the supplied views unless the model config explicitly selects keys.
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DEPTH_KEYS = ("base_0_depth",)

# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32. Supports both single images [*b, h, w, c]
    # and video clips [*b, t, h, w, c].
    # Use "..." to represent any number of batch dimensions followed by spatial dims.
    images: dict[str, at.Float[ArrayT, "..."]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Optional externally computed semantic-memory tokens.  These are kept
    # separate from robot state so a frozen memory bank can condition an
    # action policy without inflating or normalizing the proprioceptive state.
    semantic_memory: at.Float[ArrayT, "*b m d"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    fast_tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    fast_tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    fast_token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    fast_token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # Per-sample action loss mask for mixed-dataset training (e.g. single-arm / bimanual).
    action_loss_mask: at.Float[ArrayT, "*b ad"] | None = None

    # Optional frame-level validity for video observations, keyed like images.
    # Shape is [*b, T]; False means VideoFrameDataset padded/repeated that slot.
    frame_valid_masks: dict[str, at.Bool[ArrayT, "*b t"]] | None = None

    # Value model fields (for Pi0Advantage / advantage estimation).
    step_index: at.Float[ArrayT, "*b"] | None = None
    episode_T: at.Float[ArrayT, "*b"] | None = None
    terminal_reward: at.Float[ArrayT, "*b"] | None = None

    # For value training with compute_normalized_value_targets (Option B).
    episode_index: at.Float[ArrayT, "*b"] | at.Int[ArrayT, "*b"] | None = None
    frame_index: at.Float[ArrayT, "*b"] | at.Int[ArrayT, "*b"] | None = None
    # Number of frames in the current episode. Variable-length action datasets
    # use this to mask padded future waypoints at the end of each episode.
    episode_length: at.Float[ArrayT, "*b"] | at.Int[ArrayT, "*b"] | None = None
    # Precomputed scalar value target from dataset (optional; shape [B] or [B, 1]).
    value_target: at.Float[ArrayT, "*b"] | at.Float[ArrayT, "*b 1"] | None = None
    # LeRobot global row index (optional; kept by KeepModelKeys for some pipelines).
    index: at.Float[ArrayT, "*b"] | at.Int[ArrayT, "*b"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            semantic_memory=data.get("semantic_memory"),
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            action_loss_mask=data.get("action_loss_mask"),
            frame_valid_masks=data.get("frame_valid_mask"),
            # Hybrid model (Pi0.5 + FAST) specific fields
            fast_tokenized_prompt=data.get("fast_tokenized_prompt"),
            fast_tokenized_prompt_mask=data.get("fast_tokenized_prompt_mask"),
            fast_token_ar_mask=data.get("fast_token_ar_mask"),
            fast_token_loss_mask=data.get("fast_token_loss_mask"),

            # Value model fields
            step_index=data.get("step_index"),
            episode_T=data.get("episode_T"),
            terminal_reward=data.get("terminal_reward"),
            episode_index=data.get("episode_index"),
            frame_index=data.get("frame_index"),
            episode_length=data.get("episode_length"),
            value_target=data.get("value_target"),
            index=data.get("index"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        result["frame_valid_mask"] = result.pop("frame_valid_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def resolve_view_keys(
    available_keys: Sequence[str],
    image_keys: Sequence[str] | None = None,
    depth_keys: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve a stable view order without a task-specific global camera switch.

    None preserves actual observation views. Explicit keys define a fixed model
    contract; missing selected views are padded and masked by preprocessing.
    """
    available_keys = tuple(sorted(available_keys))
    depth = tuple(depth_keys) if depth_keys is not None else tuple(k for k in available_keys if "depth" in k)
    images = tuple(image_keys) if image_keys is not None else tuple(k for k in available_keys if k not in depth)
    keys = (*images, *depth)
    if not keys:
        raise ValueError("No image views were supplied or configured")
    if len(set(keys)) != len(keys):
        raise ValueError(f"Image/depth view keys must be unique and disjoint: {keys}")
    return images, depth


def _require_valid_views(valid: at.Array) -> None:
    """Reject all-masked samples eagerly and under JIT, without copying valid batches to the host."""
    def fail(invalid):
        raise ValueError("Observation has no valid image view for one or more samples; "
                         "check image_keys, image masks and frame masks, or explicitly set "
                         "allow_all_masked_images=True for a deliberate image-free ablation")

    invalid = jnp.logical_not(jnp.all(valid))
    if isinstance(invalid, jax.core.Tracer):
        def report(value):
            jax.debug.callback(fail, value)
            return jnp.int32(0)
        jax.lax.cond(invalid, report, lambda _: jnp.int32(0), invalid)
    elif bool(invalid):
        fail(invalid)


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] | None = None,
    depth_keys: Sequence[str] | None = None,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    allow_all_masked_images: bool = False,
) -> Observation:
    """Resize/augment the selected views, retaining image and video validity.

    Legacy recipes retain all actual views, including 2/3/4-camera UMI and
    ShellGame base/wrist layouts. Explicit keys select a fixed layout for mixed
    datasets; absent views inherit the observed video length and are masked.
    Depth views are resized without RGB augmentation. An all-masked sample is
    rejected unless the caller explicitly enables an image-free ablation.
    """
    batch_shape = observation.state.shape[:-1]
    image_keys, depth_keys = resolve_view_keys(tuple(observation.images), image_keys, depth_keys)
    keys = (*image_keys, *depth_keys)
    present = [key for key in keys if key in observation.images]
    if not present and not allow_all_masked_images:
        raise ValueError(f"None of the configured image views {keys} are present; "
                         f"observation provides {tuple(observation.images)}")
    # Video placeholders must keep T; do not replace an absent stream with a 4D image.
    leading_shape = observation.images[present[0]].shape[:-3] if present else batch_shape
    out_images = {}
    out_masks = {}
    out_frame_masks = {} if observation.frame_valid_masks is not None else None
    for key in keys:
        present_view = key in observation.images
        if not present_view:
            image = jnp.zeros((*leading_shape, *image_resolution, 3), dtype=jnp.float32)
            mask = jnp.zeros(batch_shape, dtype=jnp.bool_)
        else:
            image = observation.images[key]
            if image.ndim not in (len(batch_shape) + 3, len(batch_shape) + 4) or image.shape[-1] != 3:
                raise ValueError(f"Expected HWC image or THWC video for {key}; got {image.shape}")
            shape_prefix = image.shape[:-3]
            image = image.reshape((-1, *image.shape[-3:]))
            if image.shape[1:3] != image_resolution:
                image = image_tools.resize_with_pad(image, *image_resolution)
            if train and rng is not None and key not in depth_keys:
                image = image / 2.0 + 0.5
                transforms = []
                if "wrist" not in key:
                    height, width = image.shape[1:3]
                    transforms += [augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                                   augmax.Resize(width, height), augmax.Rotate((-5, 5))]
                transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
                image = jax.vmap(augmax.Chain(*transforms))(jax.random.split(rng, image.shape[0]), image)
                image = image * 2.0 - 1.0
            image = image.reshape((*shape_prefix, *image.shape[-3:]))
            mask = jnp.broadcast_to(jnp.asarray(observation.image_masks.get(key, True), dtype=jnp.bool_), batch_shape)
        if out_frame_masks is not None and image.ndim == len(batch_shape) + 4:
            frame_shape = image.shape[:-3]
            frame_mask = observation.frame_valid_masks.get(key)
            if frame_mask is None or not present_view:
                frame_mask = jnp.broadcast_to(mask[..., None], frame_shape)
            else:
                frame_mask = jnp.asarray(frame_mask, dtype=jnp.bool_)
                if frame_mask.shape != frame_shape:
                    raise ValueError(f"Frame mask for {key} has shape {frame_mask.shape}; expected {frame_shape}")
                frame_mask = frame_mask & mask[..., None]
            out_frame_masks[key] = frame_mask
            mask = mask & jnp.any(frame_mask, axis=-1)
        out_images[key] = image
        out_masks[key] = mask
    if not allow_all_masked_images:
        _require_valid_views(jnp.any(jnp.stack(tuple(out_masks.values()), axis=0), axis=0))
    # replace retains semantic memory, value targets and future optional fields.
    return dataclasses.replace(observation, images=out_images, image_masks=out_masks,
                               frame_valid_masks=out_frame_masks)


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int
    # None keeps the actual observation views for legacy recipes. Tuples make
    # camera order/padding explicit for a new recipe and its serving config.
    image_keys: tuple[str, ...] | None = dataclasses.field(default=None, kw_only=True)
    depth_keys: tuple[str, ...] | None = dataclasses.field(default=None, kw_only=True)
    allow_all_masked_images: bool = dataclasses.field(default=False, kw_only=True)

    def spec_view_keys(self, default_keys: Sequence[str] = IMAGE_KEYS) -> tuple[str, ...]:
        images, depth = resolve_view_keys(default_keys, self.image_keys, self.depth_keys)
        return (*images, *depth)

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    def configure_views(self, config: BaseModelConfig) -> None:
        self.image_keys = config.image_keys
        self.depth_keys = config.depth_keys
        self.allow_all_masked_images = config.allow_all_masked_images

    def preprocess_observation(self, rng, observation, *, train=False) -> Observation:
        return preprocess_observation(
            rng, observation, train=train,
            image_keys=getattr(self, "image_keys", None),
            depth_keys=getattr(self, "depth_keys", None),
            allow_all_masked_images=getattr(self, "allow_all_masked_images", False),
        )

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        def _restore_arg(meta):
            target_dtype = None
            if dtype is not None:
                leaf_dtype = getattr(meta, "dtype", None)
                # Only cast floating-point leaves; keep PRNG keys and integer metadata dtypes unchanged.
                if leaf_dtype is not None and np.issubdtype(np.dtype(leaf_dtype), np.floating):
                    target_dtype = dtype
            return ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=target_dtype)

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(_restore_arg, item),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)

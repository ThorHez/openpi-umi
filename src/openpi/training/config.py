"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Callable, Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_gripper as pi0_gripper
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.umi_policy as umi_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import openpi.models.pi0_discrete as pi0_discrete
import openpi.models.pi0_value as pi0_value
from openpi.transforms import make_bool_mask
from openpi.models.pi0_discrete import CheckpointWeightLoaderWithDiscreteHead
from openpi.training.weight_loaders import CheckpointWeightLoaderWithValueHead

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Optional per-key masks for normalization. See transforms.make_bool_mask.
    normalize_masks: dict[str, tuple[bool, ...]] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # Per-sample action loss mask for multi-dataset training (e.g. single-arm 7d vs bimanual 20d).
    # If set, injected into each sample as "action_loss_mask"; model uses it when present.
    # Length must match model action_dim, e.g. (1.0,)*7 + (0.0,)*25 for 7 real dims.
    action_loss_mask: Sequence[float] | None = None

    # Robot type for tokenizer (e.g. "ARM=1,G=0,H=0" for single-arm). If set, applied to
    # TokenizePrompt, TokenizeFASTInputs, and TokenizeHybridInput in model_transforms.
    robot_type: str | None = None

@dataclasses.dataclass(frozen=True)
class UmiDataConfig(DataConfig):
    action_sequence_keys: Sequence[str] = ()
    use_quantile_norm: bool = True
    prompt_from_task: bool = True


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if getattr(model_config, "fast_model_tokenizer_kwargs", None) is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


def _set_robot_type(config: DataConfig, robot_type: str) -> DataConfig:
    """Set robot_type on all tokenize transforms in model_transforms.

    robot_type uses the structured format: ARM=1|2, G=0|1, H=0|1
    e.g. "ARM=1,G=0,H=0" for single-arm without global view and without height.
    """
    tokenize_types = (
        _transforms.TokenizePrompt,
        _transforms.TokenizeFASTInputs,
        _transforms.TokenizeHybridInput,
    )
    new_inputs = tuple(
        dataclasses.replace(t, robot_type=robot_type) if isinstance(t, tokenize_types) else t
        for t in config.model_transforms.inputs
    )
    new_model_transforms = dataclasses.replace(config.model_transforms, inputs=new_inputs)
    return dataclasses.replace(config, model_transforms=new_model_transforms)


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms for UMI robot dataset.
    UMI uses end-effector control with 7-dimensional actions:
    - 3D position (x, y, z)
    - 3D rotation (axis-angle representation)
    - 1D gripper width
    """

    # Whether to apply delta action transform. UMI typically uses absolute actions,
    # so you may want to set this to True to convert to delta actions for pi0 models.
    use_delta_actions: bool = True

    # Action keys that will be used to read the action sequence from the dataset.
    # This should match the key used in your LeRobot dataset (typically "action" in singular form).
    action_sequence_keys: Sequence[str] = ("actions",)
    # action_sequence_keys: Sequence[str] = ()

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform maps dataset keys to the keys expected by the policy.
        # For UMI dataset, we map the LeRobot dataset keys to standard observation keys.
        # Note: LeRobot uses '.' instead of '/' in feature names
        # Format: "new_key": "old_key_from_dataset"
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        "camera0_rgb": "observation.camera0_rgb",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        # Data transforms that convert UMI data format to model input format.
        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputs(model_type=model_config.model_type)],
            outputs=[umi_policy.UmiOutputs()],
        )

        # Apply delta action transform if enabled.
        # For UMI, we typically want to convert position and rotation to deltas,
        # but keep gripper width absolute.
        if self.use_delta_actions:
            # First 6 actions (position + rotation) as delta, last action (gripper) as absolute
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms (tokenization, etc.)
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded(DataConfigFactory):
    """
    UMI data config that pads 7-dim actions to 32-dim AFTER normalization.
    This avoids the issue where padding values of 0 get normalized to -1.0.
    
    Key differences from LeRobotUmiDataConfig:
    - Uses 7-dim norm_stats (cleaner, no padding pollution)
    - Pads actions only (not state) to 32-dim after normalization (padding stays as 0)
    - State remains 7-dim (not wasted, Pi0.5 with discrete_state_input=False doesn't use it anyway)
    - Compatible with pi05_base pretrained weights (32-dim)
    """
    
    use_delta_actions: bool = True
    action_sequence_keys: Sequence[str] = ("actions",)
    training_mode: bool = True
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        "camera0_rgb": "observation.camera0_rgb",
                        "actions": "actions",
                        "state": "state",
                        "prompt": "task",
                    }
                )
            ]
        )
        
        # Data transforms (UmiInputs + optional DeltaActions)
        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputs(model_type=model_config.model_type)],
            outputs=[umi_policy.UmiOutputs()],
        )
        
        if self.use_delta_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        if not self.training_mode:
            # In inference mode, we need to transform the image to the desired resolution
            print("In inference mode, transforming image to 224x224")
            data_transforms = data_transforms.push(
                inputs=[_transforms.UmiImageTransform(out_res=(224, 224))],
            )
        
        # Model transforms - customized to pad AFTER normalization
        # This ensures padding values stay as 0 instead of being normalized to -1.0
        # Use PadActionsOnly since Pi0.5 with discrete_state_input=False doesn't use state
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),  # Pad actions only to 32-dim (after normalization)
            ],
        )
        
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class UmiArxInferenceDataConfig(DataConfigFactory):
    """
    UMI inference data config.
    """
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "camera0_rgb": "camera0_rgb",
                        "robot0_eef_pos": "robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
                        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
                        "robot0_gripper_width": "robot0_gripper_width",
                    }
                )
            ]
        )

        # Data transforms (UmiInputs + optional DeltaActions)
        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiArxInputs()],
            outputs=[umi_policy.UmiArxOutputs()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),  # Pad actions only to 32-dim (after normalization)
            ],
        )


        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V3(DataConfigFactory):
    """
    UMI data config that pads 7-dim actions to 32-dim AFTER normalization.
    This avoids the issue where padding values of 0 get normalized to -1.0.

    Key differences from LeRobotUmiDataConfig:
    - Uses 7-dim norm_stats (cleaner, no padding pollution)
    - Pads actions only (not state) to 32-dim after normalization (padding stays as 0)
    - State remains 7-dim (not wasted, Pi0.5 with discrete_state_input=False doesn't use it anyway)
    - Compatible with pi05_base pretrained weights (32-dim)
    """

    action_sequence_keys: Sequence[str] = ("actions",)
    training_mode: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transform
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # "camera0_rgb": "observation.camera0_rgb",
                        "actions": "actions",
                        "state": "state",
                        "prompt": "task",
                        "base_state": "base_state",
                        "state_sequence": "state_sequence",
                        "camera0_rgb_0": "camera0_rgb_0",
                        "camera0_rgb_1": "camera0_rgb_1",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                    }
                )
            ]
        )

        # Data transforms (UmiInputs + optional DeltaActions)
        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputsV2(model_type=model_config.model_type, training_mode=self.training_mode)],
            outputs=[umi_policy.UmiOutputsV2()],
        )

        # Model transforms - customized to pad AFTER normalization
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),  # Pad actions only to 32-dim (after normalization)
                _transforms.FlattenState(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V4(DataConfigFactory):
    """
    UMI data config that pads 7-dim actions to 32-dim AFTER normalization.
    """

    normalize_masks = {
        "actions": make_bool_mask(3, -7),
        "state": make_bool_mask(3, -13),
    }

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        "robot0_eef_rot_axis_angle_wrt_start": "observation.robot0_eef_rot_axis_angle_wrt_start",
                        "left_wrist_0_rgb_0": "observation.left_wrist_0_rgb_0",
                        "left_wrist_0_rgb_1": "observation.left_wrist_0_rgb_1",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputsV4()],
            outputs=[umi_policy.UmiOutputsV4()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Hybrid(DataConfigFactory):
    """Base UMI data config for hybrid (Pi0.5 + FAST) training.

    Subclass or instantiate with different mapping / normalize_masks / data_inputs_fn
    for different datasets. The create() logic (repack, data_transforms, model_transforms)
    is shared; configurable fields:

        mapping           - RepackTransform key mapping
        normalize_masks   - per-key normalization masks
        data_inputs_fn    - factory for data_transforms input (default: UmiInputsV4)
        data_outputs_fn   - factory for data_transforms output (default: UmiOutputsV4)

    Example instantiation::

        LerobotUmiDataConfig_Hybrid(
            repo_id="...",
            mapping={...},
            normalize_masks={...},
            data_inputs_fn=lambda: umi_policy.UmiInputsV4_Bimanual(),
        )
    """

    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "observation.robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "observation.robot0_gripper_width",
        "robot0_eef_rot_axis_angle_wrt_start": "observation.robot0_eef_rot_axis_angle_wrt_start",
        "left_wrist_0_rgb_0": "observation.left_wrist_0_rgb_0",
        "left_wrist_0_rgb_1": "observation.left_wrist_0_rgb_1",
        "actions": "actions",
        "prompt": "task",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7),
        "state": make_bool_mask(3, -13),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiOutputsV4()

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.mapping)
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[self.data_inputs_fn()],
            outputs=[self.data_outputs_fn()],
        )

        tokenizer_kwargs = (
            {} if getattr(model_config, "fast_model_tokenizer_kwargs", None) is None else model_config.fast_model_tokenizer_kwargs
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizeHybridInput(
                    tokenizer=_tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                    fast_tokenizer=_tokenizer.FASTTokenizer(model_config.max_token_len, **tokenizer_kwargs),
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
                _transforms.KeepModelKeys(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_Hybrid(LeRobotUmiDataConfig_Hybrid):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
        # "base_0_rgb_0": "base_0_rgb_0",
        "actions": "actions",
        "prompt": "task",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(3, -12, 3, -7, 3, -12, 3, -7),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4_Bimanual()


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(LeRobotUmiDataConfig_Hybrid):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        # "robot0_eef_desk_height": "robot0_eef_desk_height",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
        "robot1_eef_pos": "robot1_eef_pos",
        # "robot1_eef_desk_height": "robot1_eef_desk_height",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        # "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
        # "base_0_rgb_0": "base_0_rgb_0",
        "actions": "actions",
        "prompt": "task",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -12, 3, -7, 6, -12, 3, -7),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4_Bimanual_Horizon1()


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(LeRobotUmiDataConfig_Hybrid):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        # "robot0_eef_pos_desk": "robot0_eef_pos_desk",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        # "robot1_eef_pos_desk": "robot1_eef_pos_desk",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        # "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
        "base_0_rgb_0": "base_0_rgb_0",
        "base_0_depth_0": "base_0_depth_0",

        "actions": "actions",
        "prompt": "task",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -12, 3, -7, 6, -12, 3, -7),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4_Bimanual_HeadView_Depth_Horizon1()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiOutputsV4()

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.mapping)
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                _transforms.Transform_depth_to_3ch_image(depth_column_name="base_0_depth_0"),
                self.data_inputs_fn()
                ],
            outputs=[self.data_outputs_fn()],
        )

        tokenizer_kwargs = (
            {} if getattr(model_config, "fast_model_tokenizer_kwargs", None) is None else model_config.fast_model_tokenizer_kwargs
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
                _transforms.KeepModelKeys(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class WBCD_Bimamual_ImageHorizon1(DataConfigFactory):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        # "robot0_eef_pos_desk": "robot0_eef_pos_desk",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        #"robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        #"robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        # "robot1_eef_pos_desk": "robot1_eef_pos_desk",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        #"robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        #"robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        # "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
        # "base_0_rgb_0": "base_0_rgb_0",
        # "base_0_depth_0": "base_0_depth_0",

        "actions": "actions",
        "prompt": "task",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -13, 6, -13),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.WBCD_V1_Bimanual_Horizon1()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiOutputsV4()

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.mapping)
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                # _transforms.Transform_depth_to_3ch_image(depth_column_name="base_0_depth_0"),
                self.data_inputs_fn(),
                # Pad the missing left_wrist_1_rgb / right_wrist_1_rgb views with zeros + mask=False
                # so this 2-view dataset can be mixed in the same batch with 4-view datasets.
                _transforms.EnsureImageKeys(),
            ],
            outputs=[self.data_outputs_fn()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
                _transforms.KeepModelKeys(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class WBCD_Bimamual_4_views_ImageHorizon1(DataConfigFactory):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        # "robot0_eef_pos_desk": "robot0_eef_pos_desk",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        #"robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        #"robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        "left_wrist_1_rgb_0": "left_wrist_1_rgb_0",
        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        # "robot1_eef_pos_desk": "robot1_eef_pos_desk",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        #"robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        #"robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        "right_wrist_1_rgb_0": "right_wrist_1_rgb_0",
        # "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
        # "base_0_rgb_0": "base_0_rgb_0",
        # "base_0_depth_0": "base_0_depth_0",

        "actions": "actions",
        "prompt": "task",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -13, 6, -13),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.WBCD_V1_Bimanual_4_views_Horizon1()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiOutputsV4()

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.mapping)
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                # _transforms.Transform_depth_to_3ch_image(depth_column_name="base_0_depth_0"),
                self.data_inputs_fn()
                ],
            outputs=[self.data_outputs_fn()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
                _transforms.KeepModelKeys(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class WBCD_Bimamual_HeadView_Depth_ImageHorizon1(DataConfigFactory):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        # "robot0_eef_pos_desk": "robot0_eef_pos_desk",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        #"robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        #"robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        # "robot1_eef_pos_desk": "robot1_eef_pos_desk",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        #"robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        #"robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        # "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
        "base_0_rgb_0": "base_0_rgb_0",
        "base_0_depth_0": "base_0_depth_0",

        "actions": "actions",
        "prompt": "task",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -13, 6, -13),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.WBCD_V1_Bimanual_HeadView_Depth_Horizon1()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiOutputsV4()

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.mapping)
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                _transforms.Transform_depth_to_3ch_image(depth_column_name="base_0_depth_0"),
                self.data_inputs_fn()
                ],
            outputs=[self.data_outputs_fn()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
                _transforms.KeepModelKeys(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )



@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(LeRobotUmiDataConfig_Hybrid):
    """Value training variant: inherits HeadView+Depth layout, adds episode_index/frame_index passthrough."""

    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        "base_0_rgb_0": "base_0_rgb_0",
        "base_0_depth_0": "base_0_depth_0",
        "actions": "actions",
        "prompt": "task",
        "episode_index": "episode_index",
        "frame_index": "frame_index",
        "value_target": "value_target",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -12, 3, -7, 6, -12, 3, -7),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4_Bimanual_HeadView_Depth_Horizon1()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiOutputsV4()

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.mapping)
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                _transforms.Transform_depth_to_3ch_image(depth_column_name="base_0_depth_0"),
                self.data_inputs_fn()
            ],
            outputs=[self.data_outputs_fn()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
                _transforms.KeepModelKeys(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(LeRobotUmiDataConfig_Hybrid):
    """ACP (Advantage-Conditioned Policy) training variant.

    Reads ``is_positive`` from the dataset (written by ``lerobot_value_infer.py``) and appends
    ``Advantage: positive`` / ``Advantage: negative`` to the task prompt before tokenization.
    Data pipeline: RepackTransform (includes is_positive) -> ... -> ACPConditionPrompt -> TokenizePrompt.
    """

    acp_dropout: float = 0.1

    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        "base_0_rgb_0": "base_0_rgb_0",
        "base_0_depth_0": "base_0_depth_0",
        "actions": "actions",
        "prompt": "task",
        "is_positive": "is_positive",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -12, 3, -7, 6, -12, 3, -7),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4_Bimanual_HeadView_Depth_Horizon1()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiOutputsV4()

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.mapping)
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                _transforms.Transform_depth_to_3ch_image(depth_column_name="base_0_depth_0"),
                self.data_inputs_fn()
            ],
            outputs=[self.data_outputs_fn()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ACPConditionPrompt(
                    indicator_key="is_positive",
                    dropout=self.acp_dropout,
                    default_positive=True,
                ),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
                _transforms.KeepModelKeys(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP_Inference(DataConfigFactory):
    """ACP (Advantage-Conditioned Policy) inference configuration.

    For inference, always conditions on positive advantage by appending
    ``Advantage: positive`` to the task prompt before tokenization.

    Data pipeline: RepackTransform -> ... -> ACPForcePositivePrompt -> TokenizePrompt.
    """

    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        "base_0_rgb_0": "base_0_rgb_0",
        "base_0_depth_0": "base_0_depth_0",
        "actions": "actions",
        "prompt": "task",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -12, 3, -7, 6, -12, 3, -7),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4_Bimanual_HeadView_Depth_Horizon1()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiOutputsV4()

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(self.mapping)
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                _transforms.Transform_depth_to_3ch_image(depth_column_name="base_0_depth_0"),
                self.data_inputs_fn()
            ],
            outputs=[self.data_outputs_fn()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ACPForcePositivePrompt(),  # Always use positive for inference
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
                _transforms.KeepModelKeys(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_HeadView_ImageHorizon1_Hybrid(LeRobotUmiDataConfig_Hybrid):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        # "robot0_eef_pos_desk": "robot0_eef_pos_desk",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
        "robot1_eef_pos": "robot1_eef_pos",
        # "robot1_eef_pos_desk": "robot1_eef_pos_desk",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        # "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
        "base_0_rgb_0": "base_0_rgb_0",
        "actions": "actions",
        "prompt": "task",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(4, -12, 3, -7, 4, -12, 3, -7),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4_Bimanual_HeadView_Horizon1()


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V4_Hybrid(DataConfigFactory):
    """
    UMI data config for hybrid (Pi0.5 + FAST) training.
    """

    normalize_masks = {
        "actions": make_bool_mask(3, -7),
        "state": make_bool_mask(3, -13),
    }

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        "robot0_eef_rot_axis_angle_wrt_start": "observation.robot0_eef_rot_axis_angle_wrt_start",
                        "left_wrist_0_rgb_0": "observation.left_wrist_0_rgb_0",
                        "left_wrist_0_rgb_1": "observation.left_wrist_0_rgb_1",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputsV4()],
            outputs=[umi_policy.UmiOutputsV4()],
        )

        tokenizer_kwargs = (
            {} if getattr(model_config, "fast_model_tokenizer_kwargs", None) is None else model_config.fast_model_tokenizer_kwargs
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizeHybridInput(
                    tokenizer=_tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                    fast_tokenizer=_tokenizer.FASTTokenizer(model_config.max_token_len, **tokenizer_kwargs),
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V5(DataConfigFactory):
    """
    UMI data config that pads 7-dim actions to 32-dim AFTER normalization.
    """

    normalize_masks = {
        "actions": make_bool_mask(3, -7),
        "state": make_bool_mask(3, -7),
    }

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        "left_wrist_0_rgb_0": "observation.left_wrist_0_rgb_0",
                        "left_wrist_0_rgb_1": "observation.left_wrist_0_rgb_1",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputsV5()],
            outputs=[umi_policy.UmiOutputsV4()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon1(DataConfigFactory):
    """
    UMI data config for bimanual (V4).
    """

    normalize_masks = {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -12, 3, -7, 6, -12, 3, -7),
    }

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "robot0_gripper_width",
                        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
                        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
                        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
                        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
                        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
                        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
                        "robot1_eef_pos": "robot1_eef_pos",
                        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
                        "robot1_gripper_width": "robot1_gripper_width",
                        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
                        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
                        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
                        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
                        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
                        # "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
                        # "base_0_rgb_0": "base_0_rgb_0",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputsV4_Bimanual_Horizon1()],
            outputs=[umi_policy.UmiOutputsV4()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class WBCD_V1_Bimanual_Horizon1_Compute_Norm_Stats(DataConfigFactory):
    """
    UMI data config for bimanual (V4).
    """

    normalize_masks = {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -13, 6, -13),
    }

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "robot0_gripper_width",
                        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
                        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
                        #"robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
                        #"robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
                        #"left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
                        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
                        "robot1_eef_pos": "robot1_eef_pos",
                        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
                        "robot1_gripper_width": "robot1_gripper_width",
                        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
                        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
                        #"robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
                        #"robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
                        #"right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
                        # "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
                        # "base_0_rgb_0": "base_0_rgb_0",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.WBCD_V4_Bimanual_Horizon1_Compute_Norm_Stats()],
            outputs=[umi_policy.UmiOutputsV4()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon2(DataConfigFactory):
    """
    UMI data config for bimanual (V4).
    """

    normalize_masks = {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(3, -12, 3, -7, 3, -12, 3, -7),
    }

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "robot0_gripper_width",
                        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
                        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
                        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
                        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
                        "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
                        "robot1_eef_pos": "robot1_eef_pos",
                        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
                        "robot1_gripper_width": "robot1_gripper_width",
                        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
                        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
                        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
                        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
                        "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
                        # "base_0_rgb_0": "base_0_rgb_0",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputsV4_Bimanual_Horizon2()],
            outputs=[umi_policy.UmiOutputsV4()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V4_Bimanual(DataConfigFactory):
    """
    UMI data config for bimanual (V4).
    """

    normalize_masks = {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(3, -12, 3, -7, 3, -12, 3, -7),
    }

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "robot0_gripper_width",
                        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
                        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
                        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
                        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
                        "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
                        "robot1_eef_pos": "robot1_eef_pos",
                        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
                        "robot1_gripper_width": "robot1_gripper_width",
                        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
                        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
                        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
                        "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
                        "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",
                        # "base_0_rgb_0": "base_0_rgb_0",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputsV4_Bimanual()],
            outputs=[umi_policy.UmiOutputsV4()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V4_Inference(DataConfigFactory):
    """
    UMI V4 inference data config.
    """

    prompt: str = "pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box"

    normalize_masks = {
        "actions": make_bool_mask(3, -7),
        "state": make_bool_mask(3, -13),
    }

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "robot0_gripper_width",
                        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
                        "camera0_rgb": "camera0_rgb",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiArxInputs(prompt=self.prompt)],
            outputs=[umi_policy.UmiArxOutputs()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
            outputs=[_transforms.ChunkActions(target_dim=10)],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V5_Inference(DataConfigFactory):
    """
    UMI V5 inference data config.
    """

    normalize_masks = {
        "actions": make_bool_mask(3, -7),
        "state": make_bool_mask(3, -7),
    }

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "robot0_gripper_width",
                        "camera0_rgb": "camera0_rgb",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiArxInputsV5()],
            outputs=[umi_policy.UmiArxOutputs()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
            outputs=[_transforms.ChunkActions(target_dim=10)],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V4_Bimanual_Inference(DataConfigFactory):
    """
    UMI V4 bimanual inference data config.
    """
    normalize_masks = {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(3, -12, 3, 7, 3, -12, 3, 7),
    }

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        config = dataclasses.replace(config, normalize_masks=self.normalize_masks)
        return config

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "robot0_gripper_width",
                        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
                        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
                        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
                        "camera0_rgb": "camera0_rgb",
                        "robot1_eef_pos": "robot1_eef_pos",
                        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
                        "robot1_gripper_width": "robot1_gripper_width",
                        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
                        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
                        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
                        "camera1_rgb": "camera1_rgb",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiArxInputs_Bimanual()],
            outputs=[umi_policy.UmiArxOutputs()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
            outputs=[_transforms.ChunkActions(target_dim=20)],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V2(DataConfigFactory):
    """
    UMI data config V2 (10d pose, relative state).
    """

    use_delta_actions: bool = True
    action_sequence_keys: Sequence[str] = ("actions",)
    training_mode: bool = True
    use_10d_pose: bool = True
    use_relative_state: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        "camera0_rgb": "observation.camera0_rgb",
                        "actions": "actions",
                        "state": "state",
                        "prompt": "task",
                        "base_state": "base_state",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputs(model_type=model_config.model_type, use_10d_pose=self.use_10d_pose)],
            outputs=[umi_policy.UmiOutputs(use_10d_pose=self.use_10d_pose)],
        )

        if self.use_delta_actions:
            if self.use_10d_pose:
                delta_action_mask = _transforms.make_bool_mask(9, -1)
            else:
                delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        if self.use_relative_state:
            if self.use_10d_pose:
                base_state_mask = _transforms.make_bool_mask(3, -7)
            else:
                base_state_mask = _transforms.make_bool_mask(3, -4)
            data_transforms = data_transforms.push(
                inputs=[_transforms.RelativeState(base_state_mask)],
                outputs=[],
            )

        if not self.training_mode:
            print("In inference mode, transforming image to 224x224")
            data_transforms = data_transforms.push(
                inputs=[_transforms.UmiImageTransform(out_res=(224, 224))],
            )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class MultiDataConfigFactory(DataConfigFactory):
    """Config factory for multi-dataset training: multiple DataConfigFactory with optional per-dataset weights.

    state_pad_dim: if set (> 0), a PadStateOnly(state_pad_dim) transform is appended to each
        dataset's model_transforms so that all datasets produce identically-shaped state tensors
        for batching. Set to 0 or None to skip.

    use_merged_norm_stats: if True, after creating each dataset's DataConfig, merge all
        norm_stats (with optional weights) and assign the merged norm_stats to every config,
        so all datasets use the same normalization. Recommended for multi-dataset training.
    """

    repo_id: str = "multi"
    datasets: list[DataConfigFactory] = dataclasses.field(default_factory=list)
    weights: list[float] | None = None  # None = uniform; same length as datasets
    state_pad_dim: int | None = None
    use_merged_norm_stats: bool = True

    def _apply_state_pad(self, dc: DataConfig) -> DataConfig:
        """If state_pad_dim is configured, append PadStateOnly to model_transforms."""
        if not self.state_pad_dim or self.state_pad_dim <= 0:
            raise ValueError("state_pad_dim must be set and > 0")
        new_mt = dc.model_transforms.push(inputs=[_transforms.PadStateOnly(self.state_pad_dim)])
        return dataclasses.replace(dc, model_transforms=new_mt)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Return the first dataset's config (for backward compat with single-dataset code paths)."""
        if not self.datasets:
            raise ValueError("MultiDataConfigFactory.datasets must be non-empty")
        return self._apply_state_pad(self.datasets[0].create(assets_dirs, model_config))

    def create_all(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> list[DataConfig]:
        """Return DataConfig for every dataset (for multi-dataset loader)."""
        all_configs = [self._apply_state_pad(f.create(assets_dirs, model_config)) for f in self.datasets]
        if self.use_merged_norm_stats:
            stats_dict_list = [dc.norm_stats for dc in all_configs if dc.norm_stats is not None]
            if stats_dict_list:
                merge_weights = self.weights
                if merge_weights is None or len(merge_weights) != len(all_configs):
                    merge_weights = [1.0] * len(all_configs)
                weights_for_merge = [merge_weights[i] for i, dc in enumerate(all_configs) if dc.norm_stats is not None]
                merged = _normalize.merge_norm_stats_dict(stats_dict_list, weights=weights_for_merge)
                if merged:
                    all_configs = [dataclasses.replace(dc, norm_stats=merged) for dc in all_configs]
                    logging.info("Multi-dataset: merged norm_stats applied to all datasets.")
        return all_configs


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_MixedArm(DataConfigFactory):
    """
    UMI data config for mixed single-arm / bimanual training: state is flattened and padded to 100-dim
    so all samples have identical pytree structure for batching. Use with action_loss_mask (1.0)*20 + (0.0)*12.
    """

    use_delta_actions: bool = True
    action_sequence_keys: Sequence[str] = ("actions",)
    training_mode: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        "camera0_rgb": "observation.camera0_rgb",
                        "actions": "actions",
                        "state": "state",
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputs(model_type=model_config.model_type)],
            outputs=[umi_policy.UmiOutputs()],
        )
        if self.use_delta_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        if not self.training_mode:
            data_transforms = data_transforms.push(
                inputs=[_transforms.UmiImageTransform(out_res=(224, 224))],
            )
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, "discrete_state_input") else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
            ],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # --- Value training (train_value.py): optional, used when fitting value targets ---
    c_fail_coef: float = 1.0
    value_clip_min: float = -1.0
    value_clip_max: float = 0.0
    episode_metadata_path: str | None = None
    task_max_lengths: dict[int, int] | None = None

    # --- Evaluation settings (train_value.py) ---
    # Fraction of dataset reserved for validation (0 disables evaluation).
    val_ratio: float = 0.0
    # How often (in steps) to run evaluation on the validation set.
    eval_interval: int = 1000
    # Number of validation batches per evaluation round.
    eval_batches: int = 10

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Pi0Value config instances for value-only training (freeze backbone, train value head only).
_pi0_value_config = pi0_value.Pi0ValueConfig()

_pi0_value_config_umi_bimanual = pi0_value.Pi0ValueConfig(
    pi05=True,
    action_dim=32,
    action_horizon=16,
    action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
    max_token_len=756,
    num_value_bins=201,
    value_min=-1.0,
    value_max=0.0,
    value_head_dropout=0.1,
    soft_value_targets=True,
)

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Value-only training (train_value.py): Pi0 backbone + value head, fit normalized value targets.
    #
    TrainConfig(
        name="pi0_value_umi",
        model=_pi0_value_config,
        freeze_filter=_pi0_value_config.get_freeze_filter_value_head_only(),
        data=FakeDataConfig(repo_id="fake"),
        c_fail_coef=1.0,
        value_clip_min=-1.0,
        value_clip_max=0.0,
    ),
    #
    # Value training for UMI bimanual with head view + depth (HITL dataset).
    # Freezes the Pi0.5 backbone and only trains the value head (final_norm, value_fc1, value_fc2).
    # Uses episode_index / frame_index from dataset for normalized value target computation.
    #
    # Usage:
    #   python scripts/train_value.py pi0_value_umi_bimanual_headview_depth --exp_name=my_value_exp
    #
    # If ``episode_metadata_path`` is unset, episode/task metadata is read from each local dataset's
    # ``meta/episodes.jsonl`` (and ``meta/tasks.jsonl``) when present — same layout as LeRobot.
    # Optional override:
    #   --episode_metadata_path=/path/to/episode_metadata.json
    #
    # episode_metadata.json format:
    #   {
    #     "episodes": [{"episode_index": 0, "task_index": 0, "length": 500, "success": true}, ...],
    #     "task_max_lengths": {"0": 600}
    #   }
    #
    TrainConfig(
        name="pi0_value_umi_bimanual_headview_depth",
        model=_pi0_value_config_umi_bimanual,
        freeze_filter=_pi0_value_config_umi_bimanual.get_freeze_filter_value_head_only(),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
            repo_id="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
            ),
            base_config=UmiDataConfig(
                action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                robot_type="ARM=2 G=1 H=0",
            ),
        ),
        weight_loader=CheckpointWeightLoaderWithValueHead(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=1e-4,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=16,
        fsdp_devices=8,
        log_interval=10,
        save_interval=1000,
        keep_period=40_000,
        c_fail_coef=1.0,
        value_clip_min=-1.0,
        value_clip_max=0.0,
        episode_metadata_path=None,
        val_ratio=0.1
    ),
    TrainConfig(
        name="pi0_value_umi_bimanual_headview_depth_infer",
        model=_pi0_value_config_umi_bimanual,
        freeze_filter=_pi0_value_config_umi_bimanual.get_freeze_filter_value_head_only(),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
            repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_eval_failure_20260404_161149_to_20260404_161519_ep4",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_eval_failure_20260404_161149_to_20260404_161519_ep4",
            ),
            base_config=UmiDataConfig(
                action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                robot_type="ARM=2 G=1 H=0",
            ),
        ),
        weight_loader=CheckpointWeightLoaderWithValueHead(
            "/root/openpi-umi/checkpoints/pi0_value_umi_bimanual_headview_depth/my_experiment_v2/79999/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=1e-4,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=16,
        fsdp_devices=8,
        log_interval=10,
        save_interval=1000,
        keep_period=30_000,
        c_fail_coef=1.0,
        value_clip_min=-1.0,
        value_clip_max=0.0,
        episode_metadata_path=None,
        val_ratio=0.1
    ),
    TrainConfig(
        name="pi0_value_umi_bimanual_headview_depth_multi_dataset",
        model=_pi0_value_config_umi_bimanual,
        freeze_filter=_pi0_value_config_umi_bimanual.get_freeze_filter_value_head_only(),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/horizon_cloth_folding_value_training_20260401_ep116_unified",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/horizon_cloth_folding_value_training_20260401_ep116_unified",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260323",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260323",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260316",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260316",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                )
            ],
        ),
        weight_loader=CheckpointWeightLoaderWithValueHead(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=70_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        save_interval=2000,
        keep_period=40_000,
        c_fail_coef=1.0,
        value_clip_min=-1.0,
        value_clip_max=0.0,
        episode_metadata_path=None,
        val_ratio=0.1
    ),
    #
    # ACP (Advantage-Conditioned Policy) training: Pi0.5 hybrid + Advantage tag in prompt.
    # Requires dataset with ``is_positive`` column (produced by ``scripts/lerobot_value_infer.py``).
    #
    # Usage:
    #   python scripts/train.py pi05_acp_umi_bimanual_headview_depth --exp_name=my_acp_exp
    #
    TrainConfig(
        name="pi05_acp_umi_bimanual_headview_depth",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_hitl_test",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_hitl_test",
            ),
            base_config=UmiDataConfig(
                action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                robot_type="ARM=2 G=1 H=0",
            ),
            acp_dropout=0.1,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=8e-5,
            decay_steps=40_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=50_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=2,
        log_interval=10,
        save_interval=2000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_acp_umi_bimanual_headview_depth_multi_dataset",
        model=pi0_gripper.Pi0GripperConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            gripper_binary_indices=(9, 19),
            gripper_binary_threshold=0.03,
            gripper_binary_loss_weight=0.5,
            gripper_binary_close_value=0.0,
            gripper_binary_open_value=0.085,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[5.0, 1.0, 1.0, 0.5, 5.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_messy_demostration/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_messy_demostration/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260417_20260418_ep171",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260417_20260418_ep171",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260422_ep100",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260422_ep100",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_ep102",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_ep102",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
            ]
        ),
        # 从标准 pi05 checkpoint 初始化时，新增的 gripper_binary_head 会保留随机初始化；
        # 后续训练保存出的 checkpoint 会自动包含这个 head。
        weight_loader=weight_loaders.CheckpointWeightLoaderWithGripperHead(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=40000,
    ),
    TrainConfig(
        name="pi05_acp_umi_bimanual_headview_depth_bimanual_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=())
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1/params"),
    ),
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Fine-tuning UMI configs.
    #
    # These configs are for training on UMI robot dataset.
    # UMI uses end-effector control with 7D actions (position, rotation, gripper).
    TrainConfig(
        name="pi0_umi",
        # Using pi0 model for full fine-tuning
        model=pi0_config.Pi0Config(
            action_dim=7,  # UMI has 7D actions
            action_horizon=10,  # Typical action chunk size
        ),
        # Configure the UMI dataset
        data=LeRobotUmiDataConfig(
            repo_id="your_hf_username/umi_dataset",  # Replace with your HuggingFace dataset repo
            base_config=DataConfig(
                prompt_from_task=True,  # Load task instructions from the dataset
            ),
            use_delta_actions=True,  # Convert absolute actions to delta actions
        ),
        # Load from pi0 base checkpoint for transfer learning
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0_umi_low_mem_finetune",
        # Using pi0 with LoRA for memory-efficient fine-tuning
        model=pi0_config.Pi0Config(
            action_dim=7,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotUmiDataConfig(
            repo_id="your_hf_username/umi_dataset",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_umi",
        # Using pi0-FAST model for faster inference
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,  # Single-arm robot, so ~180 tokens is sufficient
        ),
        data=LeRobotUmiDataConfig(
            repo_id="your_hf_username/umi_dataset",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0_fast_umi_low_mem_finetune",
        # Using pi0-FAST with LoRA for memory-efficient fine-tuning
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotUmiDataConfig(
            repo_id="your_hf_username/umi_dataset",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_umi",
        # Using pi0.5 model - more capable than pi0, with better performance
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=7,
            action_horizon=10,
            discrete_state_input=False,  # UMI uses continuous state
        ),
        data=LeRobotUmiDataConfig(
            repo_id="/root/openpi/umi_lerobot_dataset",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Note: pi05_base has action_dim=32 (DROID), incompatible with UMI's action_dim=7
        # Training from scratch instead
        weight_loader=weight_loaders.NoOpWeightLoader(),
        # PI0.5 typically uses cosine decay schedule with lower learning rate
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=1,  # Must be divisible by number of GPUs (4)
        num_workers=0,  # Set to 0 to avoid /dev/shm space issues
        # fsdp_devices=4,  # Temporarily disabled - may be incompatible with Pi0.5
    ),
    #
    # Online UMI configs (merged from config.py.online).
    #
    TrainConfig(
        name="pi05_umi_10d_80k_95_real_umi_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=10,
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotUmiDataConfigPadded_V3(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            ),
            base_config=DataConfig(prompt_from_task=True),
            training_mode=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_10d_80k_95_real_umi_v2/my_experiment/79999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=80_000,
        batch_size=32,
        num_workers=8,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V3(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=False),
            training_mode=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=90_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=100_000,
        batch_size=64,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_compute_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_compute_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_with_desk_height",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_with_desk_height",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_horizon1_compute_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon1(
            repo_id="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_wbcd_bimanual_horizon1_compute_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=WBCD_V1_Bimanual_Horizon1_Compute_Norm_Stats(
            repo_id="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_horizon1_compute_norm_stats_fsdp_2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon1(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_hitl_test_with_value",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_hitl_test_with_value",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=2,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_horizon2_compute_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon2(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_desk_height_20260216_all_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_desk_height_20260216_all_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_headview_horizon1_compute_norm_stats",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
        ),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_ImageHorizon1_Hybrid(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_error_with_depth_20260305_clip",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_error_with_depth_20260305_clip",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V3(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v7.1_test",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v7.1_test",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True),
            training_mode=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=95_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=100_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
        ),
        data=LeRobotUmiDataConfigPadded_V4_Hybrid(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_multi_task",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
            max_token_len=756,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[1.0, 4.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
                    assets=AssetsConfig(
                        asset_id="v73_merge",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
                        # 双臂: "ARM=2,G=0,H=0"；单臂: "ARM=1,G=0,H=0"（G=全局视角, H=高度，按需改为 1）
                        robot_type="ARM=1 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
                    assets=AssetsConfig(
                        asset_id="pick_elec",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
                        # 双臂: "ARM=2,G=0,H=0"；单臂: "ARM=1,G=0,H=0"（G=全局视角, H=高度，按需改为 1）
                        robot_type="ARM=1 G=0 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=70_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        keep_period=40000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_multi_dataset_fold_clothes",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
            max_token_len=756,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[4.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_20260126",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_20260126",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_with_desk_height",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_with_desk_height",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=70_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        keep_period=40000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_two_stage_augment",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
            max_token_len=756,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[1.0, 10.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_20260126",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_20260126",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_two_stage_28_02_clip",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_two_stage_28_02_clip",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_multi_dataset_fold_clothes/my_experiment/79999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=35_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=40_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        keep_period=40000,
    ),

    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_all_data",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
            max_token_len=756,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260316",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260316",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260303_20260304_clean",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260303_20260304_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260312_clean",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260312_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260305_clean",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260305_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_v2_clean",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_v2_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_merge_fold_red_clothes_cyrus_20251217_clean",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_merge_fold_red_clothes_cyrus_20251217_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=90_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=100_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        keep_period=50_000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_horizon_folding_260322",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260316",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260316",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                # LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                #     repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                #     assets=AssetsConfig(
                #         asset_id=".",
                #         assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                #     ),
                #     base_config=UmiDataConfig(
                #         action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                #         robot_type="ARM=2 G=1 H=0",
                #     ),
                # ),
                # LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                #     repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320",
                #     assets=AssetsConfig(
                #         asset_id=".",
                #         assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320",
                #     ),
                #     base_config=UmiDataConfig(
                #         action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                #         robot_type="ARM=2 G=1 H=0",
                #     ),
                # ),
                # LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                #     repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260323",
                #     assets=AssetsConfig(
                #         asset_id=".",
                #         assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260323",
                #     ),
                #     base_config=UmiDataConfig(
                #         action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                #         robot_type="ARM=2 G=1 H=0",
                #     ),
                # ),
            ]
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_horizon_folding_260322/my_experiment/19000/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=16,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_umi_wbcd_v3_260522_h32",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=360,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.6, 0.6],
            datasets=[
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_wbcd_v3_260522_h32/260602/69999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=16,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_umi_wbcd_v1_260511",
        model=pi0_gripper.Pi0GripperConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=360,
            gripper_binary_indices=(9, 19),
            gripper_binary_threshold=0.03,
            gripper_binary_loss_weight=0.5,
            gripper_binary_close_value=0.0,
            gripper_binary_open_value=0.085,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[1.0, 1.0, 1.0],
            datasets=[
                WBCD_Bimamual_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/wbcd_260511_ori",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/wbcd_260511_ori",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                WBCD_Bimamual_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/umi/wbcd_umi_260510_260511",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/umi/wbcd_umi_260510_260511",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                WBCD_Bimamual_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/wbcd_260513",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/wbcd_260513",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithGripperHead("/root/openpi-umi/checkpoints/pi05_umi_wbcd_v1_260511/my_experiment/79999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=70_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=16,
        fsdp_devices=8,
        log_interval=10,
        keep_period=40_000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_messy_folding_gripper_binary_260422",
        model=pi0_gripper.Pi0GripperConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            gripper_binary_indices=(9, 19),
            gripper_binary_threshold=0.03,
            gripper_binary_loss_weight=0.5,
            gripper_binary_close_value=0.0,
            gripper_binary_open_value=0.085,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[5.0, 1.0, 1.0, 0.5, 5.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[1.0, 1.0, 2.0, 7.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                # LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                #     repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                #     assets=AssetsConfig(
                #         asset_id=".",
                #         assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                #     ),
                #     base_config=UmiDataConfig(
                #         action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                #         robot_type="ARM=2 G=1 H=0",
                #     ),
                # ),
                # LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                #     repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260417_20260418_ep171",
                #     assets=AssetsConfig(
                #         asset_id=".",
                #         assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260417_20260418_ep171",
                #     ),
                #     base_config=UmiDataConfig(
                #         action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                #         robot_type="ARM=2 G=1 H=0",
                #     ),
                # ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260422_ep100",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260422_ep100",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_ep102",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_ep102",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
            ]
        ),
        # 从标准 pi05 checkpoint 初始化时，新增的 gripper_binary_head 会保留随机初始化；
        # 后续训练保存出的 checkpoint 会自动包含这个 head。
        weight_loader=weight_loaders.CheckpointWeightLoaderWithGripperHead(
            "/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_messy_folding_gripper_binary_260422/my_experiment_v3/19999/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=20000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_messy_folding_260410",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_horizon_folding_260322/my_experiment_v2/59999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V5(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=25_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_freeze_vlm_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ).get_freeze_filter_freeze_vlm_only(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=40_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=20000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_bimanual",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_merge_mason_ray_cyrus_20251224_20260107_exclude_25_all_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_merge_mason_ray_cyrus_20251224_20260107_exclude_25_all_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=70_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=20000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_10d_relative",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V2(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v4_train",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v4_train",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
            use_relative_state=True,
            use_10d_pose=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=80_000,
        batch_size=32,
        num_workers=8,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v3_train",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v3_train",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=80_000,
        batch_size=32,
        num_workers=8,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_32d_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=UmiArxInferenceDataConfig(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_v2/my_experiment/79999/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_retrain",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v3",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v3",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_32d/my_experiment_v2/79999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=80_000,
        batch_size=32,
        num_workers=8,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
            prompt="pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_pick_elec_hybrid_20260126",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_pick_elec_hybrid_20260126",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
            prompt="pick up electronic components and place them into the correct boxes",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIgnoreDiscreteHead(
            "/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_pick_elec_hybrid_20260126/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v5_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_v5_20260122",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_v5_20260122",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_v5_20260122/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_freeze_vlm_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216_freeze_vlm_only",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216_freeze_vlm_only",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216_freeze_vlm_only/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_bimanual_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=())
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d",
        # Using 32-dim actions (padded from 7-dim) to be compatible with pi05_base pretrained model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # Padded to match pretrained model
            action_horizon=10,
            discrete_state_input=False,
            # Only compute loss on first 7 dimensions (real UMI actions), ignore padded dims
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v3",  # New dataset with 32-dim actions
            assets=AssetsConfig(
                # Will load norm_stats from assets/pi05_umi_32d/umi_lerobot_dataset_32d/
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v3",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Now we can load pretrained weights!
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=32,
        num_workers=8,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_32d_mp",
        # Using 32-dim actions (padded from 7-dim) to be compatible with pi05_base pretrained model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # Padded to match pretrained model
            action_horizon=10,
            discrete_state_input=False,
            # Only compute loss on first 7 dimensions (real UMI actions), ignore padded dims
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi/umi_lerobot_dataset_7d",  # New dataset with 32-dim actions
            assets=AssetsConfig(
                # Will load norm_stats from assets/pi05_umi_32d/umi_lerobot_dataset_32d/
                asset_id=".",
                assets_dir="/root/openpi/umi_lerobot_dataset_7d",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Now we can load pretrained weights!
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=4,
        num_workers=2,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_umi_multi_dataset",
        # Multi-dataset training: concatenate multiple UMI datasets with optional per-dataset weights.
        # Use scripts/train_multi_dataset.py (not train_hybrid.py) for this config.
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
        ),
        data=MultiDataConfigFactory(
            datasets=[
                LeRobotUmiDataConfigPadded(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v3",
                    assets=AssetsConfig(asset_id=".", assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v3"),
                    base_config=DataConfig(prompt_from_task=True),
                    use_delta_actions=True,
                ),
                # Add more datasets here; each can have different repo_id and norm_stats.
                # LeRobotUmiDataConfigPadded(
                #     repo_id="/path/to/second_dataset",
                #     assets=AssetsConfig(asset_id=".", assets_dir="/path/to/second_dataset"),
                #     ...
                # ),
            ],
            weights=None,  # Optional: list of float per dataset for sampling; None = uniform
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=4,
    ),
    TrainConfig(
        name="pi05_umi_32d_lora",
        # LoRA version with 32-dim actions
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Only compute loss on first 7 dimensions (real UMI actions), ignore padded dims
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfig(
            repo_id="/root/openpi/umi_lerobot_dataset_32d",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset_32d",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Load pretrained weights before LoRA fine-tuning
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=4,  # Reduced batch size per GPU
        num_workers=0,  # Set to 0 to avoid multiprocessing overhead
        fsdp_devices=1,  # Disable FSDP - use data parallel instead
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
    ),
    TrainConfig(
        name="pi05_umi_32d_lora_v3.1",
        seed=43,
        # LoRA version with 32-dim actions
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Only compute loss on first 7 dimensions (real UMI actions), ignore padded dims
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi/umi_lerobot_dataset_7d",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset_7d",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Load pretrained weights before LoRA fine-tuning
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=32,  # Reduced batch size per GPU
        num_workers=0,  # Set to 0 to avoid multiprocessing overhead
        fsdp_devices=1,  # Disable FSDP - use data parallel instead
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]}
    ),
    TrainConfig(
        name="pi05_umi_lora",
        # LoRA fine-tuning for low memory usage
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=7,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotUmiDataConfig(
            repo_id="/root/openpi/umi_lerobot_dataset_7d",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset_7d",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Training from scratch with LoRA architecture (can't load pi05_base due to action_dim mismatch)
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=80_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # Turn off EMA for LoRA training to save memory
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=8,  # Can use larger batch with LoRA
        num_workers=0,
        # Freeze non-LoRA parameters (though we're training from scratch, this defines the architecture)
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
    ),
    TrainConfig(
        name="pi05_umi_from_scratch",
        # Training pi0.5 from scratch (not recommended unless you have large dataset)
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=7,
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotUmiDataConfig(
            repo_id="/root/openpi/umi_lerobot_dataset",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # No weight loader - training from scratch
        weight_loader=weight_loaders.NoOpWeightLoader(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=1e-4,
            decay_steps=100_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=100_000,
        batch_size=4,
        num_workers=0,  # Set to 0 to avoid /dev/shm space issues
        fsdp_devices=4,  # Enable FSDP to shard model across 4 GPUs to reduce memory per GPU
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_umi_32d_clean",
        # UMI with 32-dim padding (clean version - padding stays as 0, not -1.0)
        # Uses 7-dim norm_stats + pads after normalization
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi/umi_lerobot_dataset_7d",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset_7d",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Load pretrained weights
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=4,
        num_workers=0,
        fsdp_devices=1,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

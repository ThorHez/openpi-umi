"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Callable, Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, ClassVar, Literal, Protocol, TypeAlias

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
import openpi.models.pi0_mem as pi0_mem
import openpi.models.pi0_mem_compress as pi0_mem_compress
import openpi.models.pi0_mem_fixed_grid_query_action as pi0_mem_fixed_grid_query_action
import openpi.models.pi0_mem_post_transformer as pi0_mem_post_transformer
import openpi.models.pi0_mem_pf as pi0_mem_pf
import openpi.models.pi0_mem_pf_safe as pi0_mem_pf_safe
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


@dataclasses.dataclass(frozen=True)
class ComposableDataConfig(DataConfigFactory):
    """Compile one explicit data composition without adding a class per variant.

    Dataset and normalization paths still come from ``repo_id`` / ``assets``;
    the JSON composition only declares the already existing sample semantics.
    """

    COMPOSITION_SCHEMA_VERSION: ClassVar[int] = 1
    composition: dict | None = None
    acp_dropout: float = 0.1
    training_stage: str = "sft"

    def processing_contract(self, model_config: _model.BaseModelConfig) -> dict:
        from openpi.training.composed_processing import composition_runtime_contract

        return composition_runtime_contract(self, model_config)

    def processing_sample_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Build the real local data transforms without tokenization or downloads."""
        from openpi.training.composed_processing import compose_data_config

        # create_base_config retains the actual normalization statistics. Its
        # loader short-circuits local paths; reject remote assets before calling
        # it so a finite sample check cannot initiate a download.
        assets_root = str(self.assets.assets_dir or assets_dirs)
        asset_id = self.assets.asset_id or self.repo_id
        if not pathlib.Path(assets_root).is_absolute() or "://" in str(asset_id):
            raise ValueError("COMPOSITION_DATA_INVALID: 样本预检需要本机绝对归一化参数路径")
        return compose_data_config(self, assets_dirs, model_config, include_model_transforms=False)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        from openpi.training.composed_processing import compose_data_config

        return compose_data_config(self, assets_dirs, model_config)


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
class WBCD_Bimamual_2_views_ImageHorizon1(DataConfigFactory):
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
        # "left_wrist_1_rgb_0": "left_wrist_1_rgb_0",
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
        # "right_wrist_1_rgb_0": "right_wrist_1_rgb_0",
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

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.WBCD_V1_Bimanual_2_views_Horizon1()
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
class WBCD_Bimamual_4_views_ImageHorizon1_Value(DataConfigFactory):
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
        "episode_index": "episode_index",
        "frame_index": "frame_index",
        "value_target": "value_target",
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
class WBCD_Bimamual_4_views_ImageHorizon1_ACP(DataConfigFactory):
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
        "episode_index": "episode_index",
        "frame_index": "frame_index",
        "value_target": "value_target",
        # Required by ACPConditionPrompt (written by lerobot_value_infer.py).
        "is_positive": "is_positive",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -13, 6, -13),
    })

    acp_dropout: float = 0.1

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
class WBCD_Bimamual_4_views_ImageHorizon1_ACP_Inference(DataConfigFactory):
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
class LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem(DataConfigFactory):
    """Pi0Mem bimanual wrist-only data config.

    Mirrors ``LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon1`` for the
    state/action layout, normalize masks and repack mapping, but lays the
    image side out for Pi0Mem's video encoder:

    - Image keys ``left_wrist_0_rgb_0`` / ``right_wrist_0_rgb_0`` are loaded
      across ``num_frames`` historical frames by ``VideoFrameDataset`` (the
      training script wraps the LeRobot dataset before transforms run).
    - ``transforms_video.BuildVideoTensor`` then stacks the per-frame keys
      ``<base>_<t>`` into ``<base>_video`` of shape ``(T, 3, 224, 224)``.
    - A small inline Pi0Mem video-input transform (defined in
      ``openpi.training.config_pi0_mem``) builds the state vector and the
      ``image`` / ``image_mask`` dicts that Pi0Mem expects, with each image
      having shape ``(T, 224, 224, 3)``.

    NOTE: ``ResizeImages`` is intentionally omitted in ``model_transforms``
    because frames are already 224x224 and ``image_tools.resize_with_pad``
    does not understand a leading time dimension.
    """

    # Number of historical frames to stack per image stream (T).
    num_frames: int = 2
    # Stride (in raw dataset rows) between consecutive frames.
    frame_stride: int = 1
    # Behavior near the start of an episode: 'repeat' first valid frame, or 'zero' pad.
    padding_mode: str = "repeat"
    # Number of FUTURE frames appended after the current frame (Pi0MemPF
    # past-future bottleneck training). 0 keeps the original past-only clips.
    num_future_frames: int = 0
    # Stride (in raw dataset rows) between consecutive future frames.
    future_frame_stride: int = 1
    # Single-frame image keys to expand into video (look up history in the underlying
    # LeRobot dataset). Must match keys actually stored in the dataset.
    image_keys: tuple[str, ...] = ("left_wrist_0_rgb_0", "right_wrist_0_rgb_0")

    # Same masks as the Horizon1 sibling: 20-d bimanual actions, 38-d concatenated state.
    normalize_masks = {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state":   make_bool_mask(6, -12, 3, -7, 6, -12, 3, -7),
    }

    @property
    def total_frames(self) -> int:
        """Clip length per image stream: past+current plus optional future frames."""
        return self.num_frames + self.num_future_frames

    def video_frame_config(self):
        """VideoFrameConfig consumed by the Pi0Mem training script's data loader."""
        from openpi.training.mem.video_dataset import VideoFrameConfig

        return VideoFrameConfig(
            image_keys=tuple(self.image_keys),
            num_frames=self.num_frames,
            frame_stride=self.frame_stride,
            padding_mode=self.padding_mode,
            num_future_frames=self.num_future_frames,
            future_frame_stride=self.future_frame_stride,
        )

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        return dataclasses.replace(config, normalize_masks=self.normalize_masks)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Imported lazily to avoid a hard dependency from config.py on the
        # Pi0Mem-specific helper module.
        import openpi.training.config_pi0_mem as _config_pi0_mem
        from openpi import transforms_video as _transforms_video

        # The per-frame image keys produced by VideoFrameDataset are
        #   "<image_key>_<t>" for t in [0, total_frames), laid out as
        #   [oldest_past, ..., current, future...] (future only for Pi0MemPF).
        # The repack mapping keeps those keys (they will be stacked next).
        per_frame_keys = {
            f"{k}_{t}": f"{k}_{t}"
            for k in self.image_keys
            for t in range(self.total_frames)
        }
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # State (same as Horizon1).
                        "robot0_eef_pos": "robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "robot0_gripper_width",
                        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
                        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
                        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
                        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
                        "robot1_eef_pos": "robot1_eef_pos",
                        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
                        "robot1_gripper_width": "robot1_gripper_width",
                        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
                        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
                        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
                        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
                        # Per-frame images (e.g. left_wrist_0_rgb_0_0 .. _T-1).
                        **per_frame_keys,
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                _transforms_video.BuildVideoTensor(
                    image_keys=tuple(self.image_keys),
                    num_frames=self.total_frames,
                    output_keys={k: f"{k}_video" for k in self.image_keys},
                ),
                _config_pi0_mem.UmiInputsV4_Bimanual_Video(num_frames=self.total_frames),
            ],
        )

        # Same model_transforms as the Horizon1 sibling, minus ResizeImages.
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
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
class LeRobotUmiDataConfig_Bimamual_HeadView_Depth_Horizon1_Pi0Mem(DataConfigFactory):
    """Pi0Mem data config for bimanual wrist + head RGB/depth streams.

    Mirrors ``LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid``
    on state/action layout, while expanding every image stream into a T-frame
    video tensor for Pi0Mem / Pi0MemCompress.
    """

    num_frames: int = 2
    frame_stride: int = 1
    padding_mode: str = "repeat"
    image_keys: tuple[str, ...] = (
        "left_wrist_0_rgb_0",
        "right_wrist_0_rgb_0",
        "base_0_rgb_0",
        "base_0_depth_0",
    )
    depth_image_keys: tuple[str, ...] = ("base_0_depth_0",)

    normalize_masks = {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -12, 3, -7, 6, -12, 3, -7),
    }

    def video_frame_config(self):
        """VideoFrameConfig consumed by the Pi0Mem training script's data loader."""
        from openpi.training.mem.video_dataset import VideoFrameConfig

        return VideoFrameConfig(
            image_keys=tuple(self.image_keys),
            num_frames=self.num_frames,
            frame_stride=self.frame_stride,
            padding_mode=self.padding_mode,
        )

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        return dataclasses.replace(config, normalize_masks=self.normalize_masks)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        import openpi.training.config_pi0_mem as _config_pi0_mem
        from openpi import transforms_video as _transforms_video

        per_frame_keys = {
            f"{k}_{t}": f"{k}_{t}"
            for k in self.image_keys
            for t in range(self.num_frames)
        }
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
                        "robot1_eef_pos": "robot1_eef_pos",
                        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
                        "robot1_gripper_width": "robot1_gripper_width",
                        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
                        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
                        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
                        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
                        **per_frame_keys,
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        depth_frame_transforms = [
            _transforms.Transform_depth_to_3ch_image(depth_column_name=f"{key}_{t}")
            for key in self.depth_image_keys
            for t in range(self.num_frames)
        ]
        data_transforms = _transforms.Group(
            inputs=[
                *depth_frame_transforms,
                _transforms_video.BuildVideoTensor(
                    image_keys=tuple(self.image_keys),
                    num_frames=self.num_frames,
                    output_keys={k: f"{k}_video" for k in self.image_keys},
                ),
                _config_pi0_mem.UmiInputsV4_Bimanual_HeadView_Depth_Video(num_frames=self.num_frames),
            ],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
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
class LeRobotUmiDataConfig_shellgame_Pi0Mem(DataConfigFactory):
    """Pi0Mem data config for bimanual wrist + head RGB/depth streams.

    Mirrors ``LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid``
    on state/action layout, while expanding every image stream into a T-frame
    video tensor for Pi0Mem / Pi0MemCompress.
    """

    num_frames: int = 16
    frame_stride: int = 10
    padding_mode: str = "repeat"
    num_future_frames: int = 0
    future_frame_stride: int = 1
    image_keys: tuple[str, ...] = (
        "left_wrist_0_rgb_0",
        "left_wrist_0_rgb_1",
    )
    # depth_image_keys: tuple[str, ...] = ("base_0_depth_0",)

    normalize_masks = {
        "actions": make_bool_mask(10),
        "state": make_bool_mask(10),
    }

    @property
    def total_frames(self) -> int:
        return self.num_frames + self.num_future_frames

    def video_frame_config(self):
        """VideoFrameConfig consumed by the Pi0Mem training script's data loader."""
        from openpi.training.mem.video_dataset import VideoFrameConfig

        return VideoFrameConfig(
            image_keys=tuple(self.image_keys),
            num_frames=self.num_frames,
            frame_stride=self.frame_stride,
            padding_mode=self.padding_mode,
            num_future_frames=self.num_future_frames,
            future_frame_stride=self.future_frame_stride,
        )

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        return dataclasses.replace(config, normalize_masks=self.normalize_masks)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        import openpi.training.config_pi0_mem as _config_pi0_mem
        from openpi import transforms_video as _transforms_video

        per_frame_keys = {
            f"{k}_{t}": f"{k}_{t}"
            for k in self.image_keys
            for t in range(self.total_frames)
        }
        # VideoFrameDataset emits this nested mask; keep it through Repack so
        # UmiInputs can forward it onto Observation.frame_valid_masks.
        frame_valid_mask_keys = {
            "video_frame_valid_mask": {
                k: f"video_frame_valid_mask/{k}" for k in self.image_keys
            }
        }
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        **per_frame_keys,
                        **frame_valid_mask_keys,
                        "actions": "actions",
                        "prompt": "task",
                        # Kept out of policy inputs; used only by optional
                        # episode-level diagnostics in the MEM trainer.
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                    }
                )
            ]
        )

        # depth_frame_transforms = [
        #     _transforms.Transform_depth_to_3ch_image(depth_column_name=f"{key}_{t}")
        #     for key in self.depth_image_keys
        #     for t in range(self.num_frames)
        # ]
        data_transforms = _transforms.Group(
            inputs=[
                # *depth_frame_transforms,
                _transforms_video.BuildVideoTensor(
                    image_keys=tuple(self.image_keys),
                    num_frames=self.total_frames,
                    output_keys={k: f"{k}_video" for k in self.image_keys},
                ),
                _config_pi0_mem.UmiInputsV4_Shellgame_Video(num_frames=self.total_frames),
            ],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
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
class LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(DataConfigFactory):
    """Pi0Mem data config for bimanual wrist + head RGB/depth streams.

    Mirrors ``LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid``
    on state/action layout, while expanding every image stream into a T-frame
    video tensor for Pi0Mem / Pi0MemCompress.
    """

    num_frames: int = 16
    frame_stride: int = 10
    padding_mode: str = "repeat"
    num_future_frames: int = 0
    future_frame_stride: int = 1
    image_keys: tuple[str, ...] = (
        "left_wrist_0_rgb_0",
        "left_wrist_0_rgb_1",
    )
    # depth_image_keys: tuple[str, ...] = ("base_0_depth_0",)

    normalize_masks = {
        "actions": make_bool_mask(8),
        "state": make_bool_mask(8),
    }

    @property
    def total_frames(self) -> int:
        return self.num_frames + self.num_future_frames

    def video_frame_config(self):
        """VideoFrameConfig consumed by the Pi0Mem training script's data loader."""
        from openpi.training.mem.video_dataset import VideoFrameConfig

        return VideoFrameConfig(
            image_keys=tuple(self.image_keys),
            num_frames=self.num_frames,
            frame_stride=self.frame_stride,
            padding_mode=self.padding_mode,
            num_future_frames=self.num_future_frames,
            future_frame_stride=self.future_frame_stride,
        )

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        return dataclasses.replace(config, normalize_masks=self.normalize_masks)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        import openpi.training.config_pi0_mem as _config_pi0_mem
        from openpi import transforms_video as _transforms_video

        per_frame_keys = {
            f"{k}_{t}": f"{k}_{t}"
            for k in self.image_keys
            for t in range(self.total_frames)
        }
        # VideoFrameDataset emits this nested mask; keep it through Repack so
        # UmiInputs can forward it onto Observation.frame_valid_masks.
        frame_valid_mask_keys = {
            "video_frame_valid_mask": {
                k: f"video_frame_valid_mask/{k}" for k in self.image_keys
            }
        }
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_joint_pos": "observation.robot0_joint_pos",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        **per_frame_keys,
                        **frame_valid_mask_keys,
                        "actions": "actions",
                        "prompt": "task",
                        # Optional MEM classification diagnostics use these
                        # only as labels/masks, never as policy inputs.
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                    }
                )
            ]
        )

        # depth_frame_transforms = [
        #     _transforms.Transform_depth_to_3ch_image(depth_column_name=f"{key}_{t}")
        #     for key in self.depth_image_keys
        #     for t in range(self.num_frames)
        # ]
        data_transforms = _transforms.Group(
            inputs=[
                # *depth_frame_transforms,
                _transforms_video.BuildVideoTensor(
                    image_keys=tuple(self.image_keys),
                    num_frames=self.total_frames,
                    output_keys={k: f"{k}_video" for k in self.image_keys},
                ),
                _config_pi0_mem.UmiInputsV4_Shellgame_Video_Joint(num_frames=self.total_frames),
            ],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),
            ],
            # The model predicts its padded 32-D action space, while this
            # dataset's normalization statistics describe the real 8-D
            # absolute-joint action. Slice before policy unnormalization and
            # discard the padded state output for the same reason.
            outputs=[
                _transforms.ChunkActions(target_dim=8),
                _transforms.DropKeys(keys=("state",)),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_shellgame_Base(DataConfigFactory):
    """Pi0Mem data config for bimanual wrist + head RGB/depth streams.

    Mirrors ``LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid``
    on state/action layout, while expanding every image stream into a T-frame
    video tensor for Pi0Mem / Pi0MemCompress.
    """

    # num_frames: int = 16
    # frame_stride: int = 10
    # padding_mode: str = "repeat"
    # num_future_frames: int = 0
    # future_frame_stride: int = 1
    image_keys: tuple[str, ...] = (
        "left_wrist_0_rgb_0",
        "left_wrist_0_rgb_1",
    )
    # depth_image_keys: tuple[str, ...] = ("base_0_depth_0",)

    normalize_masks = {
        "actions": make_bool_mask(7),
        "state": make_bool_mask(10),
    }

    # @property
    # def total_frames(self) -> int:
    #     return self.num_frames + self.num_future_frames

    # def video_frame_config(self):
    #     """VideoFrameConfig consumed by the Pi0Mem training script's data loader."""
    #     from openpi.training.mem.video_dataset import VideoFrameConfig

    #     return VideoFrameConfig(
    #         image_keys=tuple(self.image_keys),
    #         num_frames=self.num_frames,
    #         frame_stride=self.frame_stride,
    #         padding_mode=self.padding_mode,
    #         num_future_frames=self.num_future_frames,
    #         future_frame_stride=self.future_frame_stride,
    #     )

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        return dataclasses.replace(config, normalize_masks=self.normalize_masks)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        import openpi.training.config_pi0_mem as _config_pi0_mem
        from openpi import transforms_video as _transforms_video

        # per_frame_keys = {
        #     f"{k}_{t}": f"{k}_{t}"
        #     for k in self.image_keys
        #     for t in range(self.total_frames)
        # }
        # VideoFrameDataset emits this nested mask; keep it through Repack so
        # UmiInputs can forward it onto Observation.frame_valid_masks.
        # frame_valid_mask_keys = {
        #     "video_frame_valid_mask": {
        #         k: f"video_frame_valid_mask/{k}" for k in self.image_keys
        #     }
        # }
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "robot0_eef_pos": "observation.robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "observation.robot0_gripper_width",
                        "left_wrist_0_rgb_0": "observation.left_wrist_0_rgb_0",
                        "left_wrist_0_rgb_1": "observation.left_wrist_0_rgb_1",
                        # **per_frame_keys,
                        # **frame_valid_mask_keys,
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        # depth_frame_transforms = [
        #     _transforms.Transform_depth_to_3ch_image(depth_column_name=f"{key}_{t}")
        #     for key in self.depth_image_keys
        #     for t in range(self.num_frames)
        # ]
        data_transforms = _transforms.Group(
            inputs=[
                # *depth_frame_transforms,
                # _transforms_video.BuildVideoTensor(
                #     image_keys=tuple(self.image_keys),
                #     num_frames=self.total_frames,
                #     output_keys={k: f"{k}_video" for k in self.image_keys},
                # ),
                _config_pi0_mem.UmiInputsV4_Shellgame_Base(),
            ],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
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
class LeRobotUmiDataConfig_shellgame_Pi0Mem_Inference(LeRobotUmiDataConfig_shellgame_Pi0Mem):
    """Shellgame Pi0Mem inference pipeline without future observations."""

    num_future_frames: int = 0
    action_dim: int = 10

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create(assets_dirs, model_config)
        model_transforms = config.model_transforms.push(
            outputs=[
                _transforms.ChunkActions(self.action_dim),
                _transforms.DropKeys(keys=("state",)),
            ]
        )
        return dataclasses.replace(config, model_transforms=model_transforms)


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_shellgame_Base_Inference(LeRobotUmiDataConfig_shellgame_Base):
    """Shellgame Pi0Mem inference pipeline without future observations."""

    num_future_frames: int = 0
    action_dim: int = 7

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create(assets_dirs, model_config)
        model_transforms = config.model_transforms.push(
            outputs=[
                _transforms.ChunkActions(self.action_dim),
                _transforms.DropKeys(keys=("state",)),
            ]
        )
        return dataclasses.replace(config, model_transforms=model_transforms)


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(DataConfigFactory):
    """Pi0Mem bimanual wrist-only data config.

    Mirrors ``LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon1`` for the
    state/action layout, normalize masks and repack mapping, but lays the
    image side out for Pi0Mem's video encoder:

    - Image keys ``left_wrist_0_rgb_0`` / ``right_wrist_0_rgb_0`` are loaded
      across ``num_frames`` historical frames by ``VideoFrameDataset`` (the
      training script wraps the LeRobot dataset before transforms run).
    - ``transforms_video.BuildVideoTensor`` then stacks the per-frame keys
      ``<base>_<t>`` into ``<base>_video`` of shape ``(T, 3, 224, 224)``.
    - A small inline Pi0Mem video-input transform (defined in
      ``openpi.training.config_pi0_mem``) builds the state vector and the
      ``image`` / ``image_mask`` dicts that Pi0Mem expects, with each image
      having shape ``(T, 224, 224, 3)``.

    NOTE: ``ResizeImages`` is intentionally omitted in ``model_transforms``
    because frames are already 224x224 and ``image_tools.resize_with_pad``
    does not understand a leading time dimension.
    """

    # Number of historical frames to stack per image stream (T).
    num_frames: int = 2
    # Stride (in raw dataset rows) between consecutive frames.
    frame_stride: int = 1
    # Behavior near the start of an episode: 'repeat' first valid frame, or 'zero' pad.
    padding_mode: str = "repeat"
    # Single-frame image keys to expand into video (look up history in the underlying
    # LeRobot dataset). Must match keys actually stored in the dataset.
    image_keys: tuple[str, ...] = ("left_wrist_0_rgb_0", "right_wrist_0_rgb_0", "left_wrist_1_rgb_0", "right_wrist_1_rgb_0")

    # Same masks as the Horizon1 sibling: 20-d bimanual actions, 38-d concatenated state.
    normalize_masks = {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state":   make_bool_mask(6, -13, 6, -13),
    }

    def video_frame_config(self):
        """VideoFrameConfig consumed by the Pi0Mem training script's data loader."""
        from openpi.training.mem.video_dataset import VideoFrameConfig

        return VideoFrameConfig(
            image_keys=tuple(self.image_keys),
            num_frames=self.num_frames,
            frame_stride=self.frame_stride,
            padding_mode=self.padding_mode,
        )

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        config = super().create_base_config(assets_dirs, model_config)
        return dataclasses.replace(config, normalize_masks=self.normalize_masks)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Imported lazily to avoid a hard dependency from config.py on the
        # Pi0Mem-specific helper module.
        import openpi.training.config_pi0_mem as _config_pi0_mem
        from openpi import transforms_video as _transforms_video

        # The per-frame image keys produced by VideoFrameDataset are
        #   "<image_key>_<t>" for t in [0, num_frames).
        # The repack mapping keeps those keys (they will be stacked next).
        per_frame_keys = {
            f"{k}_{t}": f"{k}_{t}"
            for k in self.image_keys
            for t in range(self.num_frames)
        }
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        # State (same as Horizon1).
                        "robot0_eef_pos": "robot0_eef_pos",
                        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
                        "robot0_gripper_width": "robot0_gripper_width",
                        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
                        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
                        # "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
                        # "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
                        "robot1_eef_pos": "robot1_eef_pos",
                        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
                        "robot1_gripper_width": "robot1_gripper_width",
                        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
                        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
                        # "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
                        # "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
                        # Per-frame images (e.g. left_wrist_0_rgb_0_0 .. _T-1).
                        **per_frame_keys,
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                _transforms_video.BuildVideoTensor(
                    image_keys=tuple(self.image_keys),
                    num_frames=self.num_frames,
                    output_keys={k: f"{k}_video" for k in self.image_keys},
                ),
                _config_pi0_mem.WBCD_V1_Bimanual_Video(num_frames=self.num_frames),
            ],
        )

        # Same model_transforms as the Horizon1 sibling, minus ResizeImages.
        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
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
class Shellgame_Compute_Norm_Stats(DataConfigFactory):
    """
    UMI data config for bimanual (V4).
    """

    normalize_masks = {
        "actions": make_bool_mask(3, -6, 1),
        "state": make_bool_mask(3, -6, 1),
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
                        # "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
                        # "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
                        #"robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
                        #"robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
                        #"left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
                        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
                        # "robot1_eef_pos": "robot1_eef_pos",
                        # "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
                        # "robot1_gripper_width": "robot1_gripper_width",
                        # "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
                        # "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
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
            inputs=[umi_policy.Shellgame_Compute_Norm_Stats()],
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
    # Optional source-level weights used only when merging normalization
    # statistics.  This is separate from ``weights`` because the latter is a
    # per-row sampler weight: differently sized datasets need inverse-size
    # correction to realize a desired source mixture.
    norm_weights: list[float] | None = None
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
                merge_weights = self.norm_weights if self.norm_weights is not None else self.weights
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


# Additional data factories retained from multi_dataset and value_infer.
@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_DeskHeight_ImageHorizon1_Hybrid(LeRobotUmiDataConfig_Hybrid):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_desk_height": "robot0_eef_desk_height",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_desk_height": "robot1_eef_desk_height",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
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
        "state": make_bool_mask(4, -12, 3, -7, 4, -12, 3, -7),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4_Bimanual_DeskHeight_Horizon1()


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_HeadView_DeskHeight_ImageHorizon1_Hybrid(LeRobotUmiDataConfig_Hybrid):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_desk_height": "robot0_eef_desk_height",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        # "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_desk_height": "robot1_eef_desk_height",
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

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4_Bimanual_HeadView_DeskHeight_Horizon1()


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimamual_HeadView_DeskHeight_ImageHorizon2_Hybrid(LeRobotUmiDataConfig_Hybrid):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_desk_height": "robot0_eef_desk_height",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        # "left_wrist_0_rgb_0": "left_wrist_0_rgb_0",
        "left_wrist_0_rgb_1": "left_wrist_0_rgb_1",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_desk_height": "robot1_eef_desk_height",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        # "right_wrist_0_rgb_0": "right_wrist_0_rgb_0",
        "right_wrist_0_rgb_1": "right_wrist_0_rgb_1",

        "base_0_rgb_1": "base_head_rgb_1",
        "actions": "actions",
        "prompt": "task",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(4, -12, 3, -7, 4, -12, 3, -7),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiInputsV4_Bimanual_HeadView_DeskHeight_Horizon2()


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Inference(DataConfigFactory):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "observation.robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "observation.robot0_gripper_width",
        "robot0_eef_rot_axis_angle_wrt_start": "observation.robot0_eef_rot_axis_angle_wrt_start",
        "camera0_rgb": "camera0_rgb",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7),
        "state": make_bool_mask(3, -13),
    })

    prompt: str = "pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box"

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxInputs()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxOutputs()

    robot_type: str = "ARM=1 G=0 H=0"
    action_target_dim: int = 10

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

        data_inputs = self.data_inputs_fn()
        if hasattr(data_inputs, 'prompt'):
            data_inputs = dataclasses.replace(data_inputs, prompt=self.prompt)
        data_transforms = _transforms.Group(
            inputs=[data_inputs],
            outputs=[self.data_outputs_fn()],
        )

        model_transforms = _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(None),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=model_config.discrete_state_input if hasattr(model_config, 'discrete_state_input') else False,
                    robot_type=self.robot_type,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),

            ],
            outputs=[
                _transforms.DropKeys(keys=("state",)),
                _transforms.ChunkActions(target_dim=self.action_target_dim),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Single_Inference(LeRobotUmiDataConfig_Inference):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "observation.robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "observation.robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "observation.robot0_gripper_width",
        "robot0_eef_rot_axis_angle_wrt_start": "observation.robot0_eef_rot_axis_angle_wrt_start",
        "camera0_rgb": "camera0_rgb",
    })


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimanual_Inference(LeRobotUmiDataConfig_Inference):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "camera0_rgb": "camera0_rgb",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "camera1_rgb": "camera1_rgb",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(3, -12, 3, 7, 3, -12, 3, 7),
    })

    robot_type: str = "ARM=2 G=0 H=0"

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxInputs_Bimanual()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxOutputs()

    action_target_dim: int = 20


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimanual_Inference_v2(LeRobotUmiDataConfig_Inference):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "camera0_rgb": "camera0_rgb",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "camera1_rgb": "camera1_rgb",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -12, 3, 7, 6, -12, 3, 7),
    })

    robot_type: str = "ARM=2 G=0 H=0"

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxInputs_Bimanual_v2()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxOutputs()

    action_target_dim: int = 20


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimanual_Inference_With_Depth(LeRobotUmiDataConfig_Inference):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "camera0_rgb": "camera0_rgb",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "camera1_rgb": "camera1_rgb",

        # "base_0_rgb": "camera_rs_rgb",
        # "base_0_depth": "camera_rs_depth",
        "base_0_rgb": "base_0_rgb",
        "base_0_depth": "base_0_depth",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -12, 3, 7, 6, -12, 3, 7),
    })

    robot_type: str = "ARM=2 G=1 H=0"

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxInputs_Bimanual_With_Depth()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxOutputs()

    action_target_dim: int = 20

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

        data_inputs = self.data_inputs_fn()
        if hasattr(data_inputs, 'prompt'):
            data_inputs = dataclasses.replace(data_inputs, prompt=self.prompt)
        data_transforms = _transforms.Group(
            inputs=[
                _transforms.Transform_depth_to_3ch_image(depth_column_name="base_0_depth"),
                data_inputs,
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
                    robot_type=self.robot_type,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),

            ],
            outputs=[
                _transforms.DropKeys(keys=("state",)),
                _transforms.ChunkActions(target_dim=self.action_target_dim),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_Bimanual_Inference_With_Depth_Ray(LeRobotUmiDataConfig_Inference):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_pos_wrt_start": "robot0_eef_pos_wrt_start",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "camera0_rgb": "camera0_rgb",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_pos_wrt_start": "robot1_eef_pos_wrt_start",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "camera1_rgb": "camera1_rgb",

        # "base_0_rgb": "camera_rs_rgb",
        # "base_0_depth": "camera_rs_depth",
        "head_rgb": "head_rgb",
    })

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(6, -12, 3, 7, 6, -12, 3, 7),
    })

    robot_type: str = "ARM=2 G=1 H=0"

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxInputs_Bimanual_With_Depth_Ray()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxOutputs()

    action_target_dim: int = 20

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

        data_inputs = self.data_inputs_fn()
        if hasattr(data_inputs, 'prompt'):
            data_inputs = dataclasses.replace(data_inputs, prompt=self.prompt)
        data_transforms = _transforms.Group(
            inputs=[
                _transforms.Transform_depth_to_3ch_image(depth_column_name="base_0_depth"),
                data_inputs,
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
                    robot_type=self.robot_type,
                ),
                _transforms.PadActionsOnly(model_config.action_dim),
                _transforms.FlattenState(),

            ],
            outputs=[
                _transforms.DropKeys(keys=("state",)),
                _transforms.ChunkActions(target_dim=self.action_target_dim),
            ],
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfig_DeskHeight_Bimanual_Inference(LeRobotUmiDataConfig_Inference):
    mapping: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "robot0_eef_pos": "robot0_eef_pos",
        "robot0_eef_rot_axis_angle": "robot0_eef_rot_axis_angle",
        "robot0_eef_desk_height": "robot0_eef_desk_height",
        "robot0_gripper_width": "robot0_gripper_width",
        "robot0_eef_rot_axis_angle_wrt_start": "robot0_eef_rot_axis_angle_wrt_start",
        "robot0_eef_pos_wrt1": "robot0_eef_pos_wrt1",
        "robot0_eef_rot_axis_angle_wrt1": "robot0_eef_rot_axis_angle_wrt1",
        "camera0_rgb": "camera0_rgb",
        "robot1_eef_pos": "robot1_eef_pos",
        "robot1_eef_desk_height": "robot1_eef_desk_height",
        "robot1_eef_rot_axis_angle": "robot1_eef_rot_axis_angle",
        "robot1_gripper_width": "robot1_gripper_width",
        "robot1_eef_rot_axis_angle_wrt_start": "robot1_eef_rot_axis_angle_wrt_start",
        "robot1_eef_pos_wrt0": "robot1_eef_pos_wrt0",
        "robot1_eef_rot_axis_angle_wrt0": "robot1_eef_rot_axis_angle_wrt0",
        "camera1_rgb": "camera1_rgb",
    })

    robot_type: str = "ARM=2 G=0 H=1"

    normalize_masks: dict[str, tuple[bool, ...]] = dataclasses.field(default_factory=lambda: {
        "actions": make_bool_mask(3, -7, 3, -7),
        "state": make_bool_mask(4, -12, 3, 7, 4, -12, 3, 7),
    })

    data_inputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxInputs_DeskHeight_Bimanual()
    data_outputs_fn: tyro.conf.Suppress[Any] = lambda: umi_policy.UmiArxOutputs()

    action_target_dim: int = 20


@dataclasses.dataclass(frozen=True)
class ShellgameCupEvalConfig:
    """Task-level validation for absolute-joint ShellGame policies.

    This evaluates the first grasp decision without stepping a controller:
    predicted joint chunks are converted to EEF positions with MuJoCo FK and
    assigned to the nearest settled cup.  The evaluator is intentionally
    opt-in because it depends on the ShellGame raw dataset and Robosuite.
    """

    enabled: bool = False
    raw_dataset_root: str = ""
    robosuite_root: str = ""
    interval: int = 1000
    num_episodes: int = 24
    batch_size: int = 8
    num_sampling_steps: int = 4
    sample_seed: int = 260806
    selection_radius: float = 0.06


@dataclasses.dataclass(frozen=True)
class ShellgameMemoryClassifierConfig:
    """Diagnostic supervision for final cup location from history_mem.

    Labels are read from LeRobot ``meta/episodes.jsonl`` and indexed by the
    sample's episode_index. The frame range must contain only observations for
    which the label is already determined and still visible in the configured
    history window.
    """

    enabled: bool = False
    episodes_metadata_path: str = ""
    label_key: str = "final_ball_cup"
    classes: tuple[str, ...] = ("left", "middle", "right")
    min_frame_index: int = 49
    max_frame_index: int = 64
    loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    # Optional diagnostic mode: select exactly this many frame-range samples
    # per class. A value of zero uses the complete dataset split.
    overfit_samples_per_class: int = 0
    # Evaluate on the exact same balanced subset. This intentionally measures
    # memorization capacity, not held-out generalization.
    overfit_same_samples_for_validation: bool = False
    # Disable stochastic image augmentation during classifier training. This
    # is useful for a fixed-input memorization sanity check, especially for
    # video where the default preprocessing augments every frame independently.
    disable_train_augmentation: bool = False


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDataConfigPadded_V4_Norm_stats(DataConfigFactory):
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
                        # "left_wrist_0_rgb_0": "observation.left_wrist_0_rgb_0",
                        # "left_wrist_0_rgb_1": "observation.left_wrist_0_rgb_1",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputsV4_Norm_stats()],
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
class LeRobotUmiDataConfigPadded_V4_ImageHorizon1(DataConfigFactory):
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
                        # "left_wrist_0_rgb_1": "observation.left_wrist_0_rgb_1",
                        "actions": "actions",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[umi_policy.UmiInputsV4_Horizon1()],
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
class UOTConfig:
    """Sim-and-Real Co-Training 的 Ours 配置；此处把 source/target 泛化为跨场景数据。"""

    enabled: bool = False
    pair_info_path: str = ""
    target_dataset_index: int = -1
    # 每个训练 batch 从各数据集抽样的概率；论文主实验为 source:target=0.9:0.1。
    bc_dataset_probabilities: tuple[float, ...] = ()
    ot_batch_size: int = 128
    alignment_weight: float = 0.1
    embedding_cost_scale: float = 1.0
    label_cost_scale: float = 1.0
    entropy_reg: float = 5e-4
    marginal_relaxation: float = 1e-3
    sinkhorn_iterations: int = 1000
    dtw_distance_cutoff: float = 5.0
    dtw_distance_sharpness: float = 10.0
    temporal_window: int = 10


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

    # Used only by scripts/train_uot.py; ordinary training keeps its existing behavior.
    uot: UOTConfig | None = None

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Optional global validation batch size. Defaults to the training batch size.
    # This is useful for small held-out splits that cannot fill a training batch.
    eval_batch_size: int | None = None
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # If true, include detailed data-loader/per-step timing metrics in logs.
    log_perf_metrics: bool = False
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

    # Optional task-level ShellGame joint-to-FK cup-selection validation.
    shellgame_cup_eval: ShellgameCupEvalConfig = dataclasses.field(default_factory=ShellgameCupEvalConfig)
    # Optional capacity diagnostic: predict the final cup location directly
    # from compressed visual history. Disabled for normal policy training.
    shellgame_memory_classifier: ShellgameMemoryClassifierConfig = dataclasses.field(
        default_factory=ShellgameMemoryClassifierConfig
    )

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

# Isolated PF-safe stage-1 model. Keep this object shared by ``model`` and the
# freeze filter so config changes cannot silently make the two disagree.
# Both history and future bottlenecks are intentionally small because this
# version initializes directly from Pi0.5 base rather than requiring shape
# compatibility with a trained Pi0MemCompress history checkpoint.
_pi0_mem_pf_safe_shellgame_joint_model = pi0_mem_pf_safe.Pi0MemPFSafeConfig(
    pi05=True,
    action_dim=32,
    action_horizon=16,
    action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
    max_token_len=256,
    num_frames=30,
    frame_stride=2,
    memory_every=27,
    history_memory_tokens=64,
    history_resampler_depth=1,
    history_use_current_condition=True,
    history_gate_init=-6.9,
    history_gate_fixed=None,
    history_gate_lr_multiplier=5.0,
    diversity_weight=1e-3,
    current_frame_index=-1,
    current_frame_corrupt_sample_prob=0.0,
    current_frame_dropout_prob=0.0,
    current_frame_mask_prob=0.0,
    current_frame_corrupt_loss_weight=0.0,
    num_future_frames=10,
    future_frame_stride=2,
    future_latent_tokens=32,
    future_gate_init=-6.9,
    future_gate_fixed=None,
    prior_encoder_depth=2,
    lambda_prior=1.0,
    lambda_post=1.0,
    lambda_align=0.01,
    lambda_reg=1e-3,
    align_warmup_steps=2_000,
    align_ramp_steps=3_000,
    latent_variance_target=0.5,
)

# Capacity probe initialized from the trained 30-frame joint checkpoint. Only
# HistoryResampler and the linear HistoryClassifier remain trainable.
_shellgame_history_classifier_probe_model = pi0_mem_compress.Pi0MemCompressConfig(
    pi05=True,
    action_dim=32,
    action_horizon=16,
    action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
    max_token_len=256,
    # One diagnostic sample per episode: frames 0..59 are history and frame
    # 60 is current (the scripted swaps are complete at this point).
    num_frames=61,
    memory_every=1,
    history_memory_tokens=256,
    history_resampler_depth=1,
    history_use_current_condition=True,
    history_gate_fixed=1.0,
    history_gate_lr_multiplier=1.0,
    diversity_weight=0.0,
    current_frame_index=-1,
    current_frame_corrupt_sample_prob=0.0,
    current_frame_dropout_prob=0.0,
    current_frame_mask_prob=0.0,
    current_frame_corrupt_loss_weight=0.0,
    history_classifier_num_classes=3,
)

# Formal generic action model following the controlled query-action probe.  It
# retains a 60-frame stride-1 history, encodes historical patches through the
# lightweight fixed-grid temporal path, and exposes the resulting memory to
# the Pi0.5 action expert only through a dedicated learned-query interface.
_shellgame_joint_fixed_grid_query_action_model = (
    pi0_mem_fixed_grid_query_action.Pi0MemFixedGridQueryActionConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=60,
        current_frame_index=-1,
        # Build history memory, but do not route it through the old periodic
        # current-image readers.  The dedicated action cross-attention is the
        # sole history-to-action path in this controlled formal model.
        memory_every=0,
        history_memory_tokens=128,
        history_resampler_depth=1,
        history_use_current_condition=False,
        history_gate_fixed=None,
        diversity_weight=1e-3,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        temporal_width=256,
        temporal_depth=2,
        temporal_heads=8,
        spatial_pool_factor=2,
        action_memory_query_tokens=16,
        action_memory_query_width=256,
        action_memory_query_depth=2,
        action_memory_query_heads=4,
        action_memory_cross_attention_heads=8,
        action_memory_gate_init=1.0,
    )
)

# Isolated capacity experiment: compress frames 0..59 first, then run the
# compressed memory and frame 60 through pretrained SigLIP blocks.  The first
# 18 blocks keep the two token groups separate; the last 9 jointly update them.
_shellgame_history_post_transformer_probe_model = (
    pi0_mem_post_transformer.Pi0MemPostTransformerConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
        max_token_len=256,
        num_frames=61,
        history_memory_tokens=256,
        history_resampler_depth=1,
        history_use_current_condition=True,
        history_joint_start_layer=18,
        diversity_weight=0.0,
        current_frame_index=-1,
        current_frame_corrupt_sample_prob=0.0,
        current_frame_dropout_prob=0.0,
        current_frame_mask_prob=0.0,
        current_frame_corrupt_loss_weight=0.0,
        history_classifier_num_classes=3,
    )
)

# Import task recipes only after the shared config dataclasses are defined.
# This keeps task-specific data and model policy code out of this registry.
from openpi.training.mem.recipes import shellgame_semantic_action as _shellgame_semantic_action_recipe  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_memory_adapt as _shellgame_real_memory_adapt_recipe  # noqa: E402
from openpi.training.mem.recipes import shellgame_real_wrist_stage2 as _shellgame_real_wrist_stage2_recipe  # noqa: E402

_shellgame_semantic_action_train_config = (
    _shellgame_semantic_action_recipe.make_train_config()
)
_shellgame_real_wrist_stage2_train_config = (
    _shellgame_real_wrist_stage2_recipe.make_train_config()
)
_shellgame_real_memory_adapt_train_config = (
    _shellgame_real_memory_adapt_recipe.make_train_config()
)

# Temporary import compatibility for local experiment scripts. New code should
# import the task recipe directly from openpi.training.mem.recipes.
LeRobotUmiDataConfig_shellgame_Pi0Mem_AbsoluteEEF7 = (
    _shellgame_semantic_action_recipe.data_config_type()
)

# Use `get_config` if you need to get a config by name in your code.
# Per-method recipe declarations share the dataclasses and processor types above.
# Keep this facade and its get_config()/cli() API for existing training scripts.
from openpi.training.recipes.sft import _CONFIGS as _SFT_CONFIGS  # noqa: E402
from openpi.training.recipes.recap import _CONFIGS as _RECAP_CONFIGS  # noqa: E402
from openpi.training.recipes.mem import _CONFIGS as _MEM_CONFIGS  # noqa: E402
from openpi.training.recipes.rlt import _CONFIGS as _RLT_CONFIGS  # noqa: E402
from openpi.training.recipes.uot import _CONFIGS as _UOT_CONFIGS  # noqa: E402

_CONFIGS = [*_SFT_CONFIGS, *_RECAP_CONFIGS, *_MEM_CONFIGS, *_RLT_CONFIGS, *_UOT_CONFIGS]

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

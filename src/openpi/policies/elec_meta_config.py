"""Serving-only adapter for the production single-arm elec_meta recipe.

The training recipe and its dataset paths stay intact. Serving uses the
checkpoint's merged normalization assets and the same state, image, token,
and action conventions, without opening any training dataset.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import pi0_config
from openpi.models import tokenizer
from openpi.policies.elec_meta_policy import ElecMetaInputs
from openpi.policies.elec_meta_policy import ElecMetaOutputs
from openpi.training import config

ELEC_META_RECIPE = "pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta"
NORMALIZE_MASKS = {"actions": (True,) * 3 + (False,) * 7, "state": (True,) * 3 + (False,) * 13}


@dataclasses.dataclass(frozen=True)
class ElecMetaInferenceDataConfig(config.DataConfigFactory):
    """Prepared two-frame single-arm observations, rather than LeRobot rows."""

    robot_type: str | None = None

    def create(self, assets_dirs, model_config):
        del assets_dirs  # Serving must not open the original dataset directories.
        return config.DataConfig(
            repo_id=self.repo_id,
            asset_id=self.assets.asset_id or ".",
            use_quantile_norm=True,
            normalize_masks=NORMALIZE_MASKS.copy(),
            robot_type=self.robot_type,
            data_transforms=transforms.Group(inputs=[ElecMetaInputs(model_config.action_horizon)]),
            model_transforms=transforms.Group(
                inputs=[
                    transforms.ResizeImages(224, 224),
                    # Preserve training order: tokenize the (2,16) state
                    # before FlattenState, including discrete-state prompts.
                    transforms.TokenizePrompt(
                        tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        discrete_state_input=model_config.discrete_state_input,
                    ),
                    transforms.PadActionsOnly(model_config.action_dim),
                    transforms.FlattenState(),
                    transforms.PadStateOnly(model_config.action_dim),
                    transforms.KeepModelKeys(),
                ],
                outputs=[ElecMetaOutputs()],
            ),
        )

    def validate_norm_stats(self, norm_stats):
        for key, width in (("actions", 10), ("state", 16)):
            if not norm_stats or key not in norm_stats:
                raise ValueError(f"elec_meta checkpoint needs {key} normalization statistics")
            for field in ("mean", "std", "q01", "q99"):
                values = np.asarray(getattr(norm_stats[key], field, None))
                if values.shape != (width,) or values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
                    raise ValueError(f"elec_meta {key}.{field} normalization must be finite with shape ({width},)")
            for field in ("min", "max"):
                value = getattr(norm_stats[key], field, None)
                if value is not None:
                    values = np.asarray(value)
                    if values.shape != (width,) or not np.all(np.isfinite(values)):
                        raise ValueError(f"elec_meta {key}.{field} normalization must have shape ({width},)")


def adapt_elec_meta_config(train_config):
    """Return an inference copy of the named recipe, preserving model fields."""
    model = train_config.model
    if (
        train_config.name != ELEC_META_RECIPE
        or type(model) not in (pi0_config.Pi0Config, pi0_config.Pi0RTCConfig)
        or not model.pi05
        or model.action_horizon != 16
        or model.action_dim != 32
        or model.action_loss_mask != (1.0,) * 10 + (0.0,) * 22
    ):
        raise ValueError("elec-meta profile requires the original 16-step, 32D production elec_meta Pi0.5 recipe")
    source = train_config.data.datasets[0]
    metadata = dict(train_config.policy_metadata or {})
    metadata.update(
        inference_profile="elec-meta",
        rtc={"action_horizon": 16, "action_dim": 10, "model_action_dim": 32},
    )
    return dataclasses.replace(
        train_config,
        data=ElecMetaInferenceDataConfig(
            repo_id="elec-meta-inference",
            assets=config.AssetsConfig(asset_id=source.assets.asset_id or "."),
            robot_type=source.base_config.robot_type if source.base_config is not None else None,
        ),
        policy_metadata=metadata,
    )

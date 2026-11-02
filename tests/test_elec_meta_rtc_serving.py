"""Validate the real elec_meta serving recipe without opening training data."""

import copy
import dataclasses

import numpy as np
import pytest

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies import elec_meta_config
from openpi.policies import elec_meta_policy
from openpi.policies import policy_config
from openpi.policies import rtc_policy
from openpi.training import config
from scripts import serve_policy


def norm_stats():
    return {
        key: transforms.NormStats(
            mean=np.zeros(width),
            std=np.ones(width),
            q01=-np.ones(width),
            q99=np.ones(width),
        )
        for key, width in (("actions", 10), ("state", 16))
    }


def observation():
    rotation = np.tile([1, 0, 0, 0, 1, 0], (2, 1)).astype(np.float32)
    return {
        "robot0_eef_pos": np.zeros((2, 3), np.float32),
        "robot0_eef_rot_axis_angle": rotation,
        "robot0_eef_rot_axis_angle_wrt_start": rotation.copy(),
        "robot0_gripper_width": np.full((2, 1), 0.03, np.float32),
        "camera0_rgb": np.zeros((2, 3, 224, 224), np.uint8),
        "prompt": "pick electronic components",
    }


def test_inference_copy_keeps_the_production_model_and_training_recipe(monkeypatch):
    original = config.get_config(elec_meta_config.ELEC_META_RECIPE)
    inferred = elec_meta_config.adapt_elec_meta_config(original)
    assert original.model is inferred.model
    assert original.data is not inferred.data
    assert isinstance(original.data, config.MultiDataConfigFactory)
    assert original.data.weights == [1.0, 20.0, 50.0]
    assert inferred.policy_metadata["inference_profile"] == "elec-meta"
    assert "inference_profile" not in (original.policy_metadata or {})

    def no_training_assets(*_args, **_kwargs):
        pytest.fail("Serving must not open dataset assets or call a training-data factory")

    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", no_training_assets)
    data_config = inferred.data.create(original.assets_dirs, inferred.model)
    assert data_config.asset_id == "."
    assert data_config.use_quantile_norm
    assert data_config.normalize_masks == elec_meta_config.NORMALIZE_MASKS

    source = observation()
    expected = transforms.compose(
        [
            elec_meta_policy.ElecMetaInputs(),
            transforms.Normalize(norm_stats(), use_quantiles=True, key_masks=elec_meta_config.NORMALIZE_MASKS),
            *data_config.model_transforms.inputs,
        ]
    )(source)
    assert expected["state"].shape == (32,)
    assert expected["actions"].shape == (16, 32)
    assert set(expected) <= transforms.MODEL_KEYS
    assert expected["tokenized_prompt"].shape == (original.model.max_token_len,)


def test_output_projection_happens_before_action_unnormalization():
    data = elec_meta_config.adapt_elec_meta_config(config.get_config(elec_meta_config.ELEC_META_RECIPE)).data
    data_config = data.create((), pi0_config.Pi0RTCConfig(pi05=True, action_horizon=16))
    output = transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats(), use_quantiles=True, key_masks=elec_meta_config.NORMALIZE_MASKS),
        ]
    )({"actions": np.ones((16, 32)), "state": np.zeros(32)})
    assert set(output) == {"actions"}
    assert output["actions"].shape == (16, 10)
    np.testing.assert_allclose(output["actions"], 1, atol=1e-6)


def test_real_training_transforms_and_tokens_match_scene_profile(monkeypatch):
    original = config.get_config(elec_meta_config.ELEC_META_RECIPE)
    stats = {
        key: transforms.NormStats(
            mean=np.linspace(-0.2, 0.3, width),
            std=np.linspace(0.7, 1.2, width),
            q01=np.linspace(-0.8, -0.3, width),
            q99=np.linspace(0.4, 1.1, width),
        )
        for key, width in (("actions", 10), ("state", 16))
    }
    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", lambda *_args: stats)
    training = original.data.create(original.assets_dirs, original.model)
    inference = elec_meta_config.adapt_elec_meta_config(original).data.create((), original.model)
    sample = observation()
    sample["robot0_eef_pos"] = np.array([[0.02, -0.03, 0.07], [0.06, -0.01, 0.09]], dtype=np.float32)
    sample["camera0_rgb"][:] = 113
    sample["left_wrist_0_rgb_1"] = sample["camera0_rgb"][-1].copy()
    sample["actions"] = np.tile(np.array([0.02, 0.01, -0.03, 1, 0, 0, 0, 1, 0, 0.03], np.float32), (16, 1))

    def apply(data_config):
        return transforms.compose(
            [
                *data_config.data_transforms.inputs,
                transforms.Normalize(stats, use_quantiles=True, key_masks=data_config.normalize_masks),
                *data_config.model_transforms.inputs,
            ]
        )(copy.deepcopy(sample))

    expected, actual = apply(training), apply(inference)
    for key in ("state", "actions", "tokenized_prompt", "tokenized_prompt_mask"):
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    for key in expected["image"]:
        np.testing.assert_array_equal(actual["image"][key], expected["image"][key])
        np.testing.assert_array_equal(actual["image_mask"][key], expected["image_mask"][key])


@pytest.mark.parametrize("bad", ["other", "dimension", "mask"])
def test_profile_rejects_another_recipe_or_action_contract(bad):
    original = config.get_config(elec_meta_config.ELEC_META_RECIPE)
    if bad == "other":
        original = dataclasses.replace(original, name="pi05_umi_bimanual_infer")
    else:
        field = {"action_dim": 20} if bad == "dimension" else {"action_loss_mask": (1.0,) * 32}
        original = dataclasses.replace(original, model=dataclasses.replace(original.model, **field))
    with pytest.raises(ValueError, match="production elec_meta"):
        elec_meta_config.adapt_elec_meta_config(original)


@pytest.mark.parametrize("bad_key", ["actions", "state"])
def test_incompatible_checkpoint_stats_fail_explicitly(bad_key):
    stats = norm_stats()
    factory = elec_meta_config.ElecMetaInferenceDataConfig()
    factory.validate_norm_stats(stats)
    del stats[bad_key]
    with pytest.raises(ValueError, match=bad_key):
        factory.validate_norm_stats(stats)


def test_real_cli_selects_the_serving_profile(monkeypatch):
    original = config.get_config(elec_meta_config.ELEC_META_RECIPE)
    seen = {}

    def capture(train_config, checkpoint, **kwargs):
        seen.update(config=train_config, checkpoint=checkpoint, options=kwargs)
        return seen

    monkeypatch.setattr(policy_config, "create_trained_policy", capture)
    args = serve_policy.tyro.cli(
        serve_policy.Args,
        args=[
            "--profile",
            "elec-meta",
            "--rtc.enabled",
            "--rtc.mode",
            "client",
            "policy:checkpoint",
            "--policy.config",
            original.name,
            "--policy.dir",
            "/checkpoint",
        ],
    )
    serve_policy.create_policy(args)
    assert isinstance(seen["config"].data, elec_meta_config.ElecMetaInferenceDataConfig)
    assert seen["options"]["client_rtc"] is True


@pytest.mark.parametrize(("enabled", "mode"), [(False, "client"), (True, "server")])
def test_scene_profile_requires_explicit_client_rtc(enabled, mode):
    with pytest.raises(ValueError, match="elec-meta profile requires"):
        serve_policy.create_policy(
            serve_policy.Args(
                profile="elec-meta",
                rtc=rtc_policy.RTCConfig(enabled=enabled, mode=mode),
                policy=serve_policy.Checkpoint(config=elec_meta_config.ELEC_META_RECIPE, dir="/checkpoint"),
            )
        )

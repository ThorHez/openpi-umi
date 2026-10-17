"""Camera regressions that run on CPU without loading a model/checkpoint/dataset."""
import dataclasses
import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from openpi.models import model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.models import pi0_mem
from openpi.models import pi0_mem_compress
from openpi.models import pi0_mem_pf
from openpi.models_pytorch import preprocessing_pytorch

LAYOUTS = (
    ("left_wrist_0_rgb", "right_wrist_0_rgb"),
    ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
    ("left_wrist_0_rgb", "left_wrist_1_rgb", "right_wrist_0_rgb", "right_wrist_1_rgb"),
    ("base_rgb", "wrist_rgb"),
)


def observation(keys, *, frames=None):
    shape = (2, 8, 8, 3) if frames is None else (2, frames, 8, 8, 3)
    return model.Observation(
        images={key: jnp.full(shape, (i + 1) / 10) for i, key in enumerate(keys)},
        image_masks={key: jnp.ones((2,), dtype=jnp.bool_) for key in keys},
        state=jnp.zeros((2, 4)),
        semantic_memory=jnp.ones((2, 3, 4)),
        episode_length=jnp.asarray([10, 20]),
        action_loss_mask=jnp.ones((2, 4)),
    )


@pytest.mark.parametrize("keys", LAYOUTS)
@pytest.mark.parametrize("compiled", [False, True])
def test_legacy_layout_keeps_every_camera_and_pixel(keys, compiled):
    obs = observation(keys)
    preprocess = functools.partial(model.preprocess_observation, None, image_resolution=(8, 8))
    if compiled:
        preprocess = jax.jit(preprocess)
    processed = preprocess(obs)
    assert set(processed.images) == set(keys)
    for key in keys:
        np.testing.assert_array_equal(processed.images[key], obs.images[key])
        np.testing.assert_array_equal(processed.image_masks[key], [True, True])
    np.testing.assert_array_equal(processed.semantic_memory, obs.semantic_memory)
    np.testing.assert_array_equal(processed.episode_length, [10, 20])
    np.testing.assert_array_equal(processed.action_loss_mask, obs.action_loss_mask)


def test_explicit_layout_pads_missing_views_and_respects_order():
    obs = observation(("right_wrist_0_rgb",))
    keys = LAYOUTS[3]
    with pytest.raises(ValueError, match="None of the configured image views"):
        model.preprocess_observation(None, obs, image_keys=keys)
    processed = model.preprocess_observation(
        None, obs, image_keys=LAYOUTS[0], depth_keys=(), image_resolution=(8, 8)
    )
    assert tuple(processed.images) == LAYOUTS[0]
    np.testing.assert_array_equal(processed.images["left_wrist_0_rgb"], 0)
    np.testing.assert_array_equal(processed.image_masks["left_wrist_0_rgb"], [False, False])
    np.testing.assert_array_equal(processed.images["right_wrist_0_rgb"], obs.images["right_wrist_0_rgb"])


@pytest.mark.parametrize("compiled", [False, True])
def test_video_preserves_time_axis_and_pads_frame_masks(compiled):
    obs = observation(("wrist_rgb",), frames=3)
    obs = dataclasses.replace(obs, frame_valid_masks={"wrist_rgb": jnp.asarray([[False, True, True], [True, True, True]])})
    preprocess = functools.partial(
        model.preprocess_observation, None, image_keys=LAYOUTS[3], depth_keys=(), image_resolution=(8, 8)
    )
    processed = jax.jit(preprocess)(obs) if compiled else preprocess(obs)
    assert processed.images["base_rgb"].shape == (2, 3, 8, 8, 3)
    np.testing.assert_array_equal(processed.images["wrist_rgb"], obs.images["wrist_rgb"])
    np.testing.assert_array_equal(processed.frame_valid_masks["wrist_rgb"], obs.frame_valid_masks["wrist_rgb"])
    np.testing.assert_array_equal(processed.frame_valid_masks["base_rgb"], np.zeros((2, 3), dtype=bool))
    np.testing.assert_array_equal(processed.image_masks["base_rgb"], [False, False])
    # Keep the serialized mask name consistent with Observation.from_dict.
    restored = model.Observation.from_dict(processed.to_dict())
    np.testing.assert_array_equal(restored.frame_valid_masks["wrist_rgb"], obs.frame_valid_masks["wrist_rgb"])


def test_video_augmentation_and_resize_keep_time_and_validity():
    obs = observation(("base_rgb", "wrist_rgb"), frames=3)
    obs = dataclasses.replace(obs, frame_valid_masks={key: jnp.ones((2, 3), dtype=bool) for key in obs.images})
    processed = model.preprocess_observation(jax.random.key(0), obs, train=True, image_resolution=(12, 12))
    assert all(image.shape == (2, 3, 12, 12, 3) for image in processed.images.values())
    assert all(mask.shape == (2, 3) for mask in processed.frame_valid_masks.values())
    assert all(np.isfinite(image).all() for image in processed.images.values())


def test_depth_is_retained_without_rgb_augmentation():
    obs = observation(("wrist_rgb", "base_0_depth"), frames=3)
    processed = model.preprocess_observation(jax.random.key(7), obs, train=True, image_resolution=(8, 8))
    np.testing.assert_array_equal(processed.images["base_0_depth"], obs.images["base_0_depth"])
    assert not np.allclose(processed.images["wrist_rgb"], obs.images["wrist_rgb"])


@pytest.mark.parametrize("compiled", [False, True])
def test_all_masked_samples_fail_including_inside_jit(compiled):
    obs = observation(("wrist_rgb",))
    obs = dataclasses.replace(obs, image_masks={"wrist_rgb": jnp.asarray([True, False])})
    preprocess = functools.partial(model.preprocess_observation, None, image_resolution=(8, 8))
    if compiled:
        preprocess = jax.jit(preprocess)
    with pytest.raises(Exception, match="no valid image view"):
        jax.block_until_ready(preprocess(obs))
    # Deliberate memory-only ablations remain available through an explicit opt-in.
    processed = model.preprocess_observation(None, obs, allow_all_masked_images=True, image_resolution=(8, 8))
    np.testing.assert_array_equal(processed.image_masks["wrist_rgb"], [True, False])


def test_all_invalid_video_frames_are_not_counted_as_an_available_camera():
    obs = observation(("wrist_rgb",), frames=3)
    obs = dataclasses.replace(obs, frame_valid_masks={"wrist_rgb": jnp.zeros((2, 3), dtype=bool)})
    with pytest.raises(ValueError, match="no valid image view"):
        model.preprocess_observation(None, obs)


@pytest.mark.parametrize("factory,frames", [
    (pi0_config.Pi0Config, None),
    (pi0_fast.Pi0FASTConfig, None),
    (functools.partial(pi0_mem.Pi0MemConfig, num_frames=3), 3),
    (functools.partial(pi0_mem_compress.Pi0MemCompressConfig, num_frames=3), 3),
    (functools.partial(pi0_mem_pf.Pi0MemPFConfig, num_frames=3, num_future_frames=2), 5),
])
def test_explicit_specs_and_runtime_use_the_same_views(factory, frames):
    config = factory(image_keys=LAYOUTS[3], depth_keys=("base_0_depth",))
    spec, _ = config.inputs_spec(batch_size=2)
    assert tuple(spec.images) == (*LAYOUTS[3], "base_0_depth")
    assert tuple(spec.image_masks) == tuple(spec.images)
    # No backbone allocation: exercise the exact inherited model preprocessing.
    probe = SimpleNamespace()
    model.BaseModel.configure_views(probe, config)
    processed = model.BaseModel.preprocess_observation(probe, None, config.fake_obs(2))
    assert tuple(processed.images) == tuple(spec.images)
    expected_shape = (2, 224, 224, 3) if frames is None else (2, frames, 224, 224, 3)
    assert all(image.shape == expected_shape for image in processed.images.values())


def test_overlapping_rgb_and_depth_configuration_is_rejected():
    with pytest.raises(ValueError, match="unique and disjoint"):
        pi0_config.Pi0Config(image_keys=("depth",), depth_keys=("depth",)).inputs_spec()


def torch_observation(keys):
    return model.Observation(
        images={key: torch.full((2, 3, 8, 8), (i + 1) / 10) for i, key in enumerate(keys)},
        image_masks={key: torch.ones(2, dtype=torch.bool) for key in keys},
        state=torch.zeros((2, 4)),
    )


@pytest.mark.parametrize("keys", LAYOUTS)
def test_torch_keeps_legacy_camera_layouts(keys):
    obs = torch_observation(keys)
    processed = preprocessing_pytorch.preprocess_observation_pytorch(obs, image_resolution=(8, 8))
    assert tuple(processed.images) == keys
    for key in keys:
        torch.testing.assert_close(processed.images[key], obs.images[key])


def test_torch_explicit_missing_views_and_empty_images():
    obs = torch_observation(("wrist_rgb",))
    processed = preprocessing_pytorch.preprocess_observation_pytorch(
        obs, image_keys=LAYOUTS[3], depth_keys=(), image_resolution=(8, 8)
    )
    assert not processed.image_masks["base_rgb"].any()
    torch.testing.assert_close(processed.images["wrist_rgb"], obs.images["wrist_rgb"])
    with pytest.raises(ValueError, match="None of the configured image views"):
        preprocessing_pytorch.preprocess_observation_pytorch(obs, image_keys=LAYOUTS[0])
    obs = dataclasses.replace(obs, image_masks={"wrist_rgb": torch.zeros(2, dtype=torch.bool)})
    with pytest.raises(RuntimeError, match="no valid image view"):
        preprocessing_pytorch.preprocess_observation_pytorch(obs, image_resolution=(8, 8))


def test_torch_depth_is_not_color_augmented_and_video_rejected():
    obs = torch_observation(("base_0_depth",))
    processed = preprocessing_pytorch.preprocess_observation_pytorch(obs, train=True, image_resolution=(8, 8))
    torch.testing.assert_close(processed.images["base_0_depth"], obs.images["base_0_depth"])
    obs = dataclasses.replace(obs, images={"base_0_depth": torch.zeros((2, 3, 3, 8, 8))})
    with pytest.raises(ValueError, match="JAX MEM policy"):
        preprocessing_pytorch.preprocess_observation_pytorch(obs)


def collate_sample(keys, *, frames=None, frame_mask=False):
    shape = (8, 8, 3) if frames is None else (frames, 8, 8, 3)
    item = {
        "image": {key: np.full(shape, (i + 1) / 10, dtype=np.float32) for i, key in enumerate(keys)},
        "image_mask": {key: np.bool_(True) for key in keys},
        "state": np.zeros(4, dtype=np.float32),
        "actions": np.zeros((2, 4), dtype=np.float32),
    }
    if frame_mask:
        item["frame_valid_mask"] = {key: np.asarray([False, True, True]) for key in keys}
    return item


def test_mixed_two_and_four_camera_batch_keeps_actual_views():
    from openpi.training.data_loader import _collate_fn

    two, four = collate_sample(LAYOUTS[0]), collate_sample(LAYOUTS[2])
    batch = _collate_fn([two, four])
    assert set(batch["image"]) == set(LAYOUTS[2])
    np.testing.assert_array_equal(batch["image_mask"]["left_wrist_1_rgb"], [False, True])
    np.testing.assert_array_equal(batch["image"]["left_wrist_1_rgb"][0], 0)
    np.testing.assert_array_equal(batch["image"]["left_wrist_1_rgb"][1], four["image"]["left_wrist_1_rgb"])
    processed = model.preprocess_observation(None, model.Observation.from_dict(batch), image_resolution=(8, 8))
    assert set(processed.images) == set(LAYOUTS[2])
    assert all(np.any([processed.image_masks[key][i] for key in processed.images]) for i in range(2))
    # Collation must not mutate cached dataset samples.
    assert tuple(two["image"]) == LAYOUTS[0]


def test_video_collation_pads_view_and_frame_masks_before_stacking():
    from openpi.training.data_loader import _collate_fn

    wrist = collate_sample(("wrist_rgb",), frames=3)
    both = collate_sample(LAYOUTS[3], frames=3, frame_mask=True)
    batch = _collate_fn([wrist, both], image_keys=LAYOUTS[3], depth_keys=())
    assert batch["image"]["base_rgb"].shape == (2, 3, 8, 8, 3)
    np.testing.assert_array_equal(batch["frame_valid_mask"]["base_rgb"], [[False, False, False], [False, True, True]])
    np.testing.assert_array_equal(batch["frame_valid_mask"]["wrist_rgb"][0], [True, True, True])
    processed = model.preprocess_observation(None, model.Observation.from_dict(batch), image_resolution=(8, 8))
    np.testing.assert_array_equal(processed.image_masks["base_rgb"], [False, True])


def test_camera_union_does_not_relax_non_camera_fields():
    from openpi.training.data_loader import _collate_fn

    left = collate_sample(LAYOUTS[0])
    right = {**collate_sample(LAYOUTS[2]), "state": np.zeros(5)}
    with pytest.raises(ValueError):
        _collate_fn([left, right])


def test_default_order_is_identical_for_eager_jit_and_torch():
    keys = ("wrist_rgb", "base_rgb")
    obs = observation(keys)
    preprocess = functools.partial(model.preprocess_observation, None, image_resolution=(8, 8))
    eager = preprocess(obs)
    compiled = jax.jit(preprocess)(obs)
    torch_processed = preprocessing_pytorch.preprocess_observation_pytorch(
        torch_observation(keys), image_resolution=(8, 8)
    )
    assert tuple(eager.images) == tuple(compiled.images) == tuple(torch_processed.images) == ("base_rgb", "wrist_rgb")


def test_uot_paired_observations_normalize_nested_camera_unions():
    from openpi.training.data_loader import _collate_fn

    pair1 = {
        "source": collate_sample(LAYOUTS[0]),
        "target": collate_sample(("wrist_rgb",), frames=3),
        "source_label": np.int32(1),
        "target_label": np.int32(2),
    }
    pair2 = {
        "source": collate_sample(LAYOUTS[2]),
        "target": collate_sample(LAYOUTS[3], frames=3, frame_mask=True),
        "source_label": np.int32(3),
        "target_label": np.int32(4),
    }
    batch = _collate_fn([pair1, pair2])
    assert set(batch["source"]["image"]) == set(LAYOUTS[2])
    assert batch["target"]["image"]["base_rgb"].shape == (2, 3, 8, 8, 3)
    np.testing.assert_array_equal(batch["source"]["image_mask"]["left_wrist_1_rgb"], [False, True])
    np.testing.assert_array_equal(batch["target"]["frame_valid_mask"]["base_rgb"], [[False, False, False], [False, True, True]])
    np.testing.assert_array_equal(batch["source_label"], [1, 3])
    np.testing.assert_array_equal(batch["target_label"], [2, 4])
    with pytest.raises(ValueError):
        _collate_fn([pair1, {**pair2, "source": {**pair2["source"], "state": np.zeros(7)}}])


def test_norm_statistics_can_collate_empty_images_but_model_rejects_them():
    from openpi.training.data_loader import _collate_fn

    first = collate_sample(())
    second = collate_sample(())
    batch = _collate_fn([first, second])
    assert batch["image"] == batch["image_mask"] == {}
    assert batch["state"].shape == (2, 4)
    assert batch["actions"].shape == (2, 2, 4)
    with pytest.raises(ValueError, match="No image views"):
        model.preprocess_observation(None, model.Observation.from_dict(batch))


def test_internal_processor_timing_metadata_uses_union_without_relaxing_business_data():
    from openpi.training.data_loader import _collate_fn

    first = {**collate_sample(LAYOUTS[0]), "__perf": {"worker/V4_s": 0.1, "worker/base_getitem_s": 0.2}}
    second = {**collate_sample(LAYOUTS[2]), "__perf": {"worker/Horizon1_s": 0.3, "worker/base_getitem_s": 0.4}}
    batch = _collate_fn([first, second])
    timing = batch.pop("__perf")
    np.testing.assert_allclose(timing["worker/V4_s"], [0.1, 0.0])
    np.testing.assert_allclose(timing["worker/Horizon1_s"], [0.0, 0.3])
    np.testing.assert_allclose(timing["worker/base_getitem_s"], [0.2, 0.4])
    assert batch["state"].shape == (2, 4)
    assert tuple(first["__perf"]) == ("worker/V4_s", "worker/base_getitem_s")
    with pytest.raises(ValueError):
        _collate_fn([first, {**second, "actions": np.zeros((3, 4))}])


def test_nested_uot_timing_is_hoisted_for_torch_loader_batch_pop():
    from openpi.training.data_loader import _collate_fn

    pair1 = {
        "source": {**collate_sample(LAYOUTS[0]), "__perf": {"worker/V4_s": 0.1}},
        "target": {**collate_sample(LAYOUTS[0]), "__perf": {"worker/Horizon1_s": 0.2}},
        "source_label": np.int32(1),
        "target_label": np.int32(2),
    }
    pair2 = {
        "source": {**collate_sample(LAYOUTS[2]), "__perf": {"worker/Horizon1_s": 0.3}},
        "target": collate_sample(LAYOUTS[2]),
        "source_label": np.int32(3),
        "target_label": np.int32(4),
        "__perf": {"worker/paired_getitem_s": 0.5},
    }
    batch = _collate_fn([pair1, pair2])
    timing = batch.pop("__perf")
    np.testing.assert_allclose(timing["source/worker/V4_s"], [0.1, 0.0])
    np.testing.assert_allclose(timing["source/worker/Horizon1_s"], [0.0, 0.3])
    np.testing.assert_allclose(timing["target/worker/Horizon1_s"], [0.2, 0.0])
    np.testing.assert_allclose(timing["worker/paired_getitem_s"], [0.0, 0.5])
    assert "__perf" not in batch["source"]
    assert "__perf" not in batch["target"]
    np.testing.assert_array_equal(batch["target_label"], [2, 4])

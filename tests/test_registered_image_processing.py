"""CPU pixel checks for asset-selected image policies, including mixed batches."""

import ast
from pathlib import Path

import augmax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from openpi.models import model
from openpi.models_pytorch.preprocessing_pytorch import preprocess_observation_pytorch
from openpi.transforms import KeepModelKeys


def collate(items, **kwargs):
    # Execute the real collator without importing LeRobot or its video codecs.
    source = Path(__file__).parents[1] / "src/openpi/training/data_loader.py"
    node = next(node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "_collate_fn")
    scope = {"np": np, "jax": jax, "_model": model}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope["_collate_fn"](items, **kwargs)


def pixels(batch=4, size=24):
    image = np.random.default_rng(42).uniform(-0.8, 0.8, (size, size, 3)).astype(np.float32)
    return jnp.asarray(np.broadcast_to(image, (batch, size, size, 3)).copy())


def expected_augmentation(rng, image, geometric):
    height, width = image.shape[:2]
    transforms = []
    if geometric:
        transforms += [augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                       augmax.Resize(width, height), augmax.Rotate((-5, 5))]
    transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
    return augmax.Chain(*transforms)(rng, image / 2 + 0.5) * 2 - 1


@pytest.mark.parametrize("view", ["base_0_rgb", "left_wrist_0_rgb"])
def test_each_sample_uses_its_selected_chain_once_under_jit(view):
    image = pixels()
    codes = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    obs = model.Observation(images={view: image}, image_masks={view: jnp.ones(4, bool)},
                            state=jnp.zeros((4, 7)), image_augmentation={view: codes})
    rng = jax.random.key(9)
    output = jax.jit(lambda x: model.preprocess_observation(
        rng, x, train=True, image_resolution=(24, 24)
    ))(obs)
    keys = jax.random.split(rng, 4)
    np.testing.assert_allclose(output.images[view][0], expected_augmentation(
        keys[0], image[0], "wrist" not in view), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(output.images[view][1], image[1])
    np.testing.assert_allclose(output.images[view][2], expected_augmentation(
        keys[2], image[2], False), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(output.images[view][3], expected_augmentation(
        keys[3], image[3], True), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(output.image_masks[view], obs.image_masks[view])


def test_inference_ignores_augmentation_selection():
    image = pixels(batch=2)
    obs = model.Observation(images={"camera": image}, image_masks={"camera": jnp.ones(2, bool)},
                            state=jnp.zeros((2, 7)), image_augmentation={"camera": jnp.array([3, 2])})
    output = model.preprocess_observation(None, obs, train=False, image_resolution=(24, 24))
    np.testing.assert_array_equal(output.images["camera"], image)


def test_video_inherits_each_samples_policy_and_retains_validity():
    video = jnp.stack([pixels(batch=2)] * 3, axis=1)
    obs = model.Observation(images={"base_rgb": video}, image_masks={"base_rgb": jnp.ones(2, bool)},
                            state=jnp.zeros((2, 7)), image_augmentation={"base_rgb": jnp.array([1, 3])},
                            frame_valid_masks={"base_rgb": jnp.array([[True, True, False], [True, False, False]])})
    output = jax.jit(lambda x: model.preprocess_observation(
        jax.random.key(5), x, train=True, image_resolution=(24, 24)
    ))(obs)
    np.testing.assert_array_equal(output.images["base_rgb"][0], video[0])
    assert not np.allclose(output.images["base_rgb"][1], video[1])
    np.testing.assert_array_equal(output.frame_valid_masks["base_rgb"], obs.frame_valid_masks["base_rgb"])


def sample(views, policies=None):
    result = {"image": {view: np.zeros((24, 24, 3), np.uint8) for view in views},
              "image_mask": {view: np.asarray(True) for view in views}, "state": np.zeros(7, np.float32)}
    if policies is not None:
        result["image_augmentation"] = policies
    return result


def test_collation_preserves_policies_across_mixed_dataset_and_camera_padding():
    batch = collate([sample(["head"], {"head": 3}), sample(["head", "wrist"])])
    np.testing.assert_array_equal(batch["image_augmentation"]["head"], [3, 0])
    np.testing.assert_array_equal(batch["image_augmentation"]["wrist"], [1, 0])
    np.testing.assert_array_equal(batch["image_mask"]["wrist"], [False, True])
    obs = model.Observation.from_dict(batch)
    np.testing.assert_array_equal(obs.to_dict()["image_augmentation"]["head"], [3, 0])
    assert "image_augmentation" in KeepModelKeys()(batch)
    assert "image_augmentation" not in collate([sample(["head"]), sample(["head"])])


@pytest.mark.parametrize("code", [-1, 4, True, 1.5, [1]])
def test_invalid_sample_policy_is_rejected_before_jit(code):
    with pytest.raises(ValueError, match="augmentation policy"):
        collate([sample(["head"], {"head": code})])


def test_depth_never_receives_rgb_augmentation():
    with pytest.raises(ValueError, match="depth"):
        collate([sample(["base_depth"], {"base_depth": 2})])
    image = pixels(batch=2)
    obs = model.Observation(images={"base_depth": image}, image_masks={"base_depth": jnp.ones(2, bool)},
                            state=jnp.zeros((2, 7)), image_augmentation={"base_depth": jnp.array([0, 1])})
    output = model.preprocess_observation(jax.random.key(1), obs, train=True, image_resolution=(24, 24))
    np.testing.assert_array_equal(output.images["base_depth"], image)


def test_torch_policy_none_is_preserved_in_a_batch_with_color_and_geometry():
    image = torch.from_numpy(np.asarray(pixels(batch=3)).copy())
    obs = model.Observation(images={"wrist_rgb": image}, image_masks={"wrist_rgb": torch.ones(3, dtype=torch.bool)},
                            state=torch.zeros((3, 7)), image_augmentation={"wrist_rgb": torch.tensor([1, 2, 3])})
    torch.manual_seed(1)
    output = preprocess_observation_pytorch(obs, train=True, image_resolution=(24, 24))
    np.testing.assert_array_equal(output.images["wrist_rgb"][0], image[0].permute(2, 0, 1))
    assert not torch.allclose(output.images["wrist_rgb"][1], image[1].permute(2, 0, 1))
    assert not torch.allclose(output.images["wrist_rgb"][1], output.images["wrist_rgb"][2])

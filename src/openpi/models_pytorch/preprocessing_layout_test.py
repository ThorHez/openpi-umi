"""Image specifications and accepted float layouts must reach SigLIP as BCHW."""

from types import SimpleNamespace

import pytest
import torch

from openpi.models_pytorch import preprocessing_pytorch


@pytest.mark.parametrize("layout", ["BCHW", "BHWC"])
@pytest.mark.parametrize("with_missing_view", [False, True])
def test_float_image_layout_is_canonical_for_actual_and_padded_views(layout, with_missing_view):
    expected = torch.linspace(-1.0, 1.0, 2 * 3 * 8 * 12).reshape(2, 3, 8, 12)
    image = expected if layout == "BCHW" else expected.permute(0, 2, 3, 1)
    observation = SimpleNamespace(
        images={"wrist_rgb": image},
        image_masks={"wrist_rgb": torch.ones(2, dtype=torch.bool)},
        state=torch.zeros((2, 4)),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    processed = preprocessing_pytorch.preprocess_observation_pytorch(
        observation,
        image_keys=("base_rgb", "wrist_rgb") if with_missing_view else None,
        image_resolution=(8, 12),
    )
    torch.testing.assert_close(processed.images["wrist_rgb"], expected)
    assert all(value.shape == (2, 3, 8, 12) for value in processed.images.values())
    if with_missing_view:
        assert not processed.image_masks["base_rgb"].any()
        assert not processed.images["base_rgb"].any()


def test_float_depth_bhwc_is_canonical_without_rgb_augmentation():
    image = torch.linspace(-1.0, 1.0, 2 * 8 * 12 * 3).reshape(2, 8, 12, 3)
    observation = SimpleNamespace(
        images={"camera_depth": image},
        image_masks={"camera_depth": torch.ones(2, dtype=torch.bool)},
        state=torch.zeros((2, 4)),
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    processed = preprocessing_pytorch.preprocess_observation_pytorch(
        observation, train=True, image_resolution=(8, 12)
    )
    torch.testing.assert_close(processed.images["camera_depth"], image.permute(0, 3, 1, 2))

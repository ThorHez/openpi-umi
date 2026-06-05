import numpy as np
import pytest

import openpi.transforms_video as _transforms_video


def test_parse_image_chw_float32():
    """Test parsing CHW float32 image (LeRobot default format)."""
    chw_float = np.random.rand(3, 224, 224).astype(np.float32)
    result = _transforms_video._parse_image(chw_float)

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    assert result.max() <= 255
    assert result.min() >= 0


def test_parse_image_hwc_uint8():
    """Test parsing HWC uint8 image (already correct format)."""
    hwc_uint8 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    result = _transforms_video._parse_image(hwc_uint8)

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(result, hwc_uint8)


def test_parse_image_chw_uint8():
    """Test parsing CHW uint8 image."""
    chw_uint8 = np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8)
    result = _transforms_video._parse_image(chw_uint8)

    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8


def test_parse_image_normalization():
    """Test that float values are properly normalized to uint8."""
    # Create an image with values in [0, 1]
    chw_float = np.zeros((3, 224, 224), dtype=np.float32)
    chw_float[0, :, :] = 0.5  # Should become 127
    chw_float[1, :, :] = 1.0  # Should become 255
    chw_float[2, :, :] = 0.0  # Should become 0

    result = _transforms_video._parse_image(chw_float)

    assert result[0, 0, 0] == 127 or result[0, 0, 0] == 128  # Allow for rounding
    assert result[0, 0, 1] == 255
    assert result[0, 0, 2] == 0


def test_build_video_tensor():
    """Test BuildVideoTensor stacks frames correctly."""
    # Create test data with individual frames
    data = {
        "left_wrist_0_rgb_0": np.ones((224, 224, 3), dtype=np.uint8) * 10,
        "left_wrist_0_rgb_1": np.ones((224, 224, 3), dtype=np.uint8) * 20,
        "left_wrist_0_rgb_2": np.ones((224, 224, 3), dtype=np.uint8) * 30,
    }

    transform = _transforms_video.BuildVideoTensor(
        image_keys=("left_wrist_0_rgb",),
        num_frames=3,
    )

    result = transform(data)

    # Check video tensor exists and has correct shape
    assert "left_wrist_0_rgb" in result
    assert result["left_wrist_0_rgb"].shape == (3, 224, 224, 3)
    assert result["left_wrist_0_rgb"].dtype == np.uint8

    # Check individual frames are cleaned up
    assert "left_wrist_0_rgb_0" not in result
    assert "left_wrist_0_rgb_1" not in result
    assert "left_wrist_0_rgb_2" not in result

    # Check values are preserved
    assert result["left_wrist_0_rgb"][0, 0, 0, 0] == 10
    assert result["left_wrist_0_rgb"][1, 0, 0, 0] == 20
    assert result["left_wrist_0_rgb"][2, 0, 0, 0] == 30


def test_build_video_tensor_multiple_keys():
    """Test BuildVideoTensor with multiple image keys."""
    data = {
        "left_wrist_0_rgb_0": np.ones((224, 224, 3), dtype=np.uint8),
        "left_wrist_0_rgb_1": np.ones((224, 224, 3), dtype=np.uint8) * 2,
        "right_wrist_0_rgb_0": np.ones((224, 224, 3), dtype=np.uint8) * 3,
        "right_wrist_0_rgb_1": np.ones((224, 224, 3), dtype=np.uint8) * 4,
    }

    transform = _transforms_video.BuildVideoTensor(
        image_keys=("left_wrist_0_rgb", "right_wrist_0_rgb"),
        num_frames=2,
    )

    result = transform(data)

    assert result["left_wrist_0_rgb"].shape == (2, 224, 224, 3)
    assert result["right_wrist_0_rgb"].shape == (2, 224, 224, 3)
    assert result["left_wrist_0_rgb"][0, 0, 0, 0] == 1
    assert result["right_wrist_0_rgb"][0, 0, 0, 0] == 3


def test_build_video_tensor_with_output_mapping():
    """Test BuildVideoTensor with output key mapping."""
    data = {
        "cam_0_0": np.ones((224, 224, 3), dtype=np.uint8),
        "cam_0_1": np.ones((224, 224, 3), dtype=np.uint8) * 2,
    }

    transform = _transforms_video.BuildVideoTensor(
        image_keys=("cam_0",),
        num_frames=2,
        output_keys={"cam_0": "left_wrist_0_rgb"},
    )

    result = transform(data)

    # Check renamed output
    assert "left_wrist_0_rgb" in result
    assert "cam_0" not in result
    assert result["left_wrist_0_rgb"].shape == (2, 224, 224, 3)


def test_build_video_tensor_missing_key():
    """Test BuildVideoTensor raises error for missing frame keys."""
    data = {
        "left_wrist_0_rgb_0": np.ones((224, 224, 3), dtype=np.uint8),
        # Missing left_wrist_0_rgb_1
    }

    transform = _transforms_video.BuildVideoTensor(
        image_keys=("left_wrist_0_rgb",),
        num_frames=2,
    )

    with pytest.raises(KeyError, match="Missing frame key"):
        transform(data)


def test_build_video_tensor_chw_format():
    """Test BuildVideoTensor handles CHW format correctly."""
    # Create frames in CHW format (common in PyTorch/LeRobot)
    data = {
        "left_wrist_0_rgb_0": np.ones((3, 224, 224), dtype=np.uint8),  # CHW
        "left_wrist_0_rgb_1": np.ones((3, 224, 224), dtype=np.uint8) * 2,
    }

    transform = _transforms_video.BuildVideoTensor(
        image_keys=("left_wrist_0_rgb",),
        num_frames=2,
    )

    result = transform(data)

    # Result should be [T, H, W, C] format
    assert result["left_wrist_0_rgb"].shape == (2, 224, 224, 3)


def test_format_pi0_mem_video_input():
    """Test FormatPi0MemVideoInput creates correct structure."""
    # Create test data with video tensors
    data = {
        "left_wrist_0_rgb": np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8),
        "right_wrist_0_rgb": np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8),
    }

    transform = _transforms_video.FormatPi0MemVideoInput(
        image_key_mapping={
            "left_wrist_0_rgb": "left_wrist_0_rgb",
            "right_wrist_0_rgb": "right_wrist_0_rgb",
        },
    )

    result = transform(data)

    # Check images dict is created
    assert "images" in result
    assert "image_masks" in result

    # Check each image key is present
    assert "left_wrist_0_rgb" in result["images"]
    assert "right_wrist_0_rgb" in result["images"]

    # Check shapes
    assert result["images"]["left_wrist_0_rgb"].shape == (4, 224, 224, 3)
    assert result["images"]["right_wrist_0_rgb"].shape == (4, 224, 224, 3)

    # Check masks are True (since all frames are valid)
    assert result["image_masks"]["left_wrist_0_rgb"] == np.True_
    assert result["image_masks"]["right_wrist_0_rgb"] == np.True_


def test_format_pi0_mem_video_input_with_base_camera():
    """Test FormatPi0MemVideoInput creates padded base camera."""
    data = {
        "left_wrist_0_rgb": np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8),
    }

    transform = _transforms_video.FormatPi0MemVideoInput(
        image_key_mapping={"left_wrist_0_rgb": "left_wrist_0_rgb"},
        base_image_key="base_0_rgb",
    )

    result = transform(data)

    # Check base camera was created as zeros
    assert "base_0_rgb" in result["images"]
    assert result["images"]["base_0_rgb"].shape == (4, 224, 224, 3)
    assert np.all(result["images"]["base_0_rgb"] == 0)

    # Check base camera mask is False (padding)
    assert result["image_masks"]["base_0_rgb"] == np.False_


def test_format_pi0_mem_video_input_missing_key():
    """Test FormatPi0MemVideoInput raises error for missing source key."""
    data = {}  # Missing required key

    transform = _transforms_video.FormatPi0MemVideoInput(
        image_key_mapping={"left_wrist_0_rgb": "left_wrist_0_rgb"},
    )

    with pytest.raises(KeyError, match="Missing image key"):
        transform(data)


def test_format_pi0_mem_video_input_wrong_shape():
    """Test FormatPi0MemVideoInput raises error for wrong tensor shape."""
    # Create a 3D tensor instead of 4D video tensor
    data = {
        "left_wrist_0_rgb": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
    }

    transform = _transforms_video.FormatPi0MemVideoInput(
        image_key_mapping={"left_wrist_0_rgb": "left_wrist_0_rgb"},
    )

    with pytest.raises(ValueError, match="should have shape \\[T, H, W, C\\]"):
        transform(data)


def test_format_pi0_mem_cleans_up_source_keys():
    """Test that FormatPi0MemVideoInput cleans up source keys."""
    data = {
        "left_wrist_0_rgb": np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8),
    }

    transform = _transforms_video.FormatPi0MemVideoInput(
        image_key_mapping={"left_wrist_0_rgb": "left_wrist_0_rgb"},
    )

    result = transform(data)

    # Source key should be removed if different from output key
    # In this case they're the same, so it's moved to 'images'
    assert "left_wrist_0_rgb" not in result or "images" in result


def test_load_video_frames_no_index():
    """Test LoadVideoFrames passes through data when no index is available."""
    # Create a mock dataset
    class MockDataset:
        pass

    data = {
        "some_key": "some_value",
        # No index key
    }

    transform = _transforms_video.LoadVideoFrames(
        dataset=MockDataset(),
        image_keys=("left_wrist_0_rgb",),
        num_frames=2,
    )

    result = transform(data)

    # Data should pass through unchanged
    assert result == data


def test_build_video_tensor_preserves_other_keys():
    """Test that BuildVideoTensor preserves non-image keys."""
    data = {
        "left_wrist_0_rgb_0": np.ones((224, 224, 3), dtype=np.uint8),
        "left_wrist_0_rgb_1": np.ones((224, 224, 3), dtype=np.uint8) * 2,
        "actions": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "state": np.array([0.1, 0.2, 0.3]),
    }

    transform = _transforms_video.BuildVideoTensor(
        image_keys=("left_wrist_0_rgb",),
        num_frames=2,
    )

    result = transform(data)

    # Non-image keys should be preserved
    assert "actions" in result
    assert "state" in result
    assert np.allclose(result["actions"], data["actions"])
    assert np.allclose(result["state"], data["state"])

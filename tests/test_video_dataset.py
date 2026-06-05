"""Tests for video dataset and dynamic frame loading."""

import numpy as np
import pytest


def test_imports():
    """Test that all new modules can be imported."""
    from openpi import transforms_video
    from openpi.training.mem import video_dataset
    from openpi.training import config_pi0_mem

    assert transforms_video is not None
    assert video_dataset is not None
    assert config_pi0_mem is not None


def test_video_frame_config():
    """Test VideoFrameConfig dataclass."""
    from openpi.training.mem.video_dataset import VideoFrameConfig

    config = VideoFrameConfig(
        image_keys=("left_wrist_0_rgb", "right_wrist_0_rgb"),
        num_frames=4,
        frame_stride=2,
        padding_mode="zero",
    )

    assert config.image_keys == ("left_wrist_0_rgb", "right_wrist_0_rgb")
    assert config.num_frames == 4
    assert config.frame_stride == 2
    assert config.padding_mode == "zero"


def test_video_frame_config_validation():
    """Test VideoFrameConfig validation."""
    from openpi.training.mem.video_dataset import VideoFrameConfig

    # Invalid num_frames
    with pytest.raises(ValueError, match="num_frames must be >= 1"):
        VideoFrameConfig(
            image_keys=("left_wrist_0_rgb",),
            num_frames=0,
        )

    # Invalid frame_stride
    with pytest.raises(ValueError, match="frame_stride must be >= 1"):
        VideoFrameConfig(
            image_keys=("left_wrist_0_rgb",),
            num_frames=2,
            frame_stride=0,
        )

    # Invalid padding_mode
    with pytest.raises(ValueError, match="padding_mode must be"):
        VideoFrameConfig(
            image_keys=("left_wrist_0_rgb",),
            num_frames=2,
            padding_mode="invalid",
        )


def test_parse_image():
    """Test image parsing function."""
    from openpi.transforms_video import _parse_image

    # Test CHW float32 format (LeRobot default)
    chw_float = np.random.rand(3, 224, 224).astype(np.float32)
    result = _parse_image(chw_float)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    assert result.max() <= 255

    # Test HWC uint8 format (already correct)
    hwc_uint8 = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    result = _parse_image(hwc_uint8)
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8

    # Test CHW uint8 format
    chw_uint8 = np.random.randint(0, 256, (3, 224, 224), dtype=np.uint8)
    result = _parse_image(chw_uint8)
    assert result.shape == (224, 224, 3)


def test_build_video_tensor():
    """Test BuildVideoTensor transform."""
    from openpi.transforms_video import BuildVideoTensor

    # Create test data with individual frames
    data = {
        "left_wrist_0_rgb_0": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "left_wrist_0_rgb_1": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "left_wrist_0_rgb_2": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
    }

    transform = BuildVideoTensor(
        image_keys=("left_wrist_0_rgb",),
        num_frames=3,
    )

    result = transform(data)

    # Check video tensor shape
    assert "left_wrist_0_rgb" in result
    assert result["left_wrist_0_rgb"].shape == (3, 224, 224, 3)

    # Check individual keys are cleaned up
    assert "left_wrist_0_rgb_0" not in result
    assert "left_wrist_0_rgb_1" not in result
    assert "left_wrist_0_rgb_2" not in result


def test_build_video_tensor_with_mapping():
    """Test BuildVideoTensor with output key mapping."""
    from openpi.transforms_video import BuildVideoTensor

    data = {
        "cam_0_0": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "cam_0_1": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
    }

    transform = BuildVideoTensor(
        image_keys=("cam_0",),
        num_frames=2,
        output_keys={"cam_0": "left_wrist_0_rgb"},
    )

    result = transform(data)

    # Check renamed output
    assert "left_wrist_0_rgb" in result
    assert "cam_0" not in result
    assert result["left_wrist_0_rgb"].shape == (2, 224, 224, 3)


def test_format_pi0_mem_video_input():
    """Test FormatPi0MemVideoInput transform."""
    from openpi.transforms_video import FormatPi0MemVideoInput

    # Create test data with video tensors
    data = {
        "left_wrist_0_rgb": np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8),
        "right_wrist_0_rgb": np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8),
    }

    transform = FormatPi0MemVideoInput(
        image_key_mapping={
            "left_wrist_0_rgb": "left_wrist_0_rgb",
            "right_wrist_0_rgb": "right_wrist_0_rgb",
        },
    )

    result = transform(data)

    # Check images dict
    assert "images" in result
    assert "image_masks" in result
    assert "left_wrist_0_rgb" in result["images"]
    assert "right_wrist_0_rgb" in result["images"]
    assert result["images"]["left_wrist_0_rgb"].shape == (4, 224, 224, 3)

    # Check masks
    assert result["image_masks"]["left_wrist_0_rgb"] == np.True_
    assert result["image_masks"]["right_wrist_0_rgb"] == np.True_


def test_format_pi0_mem_with_base_camera():
    """Test FormatPi0MemVideoInput with base camera padding."""
    from openpi.transforms_video import FormatPi0MemVideoInput

    data = {
        "left_wrist_0_rgb": np.random.randint(0, 256, (4, 224, 224, 3), dtype=np.uint8),
    }

    transform = FormatPi0MemVideoInput(
        image_key_mapping={"left_wrist_0_rgb": "left_wrist_0_rgb"},
        base_image_key="base_0_rgb",
    )

    result = transform(data)

    # Check base camera was created
    assert "base_0_rgb" in result["images"]
    assert result["images"]["base_0_rgb"].shape == (4, 224, 224, 3)
    assert np.all(result["images"]["base_0_rgb"] == 0)
    assert result["image_masks"]["base_0_rgb"] == np.False_

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Dynamic Video Frame Loading for Pi0Mem

This document describes how to use dynamic video frame loading to significantly reduce dataset storage requirements when training Pi0Mem models.

## Overview

**Problem:** When training Pi0Mem (which uses video sequences as input), the traditional approach is to pre-expand images in the dataset. For example, if `num_frames=2`, each image is stored twice as `left_wrist_0_rgb_0` and `left_wrist_0_rgb_1`. This doubles storage requirements.

**Solution:** Store only single images in the dataset and load multiple frames dynamically during training. This can reduce storage by up to 50-80% depending on the number of frames needed.

## Quick Start

### 1. Dataset Preparation

When converting your data to LeRobot format, store only single images:

```yaml
# dataset_config.yaml
images:
  observation.left_wrist_0_rgb:
    dtype: "image"
    shape: [224, 224, 3]
    source_key: "camera0_rgb"
    per_timestep: false  # IMPORTANT: Don't expand frames
```

Use the conversion script:
```bash
python examples/umi/convert_umi_data_to_lerobot_fast_v3.py \
    --input dataset.zarr.zip \
    --output ./lerobot_dataset \
    --repo-id my_dataset \
    --task "fold the clothes" \
    --config dataset_config.yaml
```

### 2. Training with Dynamic Frame Loading

#### Method 1: Using the Example Training Script

```bash
python examples/umi/train_pi0_mem_dynamic_frames.py \
    --dataset-path ./lerobot_dataset \
    --num-frames 4 \
    --batch-size 8 \
    --num-steps 100000
```

#### Method 2: Using Standard train.py with Custom Config

Create a training configuration in `src/openpi/training/config.py` or use the provided config:

```python
from openpi.training.config_pi0_mem import LeRobotPi0MemDataConfig

# Register a config
_CONFIGS["pi0_mem_umi_dynamic"] = TrainConfig(
    name="pi0_mem_umi_dynamic",
    model=pi0_mem.Pi0MemConfig(
        num_frames=4,           # Number of temporal frames
        temporal_every=4,       # Temporal attention every N layers
        action_dim=32,
        action_horizon=16,
        max_token_len=48,
    ),
    data=LeRobotPi0MemDataConfig(
        repo_id="path/to/dataset",
        num_frames=4,           # Must match model config
        image_keys=("left_wrist_0_rgb", "right_wrist_0_rgb"),
        frame_stride=1,
        padding_mode="repeat",
    ),
    num_train_steps=100000,
    batch_size=8,
)
```

Then train:
```bash
python scripts/train.py --config-name=pi0_mem_umi_dynamic
```

#### Method 3: Programmatic Usage

```python
import openpi.models.pi0_mem as pi0_mem
from openpi.training.video_dataset import (
    VideoFrameConfig,
    create_video_data_loader,
)
from openpi.training.config_pi0_mem import LeRobotPi0MemDataConfig

# Create model config
model_config = pi0_mem.Pi0MemConfig(
    num_frames=4,
    temporal_every=4,
    action_dim=32,
    action_horizon=16,
)

# Create data config
data_config = LeRobotPi0MemDataConfig(
    repo_id="path/to/dataset",
    num_frames=4,
    image_keys=("left_wrist_0_rgb", "right_wrist_0_rgb"),
).create(assets_dirs=Path("./assets"), model_config=model_config)

# Create video frame config
video_config = VideoFrameConfig(
    image_keys=("left_wrist_0_rgb", "right_wrist_0_rgb"),
    num_frames=4,
    frame_stride=1,
    padding_mode="repeat",
)

# Create data loader with dynamic frame loading
data_loader = create_video_data_loader(
    data_config=data_config,
    model_config=model_config,
    video_config=video_config,
    batch_size=8,
    num_workers=4,
)

# Use in training loop
for observation, actions in data_loader:
    # observation.images contains video tensors [B, T, H, W, C]
    pass
```

## Architecture

### Components

1. **`VideoFrameDataset`** (`src/openpi/training/video_dataset.py`)
   - Wraps a LeRobot dataset
   - Loads multiple historical frames on-the-fly in `__getitem__`
   - Handles episode boundaries and padding

2. **`LoadVideoFrames`** (`src/openpi/transforms_video.py`)
   - Transform that loads historical frames from the dataset
   - Uses frame index information to load correct frames
   - Handles padding for start of episodes

3. **`BuildVideoTensor`** (`src/openpi/transforms_video.py`)
   - Stacks individual frames into video tensors [T, H, W, C]

4. **`FormatPi0MemVideoInput`** (`src/openpi/transforms_video.py`)
   - Formats video tensors for Pi0Mem model input
   - Creates images dict and image_masks

### Data Flow

```
LeRobot Dataset (single images)
    ↓
VideoFrameDataset.__getitem__(index)
    - Loads current frame
    - Loads num_frames-1 historical frames
    - Handles padding for episode boundaries
    - Stores frames as img_key_0, img_key_1, ...
    ↓
RepackTransform
    - Maps dataset keys to internal format
    ↓
Data Transforms (e.g., UmiInputsV4_Bimanual)
    - Process low-dimensional data
    - Reference expanded image keys if needed
    ↓
BuildVideoTensor
    - Stacks frames into video tensors [T, H, W, C]
    ↓
FormatPi0MemVideoInput
    - Creates images dict for Pi0Mem
    - Creates image_masks
    ↓
Model Transforms (Resize, Tokenize, Pad)
    - ResizeImages: applies to each frame in video
    - TokenizePrompt: processes language
    - PadStatesAndActions: pads to model dims
    ↓
Pi0Mem Model
    - Input: images [B, T, H, W, C]
    - SigLIP-MEM encoder with temporal attention
    - Output: actions
```

## Configuration Options

### VideoFrameConfig

```python
@dataclass
class VideoFrameConfig:
    image_keys: tuple[str, ...]     # Keys to expand (e.g., ("left_wrist_0_rgb",))
    num_frames: int = 2             # Number of frames to load
    frame_stride: int = 1           # Stride between frames
    padding_mode: str = "repeat"    # "repeat" or "zero"
```

### LeRobotPi0MemDataConfig

```python
@dataclass
class LeRobotPi0MemDataConfig:
    repo_id: str                    # Dataset path or HF repo
    num_frames: int = 2             # Must match Pi0MemConfig
    frame_stride: int = 1           # Frame sampling stride
    padding_mode: str = "repeat"    # Padding for episode start
    image_keys: tuple[str, ...]     # Keys to load dynamically
    mapping: dict[str, str]         # Dataset to internal key mapping
```

## Performance Considerations

### Storage Savings

| num_frames | Traditional | Dynamic Loading | Savings |
|-----------|-------------|-----------------|---------|
| 2         | 2x          | 1x              | 50%     |
| 4         | 4x          | 1x              | 75%     |
| 8         | 8x          | 1x              | 87.5%   |

### Training Speed Impact

Dynamic frame loading adds overhead because frames are loaded on-the-fly:

- **Single worker (num_workers=0):** ~10-20% slower due to sequential frame loading
- **Multiple workers (num_workers=4+):** Minimal impact (~5%) as frame loading overlaps with training

### Recommendations

1. **Use multiple workers:** Set `num_workers=4` or higher to overlap frame loading with training
2. **SSD storage:** Use fast SSD storage for the dataset to minimize frame loading latency
3. **Cache recently loaded frames:** The HuggingFace datasets library caches recently accessed data

## Advanced Usage

### Custom Frame Sampling

You can implement custom frame sampling strategies:

```python
from openpi.transforms_video import LoadVideoFrames

# Custom stride (e.g., sample every 3rd frame for longer temporal context)
video_config = VideoFrameConfig(
    image_keys=("left_wrist_0_rgb",),
    num_frames=4,
    frame_stride=3,  # Load frames: t, t-3, t-6, t-9
    padding_mode="repeat",
)
```

### Multi-Dataset Training

For training on multiple datasets with different camera setups:

```python
from openpi.training.multi_data_loader import MultiDatasetLoader

# Each dataset can have different VideoFrameConfig
dataset1_config = VideoFrameConfig(
    image_keys=("left_wrist_0_rgb",),
    num_frames=4,
)

dataset2_config = VideoFrameConfig(
    image_keys=("camera_0_rgb", "camera_1_rgb"),  # Different cameras
    num_frames=4,
)

# Create MultiDatasetLoader with per-dataset configs
```

### Inference with Dynamic Loading

For inference, you typically only need the current frame:

```python
# For inference, set num_frames=1 to disable history loading
video_config = VideoFrameConfig(
    image_keys=("left_wrist_0_rgb",),
    num_frames=1,  # Only current frame
)
```

Or manually manage frame history:

```python
class FrameBuffer:
    """Buffer to maintain frame history for inference."""

    def __init__(self, num_frames):
        self.num_frames = num_frames
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)
        if len(self.frames) > self.num_frames:
            self.frames.pop(0)

    def get_video_tensor(self):
        # Stack frames into [T, H, W, C]
        while len(self.frames) < self.num_frames:
            self.frames.insert(0, self.frames[0] if self.frames else np.zeros((224, 224, 3)))
        return np.stack(self.frames, axis=0)
```

## Troubleshooting

### Issue: Episode boundary errors

**Symptom:** Incorrect frames loaded at episode boundaries

**Solution:** Ensure your dataset has proper `episode_index` and `frame_index` fields:

```python
# Verify dataset structure
dataset = LeRobotDataset(repo_id)
print(dataset.hf_dataset[0])  # Should contain: index, episode_index, frame_index
```

### Issue: Slow data loading

**Symptom:** Training is significantly slower with dynamic loading

**Solutions:**
1. Increase `num_workers` in data loader
2. Use SSD storage instead of HDD
3. Reduce `num_frames` if possible
4. Consider using pre-expanded frames for production training after experimentation

### Issue: Out of memory

**Symptom:** GPU OOM during training

**Solution:** Video tensors [B, T, H, W, C] use more memory than single images. Reduce batch size:

```python
# With num_frames=4, you may need to reduce batch_size by ~4x compared to single-frame training
batch_size = 2  # Instead of 8
```

## Migration Guide

### From Pre-expanded Frames

If you have an existing dataset with pre-expanded frames:

1. **Option 1:** Keep using it (backward compatible)
2. **Option 2:** Re-convert with `per_timestep: false` to save space
3. **Option 3:** Use a hybrid approach - expand some frames, dynamically load others

### Converting Existing Configs

Old config (pre-expanded):
```python
mapping = {
    "left_wrist_0_rgb_0": "observation.left_wrist_0_rgb_0",
    "left_wrist_0_rgb_1": "observation.left_wrist_0_rgb_1",
}
```

New config (dynamic loading):
```python
mapping = {
    "left_wrist_0_rgb": "observation.left_wrist_0_rgb",
    # Dynamic loading creates: left_wrist_0_rgb_0, left_wrist_0_rgb_1, ...
}
```

## Examples

See `examples/umi/train_pi0_mem_dynamic_frames.py` for a complete working example.

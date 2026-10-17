from collections.abc import Sequence
import logging

import torch

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] | None = None,
    depth_keys: Sequence[str] | None = None,
    allow_all_masked_images: bool = False,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Preserve actual cameras by default, or apply an explicit padded view contract.

    Depth streams are resized without RGB augmentation. This image-only Torch
    backbone rejects video tensors; temporal MEM policies use the JAX pipeline.
    """
    depth_keys = tuple(depth_keys) if depth_keys is not None else tuple(k for k in sorted(observation.images) if "depth" in k)
    image_keys = tuple(image_keys) if image_keys is not None else tuple(k for k in sorted(observation.images) if k not in depth_keys)
    keys = (*image_keys, *depth_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Image/depth view keys must be nonempty, unique and disjoint")
    if not any(key in observation.images for key in keys) and not allow_all_masked_images:
        raise ValueError(f"None of the configured image views {keys} are present")
    batch_shape = observation.state.shape[:-1]
    device = observation.state.device

    # Track which keys were synthesized so masks can be forced to False later.
    synthesized_keys: set = set()

    out_images = {}
    for key in keys:
        if key not in observation.images:
            # Default channels-first to match the rest of the PyTorch path.
            out_images[key] = torch.zeros(
                (*batch_shape, 3, *image_resolution), dtype=torch.float32, device=device
            )
            synthesized_keys.add(key)
            continue

        image = observation.images[key]
        if image.ndim != 4:
            raise ValueError(f"PyTorch Pi0 expects a 4D image for {key}; use a JAX MEM policy for video")

        # Normalize either accepted input layout before spatial transforms.
        # SigLIP always consumes BCHW, including normalized float inputs.
        is_channels_first = image.shape[1] == 3
        if not is_channels_first and image.shape[-1] != 3:
            raise ValueError(f"PyTorch Pi0 expects three image channels for {key}; got {image.shape}")

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train and key not in depth_keys:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                height, width = image.shape[1:3]

                # Random crop and resize
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # Random crop
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # Resize back to original size
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                # Random rotation (small angles)
                # Use tensor operations instead of .item() for torch.compile compatibility
                angle = torch.rand(1, device=image.device) * 10 - 5  # Random angle between -5 and 5 degrees
                if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                    # Convert to radians
                    angle_rad = angle * torch.pi / 180.0

                    # Create rotation matrix
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # Apply rotation using grid_sample
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # Create meshgrid
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # Expand to batch dimension
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # Apply rotation transformation
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # Stack and reshape for grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # Color augmentations for all cameras
            # Random brightness
            # Use tensor operations instead of .item() for torch.compile compatibility
            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
            image = image * brightness_factor

            # Random contrast
            # Use tensor operations instead of .item() for torch.compile compatibility
            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            # Random saturation (convert to HSV, modify S, convert back)
            # For simplicity, we'll just apply a random scaling to the color channels
            # Use tensor operations instead of .item() for torch.compile compatibility
            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # Clamp values to [0, 1]
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        # Keep actual and synthesized views in the same backbone input layout.
        out_images[key] = image.permute(0, 3, 1, 2)

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key in synthesized_keys:
            # synthesized zero image -> invalid view
            out_masks[key] = torch.zeros(batch_shape, dtype=torch.bool, device=observation.state.device)
        elif key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = torch.as_tensor(observation.image_masks[key], dtype=torch.bool, device=device).expand(batch_shape)

    if not allow_all_masked_images:
        torch._assert_async(
            torch.stack(tuple(out_masks.values())).any(dim=0).all(),
            "Observation has no valid image view; check camera keys/masks or explicitly allow_all_masked_images",
        )

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        frame_valid_masks=getattr(observation, "frame_valid_masks", None),
        semantic_memory=getattr(observation, "semantic_memory", None),
        action_loss_mask=getattr(observation, "action_loss_mask", None),
    )

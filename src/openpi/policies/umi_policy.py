import dataclasses

import einops
import os
import time
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_umi_example() -> dict:
    """Creates a random input example for the UMI policy."""
    return {
        "robot0_eef_pos": np.random.rand(3),
        "robot0_eef_rot_axis_angle": np.random.rand(3),
        "robot0_gripper_width": np.random.rand(1),
        "camera0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # If CHW, convert to HWC
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    # If grayscale HxW, expand to 3 channels
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    # Normalize channels to 3 (drop alpha or repeat single channel)
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
    # Resize to 224x224 if needed
    if image.shape[0] != 224 or image.shape[1] != 224:
        try:
            from PIL import Image  # type: ignore

            pil_img = Image.fromarray(image)
            pil_img = pil_img.convert("RGB")
            _resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", getattr(Image, "BILINEAR", 2))
            pil_img = pil_img.resize((224, 224), resample=_resample)
            image = np.asarray(pil_img)
        except (ImportError, ModuleNotFoundError, ValueError, OSError, RuntimeError):
            # As a last resort, simple numpy resize (not recommended for quality)
            image = np.resize(image, (224, 224, 3))
        image = image.astype(np.uint8, copy=False)
    return image


@dataclasses.dataclass(frozen=True)
class UmiInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Combine the robot state from end-effector position, rotation (axis-angle), and gripper width.
        # For UMI, the state is 7-dimensional: [eef_pos (3D), eef_rot (3D), gripper_width (1D)]
        eef_pos = np.asarray(data["robot0_eef_pos"])
        eef_rot = np.asarray(data["robot0_eef_rot_axis_angle"])
        gripper_width = np.asarray(data["robot0_gripper_width"])
        
        # Ensure gripper_width is 1D array
        if gripper_width.ndim == 0:
            gripper_width = gripper_width[np.newaxis]
        
        # Concatenate all state components
        state = np.concatenate([eef_pos, eef_rot, gripper_width])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "camera0_rgb",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # wrist images below.
        left_wrist_camera_image = _parse_image(data["camera0_rgb"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": np.zeros_like(left_wrist_camera_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                # UMI only has one camera, so we pad the wrist images.
                "left_wrist_0_rgb": left_wrist_camera_image,
                "right_wrist_0_rgb": np.zeros_like(left_wrist_camera_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Debug print for left_wrist_0_rgb to help locate data issues
        try:
            _lw = inputs["image"]["left_wrist_0_rgb"]
            print(
                f"left_wrist_0_rgb: shape={_lw.shape}, dtype={_lw.dtype}, min={_lw.min()}, max={_lw.max()}"
            )
            # Print a single pixel sample to avoid flooding logs
            print(f"left_wrist_0_rgb[0,0]: {_lw[0,0].tolist()}")
        except (KeyError, ValueError, AttributeError, IndexError) as e:
            print(f"Failed to print left_wrist_0_rgb: {e}")

        # Persist left_wrist_0_rgb image for later debugging
        try:
            debug_root = os.environ.get("OPENPI_DEBUG_DIR", "/root/openpi/debug")
            save_dir = os.path.join(debug_root, "umi_left_wrist")
            os.makedirs(save_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            fn_base = f"left_wrist_0_rgb_{ts}_{time.time_ns()}"
            npy_path = os.path.join(save_dir, fn_base + ".npy")
            np.save(npy_path, _lw)
            # Prepare PNG (optionally convert BGR->RGB for display only)
            to_save_png = _lw
            is_bgr = os.environ.get("OPENPI_IMAGE_IS_BGR", "0").lower() in ("1", "true", "yes", "y")
            if (
                is_bgr
                and isinstance(to_save_png, np.ndarray)
                and to_save_png.ndim == 3
                and to_save_png.shape[-1] == 3
            ):
                to_save_png = to_save_png[..., ::-1]
            # Best-effort PNG save
            png_path = os.path.join(save_dir, fn_base + ".png")
            try:
                try:
                    import imageio.v2 as iio
                    iio.imwrite(png_path, to_save_png)
                except (ImportError, ModuleNotFoundError, ValueError, OSError, RuntimeError):
                    from PIL import Image  # type: ignore
                    Image.fromarray(to_save_png).save(png_path)
            except (OSError, ValueError, RuntimeError) as e:
                print(f"Failed to save PNG for left_wrist_0_rgb: {e}")
            print(
                f"Saved left_wrist_0_rgb to {npy_path} and {png_path} (BGR->RGB swap: {is_bgr})"
            )
        except (OSError, ValueError, RuntimeError) as e:
            print(f"Failed to persist left_wrist_0_rgb: {e}")

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        # if "prompt" in data:
        #     inputs["prompt"] = data["prompt"]
        inputs["prompt"] = "pick up the black bottle and place it on the white area"

        print(f"inputs: {inputs}")

        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For UMI, we only return the first 7 actions: [eef_pos (3D), eef_rot (3D), gripper_width (1D)].
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}

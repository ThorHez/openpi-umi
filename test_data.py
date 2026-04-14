# import numpy as np
import zipfile
import zarr
import os
import numpy as np

# data = np.load('/root/openpi-umi/inference_results_1/detailed_results.npz', allow_pickle=True)
# print(list(data["action_ground_truth"][0][0]))

import pyrealsense2 as rs
import numpy as np

# filters
depth_to_disp = rs.disparity_transform(True)
disp_to_depth = rs.disparity_transform(False)

spat = rs.spatial_filter()
temp = rs.temporal_filter()
hole = rs.hole_filling_filter()

# 这些参数可以当作起点再调
spat.set_option(rs.option.filter_magnitude, 2)         # 平滑强度
spat.set_option(rs.option.filter_smooth_alpha, 0.5)    # 越大越平滑
spat.set_option(rs.option.filter_smooth_delta, 20)     # 边缘阈值

temp.set_option(rs.option.filter_smooth_alpha, 0.4)
temp.set_option(rs.option.filter_smooth_delta, 20)

# hole filling 模式通常 1/2 更积极，0 更保守（不同SDK略有差异）
hole.set_option(rs.option.holes_fill, 2)

def process_depth_frame(depth_frame: rs.depth_frame) -> np.ndarray:
    f = depth_to_disp.process(depth_frame)
    f = spat.process(f)
    f = temp.process(f)
    f = disp_to_depth.process(f)
    f = hole.process(f)
    depth_u16 = np.asanyarray(f.get_data()).astype(np.uint16)
    return depth_u16



temp_dir = "/root/openpi-umi/temp_test"
temp_dir = "/root/openpi-umi/data/hitl/hitl_replay_buffer.zarr"

# if not os.path.exists(temp_dir):
#     os.makedirs(temp_dir)
#     with zipfile.ZipFile("/root/openpi-umi/data/fold_clothes/dataset_no_filter_1.zarr.zip", 'r') as zip_ref:
#             zip_ref.extractall(temp_dir)
    
    # Open zarr store
root = zarr.open(temp_dir, mode='r')

print(root["meta"]["episode_ends"].shape)
print(root["data"]["camera1_rgb"].shape)
print(list(root['data'].keys()))

# print(np.max(root["data"]["camera1_rgb"]))
# print(root["data"]["camera1_rgb"][500].max())
# print(root['data']['robot0_eef_pos'][0])
# print(root['data']['robot0_eef_rot_axis_angle'][0])
# print(root['data']['robot0_gripper_width'][0])


# mask = root["data"]["head_depth_raw"][0]==0
# print(mask.sum() / (224 * 224))


# filtered_depth = process_depth_frame(root["data"]["head_depth_raw"][0])
# mask = filtered_depth==0
# print(mask.sum() / (224 * 224))

import numpy as np

def depth_u16_to_3ch_u8(
    depth_u16: np.ndarray,
    clip_m: float = 2.0,
    depth_scale_m: float = 0.001,
    stretch_hi: bool = True,
) -> np.ndarray:
    """
    将 RealSense D435 的 uint16 深度 (batch,224,224) 转成 3 通道 uint8：
      - C0: low byte
      - C1: high byte（可选拉伸到 0..255 以增强信号）
      - C2: validity mask（有效=255，无效=0；无效定义为 depth==0）

    参数:
      depth_u16: (B,224,224) numpy uint16
      clip_m: clip 最大深度（米），例如 2.0
      depth_scale_m: 每个 uint16 单位对应多少米。D435 常见 0.001（1mm）
      stretch_hi: 是否把 high byte 做可逆拉伸，让其动态范围更大（推荐 True）

    返回:
      out: (B,224,224,3) numpy uint8
    """
    d = np.asarray(depth_u16)
    if d.dtype != np.uint16:
        raise TypeError(f"depth_u16 must be uint16, got {d.dtype}")
    if d.ndim != 3 or d.shape[1:] != (224, 224):
        raise ValueError(f"depth_u16 must be (B,224,224), got {d.shape}")

    # validity: RealSense 深度 0 通常表示 invalid
    valid = d > 0

    # clip 到指定最大深度（转换到“单位”再 clip）
    clip_units = int(round(clip_m / depth_scale_m))  # 2.0m / 0.001 = 2000
    d_clip = d.copy()
    d_clip[valid] = np.minimum(d_clip[valid], clip_units).astype(np.uint16)

    # 拆成 low / high 两个字节
    lo = (d_clip & 0xFF).astype(np.uint8)
    hi = (d_clip >> 8).astype(np.uint8)

    if stretch_hi:
        # 在 clip 范围内，hi 的最大可能值是 clip_units >> 8
        hi_max = max(int(clip_units >> 8), 1)   # 2000>>8=7
        # 可逆拉伸：hi_scaled = hi * (255//hi_max)
        scale = max(255 // hi_max, 1)
        hi = (hi.astype(np.uint16) * scale).clip(0, 255).astype(np.uint8)

    mask = (valid.astype(np.uint8) * 255)

    out = np.stack([lo, hi, mask], axis=-1)  # (B,224,224,3)
    return out


# depth_batch = root["data"]["head_depth_raw"][0:1]  # (1, 224, 224) 满足 (B,224,224)
# img3 = depth_u16_to_3ch_u8(depth_batch, clip_m=2.0, depth_scale_m=0.001, stretch_hi=True)  # (1,224,224,3)
# img3 = img3[0]  # (224, 224, 3) 用于可视化

# # 可视化 img3（3 通道: lo, hi, mask）
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# axes[0, 0].imshow(img3)
# axes[0, 0].set_title("img3 (lo, hi, mask as RGB)")
# axes[0, 0].axis("off")
# axes[0, 1].imshow(img3[:, :, 0], cmap="gray")
# axes[0, 1].set_title("channel 0: lo")
# axes[0, 1].axis("off")
# axes[1, 0].imshow(img3[:, :, 1], cmap="gray")
# axes[1, 0].set_title("channel 1: hi")
# axes[1, 0].axis("off")
# axes[1, 1].imshow(img3[:, :, 2], cmap="gray")
# axes[1, 1].set_title("channel 2: mask")
# axes[1, 1].axis("off")
# plt.tight_layout()
# plt.savefig("/root/openpi-umi/head_depth_img3_vis.png", dpi=150, bbox_inches="tight")
# print("Saved to /root/openpi-umi/head_depth_img3_vis.png")
# plt.show()


# ---- 示例 ----
# depth_batch: (B,224,224) uint16
# img3 = depth_u16_to_3ch_u8(depth_batch, clip_m=2.0, depth_scale_m=0.001, stretch_hi=True)
# img3.dtype == np.uint8, img3.shape == (B,224,224,3)

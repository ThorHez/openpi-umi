import numpy as np
import cv2
import scipy.interpolate as si
import math

import os



def inpaint_tag(img, corners, tag_scale=1.4, n_samples=16):
    # scale corners with respect to geometric center
    center = np.mean(corners, axis=0)
    scaled_corners = tag_scale * (corners - center) + center
    
    # sample pixels on the boundary to obtain median color
    sample_points = si.interp1d(
        [0,1,2,3,4], list(scaled_corners) + [scaled_corners[0]], 
        axis=0)(np.linspace(0,4,n_samples)).astype(np.int32)
    sample_colors = img[
        np.clip(sample_points[:,1], 0, img.shape[0]-1), 
        np.clip(sample_points[:,0], 0, img.shape[1]-1)
    ]
    median_color = np.median(sample_colors, axis=0).astype(img.dtype)
    
    # draw tag with median color
    img = cv2.fillPoly(
        img, scaled_corners[None,...].astype(np.int32), 
        color=median_color.tolist())
    return img


def pixel_coords_to_canonical(pts, img_shape=(2028, 2704)):
    coords = (np.asarray(pts) - np.array(img_shape[::-1]) * 0.5) / img_shape[0]
    return coords


def get_mirror_canonical_polygon():
    left_pts = [
        [540, 1700],
        [680, 1450],
        [590, 1070],
        [290, 1130],
        [290, 1770],
        [550, 1770]
    ]
    resolution = [2028, 2704]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_gripper_canonical_polygon():
    left_pts = [
        [1352, 1730],
        [1100, 1700],
        [650, 1500],
        [0, 1350],
        [0, 2028],
        [1352, 2704]
    ]
    resolution = [2028, 2704]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_finger_canonical_polygon(height=0.37, top_width=0.25, bottom_width=1.4):
    # image size
    resolution = [2028, 2704]
    img_h, img_w = resolution

    # calculate coordinates
    top_y = 1. - height
    bottom_y = 1.
    width = img_w / img_h
    middle_x = width / 2.
    top_left_x = middle_x - top_width / 2.
    top_right_x = middle_x + top_width / 2.
    bottom_left_x = middle_x - bottom_width / 2.
    bottom_right_x = middle_x + bottom_width / 2.

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = [[
        [bottom_left_x, bottom_y],
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bottom_right_x, bottom_y]
    ]]
    coords = pixel_coords_to_canonical(points, img_shape=resolution)
    return coords


def canonical_to_pixel_coords(coords, img_shape=(2028, 2704)):
    pts = np.asarray(coords) * img_shape[0] + np.array(img_shape[::-1]) * 0.5
    return pts


def draw_predefined_mask(img, color=(0,0,0), mirror=False, gripper=True, finger=True, use_aa=False):
    all_coords = list()
    if mirror:
        all_coords.extend(get_mirror_canonical_polygon())
    if gripper:
        all_coords.extend(get_gripper_canonical_polygon())
    if finger:
        all_coords.extend(get_finger_canonical_polygon())
        
    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, img.shape[:2])
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img,[pts], color=color, lineType=flag)
    return img


def get_image_transform(in_res, out_res=(224, 224), bgr_to_rgb: bool=False):
    iw, ih = in_res
    ow, oh = out_res
    rw, rh = None, None
    interp_method = cv2.INTER_AREA
    if (iw / ih) >= (ow / oh):
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR

    w_slice_start = (rw - ow) // 2
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice_start = (rh - oh) // 2
    h_slice = slice(h_slice_start, h_slice_start + oh)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih, iw, 3))
        # resize
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        # crop
        img = img[h_slice, w_slice, c_slice]
        return img
    return transform



def save_debug_image(img, stage, frame_idx=None, buffer_idx=None):
    save_dir = "debug_image"
    os.makedirs(save_dir, exist_ok=True)

    # 获取当前已有文件数量
    existing = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
    next_idx = len(existing) + 1

    # 构造文件名：stage_000123.jpg
    filename = f"{stage}_{next_idx:06d}.jpg"
    save_path = os.path.join(save_dir, filename)

    cv2.imwrite(save_path, img)
    print(f"[Saved] {save_path}")



def generate_image_pipeline(image, no_mirror=True, out_res=(224, 224)):
    #img = image.to_ndarray(format='rgb24')
    if hasattr(image, "to_ndarray"):
        img = image.to_ndarray(format="rgb24")
    else:
        img = np.array(image, copy=False)

    # 2. 确保 dtype / 连续性 / 可写
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=True)
    if (not img.flags['C_CONTIGUOUS']) or (not img.flags['WRITEABLE']):
        img = np.ascontiguousarray(img.copy())

    # mask out gripper
    img = draw_predefined_mask(img, color=(0,0,0), gripper=True, finger=False)

    ih, iw = img.shape[:2]
    resize_tf = get_image_transform(
            in_res=(iw, ih),
            out_res=out_res
        )
    img = resize_tf(img)

    # save_debug_image(img, "generate_image_pipeline")

    return img

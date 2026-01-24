# import numpy as np
import zipfile
import zarr
import os
import numpy as np

# data = np.load('/root/openpi-umi/inference_results_1/detailed_results.npz', allow_pickle=True)
# print(list(data["action_ground_truth"][0][0]))


temp_dir = "/root/openpi-umi/temp_test"

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
<<<<<<< HEAD
    with zipfile.ZipFile("/root/openpi-umi/data/fold_clothes/dataset_no_filter_1_with_desk_height_new.zarr.zip", 'r') as zip_ref:
=======
    with zipfile.ZipFile("/root/openpi-umi/data/merge_fold_clothes_cyrus_20251229.zarr.zip", 'r') as zip_ref:
>>>>>>> b467a42 (update code)
            zip_ref.extractall(temp_dir)
    
    # Open zarr store
root = zarr.open(temp_dir, mode='r')

print(root["meta"]["episode_ends"].shape)
print(root["data"]["camera1_rgb"].shape)
print(list(root['data'].keys()))
print(np.max(root["data"]["camera1_rgb"]))
print(root["data"]["camera1_rgb"][500].max())
# print(root['data']['robot0_eef_pos'][0])
# print(root['data']['robot0_eef_rot_axis_angle'][0])
# print(root['data']['robot0_gripper_width'][0])
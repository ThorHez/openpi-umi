import tempfile
import zipfile
import zarr
import os



zarr_zip_path = "/root/openpi-umi/data/dataset_fold_redclothes__20251207.zarr.zip"
tmp_dir = "/root/openpi-umi/examples/umi/analyze/tmp_zarr"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
    print(f"Extracting to temporary directory: {tmp_dir}")
    with zipfile.ZipFile(zarr_zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

# Open zarr store
root = zarr.open(tmp_dir, mode='r')
print(list(root.keys()))
print(list(root['data'].keys()))
# print(root['data']["robot0_demo_start_pose"][0])
# print(root['data']["robot0_demo_start_pose"][1])
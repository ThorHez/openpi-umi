import pandas as pd
from PIL import Image
import numpy as np
import io
import os
from pathlib import Path

# 目标目录
data_dir = Path("/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_20251217/data/chunk-000")

# 获取所有parquet文件
parquet_files = sorted(data_dir.glob("*.parquet"))
print(f"找到 {len(parquet_files)} 个parquet文件")
print("-" * 60)

non_empty_files = []
empty_files = []

for parquet_path in parquet_files:
    df = pd.read_parquet(parquet_path)
    
    if "right_wrist_0_rgb_0" not in df.columns:
        print(f"[跳过] {parquet_path.name}: 没有 right_wrist_0_rgb_0 列")
        continue
    
    # 检查每一行的图片
    has_non_empty = False
    max_values = []
    
    for row_idx in range(len(df)):
        img_data = df["right_wrist_0_rgb_0"][row_idx]
        if img_data is not None and 'bytes' in img_data:
            img = Image.open(io.BytesIO(img_data['bytes']))
            img_array = np.array(img)
            max_val = img_array.max()
            max_values.append(max_val)
            if max_val != 0:
                has_non_empty = True
                break
    
    if max_values:
        overall_max = max(max_values)
        if has_non_empty:
            non_empty_files.append(parquet_path.name)
            print(f"[非空] {parquet_path.name}: max={overall_max}, 行数={len(df)}")
        else:
            empty_files.append(parquet_path.name)
            print(f"[空]   {parquet_path.name}: max={overall_max}, 行数={len(df)}")

print("-" * 60)
print(f"\n统计结果:")
print(f"  非空图片文件 (max != 0): {len(non_empty_files)} 个")
print(f"  空图片文件 (max == 0):   {len(empty_files)} 个")

if non_empty_files:
    print(f"\n非空文件列表:")
    for f in non_empty_files:
        print(f"  - {f}")


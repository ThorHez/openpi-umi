# 快速开始指南

## 🎯 选择合适的脚本

根据你的数据集大小选择：

| 数据集大小 | 推荐脚本 | 命令 |
|-----------|---------|------|
| < 1000 episodes | 原始版本 | `convert_umi_data_to_lerobot.py` |
| 1000-5000 episodes | 并行版本 | `convert_umi_data_to_lerobot_parallel.py` |
| > 5000 episodes | 优化版本 | `convert_umi_data_to_lerobot_fast.py` |

## 📦 安装依赖

```bash
pip install lerobot datasets zarr numpy tqdm
```

## 🚀 快速运行

### 方案1：并行版本（推荐新手）

```bash
python convert_umi_data_to_lerobot_parallel.py \
    --input your_dataset.zarr.zip \
    --output ./output_lerobot \
    --repo-id your_username/dataset_name \
    --workers 8
```

### 方案2：优化版本（性能最佳）

```bash
python convert_umi_data_to_lerobot_fast.py \
    --input your_dataset.zarr.zip \
    --output ./output_lerobot \
    --repo-id your_username/dataset_name \
    --workers 8 \
    --load-batch-size 50
```

## ⚡ 性能对比

在一个包含2000个episodes（平均100帧/episode）的数据集上：

- **原始版本**: ~120分钟
- **并行版本**: ~30分钟 (4x加速)
- **优化版本**: ~15分钟 (8x加速)

## 🎛️ 常用参数

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--workers` | 工作进程数 | CPU核心数 |
| `--batch-size` | 每批处理的episodes数 | 100 (parallel) |
| `--load-batch-size` | 预加载的episodes数 | 50 (fast) |
| `--fps` | 帧率 | 30 |
| `--push-to-hub` | 自动上传到HuggingFace | false |

## 🔧 根据硬件调优

### 16GB RAM, 8核CPU
```bash
python convert_umi_data_to_lerobot_parallel.py \
    --workers 8 \
    --batch-size 100 \
    ...
```

### 32GB+ RAM, 16核+ CPU
```bash
python convert_umi_data_to_lerobot_fast.py \
    --workers 16 \
    --load-batch-size 100 \
    --process-batch-size 10 \
    ...
```

### 8GB RAM, 4核CPU（内存受限）
```bash
python convert_umi_data_to_lerobot_parallel.py \
    --workers 4 \
    --batch-size 50 \
    ...
```

## 📤 上传到HuggingFace Hub

### 方法1：转换时直接上传
```bash
# 首先登录
huggingface-cli login

# 转换并上传
python convert_umi_data_to_lerobot_fast.py \
    --input dataset.zarr.zip \
    --output ./output \
    --repo-id username/dataset_name \
    --push-to-hub
```

### 方法2：转换后再上传
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset('username/dataset_name', root='./output')
dataset.push_to_hub('username/dataset_name')
```

## 🐛 常见问题

### 问题：内存不足
**解决方案：**
- 减小 `--batch-size` 或 `--load-batch-size`
- 使用 `parallel.py` 而不是 `fast.py`
- 减少 `--workers` 数量

### 问题：速度没有提升
**解决方案：**
- 检查是否使用HDD（建议用SSD）
- 调整 `--workers` 参数
- 小数据集用原始版本更快

### 问题：进程卡住
**解决方案：**
- 检查磁盘空间（需要足够的临时空间）
- 确认zarr文件完整性
- 尝试减少workers数量

## 💡 最佳实践

1. **首次使用**：先用小数据集测试
2. **监控资源**：使用 `htop` 或 `top` 监控CPU和内存
3. **存储位置**：输出到本地SSD，完成后再移动
4. **批量处理**：参考 `benchmark_conversion.sh` 示例

## 📊 性能测试

运行benchmark脚本比较三个版本的性能：

```bash
chmod +x benchmark_conversion.sh
./benchmark_conversion.sh your_dataset.zarr.zip
```

## 🔗 相关文档

- 详细说明：[CONVERSION_README.md](CONVERSION_README.md)
- 原始脚本：`convert_umi_data_to_lerobot.py`
- 并行脚本：`convert_umi_data_to_lerobot_parallel.py`
- 优化脚本：`convert_umi_data_to_lerobot_fast.py`


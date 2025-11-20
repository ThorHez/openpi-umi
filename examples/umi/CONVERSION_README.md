# UMI to LeRobot Dataset Conversion Scripts

本目录包含三个用于将UMI zarr数据集转换为LeRobot格式的脚本，从基础版本到高度优化的多进程版本。

## 📋 脚本对比

| 脚本 | 特点 | 适用场景 | 预估速度 |
|------|------|----------|----------|
| `convert_umi_data_to_lerobot.py` | 原始单进程版本 | 小数据集、调试 | 基准速度 (1x) |
| `convert_umi_data_to_lerobot_parallel.py` | 并行处理版本 | 中等数据集 | 3-5x 加速 |
| `convert_umi_data_to_lerobot_fast.py` | 优化并行版本 | 大数据集、生产环境 | 5-10x 加速 |

## 🚀 快速开始

### 基础版本（原始脚本）

```bash
python convert_umi_data_to_lerobot.py \
    --input dataset.zarr.zip \
    --output ./umi_lerobot_dataset \
    --repo-id your_username/umi_dataset \
    --fps 30
```

### 并行版本（推荐用于中等数据集）

```bash
python convert_umi_data_to_lerobot_parallel.py \
    --input dataset.zarr.zip \
    --output ./umi_lerobot_dataset \
    --repo-id your_username/umi_dataset \
    --fps 30 \
    --workers 8 \
    --batch-size 100
```

### 优化版本（推荐用于大数据集）

```bash
python convert_umi_data_to_lerobot_fast.py \
    --input dataset.zarr.zip \
    --output ./umi_lerobot_dataset \
    --repo-id your_username/umi_dataset \
    --fps 30 \
    --workers 8 \
    --load-batch-size 50 \
    --process-batch-size 10
```

## 📚 详细说明

### 1. convert_umi_data_to_lerobot.py（原始版本）

**优点：**
- 简单直观，易于理解和调试
- 内存占用最小
- 不需要担心并发问题

**缺点：**
- 处理速度慢，单线程顺序处理
- 对于大数据集耗时很长

**适用场景：**
- 小数据集（<1000个episodes）
- 调试和测试
- 理解数据转换流程

### 2. convert_umi_data_to_lerobot_parallel.py（并行版本）

**优点：**
- 使用多进程并行处理episodes
- 每个进程独立访问zarr数据
- 可调节batch大小来平衡内存和速度
- 显著提升处理速度（3-5倍）

**缺点：**
- 每个进程需要打开zarr文件，有一定开销
- 内存占用比原始版本高

**适用场景：**
- 中等大小数据集（1000-5000个episodes）
- 内存有限的环境
- 需要稳定可靠的并行处理

**参数说明：**
- `--workers`: 工作进程数量（默认：CPU核心数）
- `--batch-size`: 每批处理的episodes数量（默认：100）

### 3. convert_umi_data_to_lerobot_fast.py（优化版本）

**优点：**
- 预加载数据到内存，避免重复IO
- 优化的批处理策略
- 最大化并行处理效率
- 最快的处理速度（5-10倍）
- 更多的图像写入线程

**缺点：**
- 内存占用最高（预加载数据）
- 需要更多系统资源

**适用场景：**
- 大数据集（>5000个episodes）
- 内存充足的环境
- 生产环境批量处理
- 需要最快处理速度

**参数说明：**
- `--workers`: 工作进程数量（默认：min(CPU核心数, 16)）
- `--load-batch-size`: 一次加载的episodes数量（默认：50）
- `--process-batch-size`: 每个worker处理的episodes数量（默认：10）

## 🎯 参数优化建议

### 根据系统配置选择参数：

#### 小型系统（4核, 8GB RAM）
```bash
python convert_umi_data_to_lerobot_parallel.py \
    --workers 4 \
    --batch-size 50 \
    ...
```

#### 中型系统（8核, 16GB RAM）
```bash
python convert_umi_data_to_lerobot_parallel.py \
    --workers 8 \
    --batch-size 100 \
    ...
```

#### 大型系统（16+核, 32GB+ RAM）
```bash
python convert_umi_data_to_lerobot_fast.py \
    --workers 16 \
    --load-batch-size 100 \
    --process-batch-size 10 \
    ...
```

### 根据数据特征调整：

**Episode较长（>100 frames）：**
- 减小 `load-batch-size`（避免内存溢出）
- 增加 `workers`（更多并行处理）

**Episode较短（<50 frames）：**
- 增加 `load-batch-size`（提高吞吐量）
- 减少 `workers`（避免进程切换开销）

**图像分辨率高：**
- 减小 `batch-size` / `load-batch-size`
- 增加 `image_writer_threads`

## 🔧 性能调优技巧

### 1. 监控资源使用

运行时监控CPU和内存使用：
```bash
# 另开一个终端
watch -n 1 'ps aux | grep convert_umi'
```

### 2. 调整workers数量

**CPU密集型**（数据处理为主）：
```bash
--workers $(nproc)  # 使用所有CPU核心
```

**IO密集型**（图像读写为主）：
```bash
--workers $(($(nproc) * 2))  # 使用2倍CPU核心数
```

### 3. 内存管理

如果遇到内存不足，尝试：
- 减小 `batch-size` 或 `load-batch-size`
- 使用 `parallel.py` 版本而不是 `fast.py`
- 减少 `workers` 数量

### 4. 存储优化

如果输出目录在网络存储上：
- 先输出到本地SSD
- 转换完成后再复制到最终位置

## 📊 性能对比示例

假设数据集：2000 episodes, 平均100 frames/episode, 224x224 RGB图像

| 脚本版本 | 耗时（估算） | CPU使用 | 内存使用 |
|---------|-------------|---------|---------|
| 原始版本 | ~120分钟 | 单核100% | ~2GB |
| 并行版本(8 workers) | ~30分钟 | 8核90% | ~6GB |
| 优化版本(8 workers) | ~15分钟 | 8核95% | ~10GB |

*注：实际性能取决于硬件配置、数据特征和系统负载*

## 🐛 故障排查

### 问题1: 内存不足 (OOM)
**解决方案：**
- 减小batch size参数
- 使用 `parallel.py` 版本
- 减少workers数量
- 增加系统swap空间

### 问题2: 进程卡住不动
**解决方案：**
- 检查zarr文件是否损坏
- 确认临时目录有足够空间
- 尝试减少workers数量
- 查看详细错误：添加 `--verbose` 标志（如果支持）

### 问题3: 转换速度没有提升
**可能原因：**
- IO瓶颈（磁盘速度限制）
- workers数量设置不当
- 数据集太小，并行开销大于收益

**解决方案：**
- 使用SSD而不是HDD
- 调整workers参数
- 小数据集使用原始版本

## 📝 完整命令示例

### 基础转换
```bash
python convert_umi_data_to_lerobot_fast.py \
    --input /path/to/dataset.zarr.zip \
    --output ./output_dataset \
    --repo-id username/dataset_name \
    --fps 30 \
    --task "pick and place task"
```

### 转换并上传到HuggingFace
```bash
# 首先登录HuggingFace
huggingface-cli login

# 转换并上传
python convert_umi_data_to_lerobot_fast.py \
    --input /path/to/dataset.zarr.zip \
    --output ./output_dataset \
    --repo-id username/dataset_name \
    --fps 30 \
    --push-to-hub \
    --workers 16
```

### 批量处理多个数据集
```bash
#!/bin/bash
# batch_convert.sh

for zarr_file in /data/datasets/*.zarr.zip; do
    basename=$(basename "$zarr_file" .zarr.zip)
    echo "Processing: $basename"
    
    python convert_umi_data_to_lerobot_fast.py \
        --input "$zarr_file" \
        --output "./datasets/$basename" \
        --repo-id "username/$basename" \
        --fps 30 \
        --workers 8
done
```

## 📞 技术支持

如果遇到问题：
1. 检查本文档的故障排查部分
2. 查看脚本源码中的注释
3. 尝试使用原始版本确认数据格式正确
4. 查看系统日志和错误信息

## 🔄 版本历史

- **v1.0** (原始): 基础单进程版本
- **v2.0** (parallel): 添加多进程支持
- **v3.0** (fast): 优化内存管理和批处理策略

## 📄 许可

这些脚本遵循OpenPI项目的许可证。


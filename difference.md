我对两个目录做了当前文件系统的只读对比。它们基于同一个 Git 提交 `6c79879`，因此差异主要是 `openpi-umi-robustness` 为跨场景 DTW + UOT 联合训练增加的代码。

核心代码差异共 6 个文件：4 个新增、2 个修改；此外还有说明文档、环境启动脚本和生成的 DTW 数据。

## 1. 新增的核心脚本

### `scripts/precompute_dtw_pairs.py`

新增离线 DTW 配对流程：

- 从 LeRobot 数据集中按 episode 读取轨迹。
- 使用左右臂末端位置拼成 6 维轨迹。
- 用 Numba 实现动态规划 DTW。
- 为每条 target episode 匹配 source episodes。
- 保存：
  - 归一化 DTW 距离；
  - target 时间点到 source 时间点的映射；
  - episode 和全局帧索引。
- 支持：
  - `--candidate-pool`：先用16点轨迹描述子预筛；
  - `--top-k`：仅保留最近的若干 source；
  - `.json.gz` 压缩输出。

当前正式 WBCD 配对使用 `candidate-pool=0`，即所有 source episode 都进行精确 DTW。

### `src/openpi/training/uot_data_loader.py`

新增两条并行数据流。

BC 数据流：

- 不再根据各数据集大小隐式决定采样比例。
- 使用 `WeightedRandomSampler` 严格按照配置中的场景概率采样。
- 每个样本的权重计算为：
  
  ```text
  dataset_probability / dataset_length
  ```

UOT 数据流：

- 读取离线 DTW 配对文件。
- 对每个 target 帧，根据 DTW 距离给 source episode 加权：
  
  ```python
  sigmoid(sharpness * (cutoff - distance))
  ```

- 从 DTW 对齐的 source 时间点中抽样。
- 再在该点附近 `±temporal_window` 帧随机扰动。
- 返回：
  
  ```text
  source observation
  target observation
  source position label
  target position label
  ```

- 位置标签直接从 LeRobot 表加载并放进共享内存，避免每个 worker 重复占用大量内存。

### `src/openpi/training/uot.py`

新增纯 JAX 的非平衡最优传输实现：

- 两两平方欧氏距离计算；
- log-domain unbalanced Sinkhorn；
- 支持两种 KL 参考测度：
  - 普通的 `a ⊗ b`；
  - `heuristic=true` 对应的对角参考 `I/N`。
- ground cost 为：

  ```text
  M = embedding_cost_scale × 视觉距离
    + label_cost_scale × 状态距离
  ```

- 运输计划 `Π` 使用 `stop_gradient`，不通过 Sinkhorn 反向传播。
- UOT 损失：

  ```text
  L_UOT = <Π, M>
  ```

状态标签不依赖模型，因此实际梯度主要通过视觉距离返回 SigLIP。

### `scripts/train_uot.py`

新增独立的联合训练入口，复用原仓库的：

- `init_logging`
- `init_wandb`
- `init_train_state`
- checkpoint 保存和恢复机制
- FSDP mesh 和 sharding

每一步同时读取：

```text
一个 BC batch
一个 DTW source/target 配对 batch
```

总损失：

```text
L_total = L_BC + alignment_weight × L_UOT
```

同时记录：

- `loss`
- `bc_loss`
- `uot_loss`
- `weighted_uot_loss`
- `uot_plan_mass`
- `uot_embedding_cost`
- `uot_label_cost`
- `grad_norm`
- `param_norm`

恢复训练、优化器状态、EMA、step 等仍沿用原 checkpoint 逻辑。

## 2. 修改的模型逻辑

### `src/openpi/models/pi0.py`

增加了约 23 行的：

```python
encode_visual_features(...)
```

它：

1. 复用模型原有的 observation 图像预处理；
2. 使用已有的 PaliGemma/SigLIP 图像编码器；
3. 对每个相机的视觉 token 做均值池化；
4. 用 `image_masks` 排除无效相机；
5. 再对有效相机特征取平均。

没有新增：

- projection head；
- 对比学习模块；
- 第二套视觉编码器。

因此 UOT 直接约束策略网络原本使用的视觉特征。

## 3. 修改的训练配置

### `src/openpi/training/config.py`

新增 `UOTConfig`，包含：

```python
enabled
pair_info_path
target_dataset_index
bc_dataset_probabilities
ot_batch_size
alignment_weight
embedding_cost_scale
label_cost_scale
entropy_reg
marginal_relaxation
sinkhorn_iterations
diagonal_reference
dtw_distance_cutoff
dtw_distance_sharpness
temporal_window
```

`TrainConfig` 中增加：

```python
uot: UOTConfig | None = None
```

因此普通配置保持兼容；只有 `train_uot.py` 会使用该字段。

新增主配置：

```text
pi05_umi_robustness
```

当前代码中的实际设置为：

- 17 个 WBCD 场景；
- target 场景索引：`9`；
- target 数据集：`0604_wbcd_hitl_with_wrist_1813`；
- BC batch：`72`；
- UOT batch：`32`；
- `alignment_weight=0.1`；
- `embedding_cost_scale=0.01`；
- `label_cost_scale=0.05`；
- `entropy_reg=0.1`；
- `marginal_relaxation=0.05`；
- `diagonal_reference=True`；
- 总训练步数：`70000`；
- checkpoint 间隔：`1000`；
- `keep_period=20000`。

另外，已有配置：

```text
pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta
```

也被加入 UOT 参数，并把初始化权重改为已有的 `59999` checkpoint，而不是原来的官方 base checkpoint。

## 4. 推理脚本修改

### `scripts/infer_local.py`

这部分主要是让本地评估适配 `pi05_umi_robustness`：

- 参数改为 Tyro CLI；
- 支持 `chunk-*` 下的 episode 文件；
- 从 Parquet 读取四路腕部相机；
- 构造双臂的十组状态字段；
- 支持双臂 20 维有效动作；
- 将模型 32 维输出裁剪到前 20 维；
- 在反归一化之前移除 padding 动作维度；
- 动态生成每个动作维度的名称和可视化子图；
- 改为评估指定的单个 episode；
- 删除保存每帧观测图像的逻辑；
- 移除 Seaborn 依赖。

## 5. 新增辅助文件和生成物

新增：

- `README_UOT_ZH.md`：实现说明、训练方式和验证结果。
- `activate.sh`：复用原仓库 `.venv` 和缓存，但让 Python 优先导入 robustness 仓库代码。
- `artifacts/dtw/elec_meta_pair_info.json.gz`：约 34.3 MB。
- `artifacts/dtw/pi05_umi_robustness_pair_info.json.gz`：约 23.5 MB。
- `inference_results/`：推理结果，不属于核心源码。

`activate.sh` 的核心作用是：

```bash
source /data2/hzl_workspace_for_pi/openpi-umi/.venv/bin/activate
export PYTHONPATH=/data2/hzl_workspace_for_pi/openpi-umi-robustness/src:...
```

所以 robustness 仓库没有重新安装一套依赖。

## 6. 两边相同、并非 robustness 独有的改动

下面这些文件虽然相对 Git 基线有本地改动，但两个目录内容完全相同，因此不是 robustness 分支新增的差异：

- `scripts/train.py`
- `scripts/train_multi_dataset.py`
- `scripts/compute_norm_stats.py`
- `src/openpi/policies/umi_policy.py`
- `src/openpi/shared/runtime_paths.py`
- `examples/umi/convert_umi_data_to_lerobot_add_target_value.py`
- `examples/umi/config/dataset_config_pick_elec_meta.yaml`

## 7. 一个文档与代码不一致之处

`README_UOT_ZH.md` 目前部分参数已经过期：

| 参数 | README | 当前代码 |
|---|---:|---:|
| `ot_batch_size` | 128 | 32 |
| `keep_period` | 30000 | 20000 |
| 初始化权重描述 | `gs://...pi05_base` | 本地缓存中的同一 base checkpoint |

实际运行应以 `config.py` 为准。

总体而言，这个 robustness 仓库没有大规模修改原训练框架，而是在原框架旁边增加了一条完整的：

```text
离线 DTW 配对
→ 概率化 BC 采样
→ DTW source/target 成对采样
→ SigLIP 特征提取
→ UOT Sinkhorn 对齐
→ BC + UOT 联合训练
```

流程。
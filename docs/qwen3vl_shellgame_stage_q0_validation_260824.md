# Qwen3-VL × 递归压缩 Memory：ShellGame Stage Q0 验证

日期：2026-08-24

## 1. 本轮目标

本轮只验证低频高层链路，不训练 action：

```text
ShellGame 短视频窗口
  → 冻结 Qwen3-VL-4B
  → 严格 event/state-delta JSON
  → deterministic schema + camera/world adapter
  → 已验证 recurrent updater
  → 最终杯位 probe
```

Qwen 和 OpenPI 使用两个独立进程。Qwen 仅生成可审计 JSONL；OpenPI 不导入
Torch/Transformers，只读取缓存。

## 2. 环境

Qwen 环境：

```bash
/root/miniconda3/bin/conda create -y -n qwen3vl_shellgame \
  --clone /data1/conda_envs/XVLA

/data1/conda_envs/qwen3vl_shellgame/bin/python -m pip install \
  --upgrade 'transformers==4.57.1'
```

实际验证版本：

```text
torch        2.7.1+cu118
transformers 4.57.1
Qwen         /data2/hzl_workspace_for_pi_mem/Qwen3-VL-4B-Instruct
GPU          NVIDIA A100-SXM4-80GB
```

OpenPI 的 `.venv` 没有修改。

## 3. 实现文件

- `src/openpi/planning/qwenvl_event_schema.py`：无外部依赖的严格公共 schema。
- `src/openpi/planning/qwenvl_subgoal_planner.py`：本地 Qwen3-VL frozen planner。
- `src/openpi/tasks/shellgame/qwenvl_event_adapter.py`：相机坐标到 world slot 的确定性 adapter 和幂等 ledger。
- `scripts/mem/build_shellgame_qwenvl_event_cache.py`：离线、可续跑、逐条 fsync 的 JSONL 缓存构建。
- `examples/shellgame/eval_qwenvl_event_recurrent_memory.py`：Qwen event 注入旧 recurrent memory 的五条件评测。

对应单元测试共 5 项，全部通过。

## 4. 数据与命令

严格使用 semantic-memory split 的前 12 个 held-out episode：

```text
8, 16, 17, 31, 47, 56, 72, 80, 90, 96, 102, 122
```

Qwen 离线标注：

```bash
CUDA_VISIBLE_DEVICES=0 \
/data1/conda_envs/qwen3vl_shellgame/bin/python \
scripts/mem/build_shellgame_qwenvl_event_cache.py \
  --split semantic-val \
  --num-episodes 12 \
  --max-retries 0 \
  --output evaluation/shellgame/qwenvl_event_cache/qwen3vl_4b_semantic_val_12ep.jsonl \
  --overwrite
```

OpenPI recurrent memory 对照：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
.venv/bin/python examples/shellgame/eval_qwenvl_event_recurrent_memory.py \
  --cache evaluation/shellgame/qwenvl_event_cache/qwen3vl_4b_semantic_val_12ep.jsonl \
  --eval-batches 2 \
  --overwrite
```

评测脚本直接从 12 个 raw NPZ 构造固定 60 帧 prefix，不展开完整 LeRobot
数据集。开发过程中曾触发一个 59GB HuggingFace parquet 临时缓存；该缓存已删除，
脚本也已改为不会再次产生这类缓存。

## 5. Stage Q0 结果

### 5.1 Qwen 离线标注

| 指标 | 结果 |
|---|---:|
| JSON schema valid | 60/60 = 100% |
| ShellGame adapter valid | 60/60 = 100% |
| reveal cup | 8/12 = 66.7% |
| swap 0 pair | 4/12 = 33.3% |
| swap 1 pair | 4/12 = 33.3% |
| swap 2 pair | 5/12 = 41.7% |
| no-event settle | 12/12 = 100% |
| 完整四事件序列 | 0/12 = 0% |

Qwen 对交换关系发生明显类别塌缩：36 个 swap 中有 35 个预测为 world-slot
`left-right`。模型置信度仍通常约 0.95，因此当前生成置信度不能作为可靠过滤器。

符号状态转移得到的最终杯位：

| 初始杯 | swap 序列 | 最终杯位准确率 |
|---|---|---:|
| GT | GT | 12/12 = 100% |
| GT | Qwen | 6/12 = 50.0% |
| Qwen | GT | 8/12 = 66.7% |
| Qwen | Qwen | 5/12 = 41.7% |

### 5.2 注入已验证 recurrent memory

| 条件 | stage memory | final memory |
|---|---:|---:|
| GT initial + GT swaps | 100% | 100% |
| GT initial + Qwen swaps | 38.9% | 50.0% |
| Qwen initial + GT swaps | 66.7% | 66.7% |
| Qwen initial + Qwen swaps | 30.6% | 41.7% |
| GT initial + wrong-episode Qwen swaps | 38.9% | 41.7% |

Neural recurrent final accuracy 与符号状态转移完全一致。因而：

1. Qwen JSON → relation code → recurrent updater 的新接口没有额外损失；
2. 当前失败不是 memory token、递归更新或 readout 导致；
3. 瓶颈是 Qwen3-VL-4B 对相同外观杯子的轨迹身份 grounding；
4. 当前 Qwen relation 伪标签不能直接加入训练集，否则会把 relation 分布污染为 `left-right`。

## 6. 可以使用与暂时不能使用的监督

当前可以使用：

- JSON/schema 格式蒸馏；
- 通用 subgoal 文本；
- 高层候选事件描述；
- 经过独立视觉 verifier 或仿真 GT 校验后的 state delta；
- no-event/keep-state 监督，但需要后续使用未泄漏 phase 的混合窗口复验。

当前不能直接使用：

- Qwen 原始 swap entity pair；
- Qwen 自报 confidence 作为 acceptance gate；
- 未经校验的 reveal cup；
- Qwen/Qwen 最终杯位作为训练真值。

## 7. 下一步质量门槛

建议保留当前分层结构，但把 Qwen 定位为 proposal generator：

```text
Qwen proposal
  + short-window visual evidence
  → grounded verifier / recurrent updater acceptance
  → Task Manager commit
```

只有当离线伪标签满足以下门槛才进入 memory 训练：

```text
schema valid                 >= 99%
event/no-event balanced acc  >= 90%
entity/state-delta accuracy  >= 90%
high-confidence ECE          可接受
wrong-episode rejection      >= 90%
```

对于当前 ShellGame，下一项最有价值的实验是：保持 Qwen 的开放词汇事件/subgoal，
用已经验证的轻量视觉 event encoder 对实体 pair 做 grounded correction；对无法通过
视觉一致性检查的 Qwen proposal 标记为 `request_reobservation`，而不是提交 memory。

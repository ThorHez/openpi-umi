# RoboMME 单任务等曝光量消融（2026-08-27）

## 目的

验证四任务统一训练中的性能不足，究竟主要来自多任务负迁移，还是来自任务自身的数据、视觉表征或状态更新难度。

本实验只改变训练数据是否按任务拆分，不增加任务定制 head，不改变模型结构、输入格式、teacher、loss 定义或测试协议。

## 公平实验协议

- 任务：PickXTimes、VideoPlaceOrder、VideoUnmask、VideoUnmaskSwap。
- 数据划分：每个任务 train/dev/test = 70/15/15 episodes。
- 初始化：四个单任务模型都从同一个统一 soft-gate checkpoint 初始化：
  `checkpoints/robomme_four_task_fixed_chunk_soft_gate_v1_260826/best/params`。
- 统一训练基线：batch size 4，每一步四个任务各取一个样本，500 steps，因此每个任务约看到 500 个训练样本。
- 单任务实验：batch size 4，125 steps，因此每个任务同样约看到 500 个训练样本。
- 优化目标：沿用当前成功的加权目标，`change_state_weight=10`、`final_state_weight=4`；不使用 trajectory/final 解耦 loss。
- 学习率：peak `1e-5`，end `3e-6`，warmup 10 steps。
- 每个任务使用 3 个随机种子：260827、260828、260829。
- checkpoint 只按 dev 指标字典序选择：final、transition、sequence、state。
- 种子选择期间完全跳过 test；每个任务选定一个种子后，只执行一次锁定 test 评估。

因此，本实验比较的是“相同每任务训练曝光量下，是否存在跨任务干扰”，而不是通过给单任务更多数据或更多梯度步来取得优势。

## 三种子验证集结果

表中数值为严格 episode/state 指标；百分数均为绝对百分比。

| Task | Seed | Best step | Dev final | Dev transition | Dev sequence | Dev state |
|---|---:|---:|---:|---:|---:|---:|
| PickXTimes | **260827** | 75 | **100.0** | **43.4** | 0.0 | **41.6** |
| PickXTimes | 260828 | 125 | 100.0 | 41.4 | 0.0 | 39.6 |
| PickXTimes | 260829 | 125 | 93.3 | 39.4 | 0.0 | 38.6 |
| VideoPlaceOrder | 260827 | 100 | 20.0 | 27.3 | 0.0 | 38.9 |
| VideoPlaceOrder | **260828** | 50 | **20.0** | **27.3** | 0.0 | **40.3** |
| VideoPlaceOrder | 260829 | 100 | 20.0 | 27.3 | 0.0 | 39.9 |
| VideoUnmask | **260827** | 25 | **40.0** | **38.7** | **6.7** | **61.2** |
| VideoUnmask | 260828 | 100 | 33.3 | 35.5 | 6.7 | 60.2 |
| VideoUnmask | 260829 | 25 | 40.0 | 38.7 | 6.7 | 61.2 |
| VideoUnmaskSwap | **260827** | 75 | **26.7** | **23.6** | 0.0 | **32.2** |
| VideoUnmaskSwap | 260828 | 75 | 13.3 | 23.6 | 0.0 | 30.7 |
| VideoUnmaskSwap | 260829 | 100 | 26.7 | 20.0 | 0.0 | 28.8 |

VideoUnmask 的 260827 和 260829 完全同分，按预先约定选择较小 seed 260827。其余任务均按上述 dev 字典序选出。

## 锁定测试集结果

统一模型基线为：
`checkpoints/robomme_four_task_fixed_chunk_soft_gate_final_ft_v1_260827/result.json`。

| Task | Training | Test final | Test transition | Test hold | Test state | Test sequence |
|---|---|---:|---:|---:|---:|---:|
| PickXTimes | Unified | 73.3 | **40.2** | 38.1 | 38.4 | 0.0 |
| PickXTimes | Single | **80.0** | 38.1 | **38.6** | **38.6** | 0.0 |
| VideoPlaceOrder | Unified | **26.7** | **32.5** | 55.3 | 54.0 | 0.0 |
| VideoPlaceOrder | Single | **26.7** | 25.0 | **57.1** | **55.2** | 0.0 |
| VideoUnmask | Unified | 20.0 | 20.0 | 67.6 | 53.5 | 0.0 |
| VideoUnmask | Single | **40.0** | **43.3** | **73.2** | **64.4** | **20.0** |
| VideoUnmaskSwap | Unified | **20.0** | **7.0** | **25.6** | **20.7** | **0.0** |
| VideoUnmaskSwap | Single | 13.3 | **7.0** | **25.6** | **20.7** | **0.0** |

四任务宏平均：

| Training | Final | Transition | State |
|---|---:|---:|---:|
| Unified | 35.0 | 24.9 | 41.6 |
| Single | **40.0** | **28.4** | **44.7** |
| Absolute delta | **+5.0** | **+3.4** | **+3.1** |

## 结论

### 1. 单任务训练不会普遍解决四任务问题

- **VideoUnmask 明显受益**：final `20.0 -> 40.0`，transition `20.0 -> 43.3`，state `53.5 -> 64.4`，并首次得到 20.0% full-sequence exact。这里存在明确的多任务负迁移。
- **PickXTimes 只有轻度 final 收益**：final `73.3 -> 80.0`，但 transition `40.2 -> 38.1`，full sequence 仍为 0。拆分训练强化了最终 readout，却没有改善完整记忆轨迹。
- **VideoPlaceOrder 基本不受益**：final 完全不变，transition 还下降 7.5 个百分点。主要瓶颈不是多任务竞争，更可能是空间关系状态的可辨识性、监督质量或 updater/readout 表达能力。
- **VideoUnmaskSwap 不受益**：final 下降一个 episode（6.7 个百分点），trajectory 指标完全不变。它的主要瓶颈仍是交换事件的视觉识别和递归状态变换，而不是统一训练造成的干扰。

### 2. 对 action 接入的判断

- **PickXTimes 可以进入 action smoke test**：80% test final 已达到此前设定的 final 门槛，但应把它视为“最终动作条件可试”，不能声称 memory trajectory 已解决。
- **VideoUnmask 值得用单任务 checkpoint 接 action 做小规模验证**：它是唯一同时提升 final、transition、state 和 sequence 的任务。
- **VideoPlaceOrder 与 VideoUnmaskSwap 暂不适合仅靠单任务继续加步后直接接 action**：本实验已经排除了“统一训练负迁移”是其主因，继续同配方训练的预期收益低。
- 不建议为了四任务整体效果维护四套定制 head。更合适的方向是保留统一结构，处理任务间梯度冲突或改进困难任务的视觉事件/状态监督。

### 3. 统计限制

每个 test 只有 15 个 episode，因此 final accuracy 每个 episode 对应 6.7 个百分点。Pick 的 `+6.7` 和 Swap 的 `-6.7` 都只相当于一个 episode，不能作为强结论；VideoUnmask 的 `+20.0` 相当于三个 episode，并且 trajectory 指标同步大幅提升，证据相对更可信。正式论文表格前应扩展 test episode 或补充 bootstrap 置信区间。

## 推荐下一步

1. 保留统一模型作为主线和参数共享基线。
2. 对 VideoUnmask 单任务 checkpoint 先做 10--20 条 action smoke test，验证 memory 提升能否转化为操作成功率。
3. PickXTimes 可并行做 10--20 条 action smoke test，但重点记录 final memory 正确而 action 失败、以及中途状态错误是否影响动作。
4. Place/Swap 不再做简单的“单任务加步”扩展；下一轮应优先检查可观测事件标签、teacher state 可分性和 change-step 的视觉混淆。
5. 若仍希望保持一个统一模型，可尝试共享 updater + 统一 readout 下的 task-balanced gradient surgery/PCGrad，或对 Unmask 提高采样但不新增任务 head。

## 产物

- Pick：`checkpoints/robomme_single_task_pick_equal_exposure_seed260827_260827/`
- Place：`checkpoints/robomme_single_task_place_equal_exposure_seed260828_260827/`
- Unmask：`checkpoints/robomme_single_task_unmask_equal_exposure_seed260827_260827/`
- Swap：`checkpoints/robomme_single_task_swap_equal_exposure_seed260827_260827/`
- 每个选中目录中的锁定测试结果：`test_visual_dependence.json`


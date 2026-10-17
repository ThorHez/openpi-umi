# Visual ceiling 状态转移蒸馏到 recurrent MEM：semantic-feedback curriculum

日期：2026-08-29

## 目的

在已验证的 visual ceiling + frozen MME action 30/30 上界基础上，检验以下最小优化能否
把 ceiling 的状态转移知识蒸馏到可部署的 recurrent MEM：

- 保留统一的 `7 entities x 5 region states` semantic table；
- 显式监督 `hold / write / swap`、write entity/region、swap pair；
- 使用 episode-local anchor ROI 和共享 anchor/pair pointer；
- operation head 显式读取 previous semantic table；
- 训练从 ceiling previous state 逐渐切换到 student previous state；
- student feedback 前向使用 straight-through hard semantic table；
- soft event gate 保持不变；
- 新增 transition、no-change、changed-field delta、final readout 和轻量 trajectory loss；
- checkpoint 最大化 `min(transition, no-change, mean final)`。

## 数据与训练

- 任务：VideoUnmask、VideoUnmaskSwap、VideoPlaceOrder；
- train/dev/test episode：210/45/45；
- 输入：固定非重叠 12 帧、4x4 SigLIP patch tokens；
- transition/no-change chunk（train）：636/4534；
- 训练步数：1200；batch size 12；seed 260901；
- curriculum：前 20% 全 teacher forcing，中间 50% 线性降到 30%，最后 30% 降到 0；
- 初始化：旧 anchor-conditioned operation checkpoint 中形状兼容的视觉/anchor 参数；
- selected checkpoint：step 1100。

## 结果

### Best checkpoint 的 final query accuracy

| 模式 / split | VideoUnmask | VideoUnmaskSwap | VideoPlaceOrder | Mean |
|---|---:|---:|---:|---:|
| Dev teacher-forced | 100.0% | 45.5% | 53.3% | 66.3% |
| Dev free-rollout | 46.7% | 9.1% | 13.3% | 23.0% |
| Test teacher-forced | 100.0% | 57.7% | 33.3% | 63.7% |
| Test free-rollout | 6.7% | 15.4% | 6.7% | **9.6%** |

### Free-rollout transition/hold

| Split | Transition exact | No-change exact | All-state exact |
|---|---:|---:|---:|
| Train | 25.9% | 24.3% | 24.5% |
| Dev | 22.3% | 23.4% | 23.3% |
| Test | 11.0% | 22.7% | 21.2% |

### Test operation accuracy

| Operation | Accuracy |
|---|---:|
| Event type | 69.7% |
| Write entity | 47.9% |
| Write region | 53.7% |
| Swap pair | 75.0% |

## 与已有方法对比

| 方法 | Test mean final |
|---|---:|
| Visual ceiling | 100.0% |
| Existing recurrent MEM fixed readout | 41.1% |
| Decomposed operation-only | 33.4% |
| Anchor-conditioned operation-only | 31.8% |
| 本次 semantic-feedback curriculum | **9.6%** |

本次方法改善了局部 `write region` 和 `swap pair`，但没有改善完整递归。正确前态能把 test
mean final 提到 63.7%，而 free rollout 下降到 9.6%，证明主要失败点是 exposure/累计状态
错误，而不是确定性 table updater 或 action backbone。

## 结论

这次实验为负结果，不应接 action。简单地增加 previous-state conditioning、delta loss 和
teacher-forcing curriculum 会让 transition head 依赖正确前态；当推理改用 student state 时，
soft gate 的 false update 和第一次错误 payload 会污染后续所有状态。训练末尾把 teacher
forcing 降到 0 也没有恢复，step 1200 反而从 best step 1100 退化。

下一步不应继续堆 trajectory/final loss。最小且可证伪的优化应是：

1. 在相同模型上从第 1 步开始 pure free rollout，消除 teacher-state dependency；
2. 单独校准 event routing，先冻结 payload，只优化 hold/event gate；
3. 比较 soft gate、straight-through hard gate 和带阈值的 hard no-change route；
4. checkpoint 继续使用 `min(transition, no-change, final)`；
5. 只有 free-rollout test final 超过旧 MEM 41.1% 后才接 MME action。

## 产物

- 模型：`src/openpi/tasks/robomme/anchor_conditioned_transition_memory.py`
- 单元测试：`src/openpi/tasks/robomme/anchor_conditioned_transition_memory_test.py`
- 训练脚本：`scripts/mem/train_robomme_anchor_transition_curriculum.py`
- checkpoint：`checkpoints/robomme_anchor_transition_curriculum_seed260901_260829/params.msgpack`
- 完整结果：`checkpoints/robomme_anchor_transition_curriculum_seed260901_260829/result.json`
- 配置：`checkpoints/robomme_anchor_transition_curriculum_seed260901_260829/training_config.json`

验证：

```text
2 passed
```

# VideoPlaceOrder learned event head + recurrent ordinal updater（2026-08-31）

## 结论

训练期使用特权事件轨迹监督、推理期只使用 RGB、7D joint state、gripper state、语言中的
ordinal 和 RGB target anchors 的可部署模型，在官方 `symbolic-grounded-subgoal/79999` 动作模型上
取得 **86.00 ± 0.00%（129/150）** 的正式闭环成功率。

该结果不使用 simulator SimpleSG/GroundSG、object pose、contact、success flag、oracle phase，
也不使用此前的 deterministic visual event correction。与相同协议的旧 causal MEM 相比提升
**16.67 percentage points**；相对 94% 的 observable-rule teacher/upper bound 仍有 8 points 差距。

## 方法

训练数据只从原始 train/dev replay 生成，test split 不物化、不参与阈值或 checkpoint 选择：

- train：70 episodes、3856 chunks、213 placement writes、11 swaps；
- dev：15 episodes、808 chunks、42 placement writes、2 swaps；
- 每个 chunk：12 RGB frames + 12 ×（7D joint + 1D gripper）；
- teacher：训练期 GT event type/payload/state trajectory；
- student event head：anchor-local RGB evidence + chunk proprio temporal evidence；
- updater：hard categorical event feedback + deterministic semantic table update；
- ordinal readout：读取 `ordered_cell_{ordinal-1}`，没有 task-specific action head。

课程训练共 3000 steps。前 1200 steps 学 event type/region/pair，随后逐步把 teacher forcing 从 1
降至 0，并加入 transition、hold、delta、trajectory 和 final ordinal losses。checkpoint 只按 dev
上的 `min(transition, hold, final ordinal)` 选择，best step 为 1300。

## 离线 dev free-rollout

| Metric | Result |
|---|---:|
| Final ordinal accuracy | 86.67% |
| Transition state exact | 93.18% |
| Hold state exact | 98.04% |
| Trajectory state exact | 97.77% |
| Full-sequence exact | 73.33% |
| Event update precision | 95.24% |
| Event update recall | 90.91% |
| Write-region accuracy | 100.00% |
| Swap-pair accuracy | 100.00% |

## 正式闭环协议和结果

- task：VideoPlaceOrder test episode 0–49；
- action：官方 MME `symbolic-grounded-subgoal/79999`；
- seeds：7、17、27；
- maximum simulator steps：1300；
- execution phase：observable RGB/gripper/EEF FSM；
- target anchors：demonstration RGB；
- MEM：冻结的 learned event head + deterministic recurrent ordinal updater。

| Method | Seed 7 | Seed 17 | Seed 27 | Mean ± sample SD | Aggregate |
|---|---:|---:|---:|---:|---:|
| Old causal MEM | 34/50 | 35/50 | 35/50 | 69.33 ± 1.15% | 104/150 |
| **Learned event + ordinal MEM** | **43/50** | **43/50** | **43/50** | **86.00 ± 0.00%** | **129/150** |
| Observable-rule teacher / upper bound | 47/50 | 46/50 | 48/50 | 94.00 ± 2.00% | 141/150 |

三个 seed 的 learned MEM 失败 episode 完全相同：
`10, 11, 17, 23, 27, 34, 35`。这表明当前剩余误差主要来自固定的 demonstration semantic
prediction / execution-stage behavior，而不是 action sampling seed。

## 合规审计

150 个 episode trace 均满足：

- `source == learned_event_mem`；
- checkpoint 指向同一个冻结的 `params.msgpack`；
- `execution_phase_source == observable_rgb_gripper_eef_fsm`；
- 不含 `place_visual_event_correction`；
- 每个 seed 恰好 50 条。

评测在 episode 24 和 48 后出现已知的 SAPIEN Vulkan
`vk::createInstanceUnique: ErrorIncompatibleDriver`。每次仅重启 renderer 并从同一
`progress.json` 续跑；基础设施异常没有计为 episode failure。

## 论文定位

86% 可以作为“完全 learned、可部署 MEM + official action”的主表候选。94% 应保留为
training-time teacher / observable structured upper bound，不能标作纯 learned MEM。两者 8-point
差距说明事件轨迹蒸馏已经吸收了 teacher 的大部分增益，但 hard episodes 上的 event/binding
泛化仍未完全达到规则 teacher。

## 产物

- 事件轨迹：`artifacts/videoplaceorder_observable_event_trajectories_v1_260831/`
- 训练脚本：`scripts/mem/train_videoplaceorder_learned_event_ordinal_memory.py`
- checkpoint：`checkpoints/videoplaceorder_learned_event_ordinal_seed260831_260831/`
- 在线推理：`scripts/mem/robomme_explicit_event_inference.py`
- 正式结果：`robomme_policy_learning/runs/evaluation/place-learned-event-mem-observable-fsm-3seed50-260831/`


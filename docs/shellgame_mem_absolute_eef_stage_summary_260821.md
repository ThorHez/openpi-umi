# ShellGame MEM + Absolute EEF 阶段总结

> 时间范围：2026-07-27 ～ 2026-08-21<br>
> 阶段状态：本轮模型与数据迭代结束<br>
> 当前推荐模型：V10 step1000<br>
> 当前正式结果：73/100 抓取成功，100/100 选杯正确

## 1. 阶段目标与最终结论

本阶段的目标，是让 Pi0/Pi0.5 在三次交换的 ShellGame 中完成两件事：

1. 从 0～59 帧视觉历史中记住球最终在哪个空间杯位；
2. 在交换结束后，根据当前视觉闭环控制机械臂靠近、下降、闭爪并抬起正确杯子。

最终形成的系统把这两个问题明确分开：

- 历史 tracker/memory 负责“选哪个杯子”；
- 当前帧和 action expert 负责“怎样持续对准并完成抓取”。

最终阶段的核心结论如下。

1. 初始 absolute-joint 模型失败的首要原因不是 JOINT_POSITION 控制器不能执行轨迹。Oracle joint replay 在左/中/右杯上均达到 30/30，说明真实未来 joint trajectory 可以被当前控制器可靠执行。
2. 原始 MEM 压缩路径没有稳定编码三次交换后的空间杯位。模型 joint 预测做 FK 后约为随机水平，正常历史、关闭 memory、打乱历史和错误 episode 历史之间差别很小。
3. 原始逐帧 learned-query 压缩破坏了跨帧稳定的空间对应关系。改成固定 2×2 网格池化、显式时空 Transformer、分阶段训练后，一次交换和完整三次交换的 held-out 视觉跟踪均可达到 100%。
4. 经过验证的 tracker/memory 可以从 joint 任务直接迁移到 absolute EEF，不需要重新训练 memory；后续只训练 action expert 和动作/时间投影。
5. Absolute EEF 比 absolute joint 更适合当前抓取任务，因为动作误差直接定义在末端笛卡尔空间，不需要模型同时学会高维 joint 到 EEF 的非线性误差补偿。
6. 后期失败已经不是选错杯，而是正确选杯后的持续 XY 对准、下降、闭爪和抬升时序问题。V6、V9、V10 的正式/筛选测试均保持 100% 选杯正确。
7. 更多纠错数据或更低 validation loss 并不自动提高闭环成功率。数据的连续性、行为保留、动作 chunk 前部的时序，以及数据是否覆盖模型真实到达的状态，比简单扩大训练步数更重要。
8. 最终 V10 step1000 在100个 episode 上得到 73/100；V6 既有正式基线为67/100。该提升方向积极，但同种子配对检验 `p=0.263`，还不能当作统计上确定的模型优势。

## 2. 最终系统结构

### 2.1 历史视觉与语义 memory

最终保留的 tracker 不是“直接把最终杯位概率交给动作头”的诊断版本，而是让 action expert 读取 compact raw memory：

```text
固定历史 RGB：frames 0..59，stride=1
    │
    ├─ frame 0 完整 SigLIP 编码
    │      └─ initial cup visual decoder
    │
    ├─ swap clips：20..29、30..39、40..49
    │      └─ SigLIP patch embedding
    │      └─ 固定 2×2 pooling：16×16 → 8×8，K=64/frame
    │      └─ width=256、depth=2 factorized temporal/spatial Transformer
    │      └─ 每次交换的 relation 表示
    │
    └─ initial state + 三次 relation
           └─ shared recurrent compact updater
           └─ raw memory：[B, 128, 64]
```

动作接口为：

```text
raw memory [B,128,64]
    └─ 16 个 learned queries，width=256
    └─ 投影到 Pi0.5 action width
    └─ action suffix tokens 对 memory 做 gated cross-attention

动态当前 RGB + 10D EEF state + phase prompt
    └─ Pi0.5 flow action expert
    └─ 16×7 absolute EEF controller action chunk
```

最终 action 模型没有接收 `final_slot` 标签、杯位 one-hot 或 simulator cup pose。运行时用于动作生成的是 raw memory token、当前视觉、机器人状态和文本 prompt。

### 2.2 输入输出契约

- 历史：固定原始帧 0～59，不随闭环向前滑动；
- 当前帧：第61个视觉位置，每次 replan 用最新 observation 更新；
- 历史 stride：1；
- action horizon：16；
- observation state：10维，`xyz(3) + rot6d(6) + gripper(1)`；
- action：7维，`world xyz(3) + world rotation-vector(3) + gripper command(1)`；
- action tensor 仍 pad 到32维，但 loss 只作用于前7维；
- 控制器：Robosuite OSC，`osc_input_type=absolute`；
- 最终执行：每次预测16步，执行前8步后重新规划，即 `replan_steps=8`。

## 3. 模型迭代路径

### 3.1 第一阶段：原始 MEM + absolute joint

最初模型为：

```text
pi0_mem_compress_evan_shellgame_openpi_joint_260727
```

训练随后改成30帧、stride=2，得到 `my_experiment_30f_s2_6gpu/23000`。训练 loss 和 eval loss 持续下降，但闭环选杯低于随机水平，因此首先排查了 action/control 语义。

关键诊断：

| 实验 | 结果 | 结论 |
|---|---:|---|
| Oracle joint replay，左杯 | 30/30 | JOINT_POSITION 可以执行数据轨迹 |
| Oracle joint replay，中杯 | 30/30 | 同上 |
| Oracle joint replay，右杯 | 30/30 | 同上 |
| 模型 joint 做 FK，正常历史 | 约32% episode majority | 预测目标约为随机 |
| 关闭 memory / 打乱 / 错误历史 | 约30%～36% | memory 对最终动作杯位贡献很弱 |
| 旧 joint full-action + temporal arm/newest gripper | 45/100 | 选杯100%，但 joint 抓取精度有限 |

Oracle replay 证明：即使原始 joint 数据保存的是 OSC 轨迹执行后的 measured `q`，它仍能被当前 JOINT_POSITION 控制器可靠重放。因此 joint command/measured-q 的语义差异值得注意，但不是这次低于33%的主因。

### 3.2 第二阶段：把 memory 与 action 分开验证

为了避免 action loss 掩盖 memory 是否真的学到语义，建立了一系列分类 probe。

| Probe | Held-out 结果 | 说明 |
|---|---:|---|
| frame 0 → initial ball cup | 100% | 球的初始位置在视觉上非常清楚 |
| 冻结旧 `history_mem` → 一次交换后杯位 | 29%～35% | 旧 memory 不可可靠解码 |
| 全量 30×256 patches → 时空 Transformer | 100% | 原始视觉特征包含足够交换信息 |
| 每帧 learned-query 压缩 K=16/32/64 | 约随机 | 不是 token 数量不足 |
| 固定网格 pooling K=64 | 100% | 稳定空间 identity 是关键 |
| K64 → M128，随机端到端训练 | 30.5% | 随机 memory bottleneck 优化失败 |
| 先训练 temporal，再冻结并训练 M128 | 100% | M128 容量足够 |
| 低 LR 联合微调 | 100% | 分阶段初始化后可稳定联合训练 |

根因由此定位为：

- 逐帧独立 learned-query 压缩并不保证 slot `k` 在不同帧表示同一个空间位置或物体；
- 交换信息依赖细粒度跨帧对应，先做无拓扑约束的空间压缩会让后续 temporal Transformer 很难恢复运动关系；
- 随机初始化的最终 memory bottleneck 会进一步阻断早期梯度，造成 token collapse 或长期停留在随机精度。

固定网格版本相对原始 history encoder 的实测效率：

| 指标（batch=6） | 原始 | 固定网格版 | 改善 |
|---|---:|---:|---:|
| 参数量 | 16.87M | 3.53M | 4.77× 减少 |
| attention score elements | 188.7M | 29.1M | 6.48× 减少 |
| 平均延迟 | 3.25 ms | 1.85 ms | 1.76× 加速 |
| 峰值显存 | 1.81 GB | 0.296 GB | 6.12× 减少 |

### 3.3 第三阶段：从一次交换扩展到三次交换

直接把一次交换模型扩展到三次交换并不成功：

- 30帧 stride=2 和60帧 stride=1 都接近随机，说明 stride 不是主因；
- naive non-causal full-history 版本最高只有40.5%；
- memory token variance 逐步塌缩；
- 单纯增加 memory token、换 endpoint projection 或延长短训练，不能稳定解决问题。

进一步控制实验得到两个重要结论：

1. 三段 swap clip 分别分类交换杯对，可以在 held-out 600个 episode 上达到100%；说明 K64 压缩没有删除运动证据。
2. 把视觉 relation 接到 compact recurrent memory，并从 frame 0 视觉预测初始杯位后，完整三次交换的 held-out joint tracking 达到100%。打乱 frame 0 或打乱 swap clips 后精度回到随机基线，证明结果确实依赖 episode-specific 历史。

因此三次交换失败的主要问题不是 memory token 容量，而是连续视觉 embedding 到 recurrent state update 之间的语义接口没有对齐，以及训练预算/初始化不足。

### 3.4 第四阶段：memory 接入 Pi0 action expert

早期使用过一个小 deterministic head，把最终杯位概率映射到 joint trajectory。它只用于证明“正确语义状态能够映射到杯子动作”，不是最终架构。

最终采用更通用的接口：

- tracker 输出 raw compact memory `[128,64]`；
- 16个 learned query 做 resample；
- Pi0.5 action suffix 通过独立 cross-attention 读取 memory；
- 不把 memory 拼成语言/视觉 prefix，也不把杯位标签传入 action expert；
- 后续 full-action 训练冻结 tracker、memory 和 memory-to-action 已验证接口，主要优化 Pi0.5 action expert。

Memory ablation 和闭环选杯结果确认：加入正常 memory 才能稳定选择正确杯子；但一旦杯子选对，joint 控制仍受到关节误差放大的影响。最终决定保留 memory，切换到 absolute EEF action。

### 3.5 第五阶段：迁移到 absolute EEF

切换动作表示时没有重训 memory。训练直接从已经验证的 tracker/memory checkpoint 初始化：

- 固定 frames 0～59 做记忆；
- 最新当前帧只进入 current-image/action 分支；
- tracker、compact memory、query resampler 和 memory cross-attention 冻结；
- 只学习 absolute EEF7 action expert。

仅使用5000条 nominal absolute-EEF 示范时：

- step1999：8/20；
- step4000：8/20；
- step5999：8/20；
- 选杯始终20/20。

这说明 memory 迁移成功，但 nominal teacher-forcing 数据不足以覆盖闭环中的横向偏差和接触前纠错状态。

## 4. Nominal absolute-EEF 数据准备

### 4.1 原始数据生成

原始数据由并行 renderer 生成：

```text
render_shellgame_phase_instruction_dataset_parallel.py
```

最终 nominal raw 数据：

```text
/data2/hzl_workspace_for_pi_mem/robosuite/outputs/
shellgame_absolute_eef_phase_instruction_dataset
```

主要生成契约：

- 5000 episodes；
- 10个并行 worker；
- 224×224，10 FPS；
- 每个 episode 155帧；
- 固定三次交换；
- `osc_input_type=absolute`；
- `action_representation=controller`；
- 不把杯子强制 attach 到夹爪；
- 帧0～59为 reveal/cover/三次交换/settle；帧60以后为机械臂动作。

并行脚本将每个 worker 写入独立 shard，成功后统一合并 episode、重写绝对路径、合并 manifest/labels，并执行原生数据验证。worker 使用互相隔离的随机数流，episode ID 不依赖 worker 数量。

### 4.2 LeRobot 转换

转换器：

```text
robosuite/scripts/convert_shellgame_to_lerobot_raw_action.py
```

输出：

```text
/data2/hzl_workspace_for_pi_mem/robosuite/outputs/
shellgame_lerobot_absolute_eef_raw7
```

最终转换审计：

- 5000 episodes，775000帧；
- 左/中/右最终标签为1645/1706/1649；
- 5000/5000原始示范成功；
- action horizon=16，action dim=7；
- wrist RGB + third-person RGB；
- position observation 使用 world absolute frame；
- phase prompt 保留。

转换中修复了三个关键问题。

1. **因果对齐**：renderer 的 row `i` 是执行同一行 action 后的 observation，因此训练必须使用 `observation[i] -> controller_action[i+1]`。
2. **末端 padding**：absolute action 的无动作语义是重复最后的 world pose，同时保持最终 gripper intent，而不是填零。
3. **rotation-vector ±π 分支**：对接近 π 的等价旋转向量统一到 dominant-axis-positive 分支，避免物理连续姿态在 action 数值上跳变约 `2π`。

## 5. On-policy / correction 数据演化

### 5.1 所有有效纠错数据共同遵守的契约

后期所有纠错集都遵守以下原则：

1. 可以使用模型 rollout 或未记录的扰动命令，把系统带到偏心状态；
2. 模型生成的错误动作、扰动动作和中间帧不作为训练监督；
3. 从扰动后/模型真实到达的 observation 开始保存；
4. 第一对监督固定为 `observation[60] -> oracle_action[61]`；
5. `action_mask=True` 的数据只能来自 Oracle；
6. 每行保存连续、完整的 horizon=16 Oracle action chunk；
7. Oracle 在下降阶段持续把 XY 指向目标杯中心，只有 XY 对齐并到达正确高度后才闭爪；
8. 生成后必须审计 episode 数量、frame/action 对齐、mask、prompt、offset、height、slot quota、gripper 连续性及 terminal padding。

这避免了两类早期错误：把模型的偏心动作学回去，以及只修正 chunk 前几步、后半段又接回会横漂的旧轨迹。

### 5.2 关键数据集

| 数据集 | 数量 | 主要设计 | 后续用途 |
|---|---:|---|---|
| nominal absolute EEF | 5000 | 完整标准示范 | 所有 EEF 训练的行为基础 |
| correction V1 | 500 | 模型偏心后 Oracle suffix | 最早50/50纠错试验 |
| correction V2 | 500 | replan-aware switch state | 15%保守纠错混合 |
| multi-height hold-Z V3 | 600 | 多高度、先横向回中并保持Z | V4训练 |
| continuous descent V4 | 1200 | 完整持续居中下降 | V5训练 |
| low-stage gated V6 | 1200 | 高/中/低下降状态，XY未对齐时禁止继续Z/闭爪 | V6/V7训练 |
| safe balanced recovery V9 | 1200 | 安全 pre-contact 状态、16方向、真实误差分层 | V9/V10训练 |
| V10 real on-policy | 150 | V10真实闭环到达状态后接 Oracle | DAgger式诊断微调 |
| V10 exact failure suffix | 6 | episode 0/1/17精确前缀重复 | 过拟合/可学习性诊断，不进入最终通用模型 |

### 5.3 连续下降 V4 数据的平衡

1200个 episode 按最终**空间杯位**平衡，而不是按初始球身份平衡：

- left/middle/right：400/400/400；
- XY offset：small/medium/large = 300/540/360；
- 对应约5～12 mm / 12～22 mm / 22～35 mm，比例25%/45%/30%；
- 8个方向，每个150；
- 多个进入点覆盖高位、下降中段和接近杯口的晚期状态；
- 每个 episode 从 frame60 起只保存 Oracle 监督。

### 5.4 Low-stage gated V6 数据

V6 继续保持1200个 episode 和每个空间杯位400条，并强化低位视觉纠偏：

- anchor stage：high/mid/late = 120/240/840；
- offset bin：360/540/300；
- 8个方向各150；
- live XY error 未达阈值时，Oracle 保持夹爪打开并继续回中；
- 只有对准后才下降到闭爪阶段。

训练 sampler 不再用固定 frame range 猜 phase，而是按 causal shift 后的 `phase_id[i+1]` 分类完整 horizon=16 chunk。

### 5.5 Safe balanced V9 数据

V9 是数据质量审计最严格的一版：

- 1200个唯一 episode seed；
- left/middle/right 各400；
- high/mid/late = 180/420/600；
- small/medium/large = 300/540/360；
- 16个 offset sector，每个75；每个杯位、每个sector各25；
- measured initial XY error 范围约5.0～36.9 mm；
- 3920个唯一的 `>5 mm` hard recovery row；
- 使用 raise → lateral at clearance → descend 的隐藏安全路径；
- 隐藏动作和隐藏帧全部不进入训练；
- 没有人工 forced-open delay；
- 112800个连续 action window 全部通过转换审计。

生成器采用 fixed design slot + retry-until-filled，避免“请求1200条，但实际类别/方向缺失”。转换采用 episode-level 多进程、resume、并行 audit，解决了早期单线程生成和转换过慢的问题。

## 6. EEF 训练版本与结果

下表是历史迭代记录。由于早期使用过 `replan=3`、后期多为5或8，并且隔离 MuJoCo 评测是在中途才完善，表中不同阶段的成功率不能全部当作严格同协议模型排名。

| 版本 | 数据/采样核心 | 代表结果 | 主要结论 |
|---|---|---:|---|
| Nominal | 5000标准示范，五阶段平衡 | 8/20 | memory正确，但闭环纠错弱 |
| Mixed V1 | nominal/correction=50/50 | 3/20 | 纠错比例过大，正常行为被破坏 |
| V2 | 85/15，保持自然时间比例 | 8/20@2000，6/20@2999 | 更保守，但更多步不一定更好 |
| V3 | 保持15%，强化switch observation | 49/100，replan3 | 纠错入口采样有效，但仍不够稳定 |
| V4 | 75/25，多高度 hold-Z | 13/20 unguarded；14/20带guard | 运行时oracle guard略有帮助但不通用，最终不采用 |
| V5 | 60/30/5/5，连续下降1200条 | 12/20 | 持续下降数据有效，但右杯/低位漂移仍存在 |
| V6 | 60/30/5/3/2，动态 phase-aware | 14/20；正式67/100 | 本阶段最稳定基线 |
| V7 | 55% nominal + 45%按真实XY误差采样 | 10/20、9/20 | 强化困难状态反而造成行为覆盖失衡 |
| V8 optimized | 每episode 16个唯一持续恢复row，不复制 | 11/20，继续后8/20 | 数据更“干净”不等于闭环更好；继续训练发生退化 |
| V9 | nominal/V6/V9=60/15/25 | 9/20→8/20；replan8为12/20 | 出现 chunk 时序与闭环状态分布错配 |
| V10 step499 | nominal/V6/V9=60/30/10，强化时序 | 12/20@replan5；14/20@replan8 | 行为保留和chunk前部时序比更多纠错更重要 |
| V10 step1000 | 继承完整train state，以3e-7续训 | 15/20筛选；正式73/100 | 当前推荐 checkpoint |
| V10 step1500/1999 | 同一recipe继续训练 | 14/20、13/20 | 后续训练非单调，最终不选最新step |

### 6.1 为什么 V7～V9 数据更复杂，效果反而下降

离线 action loss 和闭环成功率出现了明显背离。V9 validation loss 从约0.025降到0.014，但闭环从9/20降到8/20，重复运行甚至6/20。

原因主要是：

1. hard/low-error group 从少量原始 row 中反复 oversample，数据表面比例正确，但真实状态多样性不足；
2. 大量 recovery chunk 在前5步保持开爪，真正的下降、闭爪命令位于第6～12步；
3. 部署时 `replan=5` 会在动作进入关键时序前丢弃 chunk，再从中间状态重新预测；
4. 每次重新规划可能再次进入“保持开爪、继续恢复”的模式，造成迟迟不下降或不闭爪；
5. 数据主要由 V6 + 隐藏扰动 + Oracle suffix 产生，不完全覆盖训练后 V9 自己 rollout 形成的状态。

把 V9 的 replan 从5改到8后，20-episode 成功率从6～8提升到12，直接支持了 chunk/deployment mismatch 的判断。

### 6.2 V10 为什么恢复

V10 不再继续提高纠错数据占比，而是回到 V6-5999 初始化：

- 60% nominal；
- 30% V6 behavior-preservation replay；
- 10% V9 timing replay。

其中10%的 timing replay 显式覆盖：

- 2% hard recovery；
- 2% low-error recovery；
- 1% aligned continuation；
- 2% chunk前3步即明显下降；
- 2% chunk前3步内出现闭爪；
- 1% early lift。

原始 V10 训练500步，peak LR=`3e-6`，末端 LR=`3e-7`，batch=12，6 GPU，tracker/memory冻结。随后从完整 optimizer/train state 继续到2000步，不重启 warmup；step500之后一直使用终端 LR `3e-7`。

20-episode sweep 表明最优点在 step1000，继续到1500和1999会回落。因此最终 checkpoint 按闭环评测选择，而不是选择最后一步或最低 validation loss。

### 6.3 V10 on-policy 与小样本过拟合诊断

另外生成了150条 V10 真实闭环 switch-state + Oracle suffix 数据，左/中/右各50。它严格不保存模型错误动作，也没有隐藏扰动。小步 fine-tune 约为12/20，没有超过通用 V10 continuation。

对 episode 0、1、17 的6条 exact-state failure suffix 做过拟合时得到：

- uniform suffix 模型能完成下降/抓取，但 hard recenter 弱；
- 50/30/20 phase-weighted 模型能把 episode 1 的 XY error 降到2.2 mm，但会悬停、不进入下降；
- 两个模型外部 handoff 后可3/3成功；
- 单模型低 LR consolidation 仍不能稳定合并两种行为。

该实验证明 action 链路和数据可学习，但单一 action expert 在相近视觉/状态下存在“继续回中”与“进入下降/闭爪”的模式干扰。由于显式 final-slot/phase gating 不够通用，最终模型没有加入该场景专用 trick，而是选择通用的 V10 行为保留与时序 recipe。

## 7. 控制与评测链路的修复

### 7.1 固定历史语义

曾经使用 generic `main.py` 的滑动61帧窗口进行过测试，这会在闭环过程中逐步丢弃 frames 0～59，导致 tracker 的位置语义被破坏。该批结果被标记为 invalid。

最终统一为：

```text
history = raw frames 0..59，永远固定
current = 最新闭环 observation
model input = 60帧固定历史 + 1帧动态当前帧
```

### 7.2 雪花视频问题

长时间复用 MuJoCo/EGL/OpenGL readback 时出现过后半段雪花或条纹。它是 renderer/context 生命周期问题，不是模型视觉输出。

最终正式评测采用：

- 每个 episode 新建独立 MuJoCo/EGL 子进程；
- WebSocket 每4个 episode 主动重连；
- physics trace 保留最后30步；
- 所有正式视频完成后用 `ffprobe` 检查。

V10 step1000 的100个正式视频全部可读，没有雪花或截断。

### 7.3 控制变量实验形成的结论

- 多候选采样 + chunk continuity：可以降低单次 diffusion 随机性，但计算成本高，收益不稳定；
- temporal action ensemble：arm 平滑，但会平均掉关键下降/闭爪时序；
- arm ensemble + newest gripper：比全量 ensemble 更合理，但没有解决主要XY偏差；
- runtime XY-before-Z guard：使用 simulator cup pose，可提高个别版本，但不通用，因此没有作为最终方案；
- V6 持续 oracle XY、其余动作保持模型原值：31个困难episode从13/31提升到30/31，证明持续XY纠偏是主要剩余瓶颈；
- V10 step80 后切 Oracle：10/10成功，证明同一闭环状态、OSC控制器和正确杯位都可恢复，剩余问题属于 learned action suffix。

## 8. 最终模型与正式结果

### 8.1 推荐 checkpoint

```text
/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/
pi0_shellgame_old_tracker_full_absolute_eef7_mixed_correction_v10_timing_diag_260820/
absolute_eef7_v10_timing_diag_nom60_v6preserve30_v9timing10_b12_500steps_6gpu_260820/
1000
```

### 8.2 最终评测配置

- 100 isolated episodes；
- environment seed=`260813`；
- 3 swaps；
- fixed 61-frame input，history 0～59，stride=1；
- absolute EEF7；
- `replan_steps=8`；
- `max_policy_steps=150`；
- no XY-before-Z oracle guard；
- diffusion sampling steps=4；
- observe prompt：`Observe the ball moving under a cup and remember which cup contains it.`；
- grasp prompt：`The shell game has ended. Grasp and lift the cup containing the ball.`。

### 8.3 结果

| 模型/部署设置 | 总成功率 | 左杯 | 中杯 | 右杯 | 选杯正确率 |
|---|---:|---:|---:|---:|---:|
| V6 step5999，既有正式基线 | 67/100 | 23/37 | 30/39 | 14/24 | 100/100 |
| V10 step1000，replan8 | **73/100** | **25/37** | 30/39 | **18/24** | 100/100 |

同一组100个 environment seed 的配对结果：

- 两者都成功：60；
- 仅 V10 成功：13；
- 仅 V6 成功：7；
- 两者都失败：20。

V10 的 Wilson 95% 区间为63.6%～80.7%，配对精确检验 `p=0.263`。因此当前可以说 V10 step1000 是本阶段推荐 checkpoint，并显示出约6个百分点的系统级改善趋势；不能说它已经统计显著地优于 V6。

还需注意：V10 正式运行使用 replan8，既有 V6 67/100 基线使用当时的正式 unguarded 协议；二者环境 seed 相同，但不是严格的 model-only 同控制参数100-episode对照。20-episode 的 replan8同协议筛选中，V6为14/20、V10 step1000为15/20。由于本阶段已经结束，不再追加评测，但后续若需要论文级结论，应补做 V6 replan8 的独立100-episode复测。

## 9. 本阶段最重要的方法论经验

1. **先证明表示，再训练动作。** 分类 probe、shuffle/zero/wrong-history ablation 比单看 action loss 更快定位 memory 是否真的包含任务语义。
2. **固定拓扑比 learned slot 数量更重要。** K64固定网格成功，而K64 learned-query失败，说明跨帧 correspondence 是视频记忆的核心。
3. **memory capacity 与 memory optimization 是两回事。** M128随机端到端失败，但分阶段训练100%，不能把训练失败直接解释为token不足。
4. **正确选杯不等于成功抓取。** 后期100% selection 与60%～73% lift长期并存，必须分别报告 memory metric 和 action metric。
5. **闭环指标优先于离线 loss。** 更低 validation loss、更多step、更多困难样本都可能使闭环退化。
6. **纠错数据必须连续且只监督 Oracle。** 模型错误动作可以用于到达状态，不能作为标签；horizon中不能前半段Oracle、后半段重新接错误轨迹。
7. **数据比例必须保护已有技能。** V10 的关键不是新增更多数据，而是30%重放V6行为，只用10%修正时序。
8. **采样必须匹配 deployment chunk。** 如果只执行前5或8步，训练监督的关键下降/闭爪动作也必须在chunk前部有足够覆盖。
9. **absolute EEF 更适合视觉抓取纠偏。** 模型直接预测末端目标，局部误差更容易解释和修复；absolute joint 的小关节误差可能在FK后放大。
10. **评测基础设施也是实验变量。** 滑动历史、EGL雪花、WebSocket长连接和随机diffusion都曾造成误判，最终必须用固定输入契约、隔离进程和视频/trace审计。

## 10. 关键代码与实验材料

### Memory/模型

- `src/openpi/models/siglip_mem_fixed_grid_temporal.py`
- `src/openpi/models/pi0_mem_fixed_grid_temporal.py`
- `examples/shellgame/train_three_swap_pair_fixed_grid_probe.py`
- `examples/shellgame/eval_three_swap_fully_visual_relation_memory_probe.py`
- `examples/shellgame/train_three_swap_query_crossattn_pi_joint_action_probe.py`
- `examples/shellgame/train_old_tracker_full_absolute_eef.py`
- `examples/shellgame/serve_old_tracker_full_absolute_eef.py`

### 数据生成与转换

- `robosuite/scripts/render_shellgame_phase_instruction_dataset_parallel.py`
- `robosuite/scripts/convert_shellgame_to_lerobot_raw_action.py`
- `examples/shellgame/generate_onpolicy_eef_continuous_descent_dataset_v4_parallel.py`
- `examples/shellgame/generate_onpolicy_eef_low_stage_gated_dataset_v6_parallel.py`
- `examples/shellgame/generate_onpolicy_eef_safe_balanced_recovery_dataset_v9_parallel.py`
- `examples/shellgame/audit_onpolicy_eef_safe_balanced_recovery_dataset_v9.py`
- `examples/shellgame/convert_shellgame_onpolicy_safe_balanced_recovery_v9_to_lerobot_raw_action.py`

### 最终训练与评测

- `examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v10_timing_diag.py`
- `examples/shellgame/train_old_tracker_full_absolute_eef_mixed_correction_v10_continue.py`
- `examples/shellgame/eval_absolute_eef_fixed_history_xy_before_z_isolated.py`
- `examples/shellgame/run_eval_eef_v10_step1000_replan8_100ep_260821.sh`

### 已有详细报告

- `evaluation/shellgame/eef_regression_diagnosis_260820.md`
- `evaluation/shellgame/v10_failure_suffix_overfit_probe_260820_analysis.md`
- `evaluation/shellgame/v10_continue_replan8_sweep20_260821_analysis.md`
- `evaluation/shellgame/v10_step1000_replan8_formal100_260821_analysis.md`

### 关键消融原始结果

- `evaluation/shellgame/oracle_joint_replay/oracle_by_slot_30each_seed260811.json`
- `evaluation/shellgame/joint_fk_memory_ablation/15000_30f_s2_val50_s3_seed260806.json`
- `evaluation/shellgame/joint_fk_memory_ablation/23000_30f_s2_val50_s3_seed260806.json`
- `evaluation/shellgame/memory_linear_probe/23000_30f_s2_train594_val297_seed260807.json`
- `evaluation/shellgame/history_difference/history_0_59_metrics.json`
- `evaluation/shellgame/one_swap_history_probe/`
- `evaluation/shellgame/fixed_grid_temporal_optimized/`
- `evaluation/shellgame/old_tracker_query_action_closed_loop_gate_60ep_260810/old_tracker_query_action_regression.json`
- `evaluation/shellgame/oracle_joint_noise_sensitivity/formal_30ep_260811/result.json`
- `evaluation/shellgame/model_prefix_oracle_suffix/formal_val30_step5999_seed260811.json`
- `evaluation/shellgame/full_joint_grasp_checkpoint_fk/`
- `evaluation/shellgame/eef7_gt_grasp_xy_noise_extended20_260813/results.json`
- `evaluation/shellgame/eef7_mixed_v4_1999_xy_residual_stage_commonfail6_260815/analysis.md`
- `evaluation/shellgame/eef_disturbance_recovery_v4_1999_ep14_seed260813_salt260816/result.json`
- `evaluation/shellgame/eef_disturbance_recovery_v5_2999_ep14_seed260813_salt260816/result.json`
- `evaluation/shellgame/eef7_v6_step5999_right31_oracle_xy_paired_seed260813_salt260816/result.json`
- `evaluation/shellgame/eef7_v10_ft499_oracle_step80_paired10_seed260813_260820/analysis.md`

## 11. 完整实验过程与消融记录

本节补全前文为了突出主线而省略的中间实验。记录原则如下：

- 同一假设下的 smoke test、正式测试和重复测试合并为一项实验族；
- 表中优先报告 held-out、同 seed 或正式闭环结果；
- `invalid` 目录、雪花视频、滑动历史、并行首次编译超时等不作为模型证据；
- 不同 `replan_steps`、随机扩散采样和评测实现下的成功率不直接横向排序；
- “100%”若来自训练集过拟合，会明确标成 capacity control，不当作泛化结果。

### 11.1 输入窗口、prompt 与数据可见性审计

| 实验/检查 | 设置与结果 | 排除或确认的假设 |
|---|---|---|
| 训练 prompt 对齐复测 | 将早期测试的 observe/grasp 指令改回训练集原始 prompt；joint 选杯没有出现实质恢复 | prompt mismatch 不是低于随机的主因 |
| 32帧、stride=5 窗口审计 | 在动作刚开始时，因为历史不足，frame 0 被前置 padding 重复约20次；reveal/cover 又占据早期位置，真正覆盖 swap 的独立帧仅约9帧 | 早期在线窗口的有效运动证据远少于“32帧”表面数量 |
| 12帧、stride=5 讨论 | 时间跨度仍有限且采样更稀，不足以稳定保留三次交换细节，因此没有作为正式训练方案 | 单纯减少 frame 数不能解决 padding 和稀疏运动问题 |
| 30帧、stride=2 | 用于重新训练 joint 模型与多项 probe；覆盖0～59但每次交换只有5帧 | 比32×5合理，但后续证明 stride=2 不是 memory collapse 的唯一原因 |
| 60帧、stride=1 | 三次交换视觉 probe 中仍曾在 continuous-to-memory 接口上保持随机；改成显式 relation 接口后达到100% | 更多帧本身不够，语义接口和优化仍是决定因素 |
| 历史像素差异 | base camera 相邻帧 MAE 均值约0.00173，改变超过10灰度的像素约1.09%；wrist 对应约0.0165和5.36%；峰值变化集中在20/29/30/39/40/49交换边界 | 交换证据存在但稀疏，不能依赖全局图像差异直接分类 |
| frame 0 球分类 | held-out 600条，step50起即100% | reveal 阶段球大小/清晰度已经足够；本阶段没有必要再增大球 |
| 固定历史 vs 滑动历史 | 最终改为 history 永远是frames 0～59，current 单独更新；generic `main.py` 的滑动61帧结果标记 invalid | 后半段不能把原始交换历史逐步推出窗口 |

球半径没有做独立数值 sweep。之所以停止沿“增大球”方向继续，是因为 frame 0 初始杯位已经100%，而真正失败发生在跨帧 relation 和后续 memory/action 接口。

### 11.2 原始 joint 模型：控制器、FK 与 memory 消融

#### 11.2.1 checkpoint、history 和 diffusion sampling

| checkpoint/条件 | normal | memory off | shuffle history | wrong episode history | reveal only |
|---|---:|---:|---:|---:|---:|
| step15000，3次采样，episode-majority | 38% | 34% | 36% | 36% | 36% |
| step23000，3次采样，episode-majority | 32% | 30% | 32% | 32% | 32% |
| step23000，4次采样，episode-majority | 34% | 36% | 36% | 34% | 34% |

step23000 的 wrong-history 输出跟随 donor 最终杯位的比例仅28%（4次采样版本22%）。因此更长训练、更多 diffusion samples 和正常历史都没有建立可靠的“历史决定最终空间杯位”关系。训练 eval loss 下降只能说明 action reconstruction 更好，不能证明 MEM 学到了杯子身份。

#### 11.2.2 旧 memory 的线性可解码性

| checkpoint | 表示 | final spatial slot CV | target identity CV |
|---|---|---:|---:|
| 15000 | `history_mem` | 36.4% | 58.8% |
| 15000 | fused prefix | 40.4% | 46.8% |
| 23000 | `history_mem` | 32.0% | 77.1% |
| 23000 | fused prefix | 37.7% | 52.9% |

旧 memory 越训练越能保留“球最初属于哪个身份”，却没有形成“交换后位于哪个空间槽位”的可泛化表示。这解释了 action loss 下降但最终选杯仍为随机的现象。

#### 11.2.3 Oracle joint replay 与 joint 误差敏感性

| 实验 | 结果 | 结论 |
|---|---:|---|
| 数据集 measured joint trajectory 直接送 JOINT_POSITION | 左/中/右各30/30，总计90/90 | 控制器能执行真实轨迹；measured-q 语义不是主要失败原因 |
| joint bias 0° RMS | 30/30，FK XY偏差0 mm | Oracle基准 |
| joint bias 0.25° RMS | 30/30，FK XY偏差4.44 mm | 小误差仍可容忍 |
| joint bias 0.5° RMS | 22/30，FK XY偏差8.88 mm | 成功率已明显下降 |
| joint bias 1.0° RMS | 7/30，FK XY偏差17.76 mm | 绝对joint对小角误差高度敏感 |
| joint bias 2.0° RMS | 3/30，FK XY偏差35.55 mm | FK放大足以破坏抓取 |
| 仅把模型16帧 joint residual splice 到GT轨迹 | 30/30 | 模型局部误差若不持续累积，控制器仍能成功 |

这组实验同时支持两个不矛盾的结论：JOINT_POSITION 控制器本身能完成任务，但模型持续输出的微小绝对关节偏差会在 FK 后累积成厘米级 EEF 偏差。

### 11.3 一次交换：空间压缩和初始化消融

一次交换 probe 统一使用5400个训练 episode、600个 held-out episode，标签是第一次交换后的空间杯位。

| 编号 | 历史压缩/训练方式 | Held-out结果 | 实验结论 |
|---|---|---:|---|
| O1 | 冻结旧 `history_mem` + 小分类器 | 最终约29% | 旧 memory 不可可靠解码 |
| O2 | 解冻原始 HistoryResampler | 35.3% | 500步内仍在随机/多数类附近 |
| O3 | 全部30×256 SigLIP patches 后做时空 Transformer | 100% | 原始 patch 含有完整交换证据 |
| O4 | 每帧 learned-query K=16，再做时空 Transformer | 30.5% | 先压缩破坏跨帧 correspondence |
| O5 | learned-query K=32 | 30.5% | 不是K太小 |
| O6 | learned-query K=64 | 30.5% | token数增至64仍无效 |
| O7 | 固定2×2 pooling，K=64 | 100% | 固定空间拓扑是关键变量 |
| O8 | K64 temporal → 随机M128，直接端到端 | 30.5% | 随机 bottleneck 造成优化失败 |
| O9 | 先冻结 temporal、单独训练M128/readout | 100% | M128容量足够 |
| O10 | 再以低LR联合微调 | 100% | 好初始化后完整路径可稳定训练 |

还对“最初 flatten 全部 patches，再做 cross-attention”的原始想法进行了初始化对照：

- 随机初始化、300步直接 CE：68.7%；shuffle history 降至37.3%，reverse time 降至31.2%，说明该路径可以学习且确实依赖顺序；
- 用成功 fixed-grid teacher 做逐 token cosine/MSE distillation：teacher/student token cosine 达0.818，但下游仅30.5%～34.7%；
- 因此“任何预训练都会解决问题”没有得到支持，密集 token matching 反而损害语义学习；保留了“更合适的任务级初始化可能有效”这一可能性。

### 11.4 三次交换：逐层定位 memory 失败位置

#### 11.4.1 从一次交换直接扩展

| 实验 | 结果 | 定位结论 |
|---|---:|---|
| 冻结一次交换 encoder，只训 final-slot readout，30帧stride2 | 37.5% | 一次交换表征不能直接解码三次结果 |
| 全60帧、action loss=0、训练 history+readout | 最好40.5% | action loss 不是三交换失败主因 |
| 再冻结 history，只训练27层Pi reader | 36.8% | reader under-training 不是主因 |
| 三个 swap endpoint 辅助分类，只训 readout | joint 13.0%；各阶段85.8/33.5/41.8% | 旧表示主要学会第1次交换 |
| endpoint 辅助监督联合训练 | joint 19.3%；99.8/44.3/38.3% | shallow one-shot aggregation仍未学到可重复状态转移 |
| naive non-causal full-history | 最高40.5% | 只能利用弱全局相关性，不能稳定组合三次交换 |

#### 11.4.2 视觉信息是否被K64删除

| 实验 | 结果 | 结论 |
|---|---:|---|
| 每个10帧 swap clip → 被交换杯对分类 | held-out 100% | K64保留了运动证据 |
| pretrained swap encoder + 原M128 recurrent updater | 约3%～5% joint | 仅有运动encoder仍不够，接口/状态更新失败 |
| Oracle初始杯位 + 真实视觉连续 embedding | held-out随机，memory variance塌缩约3个数量级 | continuous semantic code 未被 recurrent memory 自动对齐 |
| 54条固定样本 capacity control | step150达到100% | 架构能记忆样本，但当时不能泛化 |
| 同一结构 stride=1、60帧 | step400 joint 3.43%，与stride2相当 | stride不是主因 |

#### 11.4.3 Oracle relation、token数量和 updater 结构

| 消融 | 结果 | 排除/确认 |
|---|---:|---|
| 最小 width64 shared MLP transition | 100%，roll relation后0% | 数据、标签、三步递推目标正确 |
| dense relation tokens + cross-attention extractor + MLP | 100% | token展开和cross-attention本身无问题 |
| persistent memory M=1 | step75达到100% | 单token也足够存三态 |
| persistent memory M=128 | step125达到100% | 128 token不是根因，只是收敛更慢 |
| 删除 token-axis centering | 仍约3%～6% | centering不是根因 |
| 原始稀疏 initial-slot 注入，LR=1e-3 | step250达到100% | 稀疏注入显著拖慢优化但并非不可学 |
| 同上，LR=3e-4 | step300仍6.48%，step450达到100% | 早期“结构失败”很大部分是训练预算/初始化假象 |
| M128 width64 → 1152 → 64+tanh endpoint | 约3.55% | endpoint round-trip 是主要瓶颈 |
| 仅保留64→1152，直接1152 readout | step300为23.61% | expansion也拖慢，但64压缩+tanh更严重 |
| 256个current queries直接读M128，再pool | 约3%～5% | 双重attention稀释episode signal |
| 单一显式history query读M128，再投到1152 | step400达到100% | compact memory与Pi宽度接口兼容 |

#### 11.4.4 最终视觉 relation 接口

| 条件 | initial | 三段relation | 三阶段joint tracking |
|---|---:|---:|---:|
| 正常 held-out | 100% | 100% / 100% / 100% | 100% |
| 打乱frame 0 | 33.0% | 保持100% | 33.0% |
| 打乱swap clips | 正常 | 约37.0% / 31.6% / 33.6% | 4.17%（随机1/27为3.70%） |

这个结果是 memory 阶段真正的“可泛化通过门”：初始球视觉、每段交换视觉和 compact recurrent state 都对最终结果具有因果贡献。后续 action 训练才从此 checkpoint 开始。

### 11.5 Memory-to-action 接口与 joint action 消融

| 实验 | 结果 | 解释 |
|---|---:|---|
| 直接回归frames60～75 joint trajectory | 33.3%，全部预测中杯 | action MSE鼓励条件均值，不自动形成离散语义 credit assignment |
| Pi0 flow action expert 300步 | 29.2% | 同样没有正确读出旧memory |
| 仅训练 final-slot deterministic head | 诊断性改善，但不是通用action模型 | 只证明语义到动作可映射 |
| 随机/未验证 query-action memory-only，raw vs EMA | 两者均36.7%，仅1/60杯位不同 | EMA不是随机水平的原因 |
| 该失败模型关闭cross/MLP分支 | full/cross-only/mlp-only/none约36.7/38.3/36.7/36.7% | 分支会改变动作，但未获得正确语义 |
| 该模型线性 probe | raw memory对 final slot 34.7%，对target identity 98.6%；aggregated memory为33.3%/93.1%；action queries为33.3%/33.3% | 问题是把identity误当成final slot，并在action query处丢失条件 |
| 恢复已经100%的 old tracker + query-cross-attn action | offline 59→75 endpoint 98.3% | 正确tracker和action接口可以联合工作 |
| 同模型 shuffle memory | 43.3% | 大幅下降，动作确实使用memory |
| 同模型 zero memory | 35.0% | 回到随机/中杯偏置 |
| JOINT_POSITION执行首个16步chunk | 19/20；重新开server录像复测18/20 | memory能够驱动动作向正确杯位移动；该模型没有训练抓取段，所以lift为0是预期结果 |

这一步澄清了“memory影响action”的含义：不是把最终槽位概率直接传给动作头，而是 raw compact memory 经16个 query resample 后，通过 action suffix cross-attention 改变 Pi0 flow trajectory。

### 11.6 完整 joint 抓取与运行时控制消融

#### 11.6.1 训练步数与 teacher-forced FK

- step1000 → 1999：joint RMSE下降14.5%，grasp EEF RMSE下降13.7%；
- step1999 → 5999：joint RMSE再下降10.5%，grasp EEF RMSE再下降10.1%；
- step5999 在150个 held-out teacher states 上，grasp EEF RMSE约5.51 mm，末端杯中心XY误差约7.30 mm；
- 但闭环失败 rollout 的抓取误差可达约33 mm，说明继续拟合 expert states 不能消除 covariate shift。

#### 11.6.2 夹爪、chunk 与 ensemble

以下成功率只在各自标注协议内部解释：

| 条件 | Lift | Selection | 结论 |
|---|---:|---:|---|
| step1999，20ep | 3/20 | 20/20 | 完整joint抓取基线低 |
| gripper close latch | 3/20 | 20/20 | 开合次数12.55→1.0但成功集完全不变；开合振荡是失败后的结果，不是根因 |
| step5999原始20ep | 4/20 | 20/20 | 更多训练只小幅改善 |
| step5999隔离MuJoCo | 6/20 | 20/20 | 基础设施会带来一定波动 |
| 执行完整chunk，replan16 | 7/20 | 20/20 | 减少chunk截断有帮助 |
| 4候选 + continuity | 4/20 | 20/20 | 计算增加但收益不稳定 |
| 全维 temporal ensemble | 6/20 | 20/20 | 平滑同时稀释关键时序 |
| arm ensemble + newest gripper | 8/20；扩大后45/100 | 20/20；100/100 | joint版本最佳运行时组合，但仍受EEF精度限制 |

对25个右侧压力案例还比较了 newest-only、oldest-heavy、newest-heavy、replan16 和 oracle-FK candidate。前三者只有0～1/25，replan16为4/25，oracle-FK候选为7/25，均未把右侧失败解释成单一权重或diffusion候选选择问题。

#### 11.6.3 从哪个阶段替换成Oracle

| 实验 | 结果 | 说明 |
|---|---:|---|
| 全GT轨迹 | 30/30 | 系统上限正常 |
| model全程 | 11/30 | learned joint rollout误差持续累积 |
| model到frame75，之后GT | 30/30 | 早期选杯/approach可保留 |
| model到frame89，之后GT | 30/30 | 在下降前切换仍可完全恢复 |
| model到frame109，之后GT | 13/30 | 到抓取前已经积累不可由同episode GT suffix完全消除的状态偏差 |
| model到frame119，之后GT | 11/30 | 更晚切换几乎等于model全程 |
| 从人工GT cutoff110执行GT | 30/30 | 若状态本来在专家流形，后半段控制器稳定 |

这组控制变量促成了 absolute EEF 路径：保留正常 memory 选杯过程，只替换后半段，证明选择语义正确，而 joint 表示在下降/抓取阶段累积了显著笛卡尔误差。

### 11.7 Absolute EEF：表示选择与控制瓶颈消融

| 实验 | 结果 | 结论 |
|---|---:|---|
| nominal EEF checkpoint 1999/4000/5999 | 均8/20，selection均20/20 | 更多训练步数没有解决闭环分布外状态 |
| 正常模型approach，之后GT杯中心抓取 | 20/20 | memory选择、当前OSC和后半段专家都正常 |
| GT抓取XY径向噪声12/14/16/18/20 mm | 19/20、14/20、10/20、9/20、9/20 | 抓取对厘米级横向误差敏感 |
| model全程 vs model到75/89后Oracle | 14/30 vs 30/30 / 30/30 | 失败主要在持续下降/抓取suffix，而非初始选杯 |
| model到109后Oracle | 18/30 | 到该阶段已有部分不可恢复偏心 |
| correction V2离线前三步 | XY contract pass仍0%，Z slowdown略有学习 | 早期纠错数据只学到“慢一点”，没学到“持续回中” |
| V4六个共同失败episode，只纠正下降首步XY | 0/6 | 一次性对准无效 |
| 同六例，从下降开始持续Oracle XY到闭爪 | 6/6 | 持续视觉XY纠偏是决定变量 |
| 继续纠正到闭爪后10步 | 仍6/6 | 额外闭爪后纠正不是必要条件 |
| V6右杯31例：模型XY vs 持续Oracle XY，其余动作保持模型值 | 13/31 → 30/31 | 右杯弱点主要来自持续XY残差，不是memory或Z/rotation/gripper |

V4六个共同失败例中，下降开始时XY误差8.2～22.9 mm；原模型首个close时中位15.5 mm，持续Oracle XY后降至1.0 mm。它直接决定了后续纠错数据必须覆盖完整下降，而不是只在高位做一次 recenter。

### 11.8 纠错数据设计和版本消融的完整过程

| 版本 | 相对上一版改变 | 代表结果 | 实验得到的教训 |
|---|---|---:|---|
| Nominal | 5000条成功标准示范 | 8/20 | teacher-forced正常轨迹不含闭环偏心恢复 |
| V1 | 500条模型到达状态+Oracle suffix，50%混合 | 3/20 | 纠错占比过高，破坏nominal行为 |
| V2 | 纠错降至15%，修复因果shift和action mask | 8/20@2000，6/20@2999 | 配比更稳，但尚未学会XY-before-Z contract |
| V3 | replan-aware switch observation | 49/100，replan3 | 入口状态更真实有帮助，但协议较早，不能直接和V6比较 |
| multi-height/hold-Z | 600条，高/中/低位横向回中 | 为V4训练准备 | 单一高位扰动覆盖不足 |
| V4 train | nominal/纠错/hold-Z混合 | 13/20；runtime guard 14/20 | guard仅小幅有效且使用sim cup pose，不作为最终方案 |
| continuous descent V4 data | 1200条，完整Oracle下降 | 用于V5 | 修复chunk后半段重新横漂的问题 |
| V5 train | 60/30/5/5 | 12/20 | 扰动恢复增强，但自然闭环略退化 |
| low-stage gated V6 data | 1200条，低位未对准禁止继续下降/闭爪 | 用于V6/V7 | 需要在接近杯口时继续视觉纠偏 |
| V6 train | 动态phase-aware采样，行为保留 | 14/20；67/100 | 最稳定旧基线 |
| V7 error-aware | 按真实XY error强化45%困难状态 | 10/20→9/20 | hard状态过量会破坏整体行为覆盖 |
| V8 optimized | 每episode只取16条唯一连续recovery row，避免复制 | 11/20，续训后8/20 | 去重并不能补足on-policy状态多样性；更多step会退化 |
| V9 safe-balanced | 1200唯一seed、16方向、3高度、3误差档，严格审计 | 9/20→8/20；repeat 6/20 | 数据本身正确仍可能与部署chunk时序不匹配 |
| V10 | 从V6初始化，60% nominal + 30% V6 replay + 10% V9 timing | 12/20@r5，14/20@r8；续训step1000为15/20 | 行为保护和chunk前部时序比提高纠错占比更重要 |

独立的36条件扰动恢复测试揭示了“离线恢复能力”和“自然闭环成功率”可以相反：

- V4：总体回到阈值内15/36（41.7%），最终XY误差9.53 mm；
- V5：25/36（69.4%），最终XY误差7.42 mm；
- 但 late-stage 两者都只有4/12（33.3%），V5的20ep自然闭环仍从V4的13/20降到12/20；
- 在同7个右杯 deterministic rollout 中，V4为6/7、V5为5/7。

因此 V5 的数据确实教会了受控扰动恢复，却没有充分覆盖策略自身在低位下降形成的状态。这推动了 V6 的 low-stage gated 数据，而不是继续盲目增加V5训练步数。

### 11.9 Phase sampling、replan 和离线loss消融

| 模型/条件 | Replan | Lift | 关键现象 |
|---|---:|---:|---|
| V6-5999 fresh | 5 | 14/20 | 基础评测可复现，不是代码漂移 |
| V9-2000 | 5 | 9/20 | 低于V6 |
| V9-3999 | 5 | 8/20 | val loss更低但闭环更差 |
| V9-3999随机重复 | 5 | 6/20 | diffusion方差存在 |
| V9-3999 | 8 | 12/20 | 执行到chunk第6～8步后，下降/闭爪恢复 |
| V10-250 | 5 | 9/20 | 短训尚未恢复 |
| V10-499 | 5 | 12/20 | timing replay改善close时机 |
| V10-499 | 8 | 14/20 | 与训练chunk时序更一致 |

V9 held-out 初始状态上并不比V6差：前三步XY error约8.17/8.24/8.30 mm，对应V6为8.21/8.50/8.45 mm；但V9关键close多在chunk offset 6～12，部署只执行前5步会反复重启“保持开爪、继续恢复”。这证明动态 phase-aware sampler 仍可能制造 chunk/deployment mismatch。

### 11.10 V10 on-policy、Oracle handoff 与 exact-state 过拟合

#### 11.10.1 10 seeds × 4 conditions

固定episodes 0/1/2/3/4/7/9/12/16/17，replan8，在step80切换：

| 条件 | Selection | success@150 | success@155 |
|---|---:|---:|---:|
| V10全程 | 10/10 | 6/10 | 7/10 |
| on-policy FT499全程 | 10/10 | 4/10 | 4/10 |
| V10→FT499 | 10/10 | 6/10 | 7/10 |
| V10→Oracle | 10/10 | 0/10 | 10/10 |

Oracle轨迹设计为step154才越过80 mm成功阈值，所以它在150步为0/10不是失败。FT499没有恢复V10的0/1/17三个真失败，而Oracle从完全相同的step80状态达到10/10，说明 simulator、OSC、memory和起始状态都可恢复，learned suffix才是瓶颈。

150条真实V10 on-policy状态数据（每杯50）训练后的 step250/499 均为12/20，未超过通用 V10。这说明只增加一轮on-policy suffix不足以自动改善，数据内部的phase比例仍会造成模式干扰。

#### 11.10.2 6条 exact-state suffix 过拟合

| 训练/控制 | 结果 | 结论 |
|---|---:|---|
| uniform 95-row suffix，step100 | 2/3@150 | action链路能记住两类状态 |
| uniform step299 | 0/3@155，延长到180为2/3 | 主要退化为动作时序变慢 |
| 50/30/20 recenter/descent/grasp-lift，最好checkpoint | 0/3 | episode1可回中到2.2 mm，但模型悬停不下降 |
| weighted模型执行2个chunk，再切uniform模型 | 3/3 | 两个模型学到互补的recenter与progress技能 |
| 单模型30/30/20/20低LR consolidation | 最好2/3，其余0～1/3 | 简单replay不能稳定合并两个行为模式 |

过拟合实验不是最终方案，而是验证了：数据/action loss通路可学习，真正冲突是“继续回中”与“下降/闭爪/抬升”的条件和时序。显式 final-slot/phase gate 虽能针对本任务解决模式选择，但不够通用，因此最终没有加入；V10选择更通用的行为重放与timing replay。

### 11.11 V10 continuation 与最终 checkpoint 消融

所有模型在相同20个episode、replan8下：

| checkpoint | 总成功 | 左/中/右 | Selection |
|---|---:|---:|---:|
| V6-5999 | 14/20 | 4/7、5/6、5/7 | 20/20 |
| V10-499 | 14/20 | 3/7、5/6、6/7 | 20/20 |
| V10-1000 | **15/20** | 4/7、5/6、6/7 | 20/20 |
| V10-1500 | 14/20 | 3/7、6/6、5/7 | 20/20 |
| V10-1999 | 13/20 | 3/7、4/6、6/7 | 20/20 |

训练在step500之后保持终端LR `3e-7`，仍然呈现非单调闭环结果。这是最终选择step1000而不是step1999的直接依据。正式100ep结果为73/100；相同环境seed的既有V6结果为67/100，但前文已注明二者100ep控制协议并非完全相同。

### 11.12 评测基础设施和无效实验清单

以下运行对工程定位有价值，但不得进入模型成功率比较：

1. **滑动61帧历史。** 它逐渐丢弃0～59交换帧，破坏tracker语义；对应结果全部作废。
2. **EGL雪花/条纹/180° readback flip。** 长时复用context产生视觉损坏；修复前的 `_INVALID_visual_flip`、`invalid_snow` 等目录不计入。
3. **并行首次JAX编译 ping timeout。** V10 continuation 第一次并行20ep在episode 0前即超时，改为串行隔离进程后重跑。
4. **未刷新 handoff observation/context。** 对应 `invalid_no_context_refresh`、`invalid_refresh_order` 不计入。
5. **随机seed溢出。** `failed_uint32_seed` 运行在修复确定性salt后重跑，失败目录不计入。
6. **只有1～3个episode的 smoke。** 仅验证程序、renderer、视频或checkpoint加载，除非是明确的exact-state capacity control，不报告为泛化性能。
7. **视频文件名中的 failure。** 早期只训练approach chunk的模型没有lift监督，视频后缀会写failure；选杯指标需从单独的selection统计读取。

最终正式链路固定为每episode独立MuJoCo/EGL进程、每4个episode重连WebSocket、history 0～59固定、视频 `ffprobe` 审计，并分别报告 selection 与 lift。

### 11.13 尚未作为正式实验完成的项目

- 没有进行球半径的独立对照，因为frame0视觉分类已经100%；
- 没有把 task-specific `final_slot` 分类loss作为最终训练目标，它只用于诊断；
- 没有采用需要 simulator cup pose 的runtime XY guard/Oracle candidate作为最终系统；
- 没有完成 V6 与 V10 在完全相同 `replan=8` 下各100ep的新复测；现有严格同协议证据是20ep的14/20对15/20；
- 没有证明 V10 +6个百分点具有统计显著性，当前配对检验仍为 `p=0.263`。

## 12. 阶段收口

本轮工作已经完成从“absolute joint 模型低于随机且无法判断是 memory 还是 action”的模糊状态，到以下可验证结论的转变：

- history memory 能在 held-out episode 上正确追踪三次交换；
- action expert 确实读取了 compact raw memory；
- memory 可以跨 joint/EEF 任务迁移；
- absolute EEF 闭环失败集中在正确杯位上的连续对准和动作时序；
- 数据生成、转换、mask、prompt、phase、offset、视频与闭环评测均建立了可审计契约；
- 当前可用系统在100个 episode 上达到73%抓取成功和100%选杯正确。

因此本阶段建议冻结 V10 step1000、对应数据和评测协议作为下一阶段基线，不再继续使用 step1999，也不再把 validation loss 最低点当作默认最佳模型。

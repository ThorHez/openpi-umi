# Teacher Memory Ablation Results

## 实验目的

验证在 recurrent MEM 已使用相同 teacher-compatible updater/readout 初始化的条件下，训练期 teacher latent-memory supervision 是否仍然必要。实验仅改变 memory distillation loss 的权重，避免把模型结构、初始化或训练预算差异误认为 teacher 收益。

## 对照设置

两组均使用 ShellGame 三次交换任务、每个 event 12 帧输入、4500/500 个训练/验证 episode、1000 training steps、batch size 12、seed 42，以及相同的 2.424M 可训练 visual encoder 参数。recurrent updater 和 readout 在两组中均冻结。

- **GT state only**：仅使用三个 stage 的球位置交叉熵，`lambda_state=1`、`lambda_teacher=0`。
- **+ teacher memory**：使用相同 state loss，并加入 token-aligned teacher memory loss，`lambda_state=1`、`lambda_teacher=1`。

Student 的部署输入始终只有 previous memory 与当前 12-frame event clip；relation ID、Qwen 输出和 teacher memory 都不会进入 student 推理路径。

## 主要结果

| Method | Update 1 | Update 2 | Update 3 / Final | Mean stage | Stage CE | Memory cosine |
|---|---:|---:|---:|---:|---:|---:|
| GT state only | 65.00 | 36.25 | 33.75 | 45.00 | 0.9821 | 1.0287 |
| + teacher memory | **81.67** | **74.17** | **50.00** | **68.61** | **0.6822** | **0.4167** |
| Absolute change | +16.67 | +37.92 | +16.25 | +23.61 | -30.53% | -59.49% |

Accuracy values are percentages. Final evaluation uses matching validation batches for the two runs, comprising 240 held-out samples. Periodic curve points also use matching A/B validation batches at each checkpoint, although the subset advances between checkpoints.

Teacher supervision raises mean stage accuracy by **23.61 percentage points** and final three-update accuracy by **16.25 points**. The largest gain occurs after the second recurrent update, where accuracy rises from 36.25% to 74.17%. This indicates that teacher memory primarily improves the learning of compositional state transitions rather than merely the recognition of the first visual event.

![Teacher memory ablation](figures/teacher_memory_necessity_ablation_12f_260826.png)

## 论文正文可用表述

### 中文

为了隔离 teacher memory supervision 的贡献，我们在相同的 12 帧输入、模型初始化、数据划分与训练预算下进行了严格消融。仅使用离散状态交叉熵时，模型在验证集上的平均阶段准确率为 45.00%，三次递归更新后的最终准确率为 33.75%。加入 teacher latent-memory loss 后，两项指标分别提升至 68.61% 和 50.00%，对应 23.61 和 16.25 个百分点的绝对增益。第二次更新的提升最显著（36.25%→74.17%），说明连续 teacher state 为 recurrent updater 提供了离散 readout 标签无法充分表达的状态转移几何。Teacher 仅在训练阶段使用，不增加 student 推理开销。

### English

To isolate the contribution of teacher-memory supervision, we conduct a controlled ablation with identical 12-frame inputs, initialization, data split, and optimization budget. Training with discrete state cross-entropy alone yields 45.00% mean stage accuracy and 33.75% final accuracy after three recurrent updates. Adding the latent teacher-memory objective improves these metrics to 68.61% and 50.00%, corresponding to absolute gains of 23.61 and 16.25 percentage points. The largest improvement occurs at the second update (36.25% to 74.17%), indicating that the continuous teacher state provides transition geometry that is not sufficiently captured by sparse readout labels. The teacher is used only during training and introduces no student inference overhead.

## Figure caption

**Figure X: Effect of teacher-memory supervision on recurrent state learning.** Both variants use identical 12-frame event clips, teacher-compatible initialization, and state readout supervision; the only changed factor is the latent teacher-memory loss. (a) Held-out mean stage accuracy during training. (b) Final accuracy after each recurrent update and its mean. Teacher supervision improves all stages, with the largest gain at the second update. Dashed lines denote three-way chance performance. Results are from one seed with 240 held-out samples per evaluation.

## 结论边界

该实验支持 teacher memory 是当前 recipe 中重要的训练期 representation/optimization anchor，但不能证明 teacher 在数学意义上不可替代。结果来自 ShellGame 单个 seed，论文最终版本应补充至少 3 个 seed 的均值和标准差，并在 RoboMME 任务上验证跨任务一致性。

## 资产

- 矢量 PDF：`docs/figures/teacher_memory_necessity_ablation_12f_260826.pdf`
- 可编辑 SVG：`docs/figures/teacher_memory_necessity_ablation_12f_260826.svg`
- 600 dpi PNG：`docs/figures/teacher_memory_necessity_ablation_12f_260826.png`
- 曲线数据：`evaluation/shellgame/teacher_memory_necessity_12f_260826/validation_curve.csv`
- 完整机器可读结果：`evaluation/shellgame/teacher_memory_necessity_12f_260826/result.json`

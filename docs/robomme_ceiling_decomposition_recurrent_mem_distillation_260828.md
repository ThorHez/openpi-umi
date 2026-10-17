# RoboMME ceiling 分解过程蒸馏到 recurrent MEM（2026-08-28）

## 研究问题

此前的 decomposed full-context visual ceiling 在固定 test 上达到 56/56（100%）region
accuracy。本实验检验：将 ceiling 的中间过程分解为显式局部语义操作，再蒸馏给带可学习
gate 的 recurrent MEM，是否能够把这一上界转化为可部署的 12 帧递归模型。

## 模型与监督

新的 recurrent MEM 维护一个 `7 entities x 5 states` 的显式概率表：

- entities：red、green、blue、ordered-1、ordered-2、ordered-3、ordered-4；
- states：none、episode-local region 0/1/2/3；
- 每个非重叠 12 帧 chunk 最多解码两个 micro-events；
- event type：`hold / write / swap`；
- write payload：`entity + region`；
- swap payload：四个 region 中的无序 pair（6 类）；
- write 和 swap 使用确定性的 table update；event type 使用可学习 soft gate；
- table 被编码为 `128 x 64` latent memory tokens，保留后续接 action expert 的 latent 接口。

训练集共有 210 个三任务 episode。局部标签分布为：

| micro-event | 数量 |
|---|---:|
| hold | 9,595 |
| write | 611 |
| swap | 134 |

## 实验版本

### V1：直接联合蒸馏

从第 1 步同时优化 operation、全轨迹 table CE、final table CE 和 hold consistency。
该版本暴露出两个训练问题：

1. inverse-frequency event loss 的分母错误地使用了原始样本数，使 gate loss 整体缩小约
   10 倍；
2. 尚未学会的 soft gate 从训练开始就参与长序列 rollout，少量 false positive 在几十个
   hold chunk 中累计，产生很大的 table CE 并污染局部 operation head。

固定 test final region 仅为 9.6%。

### V2：800 步 operation pretrain + 1600 步联合 rollout

修复 weighted CE 归一化。前 800 步只蒸馏 ceiling 的 operation 和 payload，随后用 100 步
ramp 接入 recurrent trajectory/final loss。

- selected step：1300；
- checkpoint selection：先最大化三任务 dev mean final，再比较最差任务和 trajectory；
- 训练期间最佳 dev mean final：46.3%，三任务最低 40.0%。

### V3：operation-only + free-rollout evaluation

训练阶段只学习 ceiling 的局部 operation；验证与测试仍从空 table 开始执行完整 free
recurrent rollout。因此它不是 teacher-forced 指标，而是检验确定性 updater 是否需要额外
trajectory loss 的对照。

- selected step：400；
- dev mean final：39.5%；
- test mean final：33.4%。

## 固定 test 结果

| 方法 | VideoUnmask | VideoUnmaskSwap | VideoPlaceOrder | 平均 |
|---|---:|---:|---:|---:|
| Visual-oracle ceiling | 100.0% | 100.0% | 100.0% | 100.0% |
| Existing recurrent MEM fixed readout | 40.0% | 34.6% | 53.3% | 41.1% |
| V1 direct joint | 6.7% | 15.4% | 6.7% | 9.6% |
| V2 staged + rollout loss | 20.0% | 19.2% | 53.3% | 30.9% |
| V3 operation-only | 20.0% | 26.9% | 53.3% | **33.4%** |

V3 的 test operation accuracy：

| 子操作 | Accuracy |
|---|---:|
| event type | 78.0% |
| write entity | 64.5% |
| write region | 47.9% |
| swap pair | 46.4% |

V2 的 test operation accuracy 为 event 64.4%、write entity 55.4%、write region 34.7%、
swap pair 39.3%。联合 rollout loss 降低了 payload 泛化，因此本次设定下 operation-only 更好。

## 结论

本次实验没有把 100% ceiling 成功蒸馏成优于现有 MEM 的 student。分解监督相对错误的直接
联合版本把 test 从 9.6% 提升到 33.4%，说明 operation inductive bias 有效；但它仍低于现有
MEM 的 41.1%，当前版本不应直接接入长规模 action evaluation。

这并不证明 recurrent 结构天然有劣势。显式 table updater 在输入 oracle operation 时与
symbolic ceiling 等价，状态转移本身可以达到 100%。实际瓶颈位于 updater 之前的视觉
payload：尤其是 `write region` 和 `swap pair`。递归会累计这些局部错误，但不是错误的来源。

## 最核心的 teacher-student mismatch

100% visual ceiling 并不是只读任意一个 12 帧窗口：

- Unmask 使用第一帧的全局颜色中心建立 episode-local region anchors；
- UnmaskSwap 使用完整 demonstration 长度推断 swap 次数/区间，并在候选 anchor 周围比较
  运动；
- PlaceOrder 使用 demonstration-only subgoal boundaries 和 grounded/visual anchor
  coordinates 初始化 ordinal table，再分析完整 swap segment；
- 当前 student 只得到固定非重叠 12 帧 SigLIP patch tokens 和任务 prompt embedding，而且
  当前 operation 特征在空间维度做了平均。

因此，本次蒸馏复制了 ceiling 的输出标签和状态更新公式，却没有复制 ceiling 做出该判断所需
的全局锚点、时序分段和 pairwise ROI 输入。尤其对 `write region` 和 `swap pair`，这是输入
条件不一致，而不是单纯增加训练步数可以解决的问题。

## 下一步建议：context-matched decomposition distillation

下一版应保留同一个 recurrent semantic table，不增加任务定制头，但改变通用输入接口：

1. 从 episode 首帧/观察阶段维护 4 个共享 anchor tokens，显式监督
   `entity/ordinal -> anchor` 初始化；
2. 每个后续 12 帧窗口不再全局 spatial mean，而是对四个 anchor 做共享 ROI/cross-attention；
3. swap head 对 6 个 anchor pair 共享打分，直接预测 pair permutation；
4. write head 用 entity query 对 4 个 anchor 做共享 pointer；
5. soft event gate 保留，但 operation loss 与 recurrent rollout loss 解耦；先训练局部操作，
   再以较小权重校准 rollout，避免 table CE 覆盖 payload 监督；
6. 对 PlaceOrder 的 privileged subgoal boundary 单独做可见输入审计：若在线验证拿不到，就只
   把它作为训练 teacher，student 必须从 RGB/proprio 学 boundary，而不能在推理时依赖该字段。

这才是对 100% ceiling 的“输入、算法和中间监督”三者同时对齐，而不是只蒸馏最终分解标签。

## 产物

- 模型：`src/openpi/tasks/robomme/decomposed_region_recurrent_memory.py`
- 单元测试：`src/openpi/tasks/robomme/decomposed_region_recurrent_memory_test.py`
- 训练脚本：`scripts/mem/train_robomme_decomposed_region_distillation.py`
- V1：`checkpoints/robomme_decomposed_region_distillation_seed260828_260828/result.json`
- V2：`checkpoints/robomme_decomposed_region_distillation_staged_seed260829_260828/result.json`
- V3：`checkpoints/robomme_decomposed_region_operation_only_seed260830_260828/result.json`


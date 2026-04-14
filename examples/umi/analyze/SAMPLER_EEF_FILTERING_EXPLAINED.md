# 为什么「滤波后」的 EEF 轨迹仍有大量毛刺？

## 结论（先答）

**sampler_v2 对 EEF 观测做的并不是“滤波/平滑”，而是“时间索引 + 插值”。**  
在按**逐帧**画出「当前步」的观测时，得到的几乎就是**原始轨迹本身**，所以毛刺会原样保留。

---

## sampler_v2 对 `robot0_eef_pos` 实际做了什么

1. **Horizon + 下采样**  
   - `key_horizon['robot0_eef_pos'] = 2`，`key_down_sample_steps = 3`  
   - 当前帧 `current_idx` 对应的观测索引为：  
     `[current_idx - 3, current_idx]`（共 2 个点）

2. **插值**  
   - 在这两个（可能非整数）索引处，对原始轨迹做**一维线性插值**（`scipy.interpolate.interp1d`），得到长度为 2 的观测序列。

3. **我们画的是哪一维？**  
   - 可视化里取的是「当前步」的观测，即 `result['robot0_eef_pos'][-1]`，也就是**插值在 `current_idx` 处的值**。  
   - 当 `current_idx` 为整数时，插值结果就等于 `raw[current_idx]`，即**原始序列在该帧的值**。

4. **取 `result['robot0_eef_pos'][0]` 能去毛刺吗？**  
   - **不能。** 在 horizon=2、down_sample_steps=3 时，`idx_with_latency = [current_idx - 3, current_idx]`，所以：  
     - `[0]` = 插值在 **current_idx - 3**（3 帧前）→ 只是**时间滞后**，整条轨迹（含毛刺）整体平移 3 帧，毛刺仍在。  
     - `[-1]` = 插值在 **current_idx**（当前帧）→ 即原始当前帧。  
   - 若想用 horizon 内信息减毛刺，可对 horizon 做**平均**，例如 `mean(result['robot0_eef_pos'], axis=0)`，相当于当前帧与 3 帧前的 2 点平均，有轻微平滑效果；要明显去毛刺仍需更长窗口的滑动平均或低通滤波。

因此：**按每一帧画出来 = 逐帧取原始轨迹，没有做任何低通/平滑。**  
毛刺来自原始数据（采集噪声、遥操作抖动等），sampler 不会消除它们。

---

## 若希望“滤波后”真的变平滑

需要在**数据或观测**上显式做平滑，例如：

- 对 zarr 里 `robot0_eef_pos` 做**滑动平均**或 **Savitzky–Golay** 等低通滤波，再喂给 sampler；或  
- 在 sampler 之后对「当前步」轨迹再做一次平滑，仅用于分析/可视化。

sampler 本身**没有**设计成去毛刺的滤波器，它只负责：时间对齐、下采样、插值，供 policy 输入使用。

# 单臂电子元器件分拣 elec_meta：RTC 接入

本入口对应现有生产配置 `pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta`：Pi0.5、预测 16 步、模型动作宽度 32、有效动作前 10 维。训练配方和权重结构不变；新增的是服务端推理适配和独立 ARX 客户端。平台网页的部署 worker 本轮未修改。

本轮验证在本地 CPU 完成，包含真实 WebSocket、JAX RTC 采样器、真实生产预处理/tokenizer 对照、异步调度和无硬件回放。端到端采样测试使用小型测试流模型，没有加载服务器上的真实 checkpoint，也没有连接机器人。

## 1. 启动场景服务

在服务器的 openpi-umi 环境中运行，替换 checkpoint 路径：

```bash
uv run scripts/serve_policy.py \
  --profile elec-meta --rtc.enabled --rtc.mode client --port 8000 \
  policy:checkpoint \
  --policy.config pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta \
  --policy.dir /absolute/path/to/checkpoint
```

需要 checkpoint 中的 `params/` 和 `assets/norm_stats.json`；PyTorch `model.safetensors` 不适用于该入口。必须使用训练保存的 merged norm stats：`actions` 宽度 10、`state` 宽度 16。启动时会检查统计的形状与有限值，不会重新计算统计，也不会读取原训练数据目录。

`--profile elec-meta` 只替换推理数据入口：

- 接收两帧单臂观测，生成训练一致的 `(2,16)` 状态和腕部图像。
- 按训练顺序归一化、tokenize，然后展平到状态宽度 32。
- 反归一化前把动作从 32 维裁剪到 10 维，去除不应反归一化的 flattened state。
- 返回 `actions(16,10)` 和 `model_actions(16,32)`，并公布场景 metadata；客户端会检查这些约定。

本场景使用物理动作前缀：客户端保留世界坐标计划，再根据新观测时刻的 TCP 重新编码成原始 10 维动作，交给服务端做训练一致的归一化。不能直接把旧的归一化相对动作向前平移后回传。

## 2. 先做无硬件回放

机器人侧只需客户端包及 UMI 几何依赖；回放不会导入部署 SDK：

```bash
python -m pip install -e 'packages/openpi-client[umi]'
```

准备一个最小输入样例：

```bash
python - <<'PY'
import numpy as np
np.savez(
    '/tmp/elec_meta_rtc_obs.npz',
    robot0_eef_pos=np.array([[0.35, 0, 0.30], [0.35, 0, 0.30]]),
    robot0_eef_rot_axis_angle=np.zeros((2, 3)),
    robot0_gripper_width=np.full((2, 1), 0.03),
    camera0_rgb=np.zeros((2, 224, 224, 3), dtype=np.uint8),
    timestamp=np.array([0, 0.125]),
)
PY
```

纯本地接口检查，不连接服务器或机器人：

```bash
python examples/umi_arx/elec_meta_rtc.py \
  --replay /tmp/elec_meta_rtc_obs.npz --mock-policy \
  --obs-pose-repr relative --action-pose-repr relative \
  --frequency 8 --prompt 'pick electronic components' --max-duration 2
```

去掉 `--mock-policy`，连接第 1 步的服务即可检查真实模型的 RPC/预处理/RTC 接口，仍不会向机器人发送动作：

```bash
python examples/umi_arx/elec_meta_rtc.py \
  --replay /tmp/elec_meta_rtc_obs.npz \
  --policy-host SERVER_IP --policy-port 8000 \
  --obs-pose-repr relative --action-pose-repr relative \
  --frequency 8 --prompt 'pick electronic components' --max-duration 2
```

上述黑图仅供连通性检查，不能用于判断模型分拣质量。真实预测若违反速度/夹爪限制，回放也会报错，不会静默修改预测。同步后应使用实际采集的 raw 观测替换样例。

NPZ 输入是部署 SDK 的原始世界坐标观测，不是已经转换过的 LeRobot 特征：

| 字段 | 单个观测形状/约定 |
| --- | --- |
| `robot0_eef_pos` | `(2,3)`，世界坐标 XYZ，米 |
| `robot0_eef_rot_axis_angle` | `(2,3)`，旋转向量，弧度 |
| `robot0_gripper_width` | `(2,1)` 或 `(2,)`，米 |
| `camera0_rgb` | `(2,224,224,3)` RGB，uint8 或 `[0,1]` 浮点 |
| `timestamp` | `(2,)`，实机时为捕获 Unix 时间 |

可为以上字段增加共同的前导样本轴，回放会按 `--frequency` 选取样本。回放采用合成的本地捕获时间，不能用于测量真实相机延迟。坐标参数支持 `abs`、`rel`、`relative`，必须与训练数据转换和原部署配置一致；本入口拒绝 `delta`，不会猜测坐标约定。上面的 `relative`、8 Hz 和任务文本仅是回放示例，不代表服务器实际训练/部署参数。

## 3. 同步后再验证实际 ARX 环境

确认原部署的频率、观测/动作坐标参数、prompt、相机顺序、延迟标定和模型结果后，才能选择 `--execute`。该选项会打开相机和控制器，并执行部署 SDK 自身的初始化；本轮没有运行此模式。

```bash
python examples/umi_arx/elec_meta_rtc.py \
  --execute --umi-deploy-root /absolute/path/to/umi-arx-deploy \
  --robot-ip ROBOT_IP --robot-port 8765 --camera-reorder 0 \
  --policy-host SERVER_IP --policy-port 8000 \
  --obs-pose-repr relative --action-pose-repr relative \
  --frequency 8 --prompt 'pick electronic components' \
  --max-duration 10 --output /absolute/path/to/recording
```

频率必须显式指定，需使用原场景已验证的控制时间尺度。相机/机器人观测延迟默认分别为 `0.17/0.20` 秒，可用 `--camera-observation-latency`、`--robot-observation-latency` 设置实际标定值。`--command-latency` 默认 `0.05` 秒。

客户端 warmup 完成后重新取观测，再启动执行。运行时旧计划继续执行，后台推理下一块：旧 world 轨迹通过位置插值和旋转 SLERP 对齐新观测时间，再重新表达为新 TCP 下的 10 维动作。新块按实际 deadline 丢弃过期行，包含观测年龄、推理和命令预留时间；不会只按 RTT 裁剪。

提交使用 `dynamic_latency=False`，避免 SDK 再次重排时间。客户端和此场景服务端禁用 keepalive ping，避免首次同步 JAX 编译触发断连；RPC 接收超时默认 120 秒，运行期 `--max-inference-seconds` 默认 2 秒。旧计划覆盖不足、推理异常或超时会结束当前 episode，不会重复执行末行动作。

提交前检查当前实测 TCP 到首点及相邻点的速度，以及夹爪范围：默认 `--max-pos-speed 0.25` m/s、`--max-rot-speed 0.6` rad/s、夹爪 `[0,0.085]` m。参数可显式调整，超界会停止，不会裁剪动作或重新安排时间。这些检查不验证碰撞或 IK。Ctrl-C、超时或异常退出时，主线程会尝试用实测 TCP/夹爪执行 hold，覆盖未执行轨迹，然后结束录像和连接；`end_episode()` 本身不负责取消机器人轨迹。

## 本地回归命令

```bash
PYTHONPATH=.:src:packages/openpi-client/src JAX_PLATFORMS=cpu python -m pytest -q \
  src/openpi/policies/elec_meta_policy_test.py \
  packages/openpi-client/src/openpi_client/umi_pose_test.py \
  packages/openpi-client/src/openpi_client/rtc_controller_test.py \
  tests/test_elec_meta_rtc_serving.py \
  tests/test_elec_meta_rtc_runner.py \
  tests/test_elec_meta_rtc_e2e.py
```

端到端测试只监听本机临时端口；受限沙盒需要允许 loopback socket。Tokenizer 缓存遵循 `OPENPI_DATA_HOME`，首次使用可能需要下载官方 tokenizer。

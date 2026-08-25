# Qwen 知识蒸馏到 direct-visual recurrent MEM

## 要验证的接口

学生 MEM 的推理输入严格为：

```text
连续 10 帧图片
  -> SigLIP 浅层 patch embedding
  -> 固定 2x2 pooling：每帧 256 -> 64 tokens
  -> width=256、depth=2 的轻量时空压缩器
  -> 连续视觉 evidence [640,64]

(previous memory [128,64], visual evidence [640,64])
  -> shared recurrent cross-attention updater
  -> next memory [128,64]
```

学生 updater 不接收 relation id、relation probability、relation logit 或 Qwen 文本输出。Qwen/仿真标签只在训练时选择教师 memory target。

## 教师与训练方式

- Qwen：`qwen3vl_shellgame_gt_event_lora_v1_260825/checkpoint-000375`。
- 已验证 held-out reveal/swap 与完整三事件序列均为 100%。
- 教师 memory：成功的 symbolic recurrent tracker step500。
- 仿真实验使用 metadata 作为干净的离线 teacher forcing；这是基于 Qwen held-out 100% 的机制等价验证，不应表述为逐 episode 实际运行 Qwen 的软标签实验。
- 冻结学生 updater/readout 和完整教师，只训练 2.42M 参数的视觉压缩器。

loss：

```text
L_memory = tokenwise cosine(student memory, teacher memory)
           + 0.1 * MSE(student memory, teacher memory)

L = L_memory + 0.25 * L_stage_slot
```

## 结果

| eval | memory cosine | stage1 | stage2 | stage3/final | stage mean |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.583 | 30.0% | 28.3% | 38.3% | 32.2% |
| 100 | 0.530 | 30.0% | 40.0% | 26.7% | 32.2% |
| 150 | 0.517 | 40.0% | 33.3% | 31.7% | 35.0% |
| 200 | 0.476 | 60.8% | 37.5% | **41.7%** | 46.7% |
| final 249 | 0.464 | 53.3% | 42.5% | **38.3%** | 44.7% |
| 300 | 0.444 | 53.3% | 40.8% | 40.0% | 44.7% |
| 350 | 0.426 | 61.7% | 40.8% | 45.8% | 49.4% |
| 400 | 0.405 | 78.3% | 60.8% | 44.2% | 61.1% |
| 450 | 0.395 | 78.3% | 71.7% | 47.5% | 65.8% |
| final 499 | 0.396 | 75.8% | 68.3% | **52.5%** | 65.6% |
| 600 | 0.293 | 90.8% | 85.8% | 69.2% | 81.9% |
| 700 | 0.216 | 95.4% | 94.2% | 82.5% | 90.7% |
| 800 | 0.187 | 87.9% | 92.9% | 85.4% | 88.8% |
| 900 | 0.138 | 98.3% | 97.9% | 94.2% | 96.8% |
| **final 999** | **0.122** | **100.0%** | **98.3%** | **94.2%** | **97.5%** |

严格历史对照：

- direct visual、随机 tracker、仅 stage slot：后期 final 约 36.7%。
- direct visual、冻结成功 memory、仅 stage slot：final 31.7%。
- 本次 memory 蒸馏：step200 41.7%，step249 38.3%，step499 52.5%，调整学习率并续训至 step999 后达到 94.2%。

## 结论

继续训练证明原来的 250 步和 500 步都明显不足。step499 后将总步数提高到 1000，并以 `4e-5 -> 4e-6` 的余弦学习率继续优化；实际恢复点 step500 的学习率约为 `2.2e-5`。最终 held-out 杯位准确率从 52.5% 提升到 94.2%，三阶段准确率达到 100.0% / 98.3% / 94.2%，memory cosine loss 从 0.396 降到 0.122。

因此，这个实验不支持“128x64 memory 容量不足”或“direct-visual recurrent 接口根本学不会”的假设。冻结 updater/readout、仅训练 2.42M 视觉压缩器已经能够接近 relation bottleneck 的表现。step450--499 看起来像平台，实际上是旧学习率尾段过低造成的欠优化；温和学习率重启后，训练与验证同步改善，并未观察到明显过拟合。

仍然存在约 5.8% 的最终误差，且误差主要出现在第三次 recurrent update。若需要进一步逼近 100%，下一步应先用固定验证集和多 seed 确认 94.2% 的方差，再针对第三次更新做难例采样或更长的小学习率收敛；当前没有证据支持优先扩大 memory token 容量或改动推理接口。

## 复现

实现：`examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py`

checkpoint：

```text
checkpoints/shellgame_qwen_distilled_direct_visual_recurrent_memory_probe/
  qwen_distilled_direct_visual_memory250_260825/499
  qwen_distilled_direct_visual_memory250_260825/999
```

第二次续训（最终有效版本）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
.venv/bin/python \
  examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py \
  --exp-name qwen_distilled_direct_visual_memory250_260825 \
  --steps 1000 \
  --warmup-steps 0 \
  --peak-lr 4e-5 \
  --decay-lr 4e-6 \
  --memory-distill-weight 1.0 \
  --stage-slot-weight 0.25 \
  --batch-size 12 \
  --fsdp-devices 6 \
  --eval-interval 100 \
  --eval-batches 20 \
  --save-interval 500 \
  --keep-period 499 \
  --num-workers 0 \
  --resume
```

第一次续训命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
.venv/bin/python \
  examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py \
  --exp-name qwen_distilled_direct_visual_memory250_260825 \
  --steps 500 \
  --warmup-steps 0 \
  --peak-lr 6e-5 \
  --memory-distill-weight 1.0 \
  --stage-slot-weight 0.25 \
  --batch-size 12 \
  --fsdp-devices 6 \
  --eval-interval 50 \
  --eval-batches 10 \
  --save-interval 250 \
  --num-workers 0 \
  --resume
```

命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
.venv/bin/python \
  examples/shellgame/train_qwen_distilled_direct_visual_recurrent_memory_probe.py \
  --exp-name qwen_distilled_direct_visual_memory250_260825 \
  --steps 250 \
  --warmup-steps 50 \
  --peak-lr 3e-4 \
  --memory-distill-weight 1.0 \
  --stage-slot-weight 0.25 \
  --batch-size 12 \
  --fsdp-devices 6 \
  --eval-interval 50 \
  --eval-batches 10 \
  --save-interval 250 \
  --num-workers 0
```

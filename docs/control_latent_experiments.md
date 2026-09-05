# tinyWAM：控制 latent 新方法与执行步骤

本分支实现可选的控制/剩余视觉因素分解、特征去相关、每视角 learned-query 压缩、
latent 变化预测，以及 Stage 2 的间歇 FDM 辅助训练。所有新功能在原配置中默认关闭。
这是待验证的实验实现，不包含新的训练权重或成功率结论。

**已有 Stage 1、Stage 2 baseline 不需要重跑。** 修正 FDM 诊断只需加载现有 checkpoint
补评估；训练下面的新结构属于新增实验。建议先跑 `slots64`，再分别做去相关、delta 和
Stage 2 辅助损失消融，不要一次启动全部配置。

## 1. 实现了什么

| 模块 | 实现与约束 | 文件 |
|---|---|---|
| 视觉分解 | DINO patch → `c:192`、`u:192`；动作模型只读取 `c` | `slim/model/control_latent.py` |
| 重建 | 用 `c` 和 `u` 重建固定 DINO 特征；压缩版必须先从 slots 解码 dense control 特征 | 同上 |
| 去相关 | 中心化和标准化后的 `c.T @ u`，对特征交叉相关矩阵求平方均值 | 同上 |
| 防坍缩 | 每个 slot 在 batch 维度计算方差，固定 query 之间的差异不能冒充样本变化 | 同上 |
| 控制 token | 双视角各 32 个 queries，加入二维位置、视角和 masked-mean 语言条件 | 同上 |
| EMA | 同步完整视觉编码器与控制表示模块，并保留 FP32 累积 shadow | `slim/model/slim_model.py` |
| Delta head | 预测 `LN(target_future) - LN(target_current)`；不再归一化差值 | `slim/model/slim_transformer.py` |
| Stage 2 辅助训练 | 按 optimizer step 调度 FDM；一个 optimizer step 的所有累积 microbatch 使用同一开关 | `slim/training/stage2.py` |
| FDM 诊断 | 训练和评估共用 action-conditioned `predict_future()` | `scripts/evaluate_latent_dynamics.py` |
| 旧权重初始化 | `control_warm_start` 显式复用兼容权重，报告重新初始化的模块 | `slim/model/initialization.py` |

表示流：

```text
固定 DINOv3: [B,392,384]
  → 控制 c: [B,392,192] / 剩余 u: [B,392,192]
  → 可选控制压缩: [B,64,192]
  → control_readout: [B,64,384]
  → 原 384-dim、16-layer、6-head MoT
```

`control_readout` 只接收 c。把 192 维 c 映射回 384 维是为了保持现有 MoT 的输入投影和
输出 decoder 兼容，不会把 u 重新送入 policy。u 不能仅凭名称解释成背景或统计独立因素。
重建分支在 `predict_action()` 中不执行；压缩版 policy 的观测流长度为
`64 current + 64 future slots + 1 proprio = 129`，原版为 785。
推理仍需执行 DINO 和压缩器，token 数下降不代表端到端按平方比例加速。

第一版实验**要求冻结 DINO**，以固定特征监督重建，避免视觉 target 跟着坍缩。
`freeze_backbone:false` 会明确报错。解冻视觉编码器需要另行设计固定重建 teacher。
本分支不与现有 wrist HR adapter 同时启用，也没有实现额外的腕部局部 cross-attention。
语言和 proprio 的 policy 接口、动作归一化、H8 和四步 flow 推理保持原配置。

## 2. 配置与消融对应关系

所有新配置位于 `configs/libero/control_latent/`。

| 实验 | Stage 1 配置 | Stage 2 配置 | 说明 |
|---|---|---|---|
| 冻结视觉对照 | `stage1_frozen_baseline.yaml` | `stage2_frozen_baseline.yaml` | 可选，用于分离“冻结 DINO”的影响 |
| 分解，无去相关 | `stage1_factorized.yaml` | `stage2_factorized.yaml` | 392 tokens；重建和方差约束仍开启 |
| 分解＋去相关 | `stage1_decorrelated.yaml` | `stage2_decorrelated.yaml` | 与上一行比较去相关项 |
| 64 控制 tokens | `stage1_slots64.yaml` | `stage2_slots64.yaml` | 建议先跑这组完整闭环 |
| 64 tokens＋delta | `stage1_slots64_delta.yaml` | `stage2_slots64_delta.yaml` | delta 只在 Stage 1 训练，Stage 2 policy-only |
| Stage 2 保持动力学 | **同一个** `stage1_slots64.yaml` checkpoint | `stage2_slots64_dynamics.yaml` | 无需另训 Stage 1 来比较这两个 Stage 2 |
| Delta＋保持动力学 | **同一个** `stage1_slots64_delta.yaml` checkpoint | `stage2_slots64_delta_dynamics.yaml` | 与 delta 的 policy-only Stage 2 配对 |

默认损失：

```text
L_repr = 1.0 * L_reconstruction + lambda_corr * L_decorrelation + 0.1 * L_variance
lambda_corr = 0（factorized）或 0.01（decorrelated / slots64）
L_FDM = L_future + lambda_delta * L_delta
lambda_delta = 0 或 0.1

Stage 1: L = 0.125 * L_IDM + L_FDM + L_repr
Stage 2: L = L_policy + L_repr + active_step * 0.1 * L_FDM
```

Stage 2 辅助分支默认每四个 optimizer step 执行一次，从第一个更新开始。
权重 0.1 是**激活 step 上的权重**，没有额外乘四；平均附加权重约为 0.025。
`L_repr` 每个训练 batch 都执行。所有权重是起始超参数，并非经过搜索的最佳值。
统计量在每张卡的本地 batch 上计算；建议 `per_device_batch_size >= 2`，梯度累积不会
增加方差统计的样本数。启用 Stage 2 辅助目标后，dataset 会为每个 batch 解码未来图像；
每四步的开关节省的是辅助模型计算，未跳过其他 batch 的未来视频解码。

## 3. 获取代码并复用现有环境

在新的目录检出实验分支，避免改变正在跑任务的代码目录：

如果实验分支还未发布、你拿到的是 `tinyWAM-control-latents.patch`，先在新目录应用补丁：

```bash
git clone https://github.com/NKUShaw/tinyWAM.git tinyWAM-control
cd tinyWAM-control
git switch -c codex/control-latent-experiments
git apply --check /absolute/path/to/tinyWAM-control-latents.patch
git apply /absolute/path/to/tinyWAM-control-latents.patch
```

然后跳过下方 clone/cd 两行，继续设置环境变量。补丁基于
`e99619ddda83af85b1fe502f5433e9a7b8fc2bcc`；若 main 后续已有重叠修改，
先处理 `git apply --check` 报告的冲突，不要强制覆盖。
实验分支发布后，可直接使用下方命令获取：

```bash
git clone --branch codex/control-latent-experiments \
  https://github.com/NKUShaw/tinyWAM.git tinyWAM-control
cd tinyWAM-control

# 指向你已经能跑 DINOv3 tinyWAM 的 Python，无需重新安装训练环境。
export SLIM_TRAIN_PYTHON=/absolute/path/to/existing/.venv/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export LIBERO_DATA_ROOT=/absolute/path/to/lerobot/libero
export DINOV3_VITS16_MODEL_DIR=/absolute/path/to/dinov3_vits16_lvd1689m
export T5_MODEL_DIR=/absolute/path/to/t5-small
export SLIM_CACHE_DIR=/absolute/path/to/slim-cache
export SLIM_LANGUAGE_CACHE=/absolute/path/to/language_embeddings  # 已有 cache 时设置
export WANDB_MODE=offline

# 必须是原 DINOv3-S + 384-dim MoT 的 dense、非 wrist-adapter checkpoint。
export BASELINE_STAGE1=/absolute/path/to/baseline/checkpoints/epoch_3_pytorch_model.pt
```

如果没有 language cache，取消 `SLIM_LANGUAGE_CACHE` 即可，模型会使用本地 T5。
运行下面的 CPU 检查需要当前环境已安装 pytest：

```bash
"$SLIM_TRAIN_PYTHON" -m pytest -q tests/test_control_latent.py
```

这些检查使用小型合成视觉特征，不下载 DINO/T5、不读取 LIBERO。

## 4. 先做 10-step GPU smoke test

示例使用 4 GPU、每卡 batch 8、梯度累积 4，global batch = 128。
若换 GPU 数量，保持 `GPU数 × 每卡batch × 累积步数 = 128` 以便比较。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$SLIM_TRAIN_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m slim.training.stage1 \
  --config configs/libero/control_latent/stage1_slots64.yaml \
  --init-checkpoint "$BASELINE_STAGE1" \
  --init_mode=control_warm_start \
  --run.name=smoke_slots64 \
  --data.per_device_batch_size=8 \
  --training.gradient_accumulation_steps=4 \
  --training.max_epochs=null \
  --training.max_train_steps=10 \
  --training.num_warmup_steps=0 \
  --training.eval_interval=1000000 \
  --training.save_interval=10
```

**`max_epochs=null` 不要漏掉。** 仓库的 `steps_from_epochs()` 优先按 epoch 推导步数，
只写 `max_train_steps=10` 不会覆盖非空的 `max_epochs`。
默认 `run.timestamp=true`，每次命令生成独立 run 目录。

检查日志中 `idm_loss/fdm_loss/reconstruction_loss/decorrelation_loss/variance_loss` 有限，
并确认能够保存 step-10 checkpoint。GPU smoke test 需要在你的训练服务器执行；
这里未启动真实 GPU 训练。

## 5. 正式训练新方法

### Stage 1：从已有 baseline 初始化新表示

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$SLIM_TRAIN_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m slim.training.stage1 \
  --config configs/libero/control_latent/stage1_slots64.yaml \
  --init-checkpoint "$BASELINE_STAGE1" \
  --init_mode=control_warm_start \
  --data.per_device_batch_size=8 \
  --training.gradient_accumulation_steps=4
```

默认训练 3 epochs，数据仍为 LIBERO all+90。结果在
`checkpoints/control_latent/stage1/control_slots64_<timestamp>/`。
日志会列出复用数量和重新初始化的参数前缀。

`control_warm_start` 的行为：

- 复用相同架构的 DINOv3、MoT、action encoder/decoder、语言条件等兼容权重。
- 新建 control encoder 和可选 delta head。
- 当 token 数从 392 改成 64 时，重新初始化 future mask tokens 与 state position table，
  因为 future token 的位置偏移也变了；不会随意截取旧表。
- 从初始化后的 online 模块同步完整 EMA 管线。
- 拒绝 DINOv2/不同 MoT 维度等不兼容权重；也拒绝把已有 control checkpoint 当作原 baseline。

这是 warm-start 新实验，不是重跑原 baseline，也不是声称新模块已经训练完成。
若希望从预训练 DINO 开始，去掉 `--init-checkpoint` 与 `--init_mode` 即可。
比较消融时应统一初始化来源、训练预算和冻结设置；warm-start 的结果不能直接宣称与
原始两阶段配方具有相同总训练成本。

### Stage 2：从对应的新 Stage 1 checkpoint 初始化

```bash
export CONTROL_STAGE1=/absolute/path/to/control_slots64_run/checkpoints/epoch_3_pytorch_model.pt

CUDA_VISIBLE_DEVICES=0,1,2,3 "$SLIM_TRAIN_PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=4 -m slim.training.stage2 \
  --config configs/libero/control_latent/stage2_slots64.yaml \
  --init-checkpoint "$CONTROL_STAGE1" \
  --data.per_device_batch_size=8 \
  --training.gradient_accumulation_steps=4
```

这里使用默认的严格 `slim_policy` 初始化，**不要再传 `control_warm_start`**。
Stage 2 默认 40 epochs、LIBERO all。先用上面 policy-only 配置做闭环评估。

再比较动力学保持时，用**同一个** `$CONTROL_STAGE1` 把配置改为：

```text
configs/libero/control_latent/stage2_slots64_dynamics.yaml
```

该配置启用 EMA，并从 Stage 1 保留完整 EMA 权重；policy-only 配置则显式跳过 EMA 权重。
对 delta 消融，Stage 1 改为 `stage1_slots64_delta.yaml`，对应 Stage 2 使用表中带 delta 的
配置。不同 token 数和 delta-head 开关的 checkpoint 不应混用，严格加载会报告不匹配。

中断后续训沿用原仓库的 `--resume-state <run>/states/step_XXXXXXXX`，不要同时传
`--init-checkpoint`。保持相同进程拓扑、配置和梯度累积；辅助目标按恢复后的
`completed_steps` 继续调度。

## 6. 闭环评估与离线 FDM 诊断

### Policy：复用已有 LIBERO / LIBERO-Plus 流程

```bash
export CONTROL_STAGE2=/absolute/path/to/new_stage2_run/checkpoints/epoch_40_pytorch_model.pt

"$SLIM_TRAIN_PYTHON" -m slim.serving.server \
  --checkpoint "$CONTROL_STAGE2" --port 10093 --bf16
```

在已有 LIBERO simulator 环境的另一个终端执行原 client 命令，保持 H8、send-state、
任务集、每任务 trial 数、seed 和 baseline 相同。server 从 checkpoint 所属 run 的
`config.yaml` 自动恢复新结构，不需要改 simulator client。完整步骤见根目录 README
的 Evaluation 部分和 `docs/environments.md`。

### FDM：现有 checkpoint 可以直接补评估

```bash
"$SLIM_TRAIN_PYTHON" scripts/evaluate_latent_dynamics.py \
  --checkpoint "$BASELINE_STAGE1" \
  --batches 50 --batch-size 8 --device cuda --bf16 \
  --output outputs/dynamics/baseline_stage1.json

"$SLIM_TRAIN_PYTHON" scripts/evaluate_latent_dynamics.py \
  --checkpoint "$CONTROL_STAGE1" \
  --batches 50 --batch-size 8 --device cuda --bf16 \
  --output outputs/dynamics/control_stage1.json
```

脚本读取每个 run 的 resolved `config.yaml` 与 `action_stats_<source>.json`，
不会重算训练集动作归一化或更新权重。使用固定 seed 从验证集抽取随机顺序的 batch，
不再只评估最前面的相邻轨迹。`val_ratio=0` 时输出 `validation_is_holdout=false`，
表示采用原仓库的训练 episode 监控子集，不能当作留出集。

| 输出 | 含义 |
|---|---|
| `fdm_future_loss` | 与训练相同的 action-conditioned future loss，默认是 normalized-L1 |
| `identity_loss` | 直接把同一 teacher 的当前表示当作未来表示 |
| `shuffled_action_loss` | 固定观测和目标，对 batch 中的动作做非零循环置换 |
| `action_sensitivity_gap` | shuffled loss − real loss；需结合动作差异大小解释 |
| `delta_loss` / `zero_delta_loss` | delta head 的 Smooth-L1 与恒零变化对照，仅 delta 模型输出 |

不同表示空间的 FDM 数值不能直接解释成相同难度下的模型优劣；重点看各自的 identity、
动作置换对照和相同协议的闭环成功率。置换动作只是诊断，不是真实反事实轨迹标签。
默认 train log 中旧的 `eval/future_latent_mse` 已更名为 `eval/fdm_future_loss`。
缺失 proprio 输入时，FDM 的 attention mask 边界也按实际 token 数修正。

## 7. 验证范围

开发检查使用 Python 3.12、PyTorch 2.6.0 CPU、Transformers 5.3.0；仓库全部 37 项
测试通过，包含 BF16 autocast 下的一次实际 backward/AdamW/EMA 更新。检查覆盖：

- policy 无法通过 u 分支读取视觉信息；重建、压缩 queries、delta head 能收到梯度。
- Stage 1/Stage 2 总损失确实包含开启的辅助项，关闭分支不要求未来图像。
- 新表示的 EMA 更新、BF16 下 FP32 shadow、teacher 无梯度。
- checkpoint 严格往返、dense warm-start、Stage 1 到 Stage 2 加载和不兼容结构报错。
- FDM 训练/评估预测一致，动作置换改变预测，诊断不修改权重。
- 新配置的继承、public/runtime schema 往返，以及按 optimizer step 的辅助调度。

真实 DINO/T5 权重加载、DeepSpeed 多 GPU、显存峰值和 LIBERO 闭环收益需要在你的环境
执行第 4–6 节验证；合成特征测试不替代这些结果。原配置的默认带 proprio 输入路径
另与修改前 Transformer 比较了 policy/IDM/FDM/联合目标的数值一致性。

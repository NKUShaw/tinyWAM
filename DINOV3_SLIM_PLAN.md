# SLIM DINOv3-S/16 迁移计划

## 目标

在保留 SLIM 官方 two-stream MoT、Stage 1/Stage 2 训练协议的前提下，将
`DINOv2-B/14` 替换为 `dinov3_vits16_lvd1689m`，评估更轻量视觉骨干对 latent
dynamics、训练效率和 LIBERO policy success 的影响。

当前运行的官方 DINOv2 Stage 1 是 baseline，不停止、不修改。

## 严格对照

| 项目 | 官方 baseline | DINOv3 实验 |
|---|---|---|
| Visual backbone | DINOv2-B/14 | DINOv3-S/16 |
| 输入分辨率 | 224 x 224 | 224 x 224 |
| 双视角 patch tokens | 512 | 392 |
| Encoder dim | 768 | 384 |
| MoT hidden dim/layers | 768 / 16 | 384 / 16 |
| Attention heads / head dim | 12 / 64 | 6 / 64 |
| Action horizon | 8 | 8 |
| Global batch | 128 | 128 |
| Stage 1 数据 | LIBERO all+90 | 相同 |
| Stage 1 训练 | 3 epoch | 相同 |
| Stage 2 数据 | LIBERO all | 相同 |
| Stage 2 训练 | 40 epoch | 相同 |

当前目标是构建轻量模型，因此主实验同时采用 DINOv3-S/16 与 384-dim MoT。该实验衡量
完整 tiny 配方，不将结果解释为纯 backbone 消融；如需归因，后续补 DINOv3-S + 768 MoT。

## 模型接口修改

```text
agent + wrist images
        |
DINOv3-S/16 online encoder
        |  [B, 392, 384]
state_input_proj: 384 -> 384
        |
official 16-layer two-stream MoT
        |
state_decoder: 384 -> 384
        |  [B, 392, 384]
DINOv3-S/16 EMA target
```

必要配置：

```yaml
model:
  vision_encoder:
    backbone_name: dinov3_vits16_lvd1689m
    image_size: 224
    freeze_backbone: false
  ema:
    enabled: true
    momentum: 0.999
  transformer:
    hidden_dim: 384
    num_layers: 16
    num_heads: 12
    num_future_tokens: 392
```

需要检查 DINOv3 是否包含 CLS/register tokens。SLIM dynamics 只使用固定顺序的 patch
tokens，不能将 CLS/register tokens 混入 392 个 future slots。

## Stage 1

保持官方目标：

```text
L_stage1 = 0.125 * L_IDM + 1.0 * L_FDM
```

- online DINOv3 参与训练，建议初始学习率 `1e-5`；
- EMA DINOv3 不接收梯度，momentum `0.999`；
- T5-Small 继续冻结并使用 language cache；
- FDM 保持官方 future-mask prediction 和 normalized-L1；
- 不加入 VICReg、SIGReg、residual FDM 或额外 policy loss；
- 3 epoch、global batch 128、seed/split seed 42。

必须额外记录：

- IDM、FDM、total loss；
- prediction error / identity error；
- shuffled-action error / real-action error；
- prediction/target std ratio；
- spatial token spread ratio；
- participation-rank ratio；
- teacher-forced 与 autoregressive 2-step error；
- step time、peak memory、tokens/s。

## Stage 2

DINOv3 Stage 2 必须从自己的 DINOv3 Stage 1 epoch-3 checkpoint 初始化，不得加载
DINOv2 Stage 1/Stage 2 模型参数。

- 保持官方 Flow-Matching policy objective；
- LIBERO all，H=8，40 epoch；
- 保持相同 action normalization、inference steps 和 evaluation seeds；
- Stage 2 是否继续更新 DINOv3，应与官方配置一致。

最终报告：

- Standard LIBERO overall/per-suite success；
- LIBERO-Plus success；
- 至少报告固定 seeds 的成功数和总 trial 数；
- 参数量、训练显存、Stage 1/Stage 2 wall time；
- 不用 Stage 1 proxy loss 代替 closed-loop policy success。

## 实验顺序

1. 完成当前官方 DINOv2 Stage 1 baseline。
2. 用同一 audit 脚本评估其 Stage 1 epoch-3 checkpoint。
3. 完成 DINOv3 backbone wrapper、shape test 和 EMA update test。
4. DINOv3 Stage 1 做 10-step、100-step smoke test。
5. 正式训练 DINOv3 Stage 1 三个 epoch。
6. 对两个 Stage 1 checkpoint 跑 matched latent audit。
7. 分别从对应 Stage 1 checkpoint 训练 Stage 2。
8. 用完全相同的 LIBERO evaluation protocol 比较。

## 实施门槛

进入完整 Stage 2 前，DINOv3 Stage 1 至少满足：

- 无 NaN/OOM，EMA 更新正常；
- FDM 优于 identity baseline；
- shuffled action 明显劣于 real action；
- 2-step autoregressive error 没有立即爆炸；
- prediction std/spread/rank 不出现严重 collapse；
- checkpoint 能独立加载并复现 validation 指标。

## 第二阶段可选消融

完成主对照后再考虑：

- DINOv3-S + 768-dim MoT，用于分离 backbone 与 MoT 宽度因素；
- DINOv3 frozen vs online+EMA；
- 224 vs 更高输入分辨率；
- DINOv3-B/16，用于分离 backbone family 与模型规模因素。

这些消融不能混入第一轮 DINOv3-S/16 + 384-dim MoT 主实验。

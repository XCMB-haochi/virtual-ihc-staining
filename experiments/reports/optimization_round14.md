# 优化记录 Round 14：base48 容量对照

日期：2026-07-17

## 依据与目标

旧 H2 实验中，base48 单模型相对双 base32 完整管线约提高 `+0.1134/+0.0991`，与一颗 base32 固定融合后约提高 `+0.1242/+0.1263`，三个 ROI 方向一致。该实验没有使用当前 A100 上统一的两阶段训练协议，也没有独立 H1 复核；增益接近 `+0.15` H2 门槛，值得做一次严格容量对照，但不足以直接推广 base64。

本轮只改变 U-Net 宽度 `32 -> 48`，检验更多参数能否在固定训练/推理管线下产生稳定、可跨 fold 的增益。

## H2 固定协议

| 项目 | 固定值 |
|---|---|
| 验证 ROI | `ROI020,ROI022,ROI024` |
| marker | `CD68` |
| seed | `2026` |
| 候选 | base48 residual U-Net，condition none |
| 对照 | Round 11 base32 residual U-Net |
| 输入 | context0，256x256 |
| 第一阶段 | 14 epoch，AdamW，lr `3e-4` cosine，weight decay `1e-4` |
| 第二阶段 | 从第一阶段 best global-MSE 初始化新 optimizer，14 epoch，lr `1e-4` cosine |
| batch / workers | `8 / 4` |
| 损失 | Uniform-7 hybrid `0.7/0.2/0.1` |
| 选模 | CD68 global-MSE，JPEG q95 |
| 完整评估 | halo64 + D4 + q95 + raw |

匹配对照直接复用：

- `runs/round11_h2_unet_cd68_stage1/`
- `runs/round11_h2_unet_cd68_stage2/`
- `reports/unet_cd68_h2.json`

候选固定输出：

- `runs/round14_h2_unet_base48_stage1/`
- `runs/round14_h2_unet_base48_stage2/`
- `runs/_round14_eval/unet_base48_h2.json`

不搜索宽度、batch、学习率、epoch、损失、快照、融合或 TTA 子集。保存每个 epoch 供审计，正式比较使用第二阶段 best global-MSE checkpoint；这是容量/可部署模型比较，而非 Round 13 的固定更新时钟因果对照。

## Promotion gate

1. H2 完整管线下，base48 相对 base32 的 q95 mean-image 与 global-MSE 都至少提高 `+0.15`；
2. ROI020、ROI022、ROI024 的两种 q95 聚合均不得回退超过 `0.05`；
3. float/raw 两种总体增益必须为正，逐 ROI 不得回退超过 `0.05`；
4. 通过 H2 后锁定全部设置，进入 H1：`ROI015,ROI018,ROI019`，重新训练匹配 base32/base48；
5. H1/H2 四个 q95 聚合增益必须全部为正，平均至少 `+0.30`，六个 ROI 均不得回退超过 `0.05`；
6. 只有通过双 fold 才允许训练 full-data，并额外评估一次固定 `50/50 base32+base48`；不搜索融合权重；
7. H2 失败则停止 base48，并且不启动 base64。

## 启动条件

- GPU1 先用 512 张训练图、64 张验证图做 batch 8 预检；
- 预检确认参数量、单 epoch 时间和峰值显存；
- 正式 run 输出目录必须不存在；
- 使用 commit `ceb7676`，不与其他本项目 GPU 任务并行，不干预服务器上其他用户进程。

## 资源与 H2 结果

小样本预检：

- 参数量 18,243,937，为 base32 的约 2.25 倍；
- batch 8 峰值 PyTorch 显存 `3.084 GiB`；
- 512/64 单 epoch `4.768` 秒；
- commit `ceb7676`，AMP 与 config 正常。

正式两阶段均完整结束，单 epoch 约 `37-38` 秒，第二阶段峰值 `3.155 GiB`。第一阶段 best 为 epoch 14 的 `69.2406/68.5670`；第二阶段按 global-MSE 选中 epoch 10：

| 模型 | 裸 q95 mean-image | 裸 q95 global-MSE | base48 - base32 |
|---|---:|---:|---:|
| base32 | 69.5464 | 68.8956 | - |
| base48 | 69.7116 | 69.0775 | +0.1652 / +0.1819 |

预注册完整 `halo64 + D4` 总体结果：

| variant | base32 | base48 | base48 - base32 |
|---|---:|---:|---:|
| q95 mean-image | 70.0121 | 70.2197 | +0.2077 |
| q95 global-MSE | 69.3435 | 69.5846 | +0.2411 |
| float mean-image | 70.0735 | 70.2974 | +0.2240 |
| float global-MSE | 69.4043 | 69.6617 | +0.2573 |

逐 ROI 增益：

| ROI | q95 mean | q95 global | float mean | float global |
|---|---:|---:|---:|---:|
| ROI020 | +0.9081 | +0.8651 | +0.9243 | +0.8811 |
| ROI022 | +0.2768 | +0.3340 | +0.2836 | +0.3407 |
| ROI024 | -0.4869 | -0.3830 | -0.4616 | -0.3576 |

base48 的四个总体指标都超过门槛，且收益不依赖 JPEG；但 ROI024 在 raw/q95、两种聚合上均大幅回退。它未通过逐 ROI gate：不进入 H1、不训练 full-data、不评估融合、不启动 base64，也不根据本折选择 ROI 特定模型。

该结果说明 A100 已消除容量实验的资源障碍，但更大 U-Net 主要强化 ROI020/022，未改善跨 ROI 泛化。完整评估证据为 `reports/unet_base48_h2.json`，匹配对照为 `reports/unet_cd68_h2.json`。

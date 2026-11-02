# 优化记录 Round 8：测试域 DAPI 均值邻近加权

日期：2026-07-17

## 目的

测试集五个 DAPI ROI 的均值集中在训练分布的较高区间。本实验不改变模型、数据顺序、增强、输入尺寸、损失组成或学习率，只按训练 ROI 的 DAPI 均值与测试 DAPI 中心的距离轻度调整逐样本 loss 权重，判断这种组成偏移是否能转化为模型增益。

## 固定设计

测试 DAPI 中心取全部测试 tile/pixel 的加权均值：

```text
test_center = 37.7882340361778
raw_weight(roi) = 1 + 0.5 * exp(-0.5 * ((roi_mean - test_center) / 1.5)^2)
weight(roi) = raw_weight(roi) / mean_over_training_tiles(raw_weight)
```

- H2 验证 ROI：`ROI020,ROI022,ROI024`
- 初始化：`runs/unet_cd68_hard_h2_uniform7_14e/best.pt`
- 模型：base32 residual U-Net，context0
- 损失：Uniform-7 hybrid `0.7/0.2/0.1`
- 优化：AdamW，`lr=1e-4`，14-epoch cosine 定义不变
- seed：2026
- 固定比较轮：epoch 5
- 实验训练 ROI 权重范围：`0.8704847-1.3023209`
- 按实际训练 tile 计算的权重均值：`1.0`

run：`runs/unet_cd68_hard_h2_uniform7_continue_testmean_rbf_a05_s15_lr1e4_14e/`。固定 epoch 5 未通过门槛后终止剩余训练，因此该 run 只保留 epoch 1-5。

## 裸验证

| epoch 5 | 逐图分 | global-MSE 分 | 相对控制 |
|---|---:|---:|---:|
| 原续训控制 | 69.6423 | 68.9628 | - |
| 测试均值 RBF 加权 | 69.5805 | 68.8882 | -0.0618 / -0.0747 |

## 固定完整管线

使用实验 seed2026 epoch 5 与原 seed2027 epoch 6 各 50%，固定 `halo64 + D4 + q95`，不搜索融合权重。完整结果位于 `reports/ensemble_continue_testmean_rbf_e5_old_s2027e6_h2.json`。

| 输出 | 控制逐图/global | 加权逐图/global | 差值 |
|---|---:|---:|---:|
| raw | 70.2233 / 69.5237 | 70.1897 / 69.4849 | -0.0336 / -0.0387 |
| log0.75 | 70.7770 / 70.1808 | 70.7574 / 70.1583 | -0.0196 / -0.0224 |

raw 逐 ROI 差值：

| ROI | 逐图分 | global-MSE 分 |
|---|---:|---:|
| ROI020 | -0.0144 | -0.0223 |
| ROI022 | -0.0442 | -0.0488 |
| ROI024 | -0.0410 | -0.0436 |

log0.75 逐 ROI 差值：

| ROI | 逐图分 | global-MSE 分 |
|---|---:|---:|
| ROI020 | -0.0111 | -0.0151 |
| ROI022 | -0.0265 | -0.0293 |
| ROI024 | -0.0208 | -0.0223 |

## 决策

预设推广门槛是两种总分均至少提高 `+0.15`、三个 ROI 均不下降，且任一 ROI 回退不得超过 `0.05`。本实验在裸验证、raw 完整管线、log0.75 完整管线及全部 ROI 上均下降，因此明确淘汰：

- 不继续 epoch 6-14；
- 不在 H1 复核；
- 不训练全数据加权模型；
- 不调整 `alpha/sigma/center` 追逐同一 H2。

测试输入的均值组成偏移是真实存在的，但仅按 ROI 均值对训练 loss 重加权不能改善当前模型。下一方向不应继续搜索同类亮度权重。

## 实现与验证

训练器新增 `--roi-weight-mode test-mean-rbf`，权重只乘逐样本 hybrid loss，不改变 DataLoader 采样顺序。派生中心、各 ROI 均值、最终权重和 tile 权重均值同时写入 config 与 checkpoint 元数据。无权重路径保持原有归约代码。

相关测试覆盖 ROI 均值、RBF 归一化、非法参数、逐样本加权公式，以及全 1 权重在 CUDA AMP 下与旧路径的 loss/梯度等价性。

# 优化记录 Round 10：ROI 均值条件化

日期：2026-07-17

## 起点

实测端点组合的服务器分为 `72.0437`，与提交前四位中心预测完全一致；平台显示 `SSIM=0.811`、`PSNR=25.505`。逐 ROI 探针已经关闭，本轮只做不依赖排行榜反馈的模型实验。

## 路线选择

最初实现的 NAF-style block U-Net 在 16 张训练图、8 张验证图的 context0 CUDA AMP 预检中有 7.25M 参数，但峰值显存达到 `4.84 GiB`，单 epoch 预检耗时约 `46.9` 秒；参数量相近的 8.11M residual U-Net + ROI FiLM 预检只有 `1.99 GiB` 和 `1.19` 秒。因此 NAF 原型不进入完整训练，也不保留该低效实现。该一次性预检目录已清理，资源数字只用于路线筛选，不作为可复现实验结论。

ROI DAPI 均值条件化优先级更高，原因是现有 patch/halo 模型无法直接观察整 ROI 的全局强度，而 ROI029 校准曾有 `+0.2435` 的服务器边际。test-mean RBF 失败只否定了按测试均值重加权训练样本，没有验证把每个 ROI 自身的输入统计交给模型。

## 固定实现

- 条件特征：同一 ROI 全部中心 DAPI tile 的像素均值，单位 `0..255`；
- 标准化：只用训练 fold 的唯一 ROI 均值，ROI 等权计算 center 与 population std；
- 验证/测试：只计算各自完整 DAPI ROI 的均值，再使用 checkpoint 中的训练 center/std；
- 注入位置：U-Net bottleneck；
- 注入方式：`feature * (1 + gamma) + beta`；
- 参数：`Linear(1, 2*C)` 等价权重和偏置全部零初始化；
- 输出：不做 DAPI 全局 residual，保持独立 sigmoid CD68 head；
- 数据截断：条件统计在 `max-train-images/max-val-images/max-images` 之前计算。

零初始化保证同 seed 下条件模型在第一个更新前与原 U-Net 逐元素同输出。FiLM 参数直接用零张量创建，不消耗随机数，因此骨干初始化与 DataLoader shuffle RNG 不因额外模块而改变。

## H2 预注册实验

| 项目 | 固定值 |
|---|---|
| 验证 ROI | `ROI020,ROI022,ROI024` |
| marker | `CD68` |
| seed | `2026` |
| 模型 | base32 residual U-Net + bottleneck ROI-mean FiLM |
| 参数量 | `8,112,705` |
| 输入 | context0，256x256 |
| 训练 | 14 epoch，batch 8，AdamW，lr `3e-4` cosine |
| 损失 | Uniform-7 hybrid `0.7/0.2/0.1` |
| 选模 | global-MSE，JPEG q95 |
| 最终比较 | halo64 + D4 + q95 + raw，不使用 ROI 输出缩放 |
| 匹配对照 | `runs/unet_cd68_hard_h2_uniform7_14e/` |

run：`runs/unet_cd68_hard_h2_uniform7_roi_mean_film_14e/`。

## 固定门槛

1. H2 完整管线的 mean-image 与 global-MSE 分都至少提高 `+0.15`；
2. ROI020、ROI022、ROI024 任一逐 ROI 分回退不得超过 `0.05`；
3. 只有同时通过 1、2 才进入 H1；H2 未通过时不搜索 std、分位数、FiLM 层数或注入位置；
4. H1 固定 `ROI015,ROI018,ROI019`，需补训完全匹配的 Uniform-7 baseline 与 conditioned run；
5. 最终要求 H1/H2 两 fold、两种聚合都为正，四项平均增益至少 `+0.30`，六个 ROI 均不得回退超过 `0.05`；
6. 未达到最终门槛时不训练全数据、不提交排行榜，也不租 GPU。

## 工程预检

- 全部单元测试：收尾复跑 `46/46` 通过；
- conditioned 训练、compact checkpoint 保存与单模型评估通过；
- conditioned test prediction 通过；
- legacy U-Net + conditioned U-Net 异构融合通过；
- conditioned checkpoint soup 的相同统计兼容与不同统计拒绝均有测试；
- 小样本 CUDA AMP 峰值：`1.9901 GiB`；
- 小样本参数量：`8,112,705`，仅比原模型多 2048 个 FiLM 参数。

## H2 结果

训练完成 14 epoch，逐 epoch 计时累计约 `20.4` 分钟，外部 wall time 约 `20.6` 分钟，峰值显存 `1.9908 GiB`。conditioned 与 baseline 都在 epoch 14 取得各自 global-MSE 最佳点。

裸 q95 验证：

| 模型 | mean-image | global-MSE | conditioned - baseline |
|---|---:|---:|---:|
| baseline | 69.1599 | 68.4839 | - |
| ROI-mean FiLM | 69.1521 | 68.5008 | -0.0079 / +0.0169 |

固定完整管线为 `halo64 + D4 + q95 + raw`，报告：`reports/ensemble_roi_mean_film_h2.json`。

| 模型 | mean-image | global-MSE | conditioned - baseline |
|---|---:|---:|---:|
| baseline | 69.5915 | 68.8996 | - |
| ROI-mean FiLM | 69.6353 | 68.9691 | +0.0439 / +0.0695 |

raw 逐 ROI 增益：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI020 | -0.0937 | -0.0434 |
| ROI022 | +0.4779 | +0.4747 |
| ROI024 | -0.2378 | -0.2224 |

FiLM 明显改善 ROI022，但同时伤害 ROI020 与 ROI024。总体两种聚合均未达到 `+0.15`，且 ROI020 mean-image、ROI024 两种聚合都超过允许的 `-0.05` 回退。

## 决策

本实验未通过预注册 H2 门槛：

- 不进入 H1；
- 不搜索 ROI std、分位数、注入位置、FiLM 层数或融合权重；
- 不训练全数据模型；
- 不生成服务器提交；
- 不租 GPU。

条件模型在 ROI022 表现出互补性，但缺少 constant/shuffled-condition 消融，不能把该收益明确归因于 ROI 均值；同时它无法稳定改善其他 ROI。后续若继续结构方向，应转向真正参数量匹配的异构复原架构或公开预训练 encoder，并继续沿用双 fold 门槛，而不是恢复逐测试 ROI 探针。

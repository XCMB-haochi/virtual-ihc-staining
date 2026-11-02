# 优化记录 Round 23：base32 + full-target context64 固定融合

日期：2026-07-18

## 立项依据

Round22 full-target context64 在 H2 halo64+D4+q95 上相对 base32 提高 `+0.173179/+0.187987`，三个 ROI 的两种聚合全部为正，但未达到该轮双 `+0.25` 强门槛。它与 base32 模型结构相同，训练视野和监督区域不同，可能存在可利用的误差互补。

Round17 中，标准 base48 单模型约 `+0.21/+0.24`，与 base32 固定 50/50 后提高到约 `+0.42/+0.42`。因此本轮只检验同样保守的固定 raw 平均，不根据 Round22 输出搜索权重。

## 固定候选

- checkpoint A：`runs/round11_h2_unet_cd68_stage2/best_global_mse.pt`；
- checkpoint A SHA-256：`f4c15098ea7479c205eeb419ff5e152b8691bc15f801a7b0ccb30ee4cd5832e3`；
- checkpoint B：`runs/round22_h2_unet_context64_fulltarget_stage2/best_global_mse.pt`；
- checkpoint B SHA-256：`f8830f86142f7c5244dc40f769fac73fedf01a47b31955cd036324afdf1f6a16`；
- 浮点输出：`0.5 * A + 0.5 * B`；
- H2：`ROI020,ROI022,ROI024`；
- 推理：halo64+D4；
- 第一阶段只评估 JPEG q95；
- 只读取 `weight_b_0.5_raw`。

固定输出：

- `runs/_round23_eval/base32_fulltarget_50_h2_q95.json`；
- q95 通过后才生成 `runs/_round23_eval/base32_fulltarget_50_h2_float.json`。

`evaluate_ensemble.py` 会同时计算训练集拟合的 calibration variants；本轮禁止读取这些 variant 进行晋级、调权或提交。禁止测试 25/75、75/25、per-ROI 权重、epoch、soup 或额外模型。

## Promotion gate

1. q95 `weight_b_0.5_raw` 相对 `reports/unet_cd68_h2.json` 的 mean-image 与 global-MSE 均至少提高 `+0.25`；
2. ROI020、ROI022、ROI024 的两种 q95 聚合均不得回退超过 `0.05`；
3. q95 通过后才运行完全相同权重的 float；两个总体增益必须为正，逐 ROI不得回退超过 `0.05`；
4. H2 通过后，在独立 H1 `ROI015,ROI018,ROI019` 从头训练与 Round22 完全相同的 full-target 两阶段模型，并与既有匹配 base32 固定 50/50；
5. H1/H2 四个 q95 总体增益必须各至少 `+0.15`、平均至少 `+0.40`，六个 ROI均不得回退超过 `0.05`；
6. 任一 gate 失败即关闭，不改融合权重，不以 calibration variant 或 direct candidate 绕过。

## Full-data 边界

只有 H1/H2 全部通过后，才另行预注册 full-data epoch 映射、seed 数和唯一提交包。Round17 已证明双折约 `+0.31` 的融合可能无法转化到服务器，因此本轮双折平均要求提高到 `+0.40`，且 full-data 不允许依据排行榜选择 epoch或权重。

## 启动条件

1. 两个 checkpoint 的 marker、模型规格、conditioning、内部 epoch 和 SHA-256 已核验；
2. q95 命令只传 `--weights 0.5 --jpeg-quality 95`，结论只读取 raw；
3. 预注册文档提交、推送并同步服务器；
4. GPU1 无本项目遗留任务，不停止或干预其他用户进程；
5. 输出文件不存在，远端 tracked tree clean。

## H2 q95 结果

固定 `weight_b_0.5_raw`：

| 输出 | base32 | 50/50 raw | 增益 |
|---|---:|---:|---:|
| mean-image | 70.012066 | 70.247542 | +0.235475 |
| global-MSE | 69.343546 | 69.583985 | +0.240439 |

逐 ROI 增益：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI020 | +0.368213 | +0.405743 |
| ROI022 | +0.162378 | +0.157449 |
| ROI024 | +0.184378 | +0.183527 |

三个 ROI 全部为正，固定融合也将 Round22 direct candidate 的总体 `+0.173/+0.188` 放大到 `+0.235/+0.240`，证明两模型有真实误差互补。完整 q95 证据为 `reports/round23_base32_fulltarget_50_h2_q95.json`。

## 决策

Round23 仍按原 gate 关闭：mean/global 分别距离 `+0.25` 门槛 `0.014525/0.009561`，因此不运行 float、不搜索权重、不读取 calibration variants，也不在本轮直接启动 H1。

该结果足够支持一个新的独立复现实验，但不能事后修改 H2 标准。Round24 保持模型配方和 50/50 权重完全不变，只在未参与上述选择的 H1 上重建 full-target 模型；其 full-data 门槛由已知 H2 数值反推并在训练前固定。

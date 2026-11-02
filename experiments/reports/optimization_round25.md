# 优化记录 Round 25：base checkpoint 的 full-target adaptation

日期：2026-07-18

## 假设

Round22/24 从头训练的 full-target 模型在 H2/H1 总体均提高，但 H1 ROI015 回退约 `0.25`；固定 50/50 可修复 ROI015，却把双折平均压到约 `+0.264`。这说明完整区域监督有价值，但 fresh context64 轨迹会重写一部分已有的跨 ROI 表征。

本轮只改变初始化：从既有 context0 base32 stage2 best 严格加载全部参数，再以 full-target context64 做一个固定 adaptation 阶段。模型、loss、数据、target halo 和正式推理都不变，不加入蒸馏或融合。

## H2 固定协议

- 验证 ROI：`ROI020,ROI022,ROI024`；
- 初始化：`runs/round11_h2_unet_cd68_stage2/best_global_mse.pt`；
- 初始化 SHA-256：`f4c15098ea7479c205eeb419ff5e152b8691bc15f801a7b0ccb30ee4cd5832e3`；
- 初始化模型：base32 residual U-Net，单 CD68，condition none；
- adaptation input / target：context64 384x384 / context64 384x384 全区域监督；
- epoch：14；batch/workers：`8/4`；seed `2026`；
- optimizer：新 AdamW，lr `1e-4` cosine 到 0，wd `1e-4`；
- loss：Uniform-7 hybrid `0.7/0.2/0.1`；
- 选模：global-MSE，JPEG q95；
- 正式评估：halo64+D4，q95 与 float；
- 输出：`runs/round25_h2_base_to_fulltarget_adapt14e/` 和 `runs/_round25_eval/base_to_fulltarget_h2.json`。

`1e-4` 与既有 stage2/continuation 协议一致，也是本轮唯一学习率。禁止搜索 lr、epoch、target halo、context halo、EMA、D4 consistency、loss、soup、融合或校准。

## Promotion gate

1. H2 q95 mean-image 与 global-MSE 相对 base32 均至少提高 `+0.25`；
2. ROI020、ROI022、ROI024 的两种 q95 聚合均不得回退超过 `0.05`；
3. float 两种总体增益必须为正，逐 ROI同样不得回退超过 `0.05`；
4. H2 通过后，在 H1 从 `runs/round17_h1_unet_base32_stage2/best_global_mse.pt` 严格初始化同样的 14-epoch adaptation；
5. H1/H2 四项 q95 增益必须各至少 `+0.15`、平均至少 `+0.40`，六个 ROI均不得回退超过 `0.05`，才允许 full-data；
6. 任一 gate 失败即关闭，不以 fresh full-target、模型融合或排行榜反馈救援。

## Full-data 边界

若双折通过，full-data adaptation 必须从当前强 full-data base checkpoint 启动；源 checkpoint、适配 epoch 和唯一提交包需另行预注册。不得把 H2/H1 checkpoint、训练标签统计或服务器分数用于 test ROI 特定选择。

## 启动条件

1. strict initialization 报告确认 source/target marker 和 model spec 完全一致；
2. Round22 full-target loader 和远端测试保持通过；
3. 本预注册提交、推送并同步服务器；
4. GPU1 资源检查通过，输出目录不存在，远端 tracked tree clean。

## 执行结果

H2 adaptation 完成全部 14 epoch。按预注册的 global-MSE 选模口径，最佳为 epoch 10：

- 裸验证 mean-image：`69.975664`；
- 裸验证 global-MSE：`69.345436`；
- 峰值显存：`4.323 GiB`；
- checkpoint：`runs/round25_h2_base_to_fulltarget_adapt14e/best_global_mse.pt`；
- SHA-256：`95dc42dc6eece8b9a17b950ca776d29169aa6ee8bd3aa5f345f00ca855ced180`。

固定使用该 checkpoint 运行 halo64+D4。完整原始结果保存为
`reports/round25_base_to_fulltarget_h2.json`。

| 口径 | 候选 mean-image | 相对 base32 | 候选 global-MSE | 相对 base32 |
|---|---:|---:|---:|---:|
| JPEG q95 | 70.212508 | +0.200442 | 69.575032 | +0.231486 |
| float | 70.285037 | +0.211572 | 69.646930 | +0.242591 |

逐 ROI 差值：

| ROI | q95 mean-image | q95 global-MSE | float mean-image | float global-MSE |
|---|---:|---:|---:|---:|
| ROI020 | +0.885845 | +0.850482 | +0.899007 | +0.863442 |
| ROI022 | +0.252344 | +0.309390 | +0.255945 | +0.313040 |
| ROI024 | -0.464424 | -0.369506 | -0.448087 | -0.353102 |

## 结论

本轮未通过 H2 gate：q95 两种总体增益均低于 `+0.25`，且 ROI024
四种口径均明显超过 `0.05` 的最大允许回退。裸验证总体提升主要来自
ROI020，不能解释为稳定的跨 ROI 泛化收益。

因此关闭 base checkpoint full-target adaptation 路线，不启动 H1，不训练
full-data，也不通过改选 epoch、学习率、融合或后处理救援本轮结果。

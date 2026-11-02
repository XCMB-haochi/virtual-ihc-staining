# CD68 第三轮优化记录

日期：2026-07-15

## 首投校准

首投 `cd68_unet_all8_context64_d4_loghalf_q95_plain.zip` 被平台完整识别：CD68、缺失图 0、异常图 0。平台显示 `SSIM=0.797`、`PSNR=25.067`，总分 `70.8233`，提交后排名 `7/19`。

总分验证了 `100 * (0.7 * SSIM + 0.3 * PSNR / 50)`，显示分量与总分的约 `0.0069` 差异来自备注只保留三位小数。plain 文件名已确认可用，无需再提交 `_fake` 包。

## 评估口径修正

原报告只计算逐图 PSNR 后平均。本轮新增第二种常见口径：先汇总全部像素的平方误差，再计算一次 PSNR。平台使用哪一种仍未确认，所有新报告同时保留：

- `psnr_mean_image` / `competition_score_mean_image_psnr`
- `psnr_global_mse` / `competition_score_global_mse_psnr`

旧留出模型在 `ROI020-ROI024`、halo64、D4、log-half、q95 下：

| 口径 | SSIM | PSNR | 综合分 |
|---|---:|---:|---:|
| 逐图 PSNR 后平均 | 0.800125 | 26.001733 | 71.6098 |
| 汇总 MSE 后计算 PSNR | 0.800125 | 24.697159 | 70.8270 |

后一总分与服务器 `70.8233` 接近，但分量不同，可能只是巧合，不能据此宣称已复现平台聚合。

## 验证偏差与困难 fold

测试 `ROI025-ROI029` 的 DAPI 均值全部在 `36.50-39.28`。原两组五 ROI 验证中包含四个均值约 `29` 的明显易样本，因此混合 fold 的 `71.61/71.94` 高估了隐藏测试表现。

本轮固定两个只含高均值 ROI 的压力测试：

- H1：`ROI015,ROI018,ROI019`，727 张
- H2：`ROI020,ROI022,ROI024`，816 张

这些 fold 用于比较相对增益，不作为服务器绝对分预测。

## 干净重训

旧权重是在 score-aligned checkpoint 选择代码完善前训练，旧日志中的 torch SSIM 与最终 scikit-image/JPEG 指标不可直接比较。本轮用当前代码重新训练相同 residual U-Net：14 epoch、seed 2026、完整 ROI 留出、q95 scikit-image 逐 epoch 验证，并按 global-MSE PSNR 综合分保存 `best.pt`。

这里的 H1/H2 是三 ROI 高亮度压力测试，而旧模型来自对应的五 ROI 留出训练。新 H1 训练额外使用 `ROI016/017`，新 H2 训练额外使用 `ROI021/023`，因此下表是“更多训练 ROI + 14 epoch + 新选模流程”的组合方案比较，不能把全部增益单独归因于 epoch 数或 checkpoint 选择。

无推理增强的最佳轮：

| Fold | 最佳 epoch | 逐图 PSNR 分 | global-MSE PSNR 分 |
|---|---:|---:|---:|
| H1 | 14 | 69.2879 | 68.6970 |
| H2 | 12 | 69.0317 | 68.2573 |

加入 halo64、D4、q95 后，在相同最终验证子集上的 log-half 比较：

| Fold | 旧模型逐图分 | 新模型逐图分 | 增益 | 旧模型 global 分 | 新模型 global 分 | 增益 |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 69.0916 | 70.3418 | **+1.2502** | 未记录 | 69.7264 | - |
| H2 | 69.5289 | 69.9298 | **+0.4009** | 68.8368 | 69.1837 | **+0.3469** |

组合方案增益在两个独立困难 fold 上复现，因此可以训练全数据新模型。H1 的旧 JSON 没有累计平方误差，不能从逐图 PSNR 均值准确还原 global-MSE PSNR，表中不猜测该值。

## 校准强度

新模型的增强结果：

| Fold | log-half 逐图/global | log-0.75 逐图/global | full-1.0 逐图/global |
|---|---:|---:|---:|
| H1 | 70.3418 / 69.7264 | 70.4941 / 69.8725 | **70.5573 / 69.9458** |
| H2 | 69.9298 / 69.1837 | **69.9448 / 69.2376** | 69.7754 / 69.1180 |

full 1.0 在 H1 最好但在 H2 回落，log 0.75 在两组均接近最优，因此全数据提交采用 log 0.75。该值来自训练标签与本地留出，不按隐藏测试 ROI 或排行榜反向拟合。

## AMP 数值修复

最终复核发现 Gaussian-11 和新增 uniform-7 SSIM 的局部矩计算原本位于 CUDA AMP 区域。真实训练图的 128 张样本上，旧半精度 Gaussian SSIM 平均比 float32 高 `0.0216`，最大偏差 `0.1199`，其中 3 张出现 `SSIM>1`。旧模型和上节 14 epoch 模型都经过该既有路径。

代码现强制两种 SSIM 的局部矩使用 float32，并加入低方差 CUDA AMP 回归测试。相同 H2、seed、14 epoch 的修复版结果：

| H2 halo64+D4+log0.75+q95 | 逐图 PSNR 分 | global-MSE PSNR 分 |
|---|---:|---:|
| 未修复训练 | 69.9448 | 69.2376 |
| float32 SSIM 矩 | **70.2182** | **69.6329** |
| 增益 | **+0.2735** | **+0.3953** |

ROI020、ROI022、ROI024 在两种口径下均上涨，因此全数据模型使用修复后的训练路径。H1 未重复训练修复版，不能宣称该项已跨两个 fold 复现；此前组合方案的跨 fold 证据与本项 H2 数值修复证据应分开理解。

## 最终优先提交

- 模型：`runs/unet_cd68_all_ampfix_14e/final.pt`
- 模型 SHA-256：`1E55A3CE94F1475D7CD2FAA2420471E097B1A3A749C1BB41CAD3926F830E30F2`
- 训练：全 25 ROI、6296 张、14 epoch、seed 2026、Gaussian-11 SSIM 主导损失，SSIM 局部矩强制 float32
- 推理：halo64、D4、log 0.75 ROI 校准、JPEG quality 95
- ZIP：`submissions/cd68_unet_all14_ampfix_context64_d4_log075_q95_plain.zip`
- ZIP SHA-256：`193B571F08A5695C18F2826AA88188AF22F9F8E6106F2D47B64D6E92AA4E55AD`
- ZIP 大小：11,433,543 bytes
- 校验：1346 个唯一 JPEG、缺失 0、多余 0、图像错误 0、CRC 正常、路径均为 `results/test/CD68/*.jpg`

该包是下一次优先提交。它尚无服务器分数，不能把本地 fold 增益直接当作榜单预期增益。

未修复的 `cd68_unet_all14_clean_context64_d4_log075_q95_plain.zip` 已降级，不应优先提交。首投模型的 raw/full/log0.75 受控包继续保留用于服务器消融。uniform-7 SSIM 训练模式已实现并通过与 scikit-image 的数值一致性测试，尚未完成模型消融，因此不进入当前提交。

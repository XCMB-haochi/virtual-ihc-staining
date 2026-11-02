# 虚拟 IHC 染色初步评估

日期：2026-07-14

## 结论

本赛题值得继续投入。当前数据没有明显的训练/测试复制泄漏，但存在可稳定复现的指标与工程优化空间。RTX 4060 Laptop 8GB 足以完成首轮 U-Net 实验，暂时不需要租用 GPU。

首投方案是单任务 CD68 U-Net + 64 px 真实邻域 halo + D4 TTA + 训练集拟合的 log-half ROI 均值缩放；服务器得分 `70.8233`，提交后排名 `7/19`。第三轮训练 14 epoch，将校准改为 log 0.75，并修复 CUDA AMP 下 SSIM 局部矩的半精度相消误差。该包服务器显示 `SSIM=0.803`、`PSNR=25.410`，排名升至 `5/19`；精确总分未提供。后续 Uniform-7 与独立 seed 集成见 `reports/optimization_round4.md`。

群内已确认 PSNR 最高 50，归一化值为 `PSNR / 50`。本地综合分按 `100 * (0.7 * SSIM + 0.3 * min(PSNR, 50) / 50)` 计算。plain 文件名协议已由首投确认；服务端 SSIM 实现和指标聚合细节仍未明确，因此本地分数用于相对比较，不视为排行榜精确复现。

## 数据审计

- 初赛数据为 Colon：6296 个训练 patch、1346 个测试 patch。
- 训练 `ROI000-ROI024`，测试 `ROI025-ROI029`，ROI 与文件名均无交集。
- 四个 target 与 DAPI 文件名严格一一对应，无缺配。
- 32,826 张图均为 `256x256` RGB JPEG 容器，但三个通道逐像素相同，实际为 8-bit 灰度。
- 未发现精确重复或明显近重复泄漏；跨集合 DAPI pHash 最小距离为 10。
- 相邻 patch 接缝 MAE 约 `3/255`，随机接缝约 `36-40/255`，文件名坐标可以重建连续 ROI。
- 所有原始 JPG 使用相同的 quality-75 量化表。
- DAPI 与 CD68 的像素、patch 均值、ROI 均值相关系数约为 `0.459/0.623/0.983`。

验证必须按完整 ROI 划分。随机 patch 切分会让相邻组织区域同时进入训练和验证，造成明显的数据泄漏。

## 经典基线

拟合仅使用 `ROI000-ROI019`，验证仅使用 `ROI020-ROI024`。

| Marker | 最佳廉价基线 | SSIM | PSNR |
|---|---|---:|---:|
| CD45RO | 256-bin 条件均值 LUT | 0.6445 | 21.12 |
| CD68 | 256-bin 条件均值 LUT | 0.6524 | 22.84 |
| HLA-DR | 256-bin 条件均值 LUT | 0.7004 | 20.44 |
| Vimentin | 256-bin 条件均值 LUT | 0.5835 | 18.82 |

暗背景使常数输出获得异常高的 SSIM，因此不能使用 DAPI identity 作为主要基线，也不能只凭视觉锐度选择模型。

## GPU 实验

下表均为 10 epoch、同一 ROI 划分、scikit-image 全量验证的未编码浮点输出。综合分采用已确认的 PSNR 归一化公式。

| 模型 | Marker | SSIM | PSNR | 综合分 |
|---|---|---:|---:|---:|
| 四任务 U-Net，SSIM 主导损失 | CD45RO | 0.7273 | 22.86 | 64.62 |
| 四任务 U-Net，SSIM 主导损失 | CD68 | 0.7893 | 25.28 | 70.42 |
| 四任务 U-Net，SSIM 主导损失 | HLA-DR | 0.7933 | 22.18 | 68.83 |
| 四任务 U-Net，SSIM 主导损失 | Vimentin | 0.7441 | 21.27 | 64.84 |
| 单任务 U-Net，SSIM 主导损失 | CD68 | **0.7951** | **25.55** | **70.99** |

结论：SSIM 主导损失优于 MSE 主导损失；若初赛确实指定 CD68，单任务模型优于共享四输出模型。四任务模型仍适合作为后续一对多创新与附加分方案。

模型输出重新编码后，quality 95 在完整验证集上略优于 quality 75 和 100。单任务 CD68 的 q95 结果为 `SSIM 0.7940 / PSNR 25.55 dB`。

## 推理与校准实验

仅使用 `ROI000-ROI019` 拟合 DAPI 到 target 的亮度关系，并在 `ROI020-ROI024` 评估。所有数值均按逐图 scikit-image 指标平均：

- patch 级平移与缩放均降低指标，应淘汰。
- ROI 级平移提高 PSNR但明显降低 SSIM，应淘汰。
- 固定 D4 TTA 将浮点综合分从 `70.9897` 提高到 `71.1366`。
- D4 输出保存为 q95 后为 `71.0528`，优于 q75 的 `70.8501` 和 q100 的 `71.0371`。
- D4 + ROI 级缩放 + q95 达到 `SSIM 0.7966 / PSNR 25.78 dB / 综合分 71.2317`。
- 将 ROI 校准缩到 log 空间半强度后，无 halo 为 `71.3645`；加入 64 px 真实邻域 halo 后为 `71.6098`。
- 第二 fold 中，无 halo D4 从 `71.2119` 提升到 log-half 的 `71.6299`；加入 halo 后进一步达到 `71.9357`，证明两项增益可迁移。

ROI 缩放在当前规则解释下属于固定输入推理：拟合系数只来自训练配对，测试时仅汇总 DAPI 与模型输出。它不使用隐藏标签或排行榜反馈拟合参数，现作为增强提交方案。

## 一手资料调研

- [VIRVS](https://www.nature.com/articles/s41597-025-05194-3) 及其 [MIT 代码](https://github.com/casus/virvs)：在多个 256x256 虚拟染色数据集上，确定性 U-Net 的 PSNR/SSIM 整体优于相同生成器的 pix2pix，支持本项目优先做像素回归。
- [MAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC11378090/)：DAPI/自发荧光到多 marker 联合生成，支持共享特征的多任务方向；其仓库不完整且无许可证，本项目不复制其代码。
- [PyramidPix2pix](https://openaccess.thecvf.com/content/CVPR2022W/CVMI/html/Liu_BCI_Breast_Cancer_Immunohistochemical_Image_Generation_Through_Pyramid_Pix2pix_CVPRW_2022_paper.html)：可借鉴多尺度像素损失，但首轮无需加入 GAN。
- [官方 PyTorch pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)：仅保留为后续消融，不进入当前主线。

GAN 和 diffusion 容易生成锐利但不存在的信号，并不适合当前 SSIM/PSNR 主导的低成本试错阶段。

## 下一阶段

1. 提交第二代 14 epoch 原名候选包，记录总分以及备注中的 SSIM、PSNR。
2. 用受控 raw/full/log0.75 包判断首投模型的校准迁移性，并向组委会确认 PSNR 聚合方式。
3. 若第二代仍明显落后榜首，再消融 uniform-7 SSIM 损失、正确训练 halo 或同架构新 seed 集成。
4. 不按排行榜反向拟合每个测试 ROI 的独立后处理参数。

当前不建议投入 GAN、diffusion、外部大模型或租用更大 GPU。

## 首份提交历史记录

使用全部 25 个训练 ROI、单任务 CD68 U-Net、8 epoch 固定训练预算、64 px 真实邻域 halo、D4 TTA、全训练 ROI 拟合的 log-half 均值缩放生成。全量 8 epoch 含约 6296 次参数更新，与开发划分 10 epoch 的约 6220 次更新相当，因此是合理的首投预算，但不代表已经证明收敛。

- 原名 ZIP：`submissions/cd68_unet_all8_context64_d4_loghalf_q95_plain.zip`
- `_fake` ZIP：`submissions/cd68_unet_all8_context64_d4_loghalf_q95_fake.zip`
- 图像：1346 张，全部为 `256x256` RGB 灰度 JPG
- ZIP 路径：分别为 `results/test/CD68/ROIxxx_rr_cc.jpg` 与 `results/test/CD68/ROIxxx_rr_cc_fake.jpg`
- JPEG quality：95
- ROI 上下文：相邻 tile 提供 64 px halo，缺失邻居使用中心反射填充
- ROI 校准：开启，仅由全训练配对拟合；`scale = sqrt(expected_mean / predicted_mean)`
- TTA：D4
- 原名 ZIP SHA-256：`6E3116DA01992B8207ED2DD5720F93F8F5BF997BC5AB6F77AD834B10D0CC331B`
- `_fake` ZIP SHA-256：`2C257DF799F2E01E928224512F443DF373DA209D1591AE031BA6D871601DBB3F`
- 模型：`runs/unet_cd68_all_8e/final.pt`
- 模型 SHA-256：`C0920BBEA248535599852E31EAE63308E0AD009B199602CB6E23806E102702F5`

两个包的像素完全相同，原本只用于确认文件名协议。平台已完整接受原名包，因此 `_fake` 包不再需要提交。

# CD68 第二轮优化记录

日期：2026-07-14

## 结论

本轮将稳健验证分从旧候选的 `71.2317` 提高到 `71.6098`。有效改动只有两项：使用同 ROI 相邻 tile 提供 64 px halo，以及把训练拟合的 ROI 均值校准缩到 log 空间半强度。两项收益均在 `ROI020-024` 与独立 `ROI015-019` fold 上复现。

当前方法仍不能证明达到榜首 `72.5557`。本地与服务器的 SSIM 实现及测试 ROI 分布不同，必须等待实际提交分数校准。

## 稳健增益

所有结果均为 quality-95 JPEG、逐图 scikit-image 指标平均后的比赛公式分数。

| Fold / 方案 | 分数 | 相对同 fold 基础方案 |
|---|---:|---:|
| ROI020-024，D4，无 halo、无校准 | 71.0528 | - |
| ROI020-024，D4，无 halo、linear-half ROI | 71.3645 | +0.3117 |
| ROI020-024，D4，halo64、无校准 | 71.2788 | +0.2260 |
| ROI020-024，D4，halo64、log-half ROI | **71.6098** | **+0.5570** |
| ROI015-019，D4，无 halo、无校准 | 71.2119 | - |
| ROI015-019，D4，无 halo、log-half ROI | 71.6299 | +0.4180 |
| ROI015-019，D4，halo64、无校准 | 71.4870 | +0.2751 |
| ROI015-019，D4，halo64、log-half ROI | **71.9357** | **+0.7238** |

校准形式：

```text
expected_mean = OLS(train ROI DAPI mean -> train ROI CD68 mean)
ratio = expected_mean / model_prediction_mean
scale = sqrt(ratio)
prediction = clip(prediction * scale, 0, 255)
```

该过程只使用训练标签拟合 OLS。测试时只读取公开 DAPI 和模型输出，不使用隐藏标签或排行榜反馈拟合参数。

## 已淘汰实验

| 实验 | 最佳结果 | 判定 |
|---|---:|---|
| halo64 微调 2 epoch | 70.8843 | 低于旧权重直接使用 halo 的 71.1434 |
| halo128，旧权重，无 TTA | 71.1557 | 仅比 halo64 高 0.0122，计算量明显更高 |
| 单层金字塔 L1 | 70.8188 | 低于 residual 基线 70.9085 |
| 单层金字塔 + EMA 0.999 | 70.5840 | 短训练预算下 EMA 滞后明显 |
| Dense U-Net block | 70.5080 | 参数更少但指标明显下降 |
| residual + pyramid 各 50% 融合 | fold1 +0.09；fold2 约 0 | 仅 3/5 确认 ROI 上涨，不保留 |
| LUT、gamma、threshold、blur | 未跨两个 fold 稳定超过半强度 ROI 校准 | 不扩大网格 |

## 最终候选

- 模型：`runs/unet_cd68_all_8e/final.pt`
- 推理：`context_halo=64`、D4、log-half ROI scale、JPEG q95
- 原名 ZIP：`submissions/cd68_unet_all8_context64_d4_loghalf_q95_plain.zip`
- `_fake` ZIP：`submissions/cd68_unet_all8_context64_d4_loghalf_q95_fake.zip`
- 原名 SHA-256：`6E3116DA01992B8207ED2DD5720F93F8F5BF997BC5AB6F77AD834B10D0CC331B`
- `_fake` SHA-256：`2C257DF799F2E01E928224512F443DF373DA209D1591AE031BA6D871601DBB3F`

两个 ZIP 的图像字节相同，只是归档内文件名是否包含 `_fake` 不同。

## 方法依据

- PyramidPix2pix 的 IHC 消融支持单个低频金字塔尺度，但本数据实测未超过基线：https://openaccess.thecvf.com/content/CVPR2022W/CVMI/html/Liu_BCI_Breast_Cancer_Immunohistochemical_Image_Generation_Through_Pyramid_Pix2pix_CVPRW_2022_paper.html
- 虚拟染色架构消融曾报告 Dense block 增益，但本数据实测不成立：https://pmc.ncbi.nlm.nih.gov/articles/PMC10201298/
- VIRVS 支持优先使用确定性 U-Net 而非 GAN，当前结果仍与这一结论一致：https://www.nature.com/articles/s41597-025-05194-3

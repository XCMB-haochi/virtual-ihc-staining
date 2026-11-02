# 优化记录 Round 39：Restormer representation-reset 双折筛查

立项：2026-07-22；完成：2026-07-23

## 立项依据与结论边界

Round38 的空间对齐分支相对 matched continuation 近似持平，说明它不能作为
成熟 U-Net 的即插即用增量，但不能排除独立初始化后重新形成表征的收益。
此前 NAFNet、ImageNet-1K ConvNeXt-Tiny U-Net 和局部 U-Net + Swin 分支均
做过 fresh `14+14` 训练；这些结果排除了若干具体配方，却没有测试纯粹的
确定性图像复原 Transformer，也没有给新骨干与成熟 base32 相同量级的训练
时钟。

本轮选择 [Restormer 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html)
及其 [MIT 官方实现](https://github.com/swz30/Restormer)作为结构依据。它使用
MDTA 和带 depth-wise convolution 的 GDFN，在全分辨率确定性复原任务上直接
优化像素损失，比 DiT/diffusion 更符合本赛题的配对回归与 PSNR/SSIM 指标。

本轮只作 H1/H2 内部筛查，不接触排行榜、不生成测试预测。它回答的是：在
相同输入、监督、增强、有效 batch 和长训练预算下，fresh Restormer 是否稳定
优于 fresh residual U-Net。它不直接声称未加 thumbnail conditioning 的模型
已经超过 Round35 部署基线。

## 固定模型

`restormer-small` 保留官方四级 encoder-decoder、MDTA、GDFN、skip 和
refinement 布局，固定为：

| 项目 | 值 |
|---|---|
| base dim | `32` |
| encoder/decoder blocks | `[2,3,3,4]` |
| refinement blocks | `2` |
| heads | `[1,2,4,8]` |
| GDFN expansion | `2.66` |
| LayerNorm | bias-free |
| 输入/输出 | 单通道 DAPI -> 单通道 CD68 |
| 输出 | sigmoid，head bias `-2` |
| 全局输入残差 | 删除；DAPI 与 IHC 不是同一成像模态 |
| 公开预训练 | 不使用 |

该缩放配置用于把参数量和有效训练预算控制在与 base32 U-Net 同一量级；若本轮
不能形成稳定机制增益，不以更大的官方 dim48 配置、DiT 或事后超参搜索救援。
实现后的精确参数量为 `6,707,543`，低于当前 conditioned base32 的约
`8.5M`，因此本轮不会把单纯参数量扩大误当作 Transformer 机制收益。

## 双折固定协议

机器可读协议为 `configs/round39_h1h2_restormer_screen.json`：

| 项目 | 固定值 |
|---|---|
| H2 | `ROI020,ROI022,ROI024` |
| H1 | `ROI015,ROI018,ROI019` |
| 输入/target | `context64/full-target64` |
| 模型 | fresh base32 residual U-Net；fresh Restormer-small |
| 训练 | 单一 70-epoch cosine，`3e-4 -> 1e-6` |
| 优化 | AdamW，weight decay `1e-4`，global grad clip `1.0` |
| loss | Uniform-7 hybrid `0.7 SSIM + 0.2 L1 + 0.1 MSE` |
| 有效 batch | 两者均为 `8` |
| micro-batch | U-Net `8x1`；Restormer `4x2` gradient accumulation |
| seed | `2039` |
| 选模 | 每 epoch q95 global-MSE；不保存逐 epoch rescue checkpoint |
| 完整评估 | halo64 + D4，同时记录 float 与 JPEG q95 |

70 epoch 对齐当前 Round31 base32 的累计训练时钟：Round26 fresh stage1 后，
依次经历 Round27、Round29、Round30、Round31 的四个 14-epoch 阶段，共
`5 * 14 = 70`。本轮使用单个长 cosine 而不是复制中途结构转换和多次重启，
因此它是匹配总更新预算的 fresh 收敛比较，不是旧轨迹的逐步复刻。

正式候选只有 Restormer direct 与固定 `50% fresh U-Net + 50% Restormer`。
禁止搜索模型宽度、block 数、学习率、epoch、seed、loss、融合权重、checkpoint
soup、逐 ROI 路由或输出校准。若 `4x2` 在 A100 预检 OOM，只允许在正式训练
前改为 `2x4`，保持有效 batch、其他配置和 gate 不变，并记录为资源适配。

## A100 预检与资源适配

提交任何正式训练结果前，在物理 GPU0 上用固定 H2 划分、64 张训练图和 32 张
验证图各跑一个 epoch。两种 Restormer 分解均保持有效 batch `8` 和每 epoch
`8` 次 optimizer step：

| 模型/分解 | 物理 GPU | 参数量 | epoch 时间 | PyTorch 峰值显存 |
|---|---:|---:|---:|---:|
| fresh U-Net `8x1` | 0 | 8,110,657 | 3.505 s | 4.294 GiB |
| Restormer `4x2` | 0 | 6,707,543 | 24.378 s | 18.428 GiB |
| Restormer `2x4` | 0 | 6,707,543 | 25.413 s | 9.309 GiB |
| Restormer `4x2` | 1 | 6,707,543 | 21.925 s | 18.428 GiB |

`2x4` 相对 GPU0 上的 `4x2` 耗时增加约 `4.2%`，峰值显存减少约 `49.5%`；
但不同 micro-batch 会改变数据增强随机数的消费顺序和 AMP 舍入轨迹，不能视为
逐步完全等价。原预注册的 `4x2` 已在两张物理 A100 上分别完整通过，正式启动
前两卡常驻显存约为 `3.75/5.75 GiB`，因此无需资源适配，正式训练保留 `4x2`。
`2x4` 结果只证明发生显存压力时存在后备配置，不参与模型或超参数选择。

服务器预检时完整测试为 `198 passed, 63 subtests passed`；`compileall`、pipeline
shell 语法和四条训练命令 dry-run 均通过。DataLoader worker 退出时会报告共享
NFS 临时目录 `Device or resource busy` 清理噪声，但训练进程返回 0，checkpoint
和指标完整；该现象不作为训练失败处理。

## 预注册 gate

mechanism gate 要求 Restormer direct 相对 fresh U-Net：

1. 完整六 ROI 与 test-like 范围的四项 q95 增益均至少 `+0.05`；
2. 完整与 test-like 的四项 float 增益全部为正；
3. H1/H2 两折的 q95 双指标均为正；
4. 任一 test-like ROI 的 q95 单项不低于 `-0.12`，双负 ROI 最多一个。

primary 固定 50/50 相对 fresh U-Net 的四项 q95 均须至少 `+0.12`，四项
float 全正、两折双正、ROI floor `-0.08`、双负 ROI 最多一个，并依赖
mechanism 通过。

若 primary 失败，Restormer direct 只有在四项 q95 均至少 `+0.20`、四项
float 全正、两折双正、ROI floor `-0.10`、双负 ROI 最多一个时，才允许进入
完整五折。两条 gate 都失败则关闭当前 Restormer-small 配方，不生成提交包。

## 官方指标澄清与评估兼容修复

正式训练完成后，组织方明确评分为逐图计算 RGB uint8 SSIM 与 PSNR，其中
SSIM 使用 `skimage.metrics.structural_similarity`、`data_range=255`、
`win_size=7`，PSNR 逐图按 `clip(PSNR/50, 0, 1)` 归一化，再对全部真实图像
平均。项目因此增加 `competition_score_official` 与逐图归一化 PSNR 诊断；本
数据集 CD68 target 的 RGB 三通道逐像素相同，所以当前单通道计算与复制为 RGB
后的官方 SSIM 数值等价。

澄清发生在训练和 gate 预注册之后。本轮仍使用原定 `best_global_mse.pt` 和两项
预注册 q95 gate，不根据结果改 checkpoint、阈值或融合权重；官方分数作为精确
诊断。后续新实验应直接以官方逐图分数选模。详细口径见
`docs/rule/organizer_clarifications_2026-07-23.md`。

第一次 OOF 启动在推理前被完整 25 ROI 保护检查拒绝，因为本协议故意只覆盖
H1/H2 六 ROI。协议增加显式 `roi_scope: subset`，评估器只过滤并校验预注册的
六 ROI；默认 `complete` 仍要求数据集和协议 ROI 完全一致。该兼容修复不修改
模型或预测。修复后服务器测试为 `202 passed, 63 subtests passed`，四个正式
训练目录均有 70 行 epoch 指标和非空 `best_global_mse.pt`。

## OOF 结果

GPU0 对 1543 张 H1/H2 验证图完成两模型 D4 OOF，耗时
`1878.791 s`（约 31 分 19 秒）。q95 结果如下；“官方分数”是组织方确认的逐图
公式，“global-MSE 分数”仅保留用于原预注册 gate。

| 范围 | 候选 | SSIM | PSNR | 官方分数 | global-MSE 分数 | 官方分数相对 U-Net |
|---|---|---:|---:|---:|---:|---:|
| 完整六 ROI | fresh U-Net | 0.800816 | 25.234748 | 71.197988 | 70.631337 | 0 |
| 完整六 ROI | Restormer direct | 0.789346 | 24.693438 | 70.070304 | 69.387593 | -1.127683 |
| 完整六 ROI | 50/50 | 0.800278 | 25.195727 | 71.136929 | 70.516860 | -0.061059 |
| test-like 五 ROI | fresh U-Net | 0.793783 | 25.091787 | 70.619912 | 70.071321 | 0 |
| test-like 五 ROI | Restormer direct | 0.780506 | 24.416970 | 69.285605 | 68.645444 | -1.334307 |
| test-like 五 ROI | 50/50 | 0.792221 | 24.943370 | 70.421479 | 69.830108 | -0.198433 |

逐折 q95 双指标增益进一步表明差异不是总体聚合造成：

| 候选 | H2 官方/global-MSE | H1 官方/global-MSE |
|---|---:|---:|
| Restormer direct | -0.705988 / -0.755906 | -1.601004 / -1.791004 |
| 50/50 | +0.099181 / +0.065063 | -0.240916 / -0.319636 |

50/50 在 H2 的小幅正增益不能跨折复现。其 test-like ROI 双指标增益为：
`ROI015=-1.529936/-1.454220`、`ROI019=-0.170494/-0.242897`、
`ROI020=+0.072079/+0.022468`、`ROI022=-0.131426/-0.145003`、
`ROI024=+0.338948/+0.293494`。Restormer direct 在这五个 ROI 上全部双负，
最差为 `ROI015=-3.949500/-3.816093`。

## Gate 与结论

mechanism、primary 和 direct diagnostic 三条 gate 全部失败，promotion 为
`allowed: false`。Restormer direct 不仅没有产生互补性收益，本身在两折均明显
弱于同预算 fresh U-Net；50/50 的 H2 局部收益被 H1 和完整/test-like 聚合反转，
且有三个 test-like ROI 双负，不能作为稳定集成成员。

因此关闭当前 `restormer-small` 配方，不扩展到五折、不生成测试预测或提交包。
本轮支持的可迁移结论是：在当前小样本、配对回归、70-epoch、无预训练条件下，
把 residual U-Net 换成同量级纯 Restormer 不能突破瓶颈；后续不再以 DiT 或更大
Transformer 作为默认方向。完整机器结果为
`reports/round39_h1h2_restormer_screen_oof.json` 与
`reports/round39_h1h2_restormer_screen_gate.json`，OOF SHA256 为
`38c90809b4ba1b96b0cea73e15ade8f50ca26ba4557f5febfd90402fc616e58d`。

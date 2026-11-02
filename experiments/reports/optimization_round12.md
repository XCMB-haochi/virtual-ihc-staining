# 优化记录 Round 12：原生 Context64 训练

日期：2026-07-17

## 启动条件与假设

本轮只在 Round 11 的 NAFNet-width32 未通过 H2 promotion gate 后启动，不与仍在运行的候选争用 GPU。

现有 U-Net 使用 context0 的 256x256 输入训练，但正式推理固定使用 halo64 的 384x384 输入。Round 6 的两颗 context64 continuation 在裸 H2 验证上分别提高 `+0.4597/+0.4942` 与 `+0.2939/+0.3052`，方向一致；此前服务器失败的两个候选都含 H2 留出模型，不是从全部训练 ROI 得到的原生 context64 模型。因此，本轮只检验一个问题：从头开始在正式推理尺寸上训练，能否稳定优于匹配的 context0 模型。

## H2 固定协议

| 项目 | 固定值 |
|---|---|
| 验证 ROI | `ROI020,ROI022,ROI024` |
| marker | `CD68` |
| seed | `2026` |
| 模型 | base32 residual U-Net |
| 候选训练输入 | context64，384x384 |
| 对照训练输入 | context0，256x256 |
| 第一阶段 | 14 epoch，AdamW，lr `3e-4` cosine，weight decay `1e-4` |
| 第二阶段 | 从第一阶段 best global-MSE 初始化新优化器，14 epoch，lr `1e-4` cosine |
| batch / workers | `8 / 4` |
| 损失 | Uniform-7 hybrid `0.7/0.2/0.1` |
| 选模 | CD68 global-MSE，JPEG q95 |
| 完整评估 | halo64 + D4 + q95 + raw |
| checkpoint | 保存每个 epoch，正式选模仍只使用 best global-MSE |

H2 对照直接复用与本协议完全匹配的 Round 11 产物：

- `runs/round11_h2_unet_cd68_stage1/`
- `runs/round11_h2_unet_cd68_stage2/`
- `runs/_round11_eval/unet_cd68_h2.json`

候选固定输出到：

- `runs/round12_h2_unet_context64_stage1/`
- `runs/round12_h2_unet_context64_stage2/`
- `runs/_round12_eval/unet_context64_h2.json`

不搜索 halo、batch、学习率、epoch、模型宽度、损失权重、快照组合或融合权重。第二阶段重新初始化优化器并重复固定 seed 的 shuffle/augmentation 序列；这一点与 Round 11 对照相同，因此不影响配对比较，但不把两阶段表述为连续的单一随机轨迹。

## Promotion gate

1. H2 完整管线下，候选相对对照的 mean-image 与 global-MSE 分都至少提高 `+0.15`；
2. ROI020、ROI022、ROI024 任一逐 ROI 分回退不得超过 `0.05`；
3. 通过 H2 后锁定全部设置，进入 H1：`ROI015,ROI018,ROI019`；
4. H1 为 context0 与 context64 各自重新训练完全匹配的两阶段模型，不复用不同 fold 的旧 checkpoint；
5. H1/H2 四个聚合增益必须全部为正，平均至少 `+0.30`，六个验证 ROI 均不得回退超过 `0.05`；
6. 只有通过第 5 条才训练全数据 context64；未通过则不搜索 context64 的局部变体，也不提交排行榜。

## 资源与解释边界

- 只使用 GPU1，不干预服务器上其他用户的进程；
- 正式训练前先用同一模型和输入尺寸做 batch 8 显存预检；若 OOM，则停止，不产生正式结果，先实现并测试梯度累积后再从头开始；
- context64 若失败，只否定该固定 base32 两阶段配方；不据此否定更大感受野或其他架构；
- 若通过双 fold，full-data 阶段再预注册 seed 数、融合方式与唯一提交包，避免依据同一验证 fold 搜索组合。

## 远程预检与状态

GPU1 上使用 512 张训练图、64 张验证图、batch 8、workers 4 做单 epoch context64 预检：

- 用时 `6.614` 秒；
- 峰值 PyTorch 显存 `4.292 GiB`；
- 8.11M 参数，AMP 正常；
- 主进程退出码 0，checkpoint 与 metrics 完整。

DataLoader workers 退出时在 NFS `TMPDIR` 上报告临时目录 `EBUSY` 清理噪声，但未影响训练、保存或退出码。正式训练继续使用 workers 4，并以 metrics、checkpoint、主进程退出码三者共同判断完成状态。

预检通过后，H2 正式两阶段训练按上述固定命令在 GPU1 顺序完成。第一、第二阶段单 epoch 均约 `48-50` 秒，第二阶段峰值 PyTorch 显存 `4.323 GiB`。

## H2 结果

第一阶段最佳为 epoch 13，裸 q95 为 `69.4628/68.7714`；第二阶段最佳为 epoch 1，裸 q95 为：

| 模型 | mean-image | global-MSE | context64 - context0 |
|---|---:|---:|---:|
| context0 baseline | 69.5464 | 68.8956 | - |
| context64 | 69.7594 | 69.1082 | +0.2130 / +0.2126 |

裸验证通过总体门槛，但预注册完整管线给出相反结论：

| 模型 | mean-image | global-MSE | context64 - context0 |
|---|---:|---:|---:|
| context0 baseline，halo64+D4+q95 | 70.0121 | 69.3435 | - |
| context64，halo64+D4+q95 | 69.8909 | 69.2267 | -0.1211 / -0.1169 |

完整管线逐 ROI 增益：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI020 | +0.1028 | +0.1094 |
| ROI022 | -0.2074 | -0.2034 |
| ROI024 | -0.2420 | -0.2278 |

context64 在不使用 TTA 的裸 q95 选模指标上提高约 `+0.21`，但完整 D4 推理后总体反而下降，ROI022 与 ROI024 也明显超过 `-0.05` 回退上限。该候选未通过 H2：不进入 H1、不训练 full-data、不搜索 halo、epoch soup 或 TTA 子集、不生成排行榜提交。

这次对照表明，原生 context64 训练并未稳定解决训练/推理尺寸不一致；至少对当前 base32 和固定 D4 路径，context0 模型从 D4/halo 获得的收益更大。后续若重新研究上下文，应把 TTA 一致性纳入训练目标或独立预注册，而不是根据本折结果选择变换。

完整评估证据已同步为 `reports/unet_context64_h2.json`；匹配对照为 `reports/unet_cd68_h2.json`。

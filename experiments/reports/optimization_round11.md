# 优化记录 Round 11：联合监督与正式 NAFNet

日期：2026-07-17

## 目标与边界

当前服务器最佳分为 `72.0437`。Round 11 只研究不依赖测试标签、排行榜反馈或逐测试 ROI 选择的模型改进。所有候选先在完整训练 ROI 留出 fold 上比较；未通过预注册门槛时不训练全数据、不生成提交包。

远程服务器有两张 A100-PCIE-40GB。两卡用于并行运行独立候选或 fold，不先实现 DDP。启动任务前检查 GPU 进程和利用率，不终止或干扰其他用户进程。长任务使用 GNU Screen，日志、缓存、环境和 checkpoint 全部位于 `/workspace`。

## 数据依据

四个 IHC 目标与 DAPI 各有 `6296` 张图、`25` 个 ROI，文件名集合完全一致。逐图去除全局均值后，同坐标抽样像素的平均 Pearson 相关为：

- CD68 与 CD45RO：`0.549`；
- CD68 与 HLA-DR：`0.481`；
- CD68 与 Vimentin：`0.336`。

这些相关性只支持优先测试联合监督，不预设它一定改善 CD68。HistoPlexer 报告多蛋白联合预测相对逐标记训练稳定提高 PSNR 与 MS-SSIM，见[论文](https://www.nature.com/articles/s42256-025-01074-y)。本项目不复制其代码，只在现有自有训练管线中独立验证多任务思想。

## H2 固定初筛

共同设置：

| 项目 | 固定值 |
|---|---|
| 验证 ROI | `ROI020,ROI022,ROI024` |
| seed | `2026` |
| 训练输入 | context0，256x256 |
| 第一阶段 | 14 epoch，AdamW，lr `3e-4` cosine，weight decay `1e-4` |
| 第二阶段 | 从第一阶段 best global-MSE 初始化新优化器，14 epoch，lr `1e-4` cosine |
| 有效 batch | `8` |
| 损失 | Uniform-7 hybrid `0.7/0.2/0.1` |
| 选模 | CD68 global-MSE，JPEG q95 |
| 完整评估 | halo64 + D4 + q95 + raw |
| 禁用项 | ROI scale、测试 ROI 特定规则、榜单反馈调参 |

DataLoader worker 数只由正式训练前的固定吞吐预检决定；匹配的单输出与多输出 U-Net 使用相同 worker 数和独立固定 shuffle generator。若 NAFNet 的 batch 8 无法放入单张 40GB A100，先补梯度累积以维持有效 batch 8，再开始正式比较。

现有服务器强基线本身来自低学习率 continuation，因此 promotion 只比较三个模型各自的第二阶段。该固定延长也避免 NAFNet 大量零初始化 block scale 在 14 epoch 短程内尚未展开而被过早否定。官方 NAFNet 使用更长的 200k-iteration、`1e-3` 学习率配方，但本轮不单独改其优化器；统一两阶段设置优先隔离架构/联合监督因素。若 NAFNet 在第二阶段末轮仍持续改善但未过门槛，只记录“当前预算下未收敛”，不据此宣称架构无效，也不临时搜索学习率。

### 对照 A：单输出 residual U-Net

- base32 residual U-Net；
- 输出仅 CD68；
- 作为同环境、同 commit、同 seed 的服务器基线重新训练。

### 候选 B：四输出 residual U-Net

- 骨干与对照 A 完全相同；
- 输出顺序：`CD68,CD45RO,HLA-DR,Vimentin`；
- 总损失固定为 `(L_CD68 + mean(L_CD45RO,L_HLA-DR,L_Vimentin)) / 2`；
- 等价通道权重：CD68 `0.5`，三个辅助标记各 `1/6`；
- 不搜索辅助权重，不增加额外损失或后处理。

### 候选 C：NAFNet-width32

- 单输出 CD68，随机初始化；
- width `32`；
- encoder blocks `[1,1,1,28]`；
- middle blocks `1`；
- decoder blocks `[1,1,1,1]`；
- 删除复原任务常用的 `output + input` 全局残差，使用独立 sigmoid 输出；
- 输出层偏置与 U-Net 对照一致，固定初始化为 `-2.0`；
- 不搜索宽度、块数或预训练权重。

结构依据为 [NAFNet 论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf)及其 [MIT 官方仓库](https://github.com/megvii-research/NAFNet)。Round 10 的临时 NAF-style U-Net 没有采用官方低分辨率集中堆叠布局，不能替代本轮正式对照。

## Promotion gate

1. H2 完整管线下，候选相对对照 A 的 mean-image 与 global-MSE 分都至少提高 `+0.15`；
2. ROI020、ROI022、ROI024 任一逐 ROI 分回退不得超过 `0.05`；
3. 通过 H2 后锁定全部超参数，进入 H1：`ROI015,ROI018,ROI019`；
4. H1 重新训练匹配的单输出 baseline，不复用不同训练范围的旧模型；
5. H1/H2 四个聚合增益必须全部为正，平均至少 `+0.30`，六个验证 ROI 均不得回退超过 `0.05`；
6. 只有通过第 5 条的候选才训练全数据；若两个候选都通过，只额外评估一次固定 `50/50` 浮点预测融合，不搜索权重；
7. 若目标是单独填补当前与 `72.5557` 的历史榜首差距，跨 fold raw 平均增益应尽量达到 `+0.60`，因为过去本地增益只兑现约 32%-38%。

## 远程资源预检

在 GPU1 上以 512 张训练图、64 张验证图、batch 8 运行单轮预检：

| 模型/设置 | 秒/epoch | 峰值显存 |
|---|---:|---:|
| 单输出 U-Net，workers 0 | 6.001 | 1.973 GiB |
| 单输出 U-Net，workers 4 | **4.954** | 1.973 GiB |
| 单输出 U-Net，workers 8 | 5.438 | 1.973 GiB |
| 四输出 U-Net，workers 4 | 5.890 | 1.982 GiB |
| NAFNet-width32，workers 4 | 19.308 | 6.282 GiB |

因此正式实验固定 workers 4。NAFNet batch 8 可直接放入单张 40GB A100，不需要梯度累积。GPU0 同期有其他用户任务和约 50%-60% 利用率，未使用；所有正式实验在空闲的 GPU1 顺序执行。

## 四输出 U-Net H2 结果

两个模型都完成固定两阶段训练。第二阶段按 global-MSE 选中的 checkpoint，其裸 q95 结果为：

| 模型 | mean-image | global-MSE |
|---|---:|---:|
| 单输出 baseline | 69.5464 | 68.8956 |
| 四输出联合监督 | 69.4499 | 68.8272 |
| 联合监督 - baseline | -0.0965 | -0.0684 |

预注册完整管线 `halo64 + D4 + q95 + raw`：

| 模型 | mean-image | global-MSE | 联合监督 - baseline |
|---|---:|---:|---:|
| 单输出 baseline | 70.0121 | 69.3435 | - |
| 四输出联合监督 | 69.9812 | 69.3648 | -0.0308 / +0.0212 |

raw 逐 ROI 增益：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI020 | +0.5528 | +0.5692 |
| ROI022 | -0.0339 | +0.0304 |
| ROI024 | -0.5529 | -0.4637 |

联合监督明显改善 ROI020，但严重伤害 ROI024；总体两种聚合也都未达到 `+0.15`。该候选未通过 H2：不进入 H1、不搜索辅助权重、不训练全数据。

## NAFNet-width32 H2 结果

第一阶段和第二阶段各完成 14 epoch。第一阶段最佳为 epoch 10，裸 q95 为 `69.4070/68.7793`；第二阶段最佳同样为 epoch 10，裸 q95 为 `69.3812/68.7553`。正式比较使用预注册的第二阶段 best global-MSE checkpoint：

| 模型 | mean-image | global-MSE | NAFNet - baseline |
|---|---:|---:|---:|
| 单输出 baseline，裸 q95 | 69.5464 | 68.8956 | - |
| NAFNet，裸 q95 | 69.3812 | 68.7553 | -0.1652 / -0.1403 |
| 单输出 baseline，halo64+D4+q95 | 70.0121 | 69.3435 | - |
| NAFNet，halo64+D4+q95 | 70.0539 | 69.4121 | +0.0418 / +0.0686 |

完整管线逐 ROI 增益：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI020 | +0.8448 | +0.8061 |
| ROI022 | +0.0218 | +0.0819 |
| ROI024 | -0.6617 | -0.5757 |

NAFNet 的完整管线总体略优于 U-Net，但两种聚合都未达到 `+0.15`，且 ROI024 严重超过 `-0.05` 回退上限。它与四输出 U-Net 呈现相似的 ROI020/ROI024 分化，因此未通过 H2：不进入 H1、不搜索宽度或学习率、不做额外排行榜提交。

两阶段单 epoch 约 `119-121` 秒，第二阶段峰值 PyTorch 显存 `6.346 GiB`。末轮验证没有继续上升，best checkpoint 位于 epoch 10，因此本结果不标记为“固定预算下仍未收敛”。官方更长训练配方仍可能改变结论，但现有跨 ROI 负迁移不足以支持十小时级救援实验。

## 审计与结论

- 远端多任务两个 config 均确认 marker 顺序为 `CD68,CD45RO,HLA-DR,Vimentin`，归一化权重为 `0.5,1/6,1/6,1/6`；
- 三份完整评估均确认只包含 `ROI020,ROI022,ROI024`，共 816 张图；
- 两阶段各自重置 seed，因此第二阶段重复第一阶段的 shuffle/augmentation 序列；匹配对照一致，不影响配对结论，但它不是单一连续随机轨迹；
- NAFNet 的全局池化使 halo64 不只影响边缘，所以失败结论限定为当前 context0 训练、halo64 推理和固定短预算配方，不泛化为 NAFNet 架构无效；
- CUDA 卷积尚未承诺逐位确定性，H2 只作初筛；本轮两个候选均未达到进入第二 fold 的门槛；
- 完整评估证据已同步为 `reports/unet_cd68_h2.json`、`reports/unet_multitask_h2.json` 与 `reports/nafnet_h2.json`；
- Round 11 已结束，不训练全数据模型，也不生成提交包。下一项固定实验为原生 context64 U-Net。

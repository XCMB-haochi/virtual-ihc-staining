# 优化记录 Round 13：多任务预训练后单任务微调

日期：2026-07-17

## 假设与公平性

Round 11 的四输出模型在 ROI020 上相对单输出 baseline 提高约 `+0.55`，但在 ROI024 上下降约 `-0.55`。这说明辅助 marker 改变了共享表征，但持续联合优化产生跨 ROI 负迁移。本轮只检验一个问题：先完成等预算多任务预训练，再移除辅助输出并做 CD68-only 微调，能否保留有用表征并消除联合目标冲突。

不能直接使用两路 Round 11 `best_global_mse.pt`：单任务 stage2 best 为 epoch 1，多任务 stage2 best 为 epoch 10，实际更新时钟不同。远端 metrics 核验如下：

| run | stage1 best epoch | stage2 best epoch |
|---|---:|---:|
| 单任务 U-Net | 10 | 1 |
| 四输出 U-Net | 10 | 10 |

因此两路都固定从 stage2 `last.pt` 初始化。它们都经历 stage1 epoch 10 加 stage2 14 epoch，即 `24 * 685 = 16,440` 次 H2 更新；不恢复 optimizer 或 scheduler。

## 初始化实现

新增显式 `--init-marker-subset`：

- 默认不启用时，`--init-checkpoint` 继续要求 marker 元数据完全一致并严格加载；
- 启用时只允许模型规格和 conditioning 完全一致的 U-Net；
- 按 marker 名称复制/重排 `head.weight` 与 `head.bias`，共享层严格逐 tensor 复制；
- 缺失或重复 marker、错误 head shape、状态 key 或模型规格不一致均拒绝；
- source/target marker、通道索引和复制 tensor 数写入 config 与 checkpoint 供审计。

本轮候选从四输出 checkpoint 的 source channel 0 取得 CD68 head，迁移后与对照均为相同的 8,110,657 参数单输出 U-Net。

## H2 固定协议

| 项目 | 固定值 |
|---|---|
| 验证 ROI | `ROI020,ROI022,ROI024` |
| marker | 仅 `CD68` |
| seed | `2026` |
| 模型 | base32 residual U-Net，condition none |
| 输入 | context0，256x256 |
| 微调 | 14 epoch，AdamW，新 optimizer，lr `5e-5` cosine 到 0，weight decay `1e-4` |
| batch / workers | `8 / 4` |
| 损失 | Uniform-7 hybrid `0.7/0.2/0.1` |
| 训练选模记录 | CD68 global-MSE，JPEG q95；保存每个 epoch，仅作诊断 |
| 主比较 checkpoint | 两路各自固定 epoch 14 `last.pt` |
| 完整评估 | halo64 + D4，同时记录 float/raw 与 JPEG q95 |

对照：

- 初始化：`runs/round11_h2_unet_cd68_stage2/last.pt`
- 输出：`runs/round13_h2_cd68_control_continue14e/`

候选：

- 初始化：`runs/round11_h2_unet_multitask_stage2/last.pt`
- 使用 `--init-marker-subset` 投影到单输出 CD68
- 输出：`runs/round13_h2_multitask_to_cd68_continue14e/`

两路各增加 `14 * 685 = 9,590` 次更新，最终历史均为 26,030 次更新。`5e-5` 是唯一学习率：它是上一阶段 `1e-4` 的减半 continuation，也缓和 CD68 损失权重从多任务的 `0.5` 切换为单任务 `1.0` 后的梯度变化。不搜索其他学习率、epoch 或冻结层数。

## H2 Promotion gate

1. 主比较只使用两路固定 epoch 14 `last.pt` 的 q95 结果；best checkpoint 只作诊断，不进入 gate、融合或提交；
2. 候选相对等时钟对照的 q95 mean-image 与 global-MSE 分都至少提高 `+0.15`；
3. 三个 ROI 的 q95 两种聚合均不得回退超过 `0.05`；
4. float/raw 两种总体增益必须均为正，逐 ROI也不得回退超过 `0.05`；
5. 候选 q95 两种总体分还必须都超过现有可部署 H2 基线 `70.0121/69.3435`；该项只判断实用价值，不作因果解释；
6. 任一条件失败即停止，不搜索 checkpoint soup、融合、辅助权重或排行榜反馈。

若通过 H2，H1 使用 `ROI015,ROI018,ROI019` 从头重建两条等时钟轨迹。两路 stage1 都固定取 14-epoch 训练轨迹的 epoch 10，stage2 固定取 epoch 14，CD68-only 微调固定取 epoch 14，以复现 H2 的 `10 + 14 + 14` 更新时钟。H1/H2 四个 q95 聚合增益必须全部为正，平均至少 `+0.30`，六个 ROI 均不得回退超过 `0.05`，才允许进入 full-data。

## 启动条件

1. marker-subset 定向测试、compileall 和 diff check 通过；
2. 远端干净 Python/PyTorch 环境完成全量测试；
3. 新代码 commit 已同步到服务器且工作树代码与 commit 一致；
4. GPU1 无本项目遗留任务，且不干预其他用户进程；
5. 候选小样本预检确认 config 中 CD68 映射为 source 0 -> target 0。

## 工程验证状态

- 本地 marker-transfer 与训练安全定向测试：`17/17` 通过；
- 本地 compileall 与 `git diff --check` 通过；
- 远端 Python 3.12 / PyTorch 2.8 环境全量测试：`74/74` 通过；
- 远端 compileall、CLI `--help` 与 `git diff --check` 通过；
- 本机 Anaconda 的 NumPy/MKL 在既有 `lstsq` 测试处发生 native abort，因此全量结果以远端干净环境为准；该缺口已由远端全量测试覆盖。

小样本迁移预检进一步确认：

- source markers 为 `CD68,CD45RO,HLA-DR,Vimentin`；
- CD68 映射为 source 0 -> target 0；
- 复制 71 个共享 tensor，按通道切片 `head.weight/head.bias`；
- 迁移后参数量为 8,110,657，峰值 PyTorch 显存 `2.064 GiB`；
- checkpoint runtime commit 为 `ceb7676`。

## H2 结果

两路训练都完成固定 14 epoch，单 epoch 约 `31-33` 秒，峰值 PyTorch 显存均为 `2.064 GiB`。固定末轮裸 q95：

| 模型 | mean-image | global-MSE | 候选 - control |
|---|---:|---:|---:|
| 单任务 control | 69.3048 | 68.6731 | - |
| 多任务转 CD68 | 69.1488 | 68.5308 | -0.1560 / -0.1424 |

完整 `halo64 + D4`：

| variant | control | 候选 | 候选 - control |
|---|---:|---:|---:|
| q95 mean-image | 69.8913 | 69.6770 | -0.2143 |
| q95 global-MSE | 69.2506 | 69.0675 | -0.1832 |
| float mean-image | 69.9627 | 69.7528 | -0.2099 |
| float global-MSE | 69.3215 | 69.1427 | -0.1789 |

逐 ROI 增益全部为负：

| ROI | q95 mean | q95 global | float mean | float global |
|---|---:|---:|---:|---:|
| ROI020 | -0.2380 | -0.1993 | -0.2329 | -0.1943 |
| ROI022 | -0.2448 | -0.2214 | -0.2424 | -0.2190 |
| ROI024 | -0.1645 | -0.1428 | -0.1589 | -0.1372 |

候选同时低于等时钟 control 和现有可部署 H2 基线 `70.0121/69.3435`。它未通过任何 promotion 条件：不进入 H1，不搜索学习率、冻结层数、辅助权重、best checkpoint、soup 或融合，不生成排行榜提交。结论限定为当前四任务等权分组预训练加固定 14-epoch CD68 微调；marker-subset 功能保留为通用、已测试的实验工具。

完整评估证据为 `reports/round13_control_h2.json` 与 `reports/round13_candidate_h2.json`。

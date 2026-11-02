# 优化记录 Round 16：base48 epoch-level GroupDRO

日期：2026-07-17

## 立项依据

Round 11-15 的多任务、NAFNet、context64、base48 与 ImageNet 预训练 ConvNeXt 都未通过逐 ROI gate。base48 和 ConvNeXt 尤其一致：ROI020 大幅提高，但 ROI024 明显下降。现有证据更符合 pooled objective 在训练域之间产生风险取舍，而不是参数量、上下文或公开预训练不足。

本轮只改变训练目标，不改变模型、增强、更新预算或推理管线。选择 epoch-level GroupDRO，而不是 V-REx：batch 8 不足以稳定同时覆盖 22 个训练 ROI，minibatch ROI 方差容易退化成样本难度方差；严谨 V-REx 还需分组采样、penalty warm-up 和额外系数搜索。

## 固定算法

训练组是当前 fold 的全部训练 ROI。对 ROI `g` 定义：

- `p_g = count_g / N`，即截断后的实际训练 tile 比例；
- 每个 stage 的 `q_g(0) = 1 / G`，不从 checkpoint 或上一 stage 继承 adversary 状态；
- 一个 epoch 内固定逐样本权重 `w_g = q_g / p_g`；
- 训练损失为 `mean(w_group(i) * hybrid_loss_i)`，因此其期望为 `sum_g q_g R_g`。

每个 epoch 的 optimizer 更新结束后，使用当前模型对同一训练折做一次确定性风险统计：

1. `augment=False`、`shuffle=False`、无反传、float32；
2. 使用同一 Uniform-7 hybrid per-sample loss；
3. 按 ROI 汇总全量平均风险 `R_g`；
4. 固定 `eta=5.0`，执行 `log_q_g <- log_q_g + eta * (R_g - mean(R))`；
5. 使用 softmax 得到下一 epoch 的 `q`。

减去公共均值只改善数值稳定性，不改变 softmax。全训练折统计让每次 adversary 更新都覆盖全部训练 ROI，同时保持原随机 DataLoader、batch、样本覆盖和 optimizer 更新次数不变。

每轮必须记录训练 ROI 的 `R_g`、更新前后 `q_g`、`q_g/p_g` 范围及 `1/sum(q_g^2)`。任何缺组、非有限风险、非正/非归一化 `q` 都立即失败。GroupDRO 不与 `test-mean-rbf`、marker weighting、EMA 或非 hybrid loss 组合。

## H2 固定协议

| 项目 | 固定值 |
|---|---|
| 验证 ROI | `ROI020,ROI022,ROI024` |
| marker | `CD68` |
| 模型 | base48 residual U-Net，condition none |
| seed | `2026` |
| 输入 | context0，256x256 |
| batch / workers | `8 / 4` |
| 损失 | Uniform-7 hybrid `0.7/0.2/0.1` + epoch-level GroupDRO |
| GroupDRO | `eta=5.0`，每 stage 从均匀 `q` 开始 |
| stage1 | 14 epoch，AdamW，lr `3e-4` cosine，weight decay `1e-4` |
| stage2 | 从 stage1 epoch 14 初始化新 optimizer；14 epoch，lr `1e-4` cosine，`q` 重置均匀 |
| 固定候选 | stage2 epoch 10，不使用候选自己的 best |
| 完整评估 | halo64 + D4，同时记录 JPEG q95 与 float/raw |

固定 stage1 epoch 14 和 stage2 epoch 10，是为了与 Round 14 base48 的实际轨迹严格匹配。候选与标准 base48 的唯一本质差异必须是 robust objective；即使 GroupDRO 的其他 epoch 更高，也不得替换正式候选。

固定输出：

- `runs/round16_h2_unet_base48_groupdro_stage1/`
- `runs/round16_h2_unet_base48_groupdro_stage2/`
- `runs/_round16_eval/unet_base48_groupdro_h2_epoch10.json`

对照证据：

- base32：`reports/unet_cd68_h2.json`；
- 标准 base48：`reports/unet_base48_h2.json`；
- Round 14 协议与结果：`reports/optimization_round14.md`。

不搜索 `eta`、初始化 `q`、风险平滑、权重裁剪、采样器、模型宽度、epoch、checkpoint、loss、TTA 或融合。

## Promotion gate

1. 相对 base32，完整 `halo64 + D4 + q95` 的 mean-image 与 global-MSE 都至少提高 `+0.15`；
2. 相对 base32，ROI020、ROI022、ROI024 的两种 q95 聚合均不得回退超过 `0.05`；
3. float/raw 两种总体增益都必须为正，逐 ROI 同样不得回退超过 `0.05`；
4. 该门槛要求候选相对标准 base48 至少修复 ROI024 约 `+0.4369/+0.3330`，同时总体 q95 最多只能损失约 `0.0577/0.0911`；
5. H2 通过后，H1 `ROI015,ROI018,ROI019` 重新训练固定时钟的 base32、标准 base48 和 GroupDRO base48；
6. H1/H2 四个 q95 聚合增益必须全部为正、平均至少 `+0.30`，六个 ROI 均不得回退超过 `0.05`，才进入 full-data。

若 H2 任一 gate 失败，或者只有非 epoch10 checkpoint 通过，则本轮淘汰；不在 H2 搜索 `eta`、切换 V-REx、做 checkpoint soup、融合或 ROI 特定救援。

## 启动条件

1. per-sample loss 与旧 scalar/weighted loss 的数值和梯度等价测试通过；
2. `q/p` 的组总贡献、非等组大小、风险平移不变性和高风险组增权测试通过；
3. 缺组、非有限风险、无效 `q` 和不兼容参数均 fail closed；
4. legacy `robust-objective=none` 路径保持不变；
5. 本地定向测试与远端全量测试、compileall、CLI 和 diff check 通过；
6. 代码 commit 推送并同步服务器，正式输出目录不存在，GPU1 无本项目并行任务。

## 实现与验证状态

- `CompetitionLoss.per_sample()` 供 GroupDRO 加权训练和风险遍历共用；legacy 无权重 scalar 路径保留原归约顺序；
- `p`、精确 group counts 与 risk loader 都使用 post-limit `train_names`；同一 ROI 内漏图或重图也会 fail closed；
- risk loader 使用独立 generator，不消耗全局或训练 shuffle RNG；
- risk pass 使用 online model、eval/inference mode、关闭 autocast、float32 前向与 loss、CPU float64 有序汇总，并恢复原 model mode；
- q update 与 `q/p` 使用 CPU float64，转 float32 后非有限或非正立即失败；
- metrics 区分 q before/after、实际训练使用的 q/p、风险、精确计数、范围和 inverse-Simpson effective groups；
- 本地全量测试 `94/94`、GroupDRO/loss 定向测试 `10/10`、compileall、CLI 与 `git diff --check` 均通过；
- 独立代码审查未发现高、中风险缺陷；固定 epoch 由启动命令显式检查 checkpoint 内部 `epoch` 字段。

## 结果

两阶段均完整跑满 14 epoch，固定 checkpoint 的内部 epoch 字段分别核验为 14 和 10：

| 阶段 | 固定 epoch | 裸 q95 mean / global | 总用时 | 平均每 epoch | 峰值显存 | effective groups（更新后） |
|---|---:|---:|---:|---:|---:|---:|
| stage1 | 14 | `69.0066 / 68.3420` | `735.8 s` | `52.56 s` | `3.084 GiB` | `4.59` |
| stage2 | 10 | `69.5714 / 68.9381` | `755.6 s`（14 epoch） | `53.97 s` | `3.155 GiB` | `7.63` |

stage1 的 global-MSE 最佳为 epoch13 的 `69.2491/68.5627`，但协议固定使用 epoch14。stage2 的最佳 global-MSE 恰好就是固定 epoch10；epoch14 已下降到 `69.3315/68.7111`，不存在靠更晚 checkpoint 救援的证据。

固定 stage2 epoch10 的完整 `halo64 + D4` 总体结果：

| 输出 | base32 mean / global | GroupDRO base48 mean / global | 相对 base32 | 相对标准 base48 |
|---|---:|---:|---:|---:|
| JPEG q95 | `70.0121 / 69.3435` | `70.1247 / 69.4853` | `+0.1126 / +0.1418` | `-0.0951 / -0.0993` |
| float/raw | `70.0735 / 69.4043` | `70.1999 / 69.5600` | `+0.1265 / +0.1556` | `-0.0975 / -0.1017` |

逐 ROI 相对 base32：

| ROI | JPEG q95 mean / global | float/raw mean / global |
|---|---:|---:|
| ROI020 | `+0.7859 / +0.7451` | `+0.7970 / +0.7561` |
| ROI022 | `+0.0589 / +0.0971` | `+0.0633 / +0.1015` |
| ROI024 | `-0.4427 / -0.3456` | `-0.4175 / -0.3202` |

相对标准 base48，GroupDRO 在 ROI024 仅修复：

- JPEG q95：`+0.0441/+0.0374`；
- float/raw：`+0.0441/+0.0374`。

这远低于通过 gate 所需约 `+0.4369/+0.3330`，并同时损失约 `0.10` 总体分。原始完整评估证据为 `reports/unet_base48_groupdro_h2_epoch10.json`。

## 决策

Round 16 淘汰，不启动 H1：

1. q95 总体增益 `+0.1126/+0.1418`，两种聚合均未达到 `+0.15`；
2. ROI024 在 q95 与 float/raw 的两种聚合中仍回退 `0.32-0.44`，明确违反逐 ROI gate；
3. GroupDRO 只小幅缓解标准 base48 的 ROI024 回退，没有改变“ROI020 大幅提高、ROI024 明显下降”的核心 trade-off；
4. 两个 stage 的 adversary 最终都集中到约 5 个有效组，固定 `eta=5.0` 的本配方没有转化为验证 ROI 稳健性。

按预注册停止：不搜索 `eta`、不切换 V-REx、不使用非固定 checkpoint、不做 soup、融合或 ROI 特定救援。该结论只淘汰当前 base48 + epoch-level GroupDRO 配方，不泛化为所有分布鲁棒方法都无效。

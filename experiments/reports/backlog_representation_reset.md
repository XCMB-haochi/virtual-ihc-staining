# 条件式后备方向：representation reset 公平比较

记录日期：2026-07-22

状态：**Round38 失败后已按独立预注册的 Round39 双折筛查立项。**

Round38 后更新：其 machine gate 已失败，aligned fusion 相对 matched control
近似持平，因此本方向已满足“可以重新评估是否立项”的触发条件；这仍不等于
自动批准训练，下一轮候选、预算和 gate 必须在执行前独立固定。

Round39 已完成上述独立固定：fresh U-Net 与 fresh Restormer-small 使用相同
H1/H2 数据、监督、增强、有效 batch 和 70-epoch 时钟比较；具体配置与 gate
见 `configs/round39_h1h2_restormer_screen.json` 和
`reports/optimization_round39.md`。本文件以下内容保留为立项时的决策依据。

## 为什么记录

Round29--38 的多数结构实验采用 matched continuation：候选与 control 从同一
成熟 checkpoint 出发，新增分支零输出初始化，并使用相同 seed、训练预算和
OOF 判定。这是判断某项信息或模块能否作为“即插即用增量”改善成熟 U-Net 的
严格设计，也能显著减少初始化方差。

但该设计的结论边界必须保留。非凸优化存在路径依赖；成熟 backbone、较低
continuation 学习率和有限更新数可能把候选限制在原有表征附近。若一个结构
需要编码器、归一化、tokenization 和解码器共同重组，matched continuation
失败不能证明它从独立初始化或合适预训练开始也失败。它只否定：

> 当前结构在固定 continuation 配方下，不能稳定地作为成熟模型的增量改进。

Round33--34 的 base48 是相关证据：fresh 固定预算明显不足，追加 continuation
后完整 OOF 又提高约 `0.23`；但充分延长后 direct 仍未超过成熟 base32。这既
说明收敛预算可能低估新模型，也说明“再练更久”不是现有差距的充分解释。

## 与现有计划的隔离

1. Round38 按已提交配置、门槛和机器 gate 完整运行，不因训练曲线修改；
2. 本文件不授权在 Round38 期间启动额外训练、搜索超参或读取排行榜反馈；
3. 只有 Round38 完成 OOF gate 后，才重新判断是否立项独立的 representation
   reset 实验；
4. Round38 若获得强且稳定的机制增益，该后备方向降级，不与其争夺算力；
5. Round38 若失败或近似持平，也不能直接启动任意大模型，仍须先单独预注册。

## 若立项，优先级

1. **确定性图像复原 Transformer**：优先考察 Restormer、SwinIR、Uformer
   一类与配对像素回归、PSNR/SSIM 更一致的架构；是否采用公开预训练须先核对
   当阶段规则；
2. **fresh aligned U-Net**：从独立初始化训练完整 U-Net 与 Round38 的空间
   对齐条件分支，检验旧 backbone 表征是否压制该信息；
3. **纯 Swin encoder-decoder**：现有 dual-branch Swin 只否定零门控全局分支，
   不等价于充分训练的纯 Swin 图像复原模型；
4. **DiT/diffusion**：保持低优先级。小规模配对数据、确定性输出和
   PSNR/SSIM 指标与生成式扩散目标不天然一致，只有在前三项证据或外部可靠
   复现明显支持时才考虑。

## 公平比较的最低要求

若后续正式执行，应在训练前固定：

- 与当前方案相同的五折 ROI、输入范围、target、增强和 OOF 口径；
- 一个从独立初始化训练的 U-Net control，以及一个候选架构；
- 按 update 数、显卡时长或预先定义的收敛判据匹配预算，而不是机械要求相同
  epoch；
- 允许架构适配的优化器、学习率和 warmup，但所有配方必须在看到完整 OOF
  结果前固定；
- 先做不接触排行榜的内部 screening，达到预注册稳定性条件后才进入完整五折；
- 完整/test-like、逐 fold、逐 ROI 和 float/q95 一致性 gate；
- 禁止根据隐藏榜单探针选择 backbone、epoch、seed 或融合权重。

该方向的目标不是为已经失败的 continuation 实验寻找事后解释，而是单独回答：
在不继承成熟 U-Net 优化轨迹时，一个充分收敛的新表示体系能否稳定超过固定
Round35 基线。

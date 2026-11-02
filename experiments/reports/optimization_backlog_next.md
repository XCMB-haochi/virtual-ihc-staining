# 后续优化路线（Round99 之后）

更新时间：2026-09-06

## 当前判断

Round91 对 Round90 固定 endpoint 的平台确认得到 `74.0287`，与此前
`74.0315` 相差 `-0.0028`，因此不构成新的调参信号。Round92 的 UNeXt2
D4-consistency 两折 screen 已完成，但直接机制和两个固定替换都未通过
分布 gate。Round93 已完成一次全标注 fresh endpoint 诊断；平台确认分为
`73.9176`（SSIM `0.831`、PSNR `26.287`），低于 Round88 的 `74.0315`，
因此不晋级，也不作为调参信号。该任务目前的主要瓶颈不是显存：
FCMAE、空间缩略图和固定配方已经能稳定工作，近期模型改动的隐藏集收益普遍
只有百分之几到十分之几。

Round59 的嵌套、更新次数匹配数据规模实验仍是最重要的正证据：监督 ROI 从
5、10、15 增加到 20 时，完整 OOF q95 均值为
`72.961672 -> 73.979437 -> 74.578788 -> 75.026439`，三条抽样曲线均严格
递增。这个结果支持“剩余收益首先来自更多有效标注”，但不允许直接把
Colon 的 epoch 或权重照搬到后续器官。

Round94 的纹理缩略图两折筛查通过后，Round95 已按原配方扩展到完整五折。
完整 OOF 仅提高 `+0.019837`，只有 3/5 folds 和 12/25 ROI 为正，最差
ROI015 为 `-0.935751`；完整 gate 失败，纹理缩略图分支关闭。Round96 的
折内、五通道、图像暴露量匹配 FCMAE 表征筛查也失败（q95 `-0.039674`，
4/10 ROI 为正）。Round97 将同一固定 Base endpoint 的推理 halo 从 128
扩大到 256，完整 OOF 只提高 `+0.010434`，仅 16/25 ROI 为正且 ROI018
下降 `-0.085058`，因此 context-scale 分支关闭。后续不再做 halo 或权重
网格；应优先寻找能在分组 OOF 中验证、且不依赖当前 Colon 分布的结构机制。

Round98 的跨标记 logit 反馈头已经完成两折运行。虽然候选和历史 Round55
auxiliary15 使用了不同目录名，但 stage1 初始化报告中的 fold-local encoder
SHA-256 完全一致，且 seed、损失、标记权重、条件和 14+14 时钟相同，因此这是
有效的 matched 比较。候选 q95 为 `74.807699`，基线为 `75.013272`，差值
`-0.205574`；该结构路线关闭。正在进行的零反馈消融只用于解释退化，不作为
调参或提交信号。

## 排序

### 1. Round93 全标注 endpoint 的固定平台确认（已关闭）

Round93 已按预注册配方完成：全部 25 个训练 ROI、all-train DAPI FCMAE
初始化、14+14 阶段时钟、context128、D4、固定 `[0,-1]` 偏移和 q95。其
`cd68_round93_full_label_auxiliary15_context128_q95_plain.zip` 已完成一次固定
确认，得分低于既有 Round88 集成。它没有诚实留出 OOF，平台分数不能反向选择
epoch、权重或后处理，也不能替代 grouped OOF。后续阶段数据发布后应重新做
分组验证，不默认沿用这个全标注 endpoint。

### 2. UNeXt2 上的 D4 consistency（已关闭）

Round92 已在现代 UNeXt2 + thumbnail FiLM 上完成 fold0/fold2 的固定 `0.1`
screen。直接候选相对 official auxiliary 的整体 q95 区间为
`[-0.017848, +0.018556]`，replacement 与 half-split 也分别只有
`[+0.015894, +0.023076]` 和 `[+0.011989, +0.015512]`；均未通过预注册
gate。因此不扩展五折、不提交、不做 consistency 权重或 D4 子集搜索。

### 3. 更细的邻域 DAPI 表示（已关闭）

Round94 的两折正结果由 fold2 主导；Round95 完整五折未复现其分布稳定性。
不再搜索纹理通道、权重、ROI route 或事后融合，也不生成测试候选。

### 4. 折内多 marker FCMAE 表征（已关闭）

Round96 将五个官方训练通道分别作为独立灰度图输入同一个 FCMAE。每个 marker
每轮恰好出现一次，固定 `160` epochs；相对 DAPI-only `800` epochs，总图像暴露
量精确匹配。fold0 optimizer steps 精确匹配，fold2 因每轮短 batch 少 `0.52%`，
不通过重复采样强行补齐。所有 validation ROI 的
五个通道都从该 fold 的预训练中排除。下游保持同一个 UNeXt2-Tiny、`14+14`
监督时钟、context128、D4、q95 和固定 `[0,-1]` offset，只测试 direct、固定
replacement 和固定 half-split。两折 gate 失败即关闭，不搜索 marker 比例或
预训练时长；通过才允许原样扩展五折。

### 5. 大上下文只做一次固定诊断（已关闭）

Round97 已完成该固定诊断：q95 仅 `+0.010434`，16/25 ROI 为正，且触发
ROI floor。保留为负结果和背景证据，不生成提交包。

### 6. 跨标记反馈（已关闭）

Round98 的新头从三个 auxiliary marker logits 预测 CD68 残差，最终投影零
初始化。两折训练和 OOF 完整，且与 Round55 auxiliary15 的编码器哈希和训练
配方匹配；q95、float、fold 和 ROI gates 全部失败。关闭该结构，不扩展五折，
不搜索头宽度、反馈权重、epoch 或提交结果。

零反馈消融进一步得到 q95 `58.925485`，远低于带反馈的 `74.807699`，说明
训练确实把大量 CD68 预测转移到了残差头；但这一路整体仍低于普通 auxiliary15，
不能通过简单关闭或缩放残差挽救。该架构及其局部变体一并关闭。

Round99 已预注册 DAPI 高通细节残差：只从输入 DAPI 的 `5x5` 高通分量学习
一个零初始化、`0.25*tanh` 有界的 CD68 logit 修正。它针对欠锐误差而不是无条件
锐化，并保持四标记 auxiliary15、fold-local FCMAE、context128 推理和固定
`[0,-1]` offset。先做 fold0/fold2；若未通过分布 gate，关闭整个细节残差族，
不继续搜索幅度、头宽度或 epoch。

Round99 已完成：候选 q95 `74.929780`，matched auxiliary15 为 `75.013272`，
差值 `-0.083492`；两折均为负，最差 ROI `-0.389315`，因此正式关闭高通
logit 残差及其简单有界变体，不扩展五折或生成提交包。下一条结构实验应改为
在解码前保留 DAPI 的全分辨率浅层特征，而不是继续在输出端叠加残差。

Round100 已完成该浅层特征实验：候选 q95 `74.953469`，相对 matched
auxiliary15 下降 `-0.059803`；两折均负，仅 4/10 ROI 正，最差 ROI
`-0.380856`。关闭该融合位置和邻近宽度/投影变体。下一步先用既有泄漏安全
OOF 缓存测量由 DAPI 和模型分歧驱动的逐图自动路由上限；只有严格 cross-fit
门控有实际增益时才实现到当前 endpoint，否则转向不同训练目标或表示。

Round101 已完成：逐图 0.1 间隔凸融合 oracle 相对固定 70/30 有
`+0.358538` 上限，但只用 DAPI 与预测分歧特征的五折 cross-fit Ridge 为
`-0.011448`，且 4/5 folds 负。关闭逐图/逐 ROI 路由及更复杂的同类选择器；
oracle 只说明目标依赖的互补存在，不是可部署增益。下一步测试固定、自动、无标签的
stride-phase 自集成，判断当前 UNeXt2 下采样与 PixelShuffle 是否存在相位误差。

Round102 已完成：四个一像素输入相位经精确中心对齐后平均，q95 为
`75.010403`，相对标准 context128 D4 为 `-0.002870`；float 为
`-0.000369`，仅 fold0 略正，最差 ROI `-0.087284`。关闭 stride-phase
TTA 及其相位权重/偏移变体。当前剩余方向应转向训练数据利用、表示学习或
固定全模型集成审计，不再继续围绕同一 UNeXt2 推理几何做微调。

Round110 的冻结多染色 FCMAE feature-guidance 也已完成两折 OOF，q95 相对
matched auxiliary15 下降 `-0.124715`，仅 3/10 ROI 为正；直接 perceptual
约束路线关闭。Round109 的特征可预测性不能作为该训练约束有效的证据。

### 7. Round103：一次固定 UNeXt2 auxiliary15 续训周期（已关闭）

Round103 不改变架构、输入条件、辅助标记、损失、数据划分或推理协议，
只从 Round55 每折的最佳 stage2 checkpoint 重新初始化优化器和 scheduler，
执行一个 `1e-4 -> 0` 的 14 epoch 余弦周期。基线为同一 checkpoint 的
context128+D4 重放，候选为续训后的 checkpoint；两折 OOF 覆盖 2706 张图。
预注册 gate 要求 q95 至少 `+0.03`、float 和两折均为正、至少 7/10 ROI
为正且最差 ROI 不低于 `-0.12`。实际候选 q95 为 `74.968910`，相对基线
`75.013272` 下降 `-0.044362`；fold0/2 为 `-0.122672/+0.029014`，仅
4/10 ROI 为正，最差 ROI `-0.345122`。gate 全部失败，故关闭无新机制的
重复续训，不追加第二周期。详见 `reports/optimization_round103.md`。

### 8. Round104：输出编码与活动度诊断（已完成）

Round104 在现有五折 OOF 上确认 q95 JPEG 不是主要瓶颈：PNG 无损相对 q95 仅
`+0.020622`。目标活动度与基线 MSE 相关 `0.5856`，高活动度四分位的 MSE
约为低活动度的四倍；公开 DAPI 特征的逐图 Ridge 均值/对比度校正抽样反而
下降，因此关闭逐图强度校正和编码网格。详见 `reports/optimization_round104.md`。

### 9. Round105：训练期活动度 hard-mix（已关闭）

候选保持 Round55 official FCMAE、UNeXt2-Tiny、四标记 auxiliary15、context64
训练和 context128+D4 推理不变，只在每个 stage 后 25% epoch 用训练集 CD68
目标活动度最高四分位替换 30% 采样位置。实际 q95 为 `74.973171`，相对基线
`75.013272` 下降 `-0.040101`；fold0/2 为 `-0.147927/+0.060933`，仅
3/10 ROI 为正，最差 `-0.273439`。gate 全部失败，故关闭该 sampler，不搜索
活动度公式、比例、启动 epoch 或第二周期。详见 `reports/optimization_round105.md`。

### 10. Round106：跨 ROI DAPI 近邻检索诊断（已关闭）

Round106 不训练模型，固定用五折 OOF 的其余四折作为检索库，以 `16x16` DAPI
形态特征检索四个邻居并平均其 CD68 target，再以 10% 权重与固定神经网络基线
融合。检索输出为 `47.247373`，固定融合为 `72.217157`，相对基线下降
`-2.768900`，五个 fold 全部下降，故关闭检索路线，不搜索 k、权重或距离度量。
详见 `reports/optimization_round106.md`。

### 11. Round107：同轨迹局部 checkpoint 平均（已关闭）

Round107 不重新训练模型。对每个 fold，读取 Round55 auxiliary15 stage2 已由
验证集选出的 global-MSE 最佳 epoch 及其相邻前后 epoch，按固定 `1:2:1` 做
参数平均；基线是同一 checkpoint 的 context128+D4 重放。两折 gate 要求 q95
至少 `+0.01`、float 和两折为正、7/10 ROI 为正且最差不低于 `-0.12`。失败即
关闭该 SWA/soup 路线，不搜索窗口、权重或 epoch；通过才扩展剩余三折。

Round107 已完成。fold0/fold2 的相邻 epoch `1:2:1` 参数平均候选 q95 为
`74.987202`，matched 基线为 `75.013272`，差值 `-0.026070`；float 差值
`-0.023432`。两个 fold 均为负，仅 4/10 ROI 为正，最差 ROI024 为
`-0.176361`，gate 全部失败。关闭 checkpoint SWA/soup 及其权重、窗口和
epoch 变体，不扩展剩余三折。详见 `reports/optimization_round107.md`。

### 12. Round112：独立 DINOv2 表示（已关闭）

Round111 完成后 q95 下降 `-0.083468`，2/10 ROI 为正，不扩展。
历史更正：Gaussian pyramid L1 在 Round2 的旧 U-Net 上已经试过，Round111
只是当前 UNeXt2 的一次复验；“从未使用”不准确。停止当前损失微调链。

Round112 改用公开自监督 DINOv2-S/14 编码器和卷积解码器，从独立监督初始化
开始，保留原数据/损失，固定比较直接输出和 50/50 融合。旧的未预训练 Swin
和 Restormer 负结果不等价于本路线已失败。配置和成本边界见
`optimization_round112.md`。同理，Round50/83/102 只否定其固定配方，不能
据此断言所有训练期局部配准或一致性机制均已被排除。

两阶段真实 smoke、离线 checkpoint 恢复和 context128+D4 推理检查通过。正式
两折 OOF 已完成：DINOv2 直接替换 q95 `73.537388`，相对 baseline
`75.013272` 下降 `-1.475885`；固定 50/50 融合为 `74.878722`，下降
`-0.134550`，全部 gate 失败。关闭该具体全骨干替换和固定融合；不调层数、学习率、权重或 epoch。

Round113 已完成两折 OOF：q95 `+0.038374`，fold0/2 为
`-0.010961/+0.084600`，仅 5/10 ROI 改善，gate 失败，不扩展。
固定 80 图消融证明联合训练后依赖 DINO 输出支路，不能据此证明预训练带来增益。
Round114 改为将 DINO 最后一层特征对齐后送入 CNN 瓶颈，经共享解码器预测，
监督训练仍从头开始。现已完成：q95 `-0.032795`、float `-0.028898`，
只有 2/10 ROI 为正，最差 `-0.567938`；该配方负向，不扩展或搜索融合位置。
Round115 固定在额外 fold1 复验 Round113 原配方，只需一折训练；确认和正式
晋级分开记录，旧 gate 不变，不能自动生成提交。见 `optimization_round115.md`。

Round115 已完成：fold1 q95 `-0.107969`、float `-0.120972`，仅 1/5 ROI
改善，确认失败；Round113 的正向结果不具备跨折复制性，DINO 残差族关闭。
Round116 固定测试 Round65 all-train DAPI FCMAE630 共享编码器与 Round55
auxiliary15 监督的统一组合，只用单卡做 fold0/fold2 screen。它检验已有
互补端点能否在单模型中结合，不改变辅助权重或 ensemble 搜索。

Round116 已完成：q95 `+0.009937`，float `+0.011407`，fold0/2 为
`-0.139333/+0.149804`，仅 3/10 ROI 改善，最差 `-0.400201`；该统一组合
关闭。Round117 固定测试 current Tiny FCMAE 的 CD68 单标记 epoch-level
GroupDRO（`eta=5`）以处理 fold/ROI 风险，不搜索 eta，仍只用一张卡。

Round117 已完成：q95 `-0.150319`、float `-0.150268`，两个 fold 均下降，
仅 3/10 ROI 改善，最差 `-0.429033`；关闭这一个 GroupDRO 配方。
2026-09-08 更正：此前“没有剩余可行训练机制”的结论没有充分依据，撤回。
用户已授权最多一到两天重新审查，期间不要求为保持 GPU 忙碌而启动训练。
已确认抽样诊断的 ROI 条件范围错误，以及训练期逐图评分的 PSNR 截断顺序问题；
正式完整 ROI OOF 未发现同样的条件偏差。保留最佳提交，后续训练须有具体依据。
详见 `reassessment_20260908.md`，不是等待新器官发布的永久暂停。

2026-09-08：用户要求继续改进，允许有依据的大模型实验；当前仅排队 GPU1，
GPU0 留给其另一项目。Round118 固定检验有效 batch 累积后的主任务优先梯度
投影，保留同向辅助梯度。依据为修正后的 6/16 冲突测量，具体见
`optimization_round118.md`。GPU 烟测通过后才允许正式两折训练。

Round119 已通过真实 smoke 并在 GPU1 启动 fold0；stage1 仅有 1 个普通 AMP
skip，stage2 投影更新全为有限值。GPU0 继续保留给用户的其他项目。

### 13. Round108：当前 UNeXt2 官方分数直接优化（已关闭）

Round108 已固定为单输出 CD68 的 `exact-score` 训练：保留官方 FCMAE800、
UNeXt2-Tiny、context128、D4 和 `[0,-1]` 偏移，只将训练目标改为官方
`0.7*SSIM + 0.3*clip(PSNR/50,0,1)`。首次 Screen 已完成 smoke，但因启动器
向单标记 recipe 多传 `--marker-loss-weights 1.0` 被训练器正确拒绝；该启动
失败不产生模型证据，修复后的同协议 Screen 已重新启动。

正式两折现已完成。exact-score q95 为 `74.840191`，matched official-FCMAE800
为 `74.987515`，下降 `-0.147324`；float 下降 `-0.149686`。两个 fold 都负，
仅 5/10 ROI 为正，最差 ROI008 为 `-0.966731`。虽然 PSNR 提高约 `0.064 dB`，
SSIM 下降约 `0.00265`，全部 gate 失败。关闭纯 exact-score 及其权重、截断、
epoch、checkpoint 变体，不做五折或提交。详见 `reports/optimization_round108.md`。

### 14. Round109：跨模态 FCMAE 特征可预测性（已通过）

Round109 使用每折严格排除验证 ROI 的 Round96 五模态 FCMAE encoder，分别提取
DAPI/CD68 stage4 空间均值特征，再用训练 ROI 的固定 Ridge 预测验证 ROI 的
CD68 表征。全量 2706 张结果的加权标准化 R2 为 `0.654895`，fold0/fold2 为
`0.671613/0.638097`；正确配对相对同 ROI 错配的余弦优势为
`0.010507/0.013841`，全部 gate 通过。该诊断授权一次固定、训练期、冻结 teacher
的 CD68 特征约束 screen，不授权 Ridge/层级/权重搜索。详见
`reports/optimization_round109.md`。

### 15. Round110：冻结多染色 FCMAE 特征引导（已关闭）

Round110 保持 Round55 auxiliary15 student 及 batch16/14+14/context128/D4/
`[0,-1]` 不变，冻结同折 Round96 五模态 FCMAE teacher，仅对每个随机 batch
前四个样本的中心 256 区域加入固定 `0.05` stage4 spatial cosine loss。真实
smoke 中该 loss 非零、梯度链和 checkpoint 完整，峰值显存 `26.583 GiB`；正式
Screen 已完成两折 OOF 和 gate：候选 q95 `74.888558`，相对 matched
auxiliary15 基线 `75.013272` 下降 `-0.124715`；float 下降 `-0.117977`。
fold0/2 分别下降 `-0.210676/-0.044168`，仅 3/10 ROI 为正，最差 ROI020 为
`-0.484634`，全部 gate 失败。关闭该直接 perceptual 路线，不搜索 teacher
层、权重、crop、子批大小、epoch 或五折扩展。详见
`reports/optimization_round110.md`。

## 已关闭或降权的方向

- Swin、Restormer、NAFNet、ConvNeXt/UNeXt2 Base 直接替换：当前配方下直接
  模型负或仅提供小的多样性，不能再当作主线容量升级。
- ImageNet-22K supervised 初始化、0.1x encoder LR、DropPath、FCMAE 1200
  epoch、目标强度 L1、辅助梯度路由、全局 affine、checkpoint selector：均有
  明确负结果或未达分布 gate。
- 单纯延长监督 epoch：各折最佳 epoch 分散且后期常回落；必须有新的正则或
  数据效率机制，不能把更长训练当作默认增益。
- 简单 gamma/softplus/局部锐化和 target-aware 频率修正：OOF oracle 上界很小
  或损害 SSIM，不值得先于结构实验。
- context halo 128->256：完整 OOF 未通过 q95、ROI 数量和 ROI floor gate；
  不再扩大 halo 或从该结果反向搜索融合。

## 执行纪律

每个昂贵新机制先判断两折 screen 是否能回答决策；通过后才做完整 grouped OOF。
全标注端点和初赛平台确认要与可迁移结构证据分开记录。后续 Liver/Stomach 数据
发布后，重新测 ROI 分组、更新次数、收敛 epoch、marker 难度和资源，不默认沿用
Colon 的 `800`、`14+14`、五折或当前 ensemble 权重。

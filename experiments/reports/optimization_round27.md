# 优化记录 Round 27：DAPI 空间统计条件化

日期：2026-07-19

## 起点与骨干判断

Round26 五折 base32/full-target 固定集成的服务器分为 `71.8249`，显示
`SSIM=0.807, PSNR=25.565`。它相对旧 raw `71.7440` 提高 `+0.0809`，
说明完整 OOF 选择可以转化为隐藏测试收益；但仍比逐测试 ROI 端点组合
`72.0437` 低 `0.2188`，同构 U-Net 集成的边际不足。

本项目并非只测试过 U-Net：NAFNet-width32、ImageNet-1K 预训练
ConvNeXt-Tiny encoder-decoder，以及 local U-Net + global Swin 双分支均已
完成 H2 对照。它们分别出现小幅总体增益但 ROI024 大幅回退，或总体直接
回退，未解决跨 ROI 泛化。DiT 的官方实现面向 latent diffusion、FID 和
数十万步 ImageNet 生成训练，不直接匹配本题确定性、配对、PSNR/SSIM 目标；
因此不把扩散式 DiT 作为下一项高成本实验。若空间信息路线失败，后备骨干
优先使用确定性图像复原 Transformer，例如
[Restormer](https://github.com/swz30/Restormer)、
[SwinIR](https://github.com/JingyunLiang/SwinIR) 或
[Uformer](https://github.com/ZhendongWang6/Uformer)。

## 空间上下文代理诊断

新增 `tools/analyze_spatial_context.py`，沿用 Round26 的五个完整 ROI fold。
每张图的 local 对照只使用 DAPI tile 自身的均值、标准差、五个分位数、
平均梯度和暗像素比例；候选额外使用：

- tile 在 ROI bounding grid 中的位置和四边距离；
- ROI occupancy、DAPI tile 均值/标准差的 ROI 统计；
- 半径 1/2 tile 的邻域摘要；
- 中心周围 `5x5` tile 的 DAPI mean/std/presence canonical grid。

固定 `HistGradientBoostingRegressor` 对 CD68 tile 均值做跨 ROI 预测。两个
模型使用相同超参数；验证 ROI 的条件特征只来自其 DAPI 输入，CD68 标签不
进入训练。该任务只是信息诊断，不估计比赛像素分。事先固定的推进条件为：
总体 RMSE 至少降低 `1%`，且至少 `4/5` fold 为正。

完整结果为 `reports/round27_spatial_context_proxy.json`：

| feature set | overall RMSE (uint8) | relative reduction |
|---|---:|---:|
| local | 7.9681 | - |
| local + ROI geometry/statistics | 7.4115 | 6.99% |
| local + `5x5` neighborhood | 7.7854 | 2.29% |
| local + all spatial features | **7.2194** | **9.40%** |

完整 spatial 候选五折 RMSE 降幅依次为 `9.98% / 0.22% / 14.34% /
8.61% / 11.70%`，五折均为正，满足推进条件。主要信号来自 ROI 几何和
全局 DAPI 统计，邻域在总体上提供额外互补；这与之前只输入单个 ROI DAPI
均值的 FiLM 不是同一条件信息。

## 固定神经网络候选

条件向量固定为代理诊断中的 9 个 local 特征与 92 个 spatial 特征，共
101 维，全部只从官方 DAPI 输入计算。候选为 base32 residual U-Net，在
bottleneck 加入小型 MLP FiLM；FiLM 最后一层 scale/shift 严格零初始化，
保证加载同一未条件化 checkpoint 后首次前向与对照逐值相同。条件只产生
全局 channel-wise affine，不产生测试输出后处理。

每折从 Round26 对应 `fulltarget_stage1/best_global_mse.pt` 初始化。已有
full-target stage2 是匹配对照；候选只重建 stage2：14 epoch、lr `1e-4`
cosine、seed 2026、batch/workers `8/4`、Uniform-7 hybrid `0.7/0.2/0.1`、
input/target `384/384`、global-MSE q95 选模。正式推理固定 halo64+D4+q95。

主候选固定为同折 conditioned/full-target 浮点输出 `50/50` 后编码 q95；
direct conditioned 只作诊断。禁止搜索条件子集、MLP 宽度、FiLM 层数、
融合权重、epoch、seed、loss、halo、TTA 或校准。

## Promotion gate

主比较为固定 50/50 候选相对 Round26 matched full-target：

1. 完整 25-ROI OOF 的 q95 mean-image 与 global-MSE 均至少提高 `+0.25`；
2. DAPI test-like 子集两项 q95 均至少提高 `+0.25`；
3. 至少四个 fold 的两项 q95 均为正，且五折平均为正；
4. 任一 test-like ROI 的两项 q95 回退不得超过 `0.08`；
5. 完整 OOF 与 test-like 的四项 float 增益均为正；
6. 任一条件失败即不生成服务器包，不搜索权重或条件子集。

通过后只生成五个 fold 固定 50/50 的等权测试集成。首包保持 raw，不混入
旧模型、ROI029 校准或逐测试 ROI 选择；获得服务器反馈后再单独判断融合。

## 当前状态

代理诊断和推进判断已完成。新增 `spatial-stats-film` 条件模式：101 维输入
先经 `101 -> 256` MLP，再产生 bottleneck 的 channel-wise scale/shift；
最后一层严格零初始化。base32 单输出参数量由 `8,110,657` 增至
`8,399,937`，增加 `289,280`，不是大模型容量实验。

训练器新增显式 `--init-add-conditioning`，只允许从未条件化、同 marker、
同 model spec 的 U-Net 复制共享权重；新增 tensor 必须全部位于
`condition_film.*`。评估、融合、校准与单模型/CV 预测器统一按 patch name
恢复条件。CV 协议新增显式 run source，Round27 只训练五条 spatial stage2，
控制和初始化分别复用 Round26 的 full-target stage2/stage1。

本机真实数据烟测完成了旧 checkpoint 初始化、1 epoch 训练、保存/strict
重载、D4 评估和测试预测；条件报告记录 816 个 101 维向量的摘要与 SHA-256，
不把 NumPy 向量直接写入 JSON。临时 run 和 prediction 已清理。最终本机
可运行测试为 `126 passed, 1 skipped, 2 deselected`；两个 deselect 均为
既有 Windows Anaconda/MKL 原生冲突。compileall、CLI、五折 dry-run 与
`git diff --check` 通过，dry-run 只展开五条 conditioned stage2。

服务器干净环境全量测试为 `129 passed, 53 subtests passed`。GPU1 的
full-target CUDA 预检使用 64 train / 32 val、batch/workers `8/4`：单
epoch 训练加验证 `2.962 s`，峰值 PyTorch 显存 `4.354 GiB`；保存后的
`condition_film.output.weight` L1 为 `52.1782`，确认条件分支获得更新。
退出时的 `/workspace/tmp-meimy/pymp-*` `EBUSY` 是既有 NFS worker
finalizer 噪声，主进程退出码为 0，config、metrics 和 checkpoint 完整。

正式 runtime commit 为 `ab9c1a8`。GPU0 保持不使用；物理 GPU1 启动前
利用率为 0%，已有约 `5.7 GiB` 的其他用户空闲进程未被停止或修改。五条
Linux 命令经 dry-run 完整展开，正式输出目录为空；任务随后在 detached
Screen `round27_spatial` 中顺序启动。五折共 70 个 epoch 全部完成，状态文件
确认五条 run 完整，未出现训练失败。

## 正式 OOF 结果

正式评估覆盖 25 个 ROI、6296 张图，报告为
`reports/round27_cv5_spatial_film_oof.json`，耗时 `930.631 s`。下表均为
固定 50/50 相对 matched full-target 的 competition score 增益：

| 范围 | 编码 | mean-image PSNR 口径 | global-MSE PSNR 口径 |
|---|---|---:|---:|
| 完整 OOF | float | +0.462288 | +0.440262 |
| 完整 OOF | q95 | **+0.464508** | **+0.442511** |
| test-like | float | +0.462320 | +0.444730 |
| test-like | q95 | **+0.463063** | **+0.445493** |

完整 OOF q95 的绝对分由 `73.406448/72.650904` 提高至
`73.870955/73.093415`；test-like 由 `71.222945/70.618812` 提高至
`71.686008/71.064305`。相对上次实际提交的 Round26 base32/full-target
50/50，完整 OOF q95 仍提高 `+0.404007/+0.394709`，test-like 提高
`+0.415923/+0.405877`。

| fold | q95 mean-image 增益 | q95 global-MSE 增益 |
|---|---:|---:|
| fold0 | +0.457623 | +0.423838 |
| fold1 | +0.449137 | +0.421021 |
| fold2 | +0.513143 | +0.495999 |
| fold3 | +0.564727 | +0.544260 |
| fold4 | +0.332096 | +0.318746 |
| 五折算术平均 | +0.463345 | +0.440773 |

九个 test-like ROI 的两项 q95 增益全部为正；最小项为 ROI006 的
`+0.097596/+0.094990`，最大项为 ROI015 的 `+1.323818/+1.236851`。
因此不存在接近 `-0.08` 回退边界的 ROI。direct spatial 诊断在完整 OOF
达到 `+0.632979/+0.598181`，test-like 达到 `+0.628174/+0.597801`，证明
条件分支本身有效；但它在实验前被限定为诊断，不因看到结果而改成首包。

## Gate 判定与唯一测试候选

全部预注册条件通过：完整 OOF 和 test-like 的 q95 双口径均超过 `+0.25`；
五折两项全部为正；所有 test-like ROI 均为正；完整 OOF 与 test-like 的
四项 float 增益也全部为正。因此允许生成固定 50/50 raw q95 候选，不搜索
融合权重、条件子集、epoch、seed、校准或旧模型混合。

测试推理加载五折各自的 full-target 与 spatial-film best checkpoint，共十个
成员；每个成员有效权重 `0.1`。完整 provenance 为
`reports/round27_cv5_fulltarget_spatialfilm50_test_prediction.json`，1346 张
测试图推理耗时 `348.564 s`。

- ZIP：`submissions/cd68_round27_cv5_fulltarget_spatialfilm50_context64_d4_raw_q95_plain.zip`；
- ZIP 大小：`11,429,599` bytes；
- ZIP SHA-256：`B182E4B352A2DA11AC794A84E99EC94825794028BF25DA335BF5368F2E9589A8`；
- prediction provenance SHA-256：`0D16D70A02FAD737E72AC483C9B09705B9B5E88646D616DDC765E2FE5D89EE8A`；
- OOF report SHA-256：`722E273D696DF4596C913B8960B7271C3D849F6DC84769E42A900ADE9FEA509F`。

服务器打包器检查 expected/actual `1346/1346`、missing/extra 0、图像错误 0、
全部 RGB；服务器 CRC 检查无错误。本机同步后独立复核得到 1346 个唯一条目、
duplicate 0、非法路径 0、CRC error 0、解码/尺寸/灰度 RGB 通道错误 0，且
三份 SHA-256 与服务器完全一致。

## 服务器结果与结论

平台返回精确分 `72.2880`，显示 `SSIM=0.813`、`PSNR=25.668`。相对
Round26 五折包 `71.8249` 提高 `+0.4631`，相对此前个人最佳 `72.0437`
提高 `+0.2443`；SSIM 和 PSNR 的三位显示值分别提高 `+0.006` 和
`+0.103`。完整 OOF 相对 Round26 实际提交候选的 q95 增益为
`+0.404007/+0.394709`，因此增益方向和数量级均迁移到隐藏集，且服务器
增益没有被 OOF 高估。

当前榜首为 `74.8890`，本包仍低 `2.6010`。结论是 DAPI 空间统计条件化
解决了此前部分跨 ROI 泛化瓶颈，但单靠固定 50/50 仍不足以接近榜首。由于
direct spatial 在服务器反馈前已经作为独立诊断完成，并在完整 OOF、
test-like 和五折均继续优于 50/50，下一轮只推进该现成端点；不根据榜单
搜索 60/40、70/30 等连续融合权重。

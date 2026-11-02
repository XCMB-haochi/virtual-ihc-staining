# 优化记录 Round 31：D4 不变的 thumbnail 环带位置编码

日期：2026-07-20

## 立项依据

Round30 direct multiscale 端点在平台得到 `73.0409`，显示
`SSIM=0.822, PSNR=25.830`，相对 Round29 提高 `+0.1576`。matched OOF
表明四级 decoder FiLM 相对同预算 continuation 基本持平，完整 q95 为
`+0.002153/-0.004446`，因此不再扩展 decoder FiLM。额外 14 epoch 续训
相对 Round29 base 则稳定提高约 `+0.26~+0.31`，说明当前 thumbnail 主线
仍有可学习空间。

Round29/30 thumbnail CNN 将 80x80 的真实 5x5 DAPI 邻域经过四次 stride-2
卷积得到 5x5 feature map，随后直接 global average pooling。该操作把中心
tile、近邻八格和远邻十六格完全混合。已有 101 维统计条件只保留各格均值、
标准差和 presence，无法替代 thumbnail CNN 学到的局部形态在不同距离上的
分布。本轮只检验：在保持 D4 不变的前提下保留 center/near/far 语义，能否
带来稳定净增益。

## 固定模型变化

新 condition mode 为 `spatial-thumbnail-ring-film`。输入仍是完全相同的
101 维 spatial statistics 加 2x80x80 thumbnail；U-Net、thumbnail 四层
CNN、stats MLP、bottleneck FiLM 和输出头全部不变。

thumbnail CNN 的 5x5x64 feature map 固定汇聚为：

| 区域 | feature cells | 语义 |
|---|---:|---|
| center | 1 | 当前目标 tile |
| inner ring | 8 | 3x3 邻域去除中心 |
| outer ring | 16 | 5x5 去除 3x3 |

三个 64 维均值拼成 192 维，经新增 `Linear(192,64)` 形成 global-average
embedding 的残差。该 Linear weight/bias 全零初始化，因此 checkpoint 转换
后的 embedding、FiLM 和最终输出必须与来源 thumbnail 模型逐值一致。新增
参数固定为 `12,352`，不是扩大 U-Net 主干容量。

每个 thumbnail 仍计算八个 D4 变换。center/inner/outer 三个集合在 D4 下
各自封闭，八路 ring-aware embedding 继续取均值，因此条件对旋转和翻转保持
不变，不需要更改训练 augmentation 或 D4 TTA。

## 匹配五折协议

机器可读协议为 `configs/round31_cv5_thumbnail_ring_film.json`，沿用
Round29/30 五折。每折从同一个 Round30 `thumbnailcontinue` best checkpoint
启动两条同预算分支：

| recipe | condition | 初始化 | 训练 |
|---|---|---|---:|
| `thumbnailcontinue` | 原 global-average thumbnail | strict | 14 epoch |
| `ringfilm` | global average + ring residual | add-conditioning | 14 epoch |

两路固定 seed `2031`、lr `1e-4`、batch/workers `8/4`、halo64、full-target
监督和 Uniform-7 hybrid loss。Round30 `thumbnailcontinue` 只作不续训 base。
不使用 Round30 multiscale checkpoint，避免把已失败的 decoder 结构带入本轮。

## 预注册判定

正式 promotion gate：

1. 相对 matched continuation，完整 OOF 与 test-like 的四项 q95 增益均
   至少为 `+0.05`；
2. 完整 OOF 与 test-like 的四项 float 增益均为正；
3. 至少 4/5 folds 的 q95 双指标为正，且五折平均为正；
4. 相对 continuation，任一 test-like ROI 不得低于 `-0.15`，且双指标为负
   的 test-like ROI 不得超过两个；
5. 相对 Round30 thumbnail base，完整/test-like 四项 q95 均至少提高
   `+0.12`。

若 promotion gate 失败，但相对 Round30 thumbnail base 的完整/test-like
四项 q95 均至少提高 `+0.15`，且五折双指标全部为正，只允许生成一次 direct
ringfilm 诊断包。两个层级都禁止搜索 ring 边界、区域权重、projection 宽度、
epoch、seed、loss、融合、校准或逐 ROI 路由。

## 当前状态

实现已完成。测试覆盖固定 conditioning metadata、旧/new condition 数值完全
一致、center/inner/outer 环带计算、D4 不变 embedding、thumbnail 模型到
ring 模型的共享 tensor 复制、首次输出逐值一致、新 projection 梯度、
checkpoint metadata/strict roundtrip，以及 Round31 五折来源和十条命令。

本机完整可运行测试为 `147 passed, 1 skipped, 2 deselected`；两个 deselect
仍是既有 Windows Anaconda/MKL `lstsq` 原生崩溃用例。相关定向测试为
`66 passed, 1 skipped`，`compileall`、CLI、协议 dry-run 和
`git diff --check` 均通过；dry-run 恰好展开五条 matched continuation 与
五条 ringfilm stage2。

服务器同步 runtime commit `d7dcff5` 后，全套 Linux 测试为
`150 passed, 53 subtests passed`，pipeline `bash -n` 和真实路径 dry-run
均通过。物理 GPU1 启动前利用率 0%，已有其他用户的 `5,749 MiB` 进程未被
停止或修改。

用 fold0 的真实 Round30 thumbnail continuation checkpoint 重建 ring 模型，
转换报告只新增：

- `condition_film.ring_projection.weight`；
- `condition_film.ring_projection.bias`。

随机 64x64 输入和完整 12,901 维 condition 上，来源与转换模型首帧
`torch.equal=True`、`max_abs_diff=0.0`。64 train / 32 val、batch/workers
`8/4`、1 epoch CUDA 预检用时 `4.467 s`，峰值 PyTorch 显存 `4.389 GiB`；
模型参数为 `8,539,393`。训练后 ring projection weight/bias L1 分别为
`1.311383/0.003609`，确认新增路径获得更新。小样本 selection score 为
`73.174420/72.250350`；该值只验证执行，不参与候选 gate。进程退出码为 0，
末尾 `pymp-* EBUSY` 仍是既有 NFS worker finalizer 噪声。

## 正式训练与 OOF 结果

runtime commit `a7fdd26` 上的十条正式 run 全部完成，随后自动执行完整
OOF。机器可读报告为 `reports/round31_cv5_thumbnail_ring_film_oof.json`，
覆盖 25 个 ROI、6,296 张图，评估耗时 `1161.794 s`。

ringfilm 相对 matched continuation 以及未续训 Round30 thumbnail base 的
差值如下。每一项依次为 mean-image PSNR 与 global-MSE PSNR 口径分数：

| 范围 | 编码 | vs continuation mean/global | vs Round30 base mean/global |
|---|---|---:|---:|
| 完整 OOF | float | -0.017920 / -0.020326 | +0.139448 / +0.146485 |
| 完整 OOF | q95 | **-0.018074 / -0.020484** | **+0.139206 / +0.146233** |
| test-like | float | +0.009260 / +0.001562 | +0.198505 / +0.200544 |
| test-like | q95 | **+0.009057 / +0.001381** | **+0.197239 / +0.199276** |

q95 绝对分为：

| 范围 | Round30 base | matched continuation | ringfilm |
|---|---:|---:|---:|
| 完整 OOF | 74.588434 / 73.792092 | 74.745714 / 73.958810 | 74.727640 / 73.938325 |
| test-like | 72.363539 / 71.739632 | 72.551721 / 71.937528 | 72.560778 / 71.938909 |

ringfilm 相对 continuation 的逐折 q95 全部为负：

| fold | mean-image 增益 | global-MSE 增益 |
|---|---:|---:|
| fold0 | -0.023260 | -0.029521 |
| fold1 | -0.007320 | -0.007994 |
| fold2 | -0.006074 | -0.003277 |
| fold3 | -0.021276 | -0.017831 |
| fold4 | -0.033020 | -0.043971 |

test-like 中 ringfilm 相对 continuation 有五个 ROI 双指标为负；最大正增益
是 ROI013 的 `+0.088228/+0.063261`，最大负增益是 ROI022 的
`-0.023531/-0.032310`。环带投影确实获得训练更新，但没有产生可泛化的
结构收益。

## Gate 判定

正式 promotion gate 失败：

1. **失败**：相对 continuation 的完整 OOF q95 两项为负，四项未达到
   `+0.05`；
2. **失败**：完整 OOF float 两项为负；
3. **失败**：0/5 folds 相对 continuation 双指标为正；
4. **失败**：虽然没有单 ROI 低于 `-0.15`，但五个 test-like ROI 双指标
   为负，超过最多两个的限制；
5. **通过**：相对 Round30 base 的完整/test-like 四项 q95 均超过
   `+0.12`。

诊断层 gate 也失败：相对 Round30 base 的完整 OOF q95 只有
`+0.139206/+0.146233`，未达到双 `+0.15`。因此不生成 ringfilm 测试包，
并关闭 center/inner/outer 环带结构路线，不搜索环边界或 projection 变体。

## Matched continuation 观察

matched continuation 相对 Round30 thumbnail base 的结果为：

| 范围 | float mean/global | q95 mean/global |
|---|---:|---:|
| 完整 OOF | +0.157368 / +0.166812 | +0.157280 / +0.166717 |
| test-like | +0.189245 / +0.198982 | +0.188182 / +0.197895 |

五折 q95 双指标全部为正，范围为 `+0.066124` 至 `+0.276792`。但
test-like ROI022 回退 `-0.168058/-0.172371`，所以这仍不是稳定的正式
结构晋级结果。结合 Round29、Round30 的 OOF 和隐藏集均已证明 continuation
方向可迁移，初赛又允许不限次数提交，因此作出独立于 ring gate 的赛务决定：
只生成一次 direct continuation 诊断包。该决定是看过 OOF 后作出的，不能
伪装为预注册 ring 候选。

## Continuation 诊断包

测试推理只加载五个 matched continuation best checkpoint，各权重 `0.2`；
1,346 张图耗时 `199.722 s`。产物为：

- ZIP：`submissions/cd68_round31_cv5_thumbnailcontinue_context64_d4_raw_q95_plain.zip`；
- ZIP 大小：`11,632,408` bytes；
- ZIP SHA-256：`58C9540B99E90688ACF4B27A20AB7C508599D954E1DD235FCA85944F73845203`；
- prediction provenance SHA-256：`6ECEAB5EF00277E08249323F8C01FDD92DC536E84E96E9F5B835C96F2FF88B09`；
- OOF report SHA-256：`04E7524E344DBE25BB22B87BBEFC436831A91F27F079AA676FA20CF86930E83F`。

服务器打包检查 expected/actual `1346/1346`、missing/extra 0、图像错误 0、
全部 RGB，独立 `unzip -tq` 无 CRC 错误。本机同步后的三份 SHA-256 与
服务器一致，条目与测试 DAPI `1346/1346` 精确对应，且
`python -m zipfile -t` 再次通过。平台结果待提交。

## 运行效率说明

服务器有 112 个逻辑 CPU。训练使用 4 个独立 DataLoader workers，`top`
对主 Python 进程显示约 100% 不代表这些 worker 未运行。此前实测 workers=4
比 0 或 8 更快，因此不增加训练 worker。

OOF evaluator 当前固定 `num_workers=0`，SSIM、JPEG roundtrip 和 PSNR
逐图串行，指标阶段确实只使用约一个 CPU 核。实时测试推理中，本项目进程
约 `115% CPU` 时物理 GPU1 已为 `100%` 利用率，说明模型推理本身是 GPU
瓶颈；增加 CPU worker 不会明显加速该部分。后续可以只并行 OOF 数据读取和
指标统计以缩短离线等待，不改变模型、预测或排行榜分数。

## 平台诊断结果与结论

continuation 诊断包得到精确分 `73.1837`，显示 `SSIM=0.823`、
`PSNR=25.911`。相对 Round30 的 `73.0409` 提高 `+0.1428`；显示 SSIM
提高 `+0.001`，PSNR 提高 `+0.081`。相对 Round29 累计提高 `+0.3004`，
相对 Round28 累计提高 `+0.6671`。相对此前记录的榜首 `74.8890`，差距
缩小到 `1.7053`。

隐藏集 `+0.1428` 与完整 OOF 的 `+0.157280/+0.166717` 只差
`0.0145~0.0239`，方向和量级高度一致；test-like OOF 的
`+0.188182/+0.197895` 略高估收益。连续两轮 continuation 均在 OOF 与
隐藏集兑现，证明当前五折协议已经能可靠筛选这一路线。

本次包完全使用原 thumbnail architecture，不包含 ring projection 或
Round30 decoder FiLM，因此增益可以明确归因于新增一阶段 continuation 与
对应的五折 best checkpoint 选择。五折最佳 epoch 为 `7/13/13/13/11`，
多数折在后程改善但均在 epoch14 前见顶。后续可以保留同源 continuation
作为 matched control，但不应假设无限重启 cosine 会持续等幅增长。

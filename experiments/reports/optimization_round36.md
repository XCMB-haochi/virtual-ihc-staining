# 优化记录 Round 36：matched context128 continuation

日期：2026-07-21

## 立项依据

Round35 的 70% base32 / 30% continued-base48 在平台取得 `73.2716`，相对
Round31 提高 `+0.0879`，证明容量多样性集成方向有效，但离已知榜首
`74.8890` 仍差 `1.6174`。继续微调相邻容量权重没有新的独立证据，已有实验
也已否定单纯更换 NAFNet、ConvNeXt、dual-branch Swin、增大宽度、多任务、
EMA、D4 consistency 和若干 conditioning/loss 变体。

当前成熟 U-Net 含四级下采样和每级双 3x3 卷积，中心输出的最大理论感受野
半径约为 100 px。halo64 推理时，靠近 256x256 提交区域边缘的输出仍可能接触
人工 padding；halo128 才能覆盖这一范围。此前 Round22 的 full-target
context64 相对 context0 已在所有 ROI 上给出一致正信号，因此本轮检验更大的
真实邻域，而不是再换模型族。

## 单变量协议

机器可读协议为 `configs/round36_cv5_context128_continue.json`。候选
`context128continue` 从 Round31 每折同一 `thumbnailcontinue` checkpoint
初始化，复用 Round32 matched control 的 seed `2032`、14 epoch、lr `1e-4`
cosine、batch/workers `8/4`、base32 residual U-Net、spatial-thumbnail-film、
Uniform-7 hybrid loss、q95 global-MSE 选模和相同五折。

唯一改变是模型输入从 halo64 增至 halo128；监督目标继续固定 halo64。因此
每个 update 的目标像素、batch、seed、训练长度和初始化均不变，不把更大的
监督区域混入“上下文”结论。Round32 `thumbnailcontinue` 是现成的严格
halo64 control，不重新训练。OOF 一次以最大 halo128 加载，再按 recipe
中心裁剪，Round31/Round34/Round32 基线仍严格接收 halo64。

固定 primary 为 `50% Round35 + 50% context128continue`，展开后是
`35% base32 + 15% continued-base48 + 50% context128continue`。不搜索 halo、
blend 权重、epoch、seed、loss、校准、checkpoint soup 或逐 ROI 路由。

## 预注册判定

首先要求 mechanism gate：`context128direct` 相对 matched
`context64control` 必须同时满足：

1. 完整与 test-like OOF 的 q95 mean/global 四项均至少提高 `+0.08`；
2. 完整与 test-like 的四项 float 增益全部为正；
3. 至少 4/5 folds 的 q95 双指标为正，且五折平均双正；
4. 任一 test-like ROI 的 q95 指标不得低于 `-0.10`，双指标为负的 ROI
   不得超过两个。

primary promotion gate 还要求 `round35_context128_50` 相对固定
`round35_shrink30` 同时满足：

1. 完整与 test-like OOF 的 q95 mean/global 四项均至少提高 `+0.10`；
2. 完整与 test-like 的四项 float 增益全部为正；
3. 至少 4/5 folds 的 q95 双指标为正；
4. 任一 test-like ROI 的 q95 指标不得低于 `-0.10`，双指标为负的 ROI
   不得超过一个；
5. mechanism gate 已通过。

两组条件全部通过才生成唯一 primary 提交包。若 primary 失败，direct 仅在其
相对 Round35 的完整/test-like 四项 q95 均至少提高 `+0.18`、5/5 folds
双正、任一 test-like ROI 不低于 `-0.12` 且至多一个 ROI 双负时，才允许生成
唯一 diagnostic 包。否则关闭 context128，不根据 OOF 或排行榜追加 halo96、
halo160 或相邻融合权重。

## 当前状态

预注册实现提交为 `be655c6`。本机完整可运行回归为
`164 passed, 1 skipped, 2 deselected`；两个 deselect 仍是既有 Windows
NumPy/MKL 原生崩溃用例。A100/Linux 完整回归为
`167 passed, 63 subtests passed`，pipeline 的 `bash -n` 通过，五折 dry-run
仅生成五条 `context128continue/stage2` 命令，全部准确使用输入/目标 halo
`128/64`、seed `2032` 和对应 Round31 同折初始化 checkpoint。

在两张物理 A100 40GB 上分别用 fold0 checkpoint 做了 64/32 张、1 epoch、
batch8 的严格初始化预检。GPU0/GPU1 峰值均为 `7.642985 GiB`，训练加验证
分别耗时 `3.356/4.746 s`；GPU1 预检时已有约 `20.386 GiB` 占用和 60% 利用率，
仍无 OOM，说明正式双卡并行有充足显存余量。末尾 `pymp-* EBUSY` 是既有 NFS
worker 临时目录清理噪声，两个进程退出码均为 0。两个预检 run 和临时 bundle
均已核验路径后清理，其他用户进程与历史日志未改动。

## 正式训练与 OOF

正式运行基于服务器 commit `520f114`。五折最佳 epoch、裸 global-MSE
selection score 和累计 epoch 时间如下：

| fold | 最佳 epoch | selection score | 累计时间 |
|---|---:|---:|---:|
| fold0 | 11 | 73.420217 | 1123.912 s |
| fold1 | 14 | 73.743851 | 2421.513 s |
| fold2 | 11 | 73.256997 | 1118.610 s |
| fold3 | 11 | 74.171342 | 2403.087 s |
| fold4 | 8 | 74.290289 | 1128.592 s |

GPU0 的 fold0/2/4 累计约 56.2 分钟，受其他任务争用的 GPU1 上 fold1/3
累计约 80.4 分钟。未来同类运行应优先给当日空闲的 GPU0 分配更多 fold；本轮
五折均已完整到 epoch 14，不迁移或重跑。正式峰值为
`7.642985--7.645061 GiB`。

OOF 在 GPU0 完成，耗时 `1678.296 s`。机器可读报告为
`reports/round36_cv5_context128_continue_oof.json`，大小 `198,516` bytes，
本机与服务器 SHA-256 均为
`5A24CA658A2A81C1C5B9D05F2E2FE0872E0B8A1998D13566A537AD0861EA0620`。
Round35 baseline 在完整/test-like、float/q95 的 20 个记录字段与原 Round35
报告逐项 bit-exact，证明最大 halo128 加载后按 recipe 裁回 halo64 没有改变
既有基线。
q95 绝对分如下，每项依次为 mean-image/global-MSE 口径：

| 范围 | Round35 | context64 control | context128 direct | 固定 50% 融合 |
|---|---:|---:|---:|---:|
| 完整 OOF | 74.869164 / 74.073735 | 74.824812 / 74.031087 | 74.865385 / 74.086165 | 74.933322 / 74.146584 |
| test-like | 72.714860 / 72.101365 | 72.578825 / 71.971816 | 72.622531 / 72.021367 | 72.739918 / 72.133349 |

context128 direct 相对 matched context64 control 的增益：

| 范围 | float mean/global | q95 mean/global |
|---|---:|---:|
| 完整 OOF | +0.039681 / +0.054212 | **+0.040573 / +0.055078** |
| test-like | +0.042232 / +0.048098 | **+0.043706 / +0.049551** |

固定 50% 融合相对 Round35 的增益：

| 范围 | float mean/global | q95 mean/global |
|---|---:|---:|
| 完整 OOF | +0.063852 / +0.072563 | **+0.064158 / +0.072850** |
| test-like | +0.024719 / +0.031664 | **+0.025058 / +0.031984** |

五折 q95 增益均双正，但幅度偏小：

| fold | mechanism mean/global | primary mean/global |
|---|---:|---:|
| fold0 | +0.040017 / +0.032335 | +0.100827 / +0.094922 |
| fold1 | +0.062115 / +0.081414 | +0.070557 / +0.074874 |
| fold2 | +0.046816 / +0.064093 | +0.040016 / +0.045340 |
| fold3 | +0.017234 / +0.015552 | +0.053761 / +0.064966 |
| fold4 | +0.037059 / +0.081330 | +0.056863 / +0.085787 |

九个 test-like ROI 的 q95 增益：

| ROI | mechanism mean/global | primary mean/global |
|---|---:|---:|
| ROI004 | +0.038925 / +0.025818 | -0.010660 / -0.020167 |
| ROI006 | +0.022748 / -0.000796 | +0.055987 / +0.055098 |
| ROI007 | +0.002590 / +0.004312 | +0.047144 / +0.050373 |
| ROI013 | +0.194791 / +0.343932 | +0.078597 / +0.134529 |
| ROI015 | +0.059427 / +0.052702 | -0.131674 / -0.140698 |
| ROI019 | +0.018855 / +0.017288 | +0.051706 / +0.076624 |
| ROI020 | +0.019967 / +0.017654 | +0.166403 / +0.169238 |
| ROI022 | +0.047344 / +0.036578 | -0.061878 / -0.057182 |
| ROI024 | +0.032011 / +0.037544 | -0.012836 / -0.002840 |

## Gate 判定

mechanism gate：

1. **失败**：完整/test-like 四项 q95 仅为
   `+0.040573/+0.055078/+0.043706/+0.049551`，均低于 `+0.08`；
2. **通过**：四项 float 增益全部为正；
3. **通过**：5/5 folds 的 q95 均双正；
4. **通过**：没有双负 test-like ROI，最差单项仅 `-0.000796`。

primary promotion gate：

1. **失败**：四项 q95 仅为
   `+0.064158/+0.072850/+0.025058/+0.031984`，均低于 `+0.10`；
2. **通过**：四项 float 增益全部为正；
3. **通过**：5/5 folds 的 q95 均双正；
4. **失败**：ROI004、ROI015、ROI022、ROI024 共四个 ROI 双负，且 ROI015
   的 `-0.131674/-0.140698` 低于 `-0.10`；
5. **失败**：mechanism gate 未通过。

direct diagnostic 也失败。context128 direct 相对 Round35 的完整 q95 为
`-0.003780/+0.012430`，test-like 为 `-0.092329/-0.079998`，且只有 fold0/1
双正，远未达到 direct gate。

因此 Round36 不生成测试预测或提交包，不追加 halo96、halo160、相邻融合权重
或排行榜探针；context128 路线关闭。方向上的小幅、五折一致正增益说明减少
人工 padding 确有作用，但收益不足以成为当前竞赛级改进。

OOF 中 GPU 推理阶段能保持高利用率；末尾 SSIM/JPEG 汇总阶段约使用 1--2 个
CPU 核，但只占总 OOF 的数分钟。训练端既有 `workers=4` 已实测最优，指标端
并行化不能保证显著缩短 28 分钟总耗时，因此按用户决定不改评估器。

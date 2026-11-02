# 优化记录 Round 21：局部 U-Net + 全局 Swin 双分支

日期：2026-07-18

## 条件启动

本轮只在 Round20 D4 等变一致性未通过 H2 promotion gate 时启动。配置在 Round20 完整训练和正式 D4 评估完成前固定，不根据其逐 epoch 走势修改。

已有负结果表明，单纯扩大 U-Net 宽度、ConvNeXt-Tiny、NAFNet、多任务、原生 context64、GroupDRO 和 loss 对齐都没有稳定解决跨 ROI 回退。公开虚拟染色方法 HEMIT 的局部卷积 + 全局 Swin 思路提供了新的结构因素；本轮只测试纯回归骨干，不引入 GAN、判别器、感知损失、PatchNCE 或公开预训练权重。

## 固定模型

架构名：`dualbranch-swin-s`。

| 部分 | 固定规格 |
|---|---|
| local 分支 | 当前 base32 residual U-Net |
| global patch embed | `4x4 / stride 4`，1 通道输入 |
| Swin dims | `[48,96,192]` |
| Swin depths | `[2,2,4]` |
| attention heads | `[3,6,12]` |
| window / shift | `8`；奇数 block shift `4` |
| 融合位置 | local 的 `1/4,1/8,1/16` 三个尺度 |
| 融合方式 | `1x1 Conv + GroupNorm + scalar gate` 残差相加 |
| gate 初始化 | 三个均严格为 `0` |
| 输出 | 原 local decoder + 单 CD68 sigmoid head |
| 参数量 | `10,410,134` |

local 分支在 global 分支前按与基线完全相同的模块顺序构造。相同 seed 下，local 初始 tensor 与 base32 U-Net 逐 tensor 相同；初始 gate 为零时，候选输出也与基线逐值相同。loader 使用独立 seeded generator，因此新增 global 参数初始化不改变样本顺序。

禁止修改 window、depth、width、heads、gate 初值、融合位置或分支比例。禁用 dropout、stochastic depth、conditioning、辅助 marker、额外 loss、EMA、test statistics 和输出校准。

## H2 固定协议

| 项目 | 固定值 |
|---|---|
| 验证 ROI | `ROI020,ROI022,ROI024` |
| marker | 仅 `CD68` |
| seed | `2026` |
| 输入 | context0，256x256 |
| batch / workers | `8 / 4` |
| 损失 | Uniform-7 hybrid `0.7/0.2/0.1` |
| stage1 | fresh，14 epoch，AdamW lr `3e-4` cosine，wd `1e-4` |
| stage2 | stage1 `best_global_mse.pt` strict init，14 epoch，lr `1e-4` cosine |
| 选模 | CD68 global-MSE，JPEG q95 |
| 完整评估 | halo64+D4，分别记录 q95 与 float |

固定输出：

- `runs/round21_h2_dualbranch_swin_s_stage1/`；
- `runs/round21_h2_dualbranch_swin_s_stage2/`；
- `runs/_round21_eval/dualbranch_swin_s_h2.json`。

匹配对照固定为 `reports/unet_cd68_h2.json` 及其 checkpoint `runs/round11_h2_unet_cd68_stage2/best_global_mse.pt`。不搜索 epoch、学习率、模型融合、checkpoint soup、TTA 子集或后处理。

## Promotion gate

1. H2 halo64+D4+q95 的 mean-image 与 global-MSE 相对 base32 均至少提高 `+0.25`；
2. ROI020、ROI022、ROI024 的两种 q95 聚合均不得回退超过 `0.05`；
3. float 两种总体增益必须为正，逐 ROI同样不得回退超过 `0.05`；
4. H2 通过后锁定全部设置，在 H1 `ROI015,ROI018,ROI019` 从头重建同样的两阶段轨迹；
5. H1/H2 四个 q95 总体增益必须各至少 `+0.15`、平均至少 `+0.40`，六个 ROI 均不得回退超过 `0.05`；
6. 任一 gate 失败即淘汰，不调整 gate、结构或 loss，也不通过与旧模型融合救援；
7. 双 fold 通过后才训练 full-data，且 full-data epoch 映射必须在启动前另行根据两折同 epoch 指标固定。

## 启动条件

1. 模型任意支持尺寸 forward/backward、参数量、零 gate 因果对照、optimizer 覆盖、checkpoint strict roundtrip 和 D4 形状测试通过；
2. 本地和远端全量测试、compileall、CLI 与 diff check 通过；
3. 预注册文档提交、推送并同步服务器；
4. GPU1 小样本 batch 8 预检确认显存、吞吐、三个 gate 梯度与 checkpoint metadata；
5. 正式输出目录不存在，GPU 资源检查确认不干预其他用户任务。

## 工程验证

- `dualbranch-swin-s` 及相邻本地测试通过；完整本机可运行集合为 `103 passed, 1 skipped, 1 deselected`；
- 远端全量测试 `106 passed, 1 skipped, 45 subtests passed`；
- 512 train / 64 val、batch 8 单 epoch 预检用时 `8.56 s`，峰值 PyTorch 显存 `2.463 GiB`；
- 预检结束后三个 fusion gate 为 `[-0.004903,0.004171,0.005789]`，均已离开零值，证明 global 分支获得训练信号；
- 正式 runtime commit 为 `7a6e972e2fd20364401f77e6802a33e9552834ff`，物理 GPU1；既有 GPU 进程未被停止或修改。

## H2 训练结果

两阶段各完整训练 14 epoch。stage1 每 epoch 约 `58-62 s`、峰值 `2.462 GiB`；stage2 每 epoch约 `45-62 s`、峰值 `2.500 GiB`。

| 阶段 | best epoch | 候选裸 q95 mean / global | 匹配 base32 | 候选 - base32 |
|---|---:|---:|---:|---:|
| stage1 | 10 | 69.287802 / 68.578801 | 69.334928 / 68.636064 | -0.047126 / -0.057263 |
| stage2 | 1 | 69.490524 / 68.829995 | 69.546398 / 68.895627 | -0.055874 / -0.065632 |

stage1 best checkpoint SHA-256 为 `ae3285e276a9c1a18d888ade0cece783e6e6d08494fd94ece2e610b8b2875d61`，gate 为 `[-0.078794,0.035242,0.020219]`。stage2 best checkpoint SHA-256 为 `07851b48e45d1afa9e982284d1b1c095725c464a7dfee659ece1bf0d19a2d127`，gate 为 `[-0.078174,0.033888,0.017749]`。global 分支持续参与，但两个阶段的裸选模指标均未超过匹配 base32。

## 正式评估

固定使用 stage2 epoch 1 的 `best_global_mse.pt`，运行 halo64+D4：

| 输出 | base32 mean / global | 候选 mean / global | 候选 - base32 |
|---|---:|---:|---:|
| q95 | 70.012066 / 69.343546 | 69.948840 / 69.275554 | -0.063226 / -0.067991 |
| float | 70.073465 / 69.404339 | 70.011665 / 69.337794 | -0.061799 / -0.066545 |

q95 逐 ROI 增益：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI020 | -0.020119 | -0.022897 |
| ROI022 | -0.146204 | -0.155215 |
| ROI024 | -0.024489 | -0.024270 |

float 同样三个 ROI 全部为负；ROI022 回退 `-0.143967/-0.152970`，是主要失败来源。完整证据保存为 `reports/round21_dualbranch_swin_s_h2.json`。

## 决策

Round21 淘汰，不启动 H1 或 full-data：两个 q95 总体指标均为负，远未达到 `+0.25`，且 ROI022 明显违反 `-0.05` 回退上限。按预注册规则，不搜索 Swin 深度、窗口、gate 初值、学习率、loss、epoch、与旧模型融合或排行榜提交。结论限定为当前轻量、零门控、无预训练 dual-branch 配方；它反驳“仅增加轻量全局注意力即可突破当前 base32”的假设。

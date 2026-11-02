# 优化记录 Round 15：ImageNet-1K 预训练 ConvNeXt-Tiny U-Net

日期：2026-07-17

## 路线依据

Round 11-14 的持续多任务、NAFNet、原生 context64 与 base48 都出现跨 ROI 回退；base48 虽将总体 q95 提高 `+0.2077/+0.2411`，却使 ROI024 下降 `-0.4869/-0.3830`。这反驳了“只需更多参数或更大上下文”的假设。训练 ROI tile 数 CV 仅 17.1%，ROI024 本身有 288 张、不是稀有域，DAPI 均值也与 ROI020/022 接近，因此不优先做静态 ROI 数量平衡。

公开预训练 encoder 是尚未严格测试且与上述改动正交的强先验。规则允许公开预训练权重；本轮固定使用 torchvision `0.23.0` 的 `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`，不搜索其他 encoder、权重版本或预训练数据。

## 固定模型

- encoder：torchvision ConvNeXt-Tiny 的四级多尺度 features，官方 stochastic-depth 设置保持不变；
- 输入：DAPI `0..1` 灰度复制为 3 通道，仅应用固定 ImageNet mean/std；不调用包含 resize/crop 的 weights transform；
- decoder：逐级 bilinear 上采样、encoder skip、GroupNorm + SiLU 卷积块；
- 额外 full-resolution DAPI shallow skip，缓解 stride-4 stem 丢失像素细节；
- 单输出 sigmoid CD68 head，bias 初始化为 `-2.0`；
- context0 训练；不使用辅助 marker、ROI 权重、测试统计、输出校准、感知损失或 GAN；
- fresh stage1 才加载公开 encoder 权重；checkpoint 构建、推理和 stage2 初始化均不得触发下载；
- config/checkpoint 记录 torchvision 版本、weight enum、URL、缓存文件和完整 SHA-256。

ConvNeXt-Tiny 使用 LayerNorm，避免 batch 8 下预训练 BatchNorm running statistics 与灰度病理域冲突。该完整候选直接与可部署 base32 比较；不额外训练随机初始化 ConvNeXt，因为本轮问题是“该方法是否提高获奖概率”，不是拆分架构与预训练的因果贡献。

## H2 固定协议

| 项目 | 固定值 |
|---|---|
| 验证 ROI | `ROI020,ROI022,ROI024` |
| marker | `CD68` |
| seed | `2026` |
| 输入 | context0，256x256 |
| batch / workers | `8 / 4`；仅 OOM 时停止并实现梯度累积，不降低有效 batch |
| 损失 | Uniform-7 hybrid `0.7/0.2/0.1` |
| stage1 | 14 epoch；encoder lr `3e-5`，decoder/head lr `3e-4`，AdamW，weight decay `1e-4`，cosine 到 0 |
| stage2 | 从 stage1 best global-MSE 初始化新 optimizer；14 epoch；encoder lr `1e-5`，decoder/head lr `1e-4`，cosine 到 0 |
| 选模 | CD68 global-MSE，JPEG q95 |
| 完整评估 | halo64 + D4，同时记录 float/raw 与 JPEG q95 |

固定输出：

- `runs/round15_h2_convnext_tiny_unet_stage1/`
- `runs/round15_h2_convnext_tiny_unet_stage2/`
- `runs/_round15_eval/convnext_tiny_unet_h2.json`

匹配性能对照为 `reports/unet_cd68_h2.json`。不搜索冻结 epoch、encoder/decoder 学习率、decoder 宽度、drop-path、输入尺寸、loss、TTA 子集、checkpoint soup 或融合权重。

## Promotion gate

1. 完整 `halo64 + D4 + q95` 下，相对 base32 的 mean-image 与 global-MSE 都至少提高 `+0.15`；
2. ROI020、ROI022、ROI024 的两种 q95 聚合均不得回退超过 `0.05`；
3. float/raw 两种总体增益都必须为正，逐 ROI 同样不得回退超过 `0.05`；
4. 总体过线但 ROI024 或任一 ROI 失败时仍立即淘汰，不以 pooled 分数掩盖域退化；
5. H2 通过后锁定全部设置，在 H1 `ROI015,ROI018,ROI019` 从同一公开初始权重重新训练；
6. H1/H2 四个 q95 聚合增益必须全部为正，平均至少 `+0.30`，六个 ROI 均不得回退超过 `0.05`；
7. 双 fold 通过后才训练 full-data；H2 失败时不追加 ROI 平衡、冻结层搜索或融合救援。

## 启动条件

1. 新模型 forward/backward、任意可接受尺寸、参数分组、checkpoint roundtrip 和“加载 checkpoint 不下载权重”测试通过；
2. legacy U-Net/NAFNet 和严格 checkpoint 行为保持通过；
3. 本地定向测试、远端全量测试、compileall、CLI 与 diff check 通过；
4. 官方权重下载到 `/workspace` 范围内并记录完整 SHA-256；
5. GPU1 小样本 batch 8 预检确认显存、吞吐、差分学习率和 provenance；
6. 正式输出目录不存在，代码 commit 已推送并同步服务器，且不干预其他用户进程。

## 工程验证状态

- 模型总参数量：34,927,649；
- 新增 forward/backward、归一化、shallow skip、显式权重初始化、完整 SHA、差分 LR、legacy optimizer、禁止隐式下载和真实 strict checkpoint roundtrip 测试；
- 本地新模型测试：`10/10` 通过；相邻 model/checkpoint/train safety 测试通过；
- 远端 Python 3.12 / PyTorch 2.8 / torchvision 0.23 全量测试：`84/84` 通过；
- 本地与远端 compileall、CLI 和 `git diff --check` 通过；
- checkpoint factory 的真实 state dict 恢复在权重下载函数被强制设为异常时仍通过，确认推理/stage2 不会隐式联网；
- 本机全量测试仍受既有 NumPy/MKL `lstsq` native abort 阻断，该缺口已由远端全量测试覆盖。

## 权重与远程预检

官方权重由 torchvision 直接下载并通过内置短哈希校验：

- URL：`https://download.pytorch.org/models/convnext_tiny-983f1562.pth`；
- 位置：`/workspace/.cache/torch/hub/checkpoints/convnext_tiny-983f1562.pth`；
- 大小：114,419,221 bytes；
- 完整 SHA-256：`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`；
- torchvision version：`0.23.0+cu128`；
- 实际加载 encoder tensors：178。

GPU1 使用 512 张训练图、64 张验证图做 batch 8 单 epoch 预检：

- 用时 `18.651` 秒；
- 峰值 PyTorch 显存 `2.480 GiB`；
- decoder/head：7,109,057 参数，初始 lr `3e-4`；
- encoder：27,818,592 参数，初始 lr `3e-5`；
- 总参数 34,927,649，AMP、forward/backward、验证与 checkpoint 保存正常；
- runtime commit：`9409bde`。

预检退出时仍出现已知 NFS multiprocessing 临时目录 `EBUSY` 清理噪声；主进程退出码 0，config、metrics 和 checkpoint 完整，不影响正式训练判断。

## H2 训练结果

两阶段训练均完整跑满 14 epoch：

| 阶段 | 最佳 epoch | 裸验证 q95 mean-image / global-MSE | 总用时 | 平均每 epoch | 峰值显存 |
|---|---:|---:|---:|---:|---:|
| stage1 | 12 | `69.1455 / 68.5723` | `683.7 s` | `48.84 s` | `2.480 GiB` |
| stage2 | 10 | `69.5264 / 68.9424` | `693.0 s` | `49.50 s` | `2.612 GiB` |

最终严格使用 stage2 的 `best_global_mse.pt` 做完整 `halo64 + D4` 评估，没有根据结果改选 epoch、学习率或后处理。

## 完整评估与对照

总体结果：

| 输出 | base32 mean / global | ConvNeXt mean / global | 差值 |
|---|---:|---:|---:|
| JPEG q95 | `70.0121 / 69.3435` | `70.0858 / 69.4614` | `+0.0737 / +0.1179` |
| float/raw | `70.0735 / 69.4043` | `70.1622 / 69.5371` | `+0.0887 / +0.1328` |

逐 ROI 差值：

| ROI | JPEG q95 mean / global | float/raw mean / global |
|---|---:|---:|
| ROI020 | `+0.6369 / +0.6190` | `+0.6544 / +0.6364` |
| ROI022 | `-0.0940 / -0.0004` | `-0.0891 / +0.0045` |
| ROI024 | `-0.2760 / -0.2078` | `-0.2540 / -0.1857` |

原始评估证据保存为 `reports/convnext_tiny_unet_h2.json`。

## 决策

Round 15 淘汰，不启动 H1：

1. q95 总体增益 `+0.0737/+0.1179`，两种聚合均未达到 `+0.15`；
2. ROI022 的 mean-image 回退超过 `0.05`；
3. ROI024 在 q95 和 float/raw 的两种聚合中都回退 `0.18-0.28`，明确违反逐 ROI gate；
4. 结果再次呈现“ROI020 大幅提升、ROI024 明显下降”的同一跨域 trade-off，说明公开预训练与更大模型仍未解决核心瓶颈。

因此不追加冻结层、学习率、encoder、decoder 或融合搜索。下一条主线应直接优化训练 ROI 间的稳健目标，并在实验前固定 GroupDRO/V-REx 的单一路线和 gate。

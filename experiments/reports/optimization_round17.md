# 优化记录 Round 17：统一协议 base32 + base48 固定融合

日期：2026-07-18

## 立项依据

Round 14 的标准 base48 在总体 q95 上相对 base32 提高 `+0.2077/+0.2411`，但 ROI024 回退 `-0.4869/-0.3830`。Round 16 GroupDRO 只修复其中约 `0.04`，不足以解决跨 ROI trade-off。继续训练新模型的证据收益已经很低；现有两个容量不同的 U-Net 是否有可利用的残差互补，可以用一次固定融合直接回答。

旧 base48 + base32 实验使用了不同训练轨迹，且较高结果依赖 log-space ROI 校准；其 raw 50/50 本身不构成本轮正证据。因此本轮不复用旧权重搜索或校准结论，只测试当前 A100 统一协议下的固定 raw 融合。

## 固定候选

- checkpoint A：`runs/round11_h2_unet_cd68_stage2/best_global_mse.pt`，base32 residual U-Net，内部 epoch 1；
- checkpoint B：`runs/round14_h2_unet_base48_stage2/best_global_mse.pt`，base48 residual U-Net，内部 epoch 10；
- 浮点预测：`0.5 * A + 0.5 * B`；
- H2：`ROI020,ROI022,ROI024`；
- 推理：context halo64 + D4；
- 第一阶段只评估 JPEG q95；
- 不使用 ROI calibration、输出缩放、test statistics、checkpoint soup 或 ROI 特定权重。

固定输出：

- `runs/_round17_eval/base32_base48_50_q95.json`；
- 通过 q95 gate 后才运行 `runs/_round17_eval/base32_base48_50_float.json`。

`tools/evaluate_ensemble.py` 由于历史接口会同时计算 calibration variants；本轮只允许读取 `weight_b_0.5_raw`。其他 variant 即使更高也不属于候选，不得用于晋级、改权重或后续提交。

## Promotion gate

1. 相对 `reports/unet_cd68_h2.json`，q95 mean-image 与 global-MSE 都至少提高 `+0.15`；
2. ROI020、ROI022、ROI024 的两种 q95 聚合均不得回退超过 `0.05`；
3. q95 通过后，以完全相同的 50/50 raw 融合运行 float；两种总体增益必须为正，逐 ROI 不得回退超过 `0.05`；
4. H2 通过后，H1 `ROI015,ROI018,ROI019` 从头重建匹配的 base32/base48 轨迹，再评估固定 50/50；
5. H1/H2 四个 q95 聚合增益必须全部为正、平均至少 `+0.30`，六个 ROI 均不得回退超过 `0.05`，才训练 full-data 双模型；
6. H2 任一 gate 失败立即淘汰，不测试 25/75、75/25、ConvNeXt、GroupDRO、三模型融合或 calibration。

H1 的匹配基线固定为本轮从头训练的 base32 stage2 `best_global_mse.pt`，不得使用旧的单阶段 H1 checkpoint。按已经完成的 H2 增益计算，H1 的两个 q95 总体增益必须各自严格为正，且二者之和至少为 `0.3593583464`，四项总体增益的平均值才能达到 `+0.30`。

## Full-data epoch 映射预注册

`--train-all` 没有验证集，因此不会产生可用于选模的 `best_global_mse.pt`。若 H1/H2 gate 通过，full-data checkpoint 采用以下固定映射，不依据 full-data 预测或服务器分数选 epoch：

1. 对每个容量（base32、base48）和每个阶段（stage1、stage2）分别处理；
2. 对 epoch 1--14，计算 H1 与 H2 对应 `metrics.jsonl` 中 bare q95 `selection_scores["global-mse"]` 的等权平均；
3. 固定选择平均值最大的 epoch；精确并列时选择更早的 epoch；
4. full-data 每个阶段仍完整训练 14 epochs，并使用 `--save-every-epoch` 保持与验证阶段相同的 cosine 时钟；
5. stage2 从 full-data stage1 的固定 epoch snapshot 初始化，最终推理使用 full-data stage2 的固定 epoch snapshot；
6. 最终只生成 base32/base48 raw 50/50、halo64、D4、JPEG q95 候选，不搜索权重、校准或 ROI 特定规则。

该映射仅使用两折训练期间预先定义的同名验证指标；H1 完成后只代入数值执行，不修改规则。

## 启动条件

1. 两个 checkpoint 的内部 epoch、base channels、model spec 与 marker 已显式核验；
2. 远端 tracked tree clean，无本项目 GPU 任务；
3. 预注册文档已在评估前提交并推送；
4. 固定命令仅传入 `--weights 0.5 --jpeg-quality 95`，结果只读取 raw 记录。

## H2 结果

正式结果保存为 `reports/base32_base48_50_h2_q95.json` 与 `reports/base32_base48_50_h2_float.json`；它们分别对应预注册时的 `runs/_round17_eval/base32_base48_50_q95.json` 与 `runs/_round17_eval/base32_base48_50_float.json`。所有结论只读取精确名称为 `weight_b_0.5_raw` 的记录。

| 输出 | base32 mean/global | 50/50 raw mean/global | 增益 mean/global |
|---|---:|---:|---:|
| q95 | 70.012066 / 69.343546 | 70.430584 / 69.765669 | +0.418518 / +0.422124 |
| float | 70.073465 / 69.404339 | 70.497133 / 69.831613 | +0.423668 / +0.427274 |

q95 逐 ROI 增益：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI020 | +0.785174 | +0.743490 |
| ROI022 | +0.376054 | +0.393125 |
| ROI024 | +0.128445 | +0.170918 |

float 的三个 ROI、两种聚合也全部为正。H2 满足总体增益、逐 ROI 回退和 float 一致性三项 gate，晋级 H1；校准 variants 未参与判断。

## H1 结果

H1 使用 `ROI015,ROI018,ROI019`，共 727 张验证图。base32 与 base48 均从头训练 14 epochs stage1，再从各自 stage1 `best_global_mse.pt` 严格初始化 14 epochs stage2；两个最终 checkpoint 的内部最佳 epoch 均为 12。

正式结果：

- `reports/unet_base32_h1.json`；
- `reports/unet_base48_h1.json`；
- `reports/base32_base48_50_h1_q95.json`；
- `reports/base32_base48_50_h1_float.json`。

| 输出 | base32 mean/global | base48 mean/global | 50/50 raw mean/global | 融合相对 base32 |
|---|---:|---:|---:|---:|
| q95 | 70.374826 / 69.842100 | 70.587768 / 70.056627 | 70.572124 / 70.039262 | +0.197299 / +0.197163 |
| float | 70.447770 / 69.914658 | 70.662305 / 70.130790 | 70.646159 / 70.112871 | +0.198389 / +0.198213 |

H1 q95 逐 ROI 融合增益：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI015 | +0.082391 | +0.087776 |
| ROI018 | +0.179075 | +0.191937 |
| ROI019 | +0.279145 | +0.269239 |

H1 的 float 逐 ROI 增益也全部为正。H1/H2 四个 q95 总体增益为 `+0.197299`、`+0.197163`、`+0.418518`、`+0.422124`，平均 `+0.308776`；六个 ROI 的两种 q95 聚合均未回退。候选通过双折 promotion gate，进入 full-data。

## 固定 full-data 时钟

把 H1/H2 同 epoch 的 bare q95 global-MSE 按预注册规则等权平均后，固定结果为：

| 容量 | stage1 epoch / 平均分 | stage2 epoch / 平均分 |
|---|---:|---:|
| base32 | 13 / 68.726429 | 12 / 68.999314 |
| base48 | 13 / 68.957841 | 12 / 69.280885 |

因此两种容量均完整训练两个 14-epoch full-data stage；stage2 从 stage1 `epoch_013.pt` 初始化，最终推理读取 stage2 `epoch_012.pt`。

## Full-data 与提交包结果

四个 full-data stage 均使用全部 6296 张训练图，完成 14 epochs 并保存 14 个快照。stage2 的初始化均为 strict marker-compatible load。固定 checkpoint 为：

| 容量 | stage1 epoch 13 SHA-256 | stage2 epoch 12 SHA-256 | stage2 epoch 14 train loss | 峰值显存 |
|---|---|---|---:|---:|
| base32 | `6c85c6ed...08d9e` | `966ba7d8...9934f` | 0.124213 | 2.003 GiB |
| base48 | `b25420a8...a073b` | `e4a9e3b3...7ddf5` | 0.121486 | 3.155 GiB |

最终预测严格使用 stage2 epoch 12 的 base32/base48 float 50/50、halo64、D4、JPEG q95，不使用 ROI scale 或任何 calibration。输出 1346 张测试 JPEG。

提交包：`submissions/cd68_round17_base32_base48_50_context64_d4_raw_q95_plain.zip`。

- 大小：11,227,293 bytes；
- SHA-256：`84abb4d664144eb5c8959c68047d31e29167b03db3aa746179679726cf3ba754`；
- 1346 个条目、1346 个唯一条目、内部路径全部匹配 `results/test/CD68/*.jpg`；
- missing 0、extra 0、image error 0，1346 张均为 RGB 且三通道一致；
- ZIP CRC 检查通过；远端与本机文件大小及 SHA-256 一致。

完整机器可读清单位于 `reports/round17_full_submission.json`。

## 服务器结果

- 精确总分：`71.5953`；
- 平台显示指标：`SSIM=0.804`、`PSNR=25.523`；
- 相对旧 full-data raw 50/50 基线 `71.7440`：`-0.1487`；
- 相对当前最佳端点组合 `72.0437`：`-0.4484`。

Round17 没有兑现双折离线增益。显示 PSNR 比旧 raw 的 `25.414` 高 `0.109 dB`，但显示 SSIM 从 `0.807` 降到 `0.804`，总分仍明显下降。该结果淘汰当前 full-data base32/base48 50/50 全局候选；不继续搜索全局融合权重、epoch、输出缩放或 calibration。双折门槛只能说明两个容量在 H1/H2 留出 ROI 上存在互补性，不能再被当作当前 test ROI 的可靠服务器收益预测器。

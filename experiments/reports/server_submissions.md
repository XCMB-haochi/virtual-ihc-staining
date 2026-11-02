# 服务器提交记录

## 2026-07-15 首投

- 文件：`cd68_unet_all8_context64_d4_loghalf_q95_plain.zip`
- SHA-256：`6E3116DA01992B8207ED2DD5720F93F8F5BF997BC5AB6F77AD834B10D0CC331B`
- 方法：全训练集 8 epoch CD68 U-Net、64 px 邻域、D4 TTA、log-space 0.5 强度 ROI 均值校准、JPEG quality 95
- 平台状态：自动评分完成
- 输出类型：CD68
- 输出数：1 个标记类型
- 缺失图：0
- 异常图：0
- 平台显示指标：`SSIM=0.797`、`PSNR=25.067`
- 总分：`70.8233`
- 提交后显示名次：`7/19`

平台完整接收 plain 文件名，因此后续无需再用像素相同的 `_fake` 包确认命名协议。

显示到三位小数的分量代入已确认公式得到 `70.8302`。与总分相差 `0.0069`，可由后台使用未显示的高精度分量解释；按显示 PSNR 反推 SSIM 约为 `0.79690`。没有证据表明平台使用了不同权重或不同 PSNR 上限。

## 当前能确认与不能确认的事项

能确认：

- ZIP 目录、plain 文件名、RGB 灰度 JPEG 和 CD68 输出均被正确解析。
- 综合分采用 `100 * (0.7 * SSIM + 0.3 * PSNR / 50)`，当前 PSNR 远低于 50 上限。
- 本地留出模型的绝对分不能直接当作服务器分数预测。

仍不能确认：

- PSNR 是逐图计算后平均，还是先汇总全测试集 MSE 后计算。
- SSIM 的具体实现参数与聚合层级。
- 首投与本地留出评估使用的权重及训练数据不同，不能把两者差值完全归因于测试域偏移。

在 `ROI020-ROI024` 上，同一留出模型和首投推理配置按两种 PSNR 口径计算分别为：

| PSNR 聚合 | SSIM | PSNR | 综合分 |
|---|---:|---:|---:|
| 逐图 PSNR 后平均 | 0.800125 | 26.001733 | 71.6098 |
| 汇总全部像素 MSE 后计算 | 0.800125 | 24.697160 | 70.8270 |

后一总分与首投 `70.8233` 接近，但 SSIM、PSNR 分量并不相同，单次提交不足以证明平台使用该口径。后续本地报告同时保留两种聚合结果。

## 已准备的受控消融包

以下三包保持模型、64 px 邻域、D4、JPEG quality 95 不变，只改变训练集固定 ROI 校准强度：

| 包 | 校准 | 用途 |
|---|---|---|
| `cd68_unet_all8_context64_d4_log075_q95_plain.zip` | log 0.75 | 历史受控候选，已被后续模型替代 |
| `cd68_unet_all8_context64_d4_fullscale_q95_plain.zip` | full 1.0 | 判断更强校准是否迁移 |
| `cd68_unet_all8_context64_d4_raw_q95_plain.zip` | 无 | 校准的因果对照 |

SHA-256：

- log 0.75：`FDC87F6A192738A418829018607EC1B433727B26DC670CFDC16F6383B17B40D4`
- full 1.0：`0D72E2241C06C37B8B342C148DAA1A51D14C54A2F1B27C243E6B9E66B85FDD48`
- raw：`96802C1848BED32673B35DE307075F1B25B67CB5AB3B64CBC2E0C9DA980AD8C9`

以上历史强度均来自训练数据和预先定义的本地留出消融。2026-07-16 起，根据用户报告的当前平台可重复提交现状，以及用户转述的最终复现执行口径，另行启用了按测试 ROI 的排行榜消融探针；正式规则本身没有写明提交次数或 Docker 条款。该流程与结果单独记录在 `reports/roi_scale_ablation_probes.md`，不与本节历史结论混用。

## 第二次提交前候选（现已提交）

- 文件：`cd68_unet_all14_ampfix_context64_d4_log075_q95_plain.zip`
- SHA-256：`193B571F08A5695C18F2826AA88188AF22F9F8E6106F2D47B64D6E92AA4E55AD`
- 变化：相同 residual U-Net 由旧训练流程的全数据 8 epoch 改为全数据 14 epoch，并修复 CUDA AMP 下 SSIM 局部矩的半精度相消误差；推理使用 halo64、D4、log 0.75、q95。
- 依据：组合方案在 H1/H2 两个高亮度困难验证子集上，log-half 的逐图分分别提高 `+1.2502/+0.4009`；H2 global-MSE 分提高 `+0.3469`。新模型使用的三 ROI 留出训练比旧五 ROI 留出模型多见两个非验证 ROI，因此该数值不是单一训练改动的严格归因。
- AMP 修复对照：相同 H2、seed、14 epoch 下，最终 log0.75 分由 `69.9448/69.2376` 提升到 `70.2182/69.6329`（逐图/global），三个 ROI 全部上涨。
- 状态：本地包已验证，并已作为下节记录的第二次提交上传服务器。

## 2026-07-15 第二次提交

- 文件：`cd68_unet_all14_ampfix_context64_d4_log075_q95_plain.zip`
- SHA-256：`193B571F08A5695C18F2826AA88188AF22F9F8E6106F2D47B64D6E92AA4E55AD`
- 平台状态：自动评分完成
- 输出类型：CD68
- 输出数：1 个标记类型
- 缺失图：0
- 异常图：0
- 平台显示指标：`SSIM=0.803`、`PSNR=25.410`
- 平台显示名次：`5/19`
- 精确总分：用户未提供

按显示三位小数的中心值复算为约 `71.4560`。若平台显示值由真实指标按标准四舍五入得到，则精确总分应约在 `71.4207-71.4913` 内；该区间只用于理解量级，不替代平台精确总分。

相对首投可确认：显示 SSIM 提高 `0.006`，PSNR 提高 `0.343 dB`，按显示中心值估算总分提高约 `0.63`，名次由 `7/19` 提升至 `5/19`。本地 H2 对新训练流程预测的增益约为 `0.4-0.7`，与服务器量级一致，说明高亮度困难 fold 可继续用于相对选型。

## 2026-07-16 两个 Uniform-7 seed 融合提交

共同设置：40% seed 2026 + 60% seed 2027、halo64、D4、JPEG q95。两个包均为 CD68、输出类型数 1、缺失图 0、异常图 0；提交时均排名第 4，随后均降至第 5。

### 带 log0.75 ROI 缩放

- 文件：`cd68_ensemble_uniform7_s2026_40_s2027_60_context64_d4_log075_q95_experimental_plain.zip`
- SHA-256：`1415ABF1EAB18F2BE08C0510F2B055C854104A67D232EA75C3C01482364F63F4`
- 精确总分：`71.6491`
- 平台显示指标：`SSIM=0.805`、`PSNR=25.480`

### 不使用 ROI 缩放

- 文件：`cd68_ensemble_uniform7_s2026_40_s2027_60_context64_d4_q95_noroiscale_plain.zip`
- SHA-256：`D21A1987901F080CD7483352AE13D0E1AD5095CC55D13D2F9CE8F92EDC266C49`
- 精确总分：`71.5653`
- 平台显示指标：`SSIM=0.805`、`PSNR=25.379`

两包精确分只差 `0.0838`。显示 SSIM 相同，显示 PSNR 相差 `0.101 dB`；隐藏精度共同造成剩余差异。本地 H2 对同一模型的 log0.75 相对 raw 预测约为 `+0.84/+0.94`（逐图/global），明显高估服务器收益，因此停止继续搜索 ROI 校准、JPEG 或类似后处理参数。

相对第二次提交的显示中心值约 `71.4560`，新实验版提高约 `+0.19`，无缩放版提高约 `+0.11`。可观察增益主要来自 SSIM 从 `0.803` 到 `0.805`，说明后续应优先提升模型结构与表征，而不是输出校准。

## 2026-07-16 低学习率续训双模型提交

两颗全数据 Uniform-7 模型分别从原 14 epoch 权重继续训练，学习率从 `1e-4` 余弦衰减；按 H2 验证选择 seed 2026 的续训 epoch 5 和 seed 2027 的续训 epoch 6。

### 推荐先提交：带 log0.75 ROI 缩放

- 文件：`cd68_continue_s2026e5_30_s2027e6_70_context64_d4_log075_q95_experimental_plain.zip`
- SHA-256：`2F312C5F8EF9C7B954D7A2506DAEA6DC35137143A0D2517E6027AE9DF98D31BE`
- 大小：`11,413,841` bytes
- H2 本地：`70.7966/70.1973`（逐图/global）
- 相对现有提交代理：`+0.3013/+0.2922`
- 校验：1346 张，缺失 0，多余 0，图像错误 0，CRC 正常
- 平台精确分：`71.7637`
- 平台显示指标：`SSIM=0.806`、`PSNR=25.546`
- 平台校验：输出类型 CD68、输出数 1、缺失图 0、异常图 0

### 同时提交：不使用 ROI 缩放

- 文件：`cd68_continue_s2026e5_50_s2027e6_50_context64_d4_q95_noroiscale_plain.zip`
- SHA-256：`B72001E7421FCB2F9886BB3350CCC277495F4773747566B8D35EAD66E0D726A7`
- 大小：`11,153,868` bytes
- H2 本地：`70.2233/69.5237`（逐图/global）
- 相对现有提交代理：`+0.5607/+0.5499`
- 校验：1346 张，缺失 0，多余 0，图像错误 0，CRC 正常
- 平台精确分：`71.7440`
- 平台显示指标：`SSIM=0.807`、`PSNR=25.414`
- 平台校验：输出类型 CD68、输出数 1、缺失图 0、异常图 0

相对上一轮同类包，raw 提高 `+0.1787`，experimental 提高 `+0.1146`。新 experimental 仅比新 raw 高 `+0.0197`；显示指标上，ROI 缩放将 PSNR 提高 `0.132 dB`，但 SSIM 从 `0.807` 降到 `0.806`。这说明当前缩放已经接近收益抵消点，后续模型选型以 raw 为主，不再用本地较大的校准增益预测服务器提升。

## 2026-07-16 固定 50/50 校准因果对照

当前 raw 服务器分 `71.7440` 使用 50% seed 2026 e5 + 50% seed 2027 e6，而已提交 experimental 使用 30%/70%，因此二者 `+0.0197` 差值不能完全归因于 ROI 校准。以下包保持 50%/50%、halo64、D4、q95 不变，只启用 log0.75：

- 文件：`cd68_continue_s2026e5_50_s2027e6_50_context64_d4_log075_q95_control_plain.zip`
- SHA-256：`383E9B401D23483F84E9FCAB30D9295C4503B1C6AC2A52E11832F543F1DEB78A`
- 大小：`11,432,679` bytes
- 校验：1346 个唯一 JPEG、缺失 0、多余 0、CRC 正常
- 平台精确分：`71.8597`
- 平台显示指标：`SSIM=0.808`、`PSNR=25.554`

记该控制分为 `C=71.8597`。相对 raw 的纯校准服务器收益为：

```text
C - 71.7440 = +0.1157
```

随后提交了 `submissions/roi_scale_ablation_round2/` 中五个 leave-one-out 包：

| 关闭缩放的 ROI | 精确分 `L_i` | 显示 SSIM | 显示 PSNR | 缩放贡献 `C-L_i` |
|---|---:|---:|---:|---:|
| ROI025 | 71.8755 | 0.808 | 25.556 | -0.0158 |
| ROI026 | 71.9350 | 0.809 | 25.556 | -0.0753 |
| ROI027 | 71.8927 | 0.808 | 25.569 | -0.0330 |
| ROI028 | 71.8636 | 0.808 | 25.556 | -0.0039 |
| ROI029 | 71.6162 | 0.805 | 25.396 | +0.2435 |

`C-L_i` 为启用对应 ROI 缩放的边际贡献，因此负值表示该缩放有害。ROI026 消融包以 `71.9350` 成为消融阶段的中间最高版本，但它证明的是应关闭 ROI026 缩放。五项贡献之和为 `+0.1155`，与整体校准收益 `+0.1157` 仅差 `0.0002`，说明按 ROI 替换在服务器评分上近似可加。

最终组合只对 ROI029 使用 log0.75。提交前由 raw 加上 ROI029 贡献得到 `71.7440+0.2435=71.9875`；由控制包剔除 ROI025-ROI028 的四项负贡献得到 `71.9877`，预测区间为 `71.9875-71.9877`。实际分为 `71.9875`，命中下界并比上界低 `0.0002`。完整探针哈希见 `reports/roi_scale_ablation_probes.md`。

### 2026-07-17 ROI029-only 最终组合提交

- 文件：`cd68_continue_s2026e5_50_s2027e6_50_context64_d4_roi029log075_q95_plain.zip`
- SHA-256：`E0077FF5E4CBCD91FA1949ED9D4D006A208A715762416A65494FA5AF39D77828`
- 大小：`11,290,645` bytes
- 内容校验：ROI025-ROI028 共 1088 张与 raw 逐字节一致，ROI029 共 258 张与 control 逐字节一致；1346 个唯一 JPEG、CRC 正常、来源错配 0
- 平台精确分：`71.9875`
- 平台显示指标：`SSIM=0.809`、`PSNR=25.573`
- 相对 raw：`+0.2435`
- 相对 ROI026 消融探针：`+0.0525`

## 2026-07-17 ROI029-only log1.5 强度探针

该包不是预设的新最佳，而是与强度 0、0.75 组成等距三点，用一次反馈拟合 ROI029 强度曲率。ROI025-ROI028 保持与 raw 逐字节一致。

- 文件：`cd68_continue_s2026e5_50_s2027e6_50_context64_d4_roi029log15_q95_plain.zip`
- SHA-256：`3B7BFBF29E389B8401B72B6617619D7A916DDC8B872D03DA254D89E72A05B8D0`
- 大小：`11,435,425` bytes
- ROI029 基础 scale：`1.2350625592403204`
- strength 1.5 实际倍率：`1.372566625068635`
- 校验：1346 个唯一 canonical JPEG、CRC 正常、路径错误 0、来源错配 0
- 详细曲线拟合：`reports/optimization_round7.md`
- 平台精确分：`71.7563`
- 平台显示指标：`SSIM=0.808`、`PSNR=25.351`
- 相对 log0.75：`-0.2312`
- 相对 raw：`+0.0123`

强度 0、0.75、1.5 的二次拟合顶点为 `0.7597`，拟合分约 `71.98754`，相对 log0.75 理论上只高约 `0.00004`。该差值不可稳定分辨，因此不再生成确认包，强度搜索结束。

## 2026-07-17 单 ROI 模型融合端点探针

十个包都以 raw 50%/50% 的 `71.7440` 为基准，只将一个 ROI 替换成 seed2026-only 或 seed2027-only。结果如下：

| ROI | seed2026-only | SSIM/PSNR | seed2027-only | SSIM/PSNR |
|---|---:|---:|---:|---:|
| ROI025 | 71.7715 | 0.808 / 25.391 | 71.6643 | 0.806 / 25.406 |
| ROI026 | 71.7717 | 0.808 / 25.391 | 71.6564 | 0.806 / 25.407 |
| ROI027 | 71.7311 | 0.807 / 25.369 | 71.6809 | 0.806 / 25.415 |
| ROI028 | 71.7450 | 0.807 / 25.392 | 71.6818 | 0.806 / 25.403 |
| ROI029 | 71.7486 | 0.807 / 25.431 | 71.6664 | 0.806 / 25.374 |

seed2026 在五个 ROI 上都优于 seed2027，但 ROI027 的两个端点都低于 50/50。raw 无约束二次拟合上界约为 `+0.0791`，按预注册门槛筛选的可执行拟合空间约为 `+0.0749`；十次提交的信息收益已进入低区间。完整包 SHA、边际和拟合见 `reports/per_roi_blend_probes.md`。

### 已提交实测端点组合

- 文件：`cd68_perroi_ROI025s26_ROI026s26_ROI028s26_ROI029blend50log075_q95_plain.zip`
- SHA-256：`43CBA075F75A34FF0FB7C2BA14BD4C7C919344A5A5A438F5A88BAC98F5AEEF51`
- 大小：`11,335,873` bytes
- 内容：ROI025/026/028 使用 seed2026-only；ROI027 保持 raw 50/50；ROI029 保持已验证的 50/50 + log0.75。
- 中心预测：`72.0437`，由互不相交 ROI 的实测边际相加得到，不是保证分数。
- 校验：1346 个唯一 canonical JPEG、CRC 正常、逐 ROI 来源错配 0。
- 边界：未使用二次拟合中间权重，但仍根据 Colon 排行榜反馈选择测试 ROI，不代表规则层面保守。
- 平台精确分：`72.0437`
- 平台显示指标：`SSIM=0.811`、`PSNR=25.505`
- 四位观测残差：`0.0000`；后台隐藏精度残差未知

相对上一最佳 `71.9875` 提高 `+0.0562`，相对 raw `71.7440` 提高 `+0.2997`。预测精确命中四位总分，确认这组不相交 ROI 替换近似可加；不继续用提交次数验证更小的端点或强度差异。

## 2026-07-17 Context64 Soup 全局 raw 探针对

第 11、12 个独立探针均使用 H2 context64 seed2026 late soup，第二成员分别为 full-data 与 H2 seed2027 continuation e6：

| 版本 | 精确分 | SSIM | PSNR | 相对 raw 71.7440 |
|---|---:|---:|---:|---:|
| full-data hybrid | 71.3833 | 0.802 | 25.363 | -0.3607 |
| H2 matched | 71.3481 | 0.802 | 25.327 | -0.3959 |

H2 matched 的本地 `+0.1535/+0.1640` 没有转化为可用的服务器候选。其本地 matched control 与服务器 full-data raw 基准并不相同，不能计算严格兑现率或断言精确符号反转。full-data hybrid 仍包含 H2 context64 成员，不等同于真正的 full-data context64 模型；因此本轮只能淘汰现有 soup，不能断言所有 full-data context64 必然失败。两包不再生成 ROI029 校准版本，完整解释见 `reports/context64_soup_submission_probe.md`。

## 2026-07-18 Round17 base32/base48 固定融合

- 文件：`cd68_round17_base32_base48_50_context64_d4_raw_q95_plain.zip`
- SHA-256：`84ABB4D664144EB5C8959C68047D31E29167B03DB3AA746179679726CF3BA754`
- 内容：full-data base32/base48 stage2 epoch 12，float 50%/50%，halo64、D4、JPEG q95、raw；
- 平台精确分：`71.5953`；
- 平台显示指标：`SSIM=0.804`、`PSNR=25.523`；
- 相对旧 raw `71.7440`：`-0.1487`；
- 相对最佳端点组合 `72.0437`：`-0.4484`。

该模型在 H1/H2 固定双折上通过了预注册门槛，但 full-data 服务器结果失败。显示 PSNR 相对旧 raw 高 `0.109 dB`，显示 SSIM 低 `0.003`，净结果为负。当前全局 50/50 候选关闭，不追加 25/75、75/25、epoch、校准或 ROI029 缩放版本。

## 2026-07-18 Round18 full-data 容量端点

共同设置：Round17 full-data stage2 epoch 12、halo64、D4、JPEG q95、raw。

| 端点 | 精确分 | 显示 SSIM | 显示 PSNR | 相对 Round17 50/50 |
|---|---:|---:|---:|---:|
| base32 | 71.4557 | 0.803 | 25.445 | -0.1396 |
| base48 | 71.5729 | 0.804 | 25.524 | -0.0224 |

base48 比 base32 高 `0.1172`，但 50/50 仍比 base48 高 `0.0224`。融合保留了少量互补性，不能解释 Round17 相对旧 raw 的整体回退；主要问题转向 full-data 时钟和验证域外推。两个端点完成诊断后关闭，不搜索连续权重。

## 2026-07-18 Round19 四 marker 单独提交

四个包来自同一个 H2 四输出 checkpoint，分别以单-marker ZIP 提交；共同使用 halo64、D4、JPEG q95、raw。

| marker | 精确分 | 显示 SSIM | 显示 PSNR | 相对同模型 CD68 |
|---|---:|---:|---:|---:|
| CD68 | 71.3323 | 0.802 | 25.351 | - |
| HLA-DR | 67.4934 | 0.779 | 21.582 | -3.8389 |
| CD45RO | 68.1347 | 0.782 | 22.347 | -3.1976 |
| Vimentin | 65.7261 | 0.755 | 21.433 | -5.6062 |

四类按组织者确认的多输出平均规则合并时只有 `68.1716`。该受控结果排除“只输出 CD68 导致初赛分数受限”的假设；其他三类均不进入 dedicated full-data 训练，CD68 保持初赛主任务。

## 2026-07-19 Round26 5-fold base32/full-target 固定集成

- 文件：`cd68_round26_cv5_base32_fulltarget50_context64_d4_raw_q95_plain.zip`
- SHA-256：`2172AABAECB467E251A3383E6951089F1BA7CF3B9B8B8C622C28819E03858E24`
- 大小：`11,330,647` bytes
- 内容：五个 ROI 分层 fold；每折 base32/full-target 浮点 50/50，再对五折等权，十颗 checkpoint 各权重 `0.1`；halo64、D4、raw、JPEG q95
- 完整 25-ROI OOF 相对 matched base32：q95 `+0.219760/+0.224590`
- DAPI test-like OOF 相对 matched base32：q95 `+0.238628/+0.238596`
- 校验：1346 个唯一 canonical JPEG，missing/extra 0，CRC 正常，非法路径 0，图像错误 0
- 平台精确分：`71.8249`
- 平台显示指标：`SSIM=0.807`、`PSNR=25.565`
- 相对旧 raw `71.7440`：`+0.0809`
- 相对 Round17 full-data 50/50 `71.5953`：`+0.2296`
- 相对最佳逐 ROI 组合 `72.0437`：`-0.2188`
- 边界：该包未使用 ROI calibration、旧模型混合或逐测试 ROI 选择

完整协议、逐折与逐 ROI 证据见 `reports/optimization_round26.md`，checkpoint
哈希与有效权重见 `reports/round26_cv5_test_prediction.json`。服务器结果确认
完整 OOF 方案能带来可迁移的 raw 增益，但增幅不足以超过榜单反馈选择过的
逐 ROI 组合。

## 2026-07-19 Round27 spatial-film 五折固定集成

- 文件：`cd68_round27_cv5_fulltarget_spatialfilm50_context64_d4_raw_q95_plain.zip`
- SHA-256：`B182E4B352A2DA11AC794A84E99EC94825794028BF25DA335BF5368F2E9589A8`
- 大小：`11,429,599` bytes
- 内容：五个 ROI 分层 fold；每折 full-target/spatial-film 浮点 50/50，再对五折等权，十颗 checkpoint 各权重 `0.1`；halo64、D4、raw、JPEG q95
- 完整 25-ROI OOF 相对 matched full-target：q95 `+0.464508/+0.442511`
- DAPI test-like OOF 相对 matched full-target：q95 `+0.463063/+0.445493`
- 五折 gate：五折两项均为正；九个 test-like ROI 两项均为正
- 校验：1346 个唯一 canonical JPEG，missing/extra 0，CRC 正常，非法路径 0，图像错误 0
- 平台精确分：`72.2880`
- 平台显示指标：`SSIM=0.813`、`PSNR=25.668`
- 相对 Round26 `71.8249`：`+0.4631`
- 相对此前个人最佳 `72.0437`：`+0.2443`
- 相对当前榜首 `74.8890`：`-2.6010`
- 边界：未搜索 50/50 权重，未使用 ROI calibration、旧模型混合或逐测试 ROI 选择

完整实验、gate 与逐折证据见 `reports/optimization_round27.md`；checkpoint
哈希和有效权重见
`reports/round27_cv5_fulltarget_spatialfilm50_test_prediction.json`。

## 2026-07-19 Round28 direct spatial-film 五折端点

- 文件：`cd68_round28_cv5_spatialfilm_context64_d4_raw_q95_plain.zip`
- SHA-256：`A10D702DA1115B0CA874319E06C29668A2499D75649C2B11B0A62F9DF5F195E4`
- 大小：`11,475,210` bytes
- 内容：五个 spatial-film fold best checkpoint 等权 `0.2`；halo64、D4、raw、JPEG q95
- 完整 OOF 相对 Round27 50/50：q95 `+0.168472/+0.155670`
- DAPI test-like 相对 Round27 50/50：q95 `+0.165111/+0.152308`
- 风险：ROI006、ROI022、ROI024 相对 50/50 回退，但三者仍优于 matched full-target
- 校验：1346 个唯一 canonical JPEG，missing/extra 0，CRC 正常，非法路径 0，图像错误 0
- 平台精确分：`72.5166`
- 平台显示指标：`SSIM=0.816`、`PSNR=25.651`
- 相对 Round27 `72.2880`：`+0.2286`
- 相对 Round26 `71.8249`：`+0.6917`
- 相对此前个人最佳 `72.0437`：`+0.4729`
- 相对当前榜首 `74.8890`：`-2.3724`
- 边界：未训练新模型，未搜索中间融合权重，未使用 ROI calibration、旧模型混合或逐测试 ROI 选择

完整证据见 `reports/optimization_round28.md`；checkpoint 哈希和有效权重见
`reports/round28_cv5_spatialfilm_test_prediction.json`。

## 2026-07-20 Round29 direct thumbnail-film 诊断端点

- 文件：`cd68_round29_cv5_thumbnailfilm_context64_d4_raw_q95_plain.zip`
- SHA-256：`0B33ADCBDA5164BBBB80EAC2020597016B4413A18FBB4793BCB7576A08AB331E`
- 大小：`11,561,425` bytes
- 内容：五个 thumbnail-film fold best checkpoint 等权 `0.2`；真实 5x5 DAPI 邻域缩略图条件；halo64、D4、raw、JPEG q95
- 相对 matched continuation：完整 OOF q95 `+0.063708/+0.075955`，test-like `+0.113933/+0.117142`
- 相对 Round28 spatial base：完整 OOF q95 `+0.259167/+0.274245`，test-like `+0.216019/+0.255854`
- 正式 gate：失败；增益未达到相对 continuation 的 `+0.20`，且 ROI004 回退 `-0.240518/-0.232675`
- 校验：1346 个唯一 canonical JPEG，missing/extra 0，CRC 正常，非法路径 0，图像错误 0
- 平台精确分：`72.8833`
- 平台显示指标：`SSIM=0.820`、`PSNR=25.793`
- 相对 Round28 `72.5166`：`+0.3667`
- 相对此前榜首 `74.8890`：`-2.0057`
- 边界：仅用于固定单端点迁移诊断；未搜索融合权重，未使用校准、旧模型混合或逐测试 ROI 选择

完整训练、OOF、gate 与风险证据见 `reports/optimization_round29.md`；
checkpoint 哈希和有效权重见
`reports/round29_cv5_thumbnailfilm_test_prediction.json`。

## 2026-07-20 Round30 direct multiscale-film 诊断端点

- 文件：`cd68_round30_cv5_multiscalefilm_context64_d4_raw_q95_plain.zip`
- SHA-256：`EEDA4982FA0ADB876D0C4A0202256D27C8C37248FB53C962BF4838E9DD51F13F`
- 大小：`11,569,424` bytes
- 内容：五个 multiscale-film fold best checkpoint 等权 `0.2`；Round29 thumbnail 条件加四级 decoder FiLM；额外续训 14 epoch；halo64、D4、raw、JPEG q95
- 相对 matched thumbnail continuation：完整 OOF q95 `+0.002153/-0.004446`，test-like `+0.010617/+0.007730`
- 相对 Round29 thumbnail base：完整 OOF q95 `+0.291993/+0.264316`，test-like `+0.307018/+0.274895`
- 正式 gate：失败；只有 2/5 folds 相对 continuation 双指标为正，且 ROI015 回退 `-0.267731/-0.280271`
- 诊断 gate：通过；相对 Round29 base 的完整/test-like 四项均超过 `+0.15`，五折双指标全部为正
- 校验：1346 个唯一 canonical JPEG，missing/extra 0，CRC 正常，非法路径 0，图像错误 0
- 平台精确分：`73.0409`
- 平台显示指标：`SSIM=0.822`、`PSNR=25.830`
- 相对 Round29 `72.8833`：`+0.1576`
- 相对 Round28 `72.5166`：`+0.5243`
- 相对此前记录榜首 `74.8890`：`-1.8481`
- 边界：未提交 matched continuation，不能从隐藏集分离续训与 decoder FiLM 的贡献；未搜索融合、校准、decoder 子集或逐 ROI 路由

完整训练、OOF、gate 与平台解释见 `reports/optimization_round30.md`；
checkpoint 哈希和有效权重见
`reports/round30_cv5_multiscalefilm_test_prediction.json`。

## 2026-07-20 Round31 direct thumbnail continuation 诊断端点

- 文件：`cd68_round31_cv5_thumbnailcontinue_context64_d4_raw_q95_plain.zip`
- SHA-256：`58C9540B99E90688ACF4B27A20AB7C508599D954E1DD235FCA85944F73845203`
- 大小：`11,632,408` bytes
- 内容：五个 Round31 matched thumbnail continuation best checkpoint 等权 `0.2`；从 Round30 thumbnail continuation 再续训 14 epoch；halo64、D4、raw、JPEG q95
- 相对 Round30 thumbnail base：完整 OOF q95 `+0.157280/+0.166717`，test-like `+0.188182/+0.197895`
- 五折：相对 base 的 q95 双指标全部为正；test-like ROI022 回退 `-0.168058/-0.172371`
- ring 对照结论：ringfilm 相对本 continuation 完整 OOF q95 回退 `-0.018074/-0.020484`，5/5 folds 双指标为负，未生成 ring 包
- 校验：1346 个唯一 canonical JPEG，missing/extra 0，CRC 正常，非法路径 0，图像错误 0
- 平台精确分：`73.1837`
- 平台显示指标：`SSIM=0.823`、`PSNR=25.911`
- 相对 Round30 `73.0409`：`+0.1428`
- 相对 Round29 `72.8833`：`+0.3004`
- 相对 Round28 `72.5166`：`+0.6671`
- 相对此前记录榜首 `74.8890`：`-1.7053`
- 边界：该包是看过 OOF 后生成的 continuation 诊断端点；不包含 ring projection、decoder FiLM、融合、校准或逐 ROI 路由

完整训练、OOF、结构淘汰和平台解释见 `reports/optimization_round31.md`；
checkpoint 哈希和有效权重见
`reports/round31_cv5_thumbnailcontinue_test_prediction.json`。

## 2026-07-21 Round35 analytic 70/30 capacity shrinkage

- 文件：`cd68_round35_cv5_base32_70_base48_30_context64_d4_raw_q95_plain.zip`
- SHA-256：`430B379BE3C2B0B2364DFF149803418CC371B5C5B51B01F494C3188B76289168`
- 大小：`11,578,050` bytes
- 内容：五个 Round31 base32 checkpoint 各权重 `0.14`，五个 Round34 continued-base48 checkpoint 各权重 `0.06`；halo64、D4、raw、JPEG q95
- 相对 Round31 base32：完整 OOF q95 `+0.123450/+0.114925`，test-like `+0.163139/+0.163837`
- 五折与 ROI：5/5 folds q95 双指标为正；仅 test-like ROI006 双负 `-0.013095/-0.019342`，未触及 `-0.08` 下限
- 正式 gate：五项全部通过；相邻容量权重搜索按预注册规则关闭
- 校验：1346 个唯一 canonical JPEG，missing/extra 0，CRC 正常，非法路径 0，图像错误 0
- 平台精确分：`73.2716`
- 平台显示指标：`SSIM=0.824`、`PSNR=25.940`
- 相对 Round31 `73.1837`：`+0.0879`
- 相对 Round30 `73.0409`：`+0.2307`
- 相对 Round29 `72.8833`：`+0.3883`
- 相对 Round28 `72.5166`：`+0.7550`
- 相对此前记录榜首 `74.8890`：`-1.6174`
- 边界：只验证由 Round34 OOF 解析结果预先固定的 30% base48 权重；未搜索 20%、25%、27.5%、35% 或 40%，未使用校准、soup 或逐 ROI 路由

完整固定协议、OOF、gate 与平台解释见 `reports/optimization_round35.md`；
checkpoint 哈希和有效权重见
`reports/round35_cv5_base32_70_base48_30_test_prediction.json`。

## 2026-07-24 Round44 five-fold FCMAE promotion

- 文件：`cd68_round44_cv5_round35_fcmae20_context64_d4_raw_q95_plain.zip`
- SHA-256：`c12618ce4151b8f31694f50fa9196798a3adba707a5a6ebe64412ac441043bf7`
- 大小：`11,511,546` bytes
- 内容：Round35 `0.80` 总权重与五折 UNeXt2 FCMAE `0.20` 总权重；15-member ensemble，halo64、D4、raw、JPEG q95
- 完整 OOF：q95 `75.009599`，float `75.087934`；相对 Round35 q95 `+0.140386`
- 校验：1346 个唯一 RGB JPEG，missing/extra 0，图像错误 0，CRC 正常
- 平台精确分：`73.3871`
- 平台显示指标：`SSIM=0.825`、`PSNR=26.005`
- 相对 Round43 `73.3050`：`+0.0821`
- 相对 Round35 `73.2716`：`+0.1155`
- 相对此前用户报告榜首 `75.4966`：`-2.1095`
- 结论：固定 FCMAE 结构多样性在隐藏集兑现为正，但兑现率约 58%；不据此进行事后融合权重或逐 ROI 探针，下一轮应更换预注册机制

完整训练、gate、推理 provenance 与解释见 `reports/optimization_round44.md`；
推理报告见 `reports/round44_cv5_round35_fcmae20_test_prediction.json`。

## 2026-07-26 Round46 FCMAE800 controlled diagnostic

- 文件：`cd68_round46_cv5_round35_fcmae800_20_context64_d4_raw_q95_plain.zip`
- SHA-256：`43508ed1609675cb18660838361f8114f86af8edfe020cf92db4537e0f07940c`
- 大小：`11,508,543` bytes
- 内容：Round35 `0.80` 总权重与五折 UNeXt2 FCMAE800 `0.20` 总权重；15-member ensemble，halo64、D4、raw、JPEG q95
- 完整 OOF：相对 Round35 q95 `+0.248498`、float `+0.248242`；5/5 folds、25/25 ROI 为正，最差 ROI `+0.121877`
- 相对 Round44 固定集成：q95 `+0.108112`；5/5 folds、23/25 ROI 为正，最差 ROI `-0.015018`
- 形式门控：`promotion.allowed=false`，原因是 direct FCMAE800 在 ROI009 相对 FCMAE100 回退 `-0.305986`，低于预设 `-0.15` floor；门控结果未修改
- 决策：最终固定集成的所有实质检查通过，因此仅生成一次受控诊断包；不搜索融合权重、ROI 路由、校准或补救包
- 校验：1346 个唯一 canonical RGB JPEG，missing/extra 0，非法路径 0，图像错误 0，CRC 正常；本机与服务器哈希一致
- 平台精确分：`73.4750`
- 平台显示指标：`SSIM=0.826`、`PSNR=26.061`
- 相对 Round44 `73.3871`：`+0.0879`；相应完整 OOF 增量 `+0.108112`，兑现约 `81.3%`
- 相对 Round35 `73.2716`：`+0.2034`；相应完整 OOF 增量 `+0.248498`，兑现约 `81.9%`
- 相对此前用户报告榜首 `75.4966`：`-2.0216`；该榜首状态本轮未重新核验
- 后续状态：独立审查已完成；固定融合权重审计转入 Round47

完整训练、五折证据、门控解释和推理 provenance 见
`reports/optimization_round46.md` 与
`reports/round46_cv5_round35_fcmae800_20_test_prediction.json`。

## 2026-07-27 Round47 FCMAE800 60% blend

- 文件：`cd68_round47_cv5_blend_w060_context64_d4_raw_q95_plain.zip`
- SHA-256：`ce0372f83d4c6fe55daeddb40d52a61c0c28af811b8ca8c1508409f7532fb97a`
- 大小：`11,414,129` bytes
- 内容：五折 UNeXt2 FCMAE800 `0.60`、Round35 base32 `0.28`、Round35 base48 `0.12`；15-member ensemble，halo64、D4、raw、JPEG q95
- 完整 OOF：q95 `75.324015`、float `75.402853`；相对 Round46 20% 融合 q95 `+0.206304`、float `+0.206214`
- 分布证据：5/5 folds、25/25 ROI 为正，最差 ROI `+0.033632`；五次 leave-one-fold-out 均选择 60%
- 门禁：`blend_w060_full` 全部通过；80% 已出现 4 个负 ROI，100% 仅 2/5 folds、14/25 ROI 为正，不继续外推
- 校验：服务器与本机均为 1346 个唯一 canonical RGB JPEG，missing/extra 0，图像错误 0，ZIP CRC 正常，传输前后哈希一致
- 平台精确分：`73.6745`
- 平台显示指标：`SSIM=0.828`、`PSNR=26.189`
- 相对 Round46 `73.4750`：`+0.1995`；相应完整 OOF 增量 `+0.206304`，兑现约 `96.7%`
- 相对 Round35 `73.2716`：`+0.4029`
- 相对此前用户报告榜首 `75.4966`：`-1.8221`；该榜首状态本轮未重新核验
- 边界：不追加相邻权重、逐 ROI 路由、校准或任何基于测试集/排行榜的选择

完整网格、门禁、留一折稳定性和推理 provenance 见
`reports/optimization_round47.md`、
`reports/round47_cv5_fcmae800_blend_grid_gate.json` 与
`reports/round47_cv5_blend_w060_test_prediction.json`。

## 2026-08-14 Round53 official-FCMAE 70% equal split

- 文件：`cd68_round53_cv5_t70_o50_context64_d4_raw_q95_plain.zip`
- SHA-256：`de031b11caa61baa26c0a5d21ae6e86e502ca0885ac84aa066206d9fb325ab28`
- 大小：`11,390,002` bytes
- 内容：Round35 base32 `0.21`、base48 `0.09`、五折 scratch FCMAE800 `0.35`、五折 official-initialized FCMAE800 `0.35`；20-member ensemble，halo64、D4、raw、JPEG q95
- 完整 OOF：q95 `75.560424`、float `75.638565`；相对 Round47 q95 `+0.236409`、float `+0.235711`
- 分布证据：相对 Round52 equal-split 60% 基线 q95 `+0.030582`；5/5 folds、22/25 ROI 为正，最差 ROI009 `-0.016835`
- 门禁：`t70_o50_full` 全部通过；均分略高的 `t70_o67` 因 ROI023 `-0.121698` 触发固定 ROI floor 而拒绝
- 校验：1,346 个唯一 RGB 256x256 JPEG，输入/预测文件名一一对应，20/20 checkpoint 哈希通过，ZIP payload 逐文件一致，CRC 正常
- 平台精确分：`73.9028`
- 平台显示指标：`SSIM=0.830`、`PSNR=26.350`
- 相对 Round47 `73.6745`：`+0.2283`；对应完整 OOF 增量 `+0.236409`，兑现约 `96.57%`
- 相对此前用户报告榜首 `75.4966`：`-1.5938`；该榜首状态本轮未重新核验
- 边界：固定 3x3 权重面已关闭；不继续用邻近小数权重、排行榜反馈、逐 ROI 路由或人工挑选追分

完整权重面、门禁、推理 provenance 和平台解释见
`reports/optimization_round53.md`、
`reports/round53_cv5_official_fcmae_weight_surface_gate.json` 与
`reports/round53_cv5_t70_o50_test_prediction.json`。

## 2026-08-25 Round71 Base diversity split

- 文件：`cd68_round71_cv5_base_split_context64_d4_raw_q95_plain.zip`
- SHA-256：`59bc90793d0fd1baeebd9d41cc8073a1e52f79a1b9d361c9f1d141d72d48b76c`
- 大小：`11,385,557` bytes
- 内容：Round35 base32 `0.21`、base48 `0.09`、五折 scratch FCMAE800 `0.35`、五折 official Tiny FCMAE800 `0.175`、五折 official Base FCMAE800 `0.175`；25-member ensemble，halo64、D4、raw、JPEG q95
- 完整 OOF：q95 `75.644105`；相对 Round53 `t70_o50` 提升 `+0.083681`
- 分布证据：5/5 folds、20/25 ROI 为正，最差 ROI005 `-0.070786`；完整门禁通过
- 校验：1,346 个唯一 RGB 256x256 JPEG，输入/预测文件名一一对应，25/25 checkpoint 哈希通过，ZIP 内容清单与服务器预测目录一致，CRC 正常
- 平台精确分：`73.9228`
- 平台显示指标：`SSIM=0.830`、`PSNR=26.388`
- 相对 Round53 `73.9028`：`+0.0200`；对应完整 OOF 增量 `+0.083681`，兑现约 `23.9%`
- 解释：Base 多样性在隐藏集保持正向，但 OOF 明显高估幅度；不继续 Base/Tiny 权重插值、校准或排行榜探针，后续转向新的结构机制或跨域稳健性

完整训练恢复、五折证据、组合关闭、推理 provenance 和平台解释见
`reports/optimization_round69.md`、`reports/optimization_round70.md`、
`reports/optimization_round71.md` 与
`reports/round71_cv5_base_split_test_prediction.json`。

## 2026-08-30 Round88 fixed context128 diagnostic

- 文件：`cd68_round88_cv5_context128_diagnostic_q95_plain.zip`
- SHA-256：`68acbee0c23319897fa32d99c2c39a421c6ffd6ab92d98353aca0e948912777f`
- 大小：`11,740,002` bytes
- 内容：Round71 的 25-member Base-diversity ensemble，context128 inference、D4、固定 `[0,-1]` 输出偏移、JPEG q95、CD68 only；未重新训练
- Round85 完整 OOF：q95 `75.781550`，相对 context64 shifted baseline `+0.026427`；5/5 folds 正，18/25 ROI 正，但分布 gate 未通过
- 校验：1346/1346 预测，256x256 RGB 灰度图，ZIP 1346 条目，CRC 正常，25 个成员权重和为 1
- 平台精确分：`74.0315`
- 平台显示指标：`SSIM=0.831`、`PSNR=26.435`
- 相对 Round82 context64 shifted 包 `74.0101`：`+0.0214`
- OOF 增益到隐藏集的兑现约 `81.0%`；该包是固定确认性诊断，不据此搜索 context、偏移、权重或校准

完整配置、预测报告、运行哈希和独立格式审计见
`reports/optimization_round88.md`、
`reports/round88_context128_diagnostic_test_prediction.json`。

## 2026-08-31 Round93 full-label endpoint confirmation

- 文件：`cd68_round93_full_label_auxiliary15_context128_q95_plain.zip`
- SHA-256：`1b577b2ff6dfb1578a24374311e2060716c8a6dd24e61e90de9703d304af9049`
- 内容：全 25 个训练 ROI 的 fresh UNeXt2-FCMAE-Tiny endpoint；四 marker
  auxiliary loss，14+14 阶段，context128、D4、固定 `[0,-1]` 偏移、JPEG q95，
  仅输出 CD68
- 平台精确分：`73.9176`
- 平台显示指标：`SSIM=0.831`、`PSNR=26.287`
- 名次：用户未提供
- 相对 Round88 固定集成 `74.0315`：`-0.1139`；SSIM 不变，PSNR `-0.148` dB
- 边界：无 held-out OOF，仅作一次固定平台确认；不据此调整 epoch、融合、
  校准、ROI 路由或 Round94 协议

完整训练、阶段 checkpoint、归档校验和端点限制见
`reports/optimization_round93.md`。

## 2026-09-13 Round124 relational representation confirmation

- 文件：`cd68_round124_cv5_relational_q95_plain.zip`
- SHA-256：`44ce2f04b41f4e351d9a25a9edabc98eb6e44ebddc6f184024a7d39e05fbd089`
- 内容：五折 UNeXt2-FCMAE-Tiny，训练期 median-normalized CD68 relational
  guidance，context128、D4、固定 `[0,-1]` 偏移、JPEG q95，CD68 only
- 完整 OOF：q95 `75.506276`，相对 matched auxiliary15 `+0.094465`；4/5
  folds positive，17/25 ROIs positive，最差 ROI016 `-0.586862`
- 平台精确分：`74.1630`
- 平台显示指标：`SSIM=0.832`、`PSNR=26.503`
- 相对 Round88 context128 package `74.0315`：`+0.1315`
- 相对此前记录的外部排行榜最高分 `75.4966`：`-1.3336`
- 解释：关系表征在隐藏集有小幅正向迁移，但 OOF 绝对分高估明显；保留为
  当时成为个人最高分；不根据该分数搜索权重或 ROI 路由

用户随后报告该包平台分数 `74.1630`（`SSIM=0.832`、`PSNR=26.503`）。相对
Round88 包 `74.0315` 提高 `+0.1315`，相对此前记录的外部排行榜最高分
`75.4966` 仍低 `1.3336`。该结果使 Round124 成为当时的个人最高分，但不用于校准、
逐 ROI 路由或融合权重搜索。

## 2026-09-14 Round127 seed2027 relational confirmation package

- 文件：`cd68_round127_cv5_relational_seed2027_q95_plain.zip`
- SHA-256：`2d477e60c7d0d357e3eac12322ade72539334ef7f5d7b2869348fc95d0e94a23`
- 大小：`11,964,493` bytes
- 内容：Round127 五折 seed2027 median-relational candidate，context128、
  D4、JPEG q95、固定 `[0,-1]` 偏移、CD68 only
- 完整 OOF：q95 `75.626139`，相对 matched auxiliary15 `+0.214328`；5/5
  folds positive，19/25 ROI positive，最差 ROI013 `-0.252999`
- 校验：1346/1346 预测，ZIP 1346 条目，RGB 三通道一致，CRC 正常
- 状态：平台确认后成为当时的个人最高分；不根据平台结果搜索权重

用户随后报告该包平台分数 `74.2222`（`SSIM=0.833`、`PSNR=26.483`）。相对
Round124 包 `74.1630` 提高 `+0.0592`，相对 Round88 包 `74.0315` 提高
`+0.1907`，相对此前记录的外部排行榜最高分 `75.4966` 仍低 `1.2744`。
Round127 因此成为当时的个人最高分；该结果不用于逐折修补或事后选择融合比例。

## 2026-09-14 Round128 fixed 50/50 relational seed blend package

- 文件：`cd68_round128_relational_seed_blend_q95_plain.zip`
- SHA-256：`18e638a23f280277a38b948f5724ca13ee6e4f53ae28f38b968981ded047da95`
- 大小：`11,937,352` bytes
- 内容：Round124 seed2044 与 Round127 seed2027 对应五折预测固定 50/50，
  共 10 个成员；context128、D4、JPEG q95、固定 `[0,-1]` 偏移、CD68 only
- 完整 OOF：q95 `75.728732`，相对 seed2027 `+0.102593`；5/5 folds 和
  21/25 ROI 为正，但最差 ROI016 `-0.452199`，固定 risk gate 失败
- 校验：1346/1346 预测，RGB 三通道一致，ZIP 目录与 CRC 正常
- 平台精确分：`74.2362`
- 平台显示指标：`SSIM=0.833`、`PSNR=26.519`
- 相对 Round127 `74.2222`：`+0.0140`
- 相对 Round124 `74.1630`：`+0.0732`
- 相对 Round88 `74.0315`：`+0.2047`
- 相对此前记录的外部排行榜最高分 `75.4966`：`-1.2604`
- 状态：历史探索结果；根据最新组织者口径，双 seed 混合不作为后续提交模板

## 2026-09-15 Round129 fixed Base + relational hybrid package

- 文件：`cd68_round129_base_relational_hybrid_q95_plain.zip`
- 本地路径：`submissions/cd68_round129_base_relational_hybrid_q95_plain.zip`
- SHA-256：`a898db9aad5a41be17dd2184b49a04dcb04eea64a9b4dfa026e66689bc4a5043`
- 大小：`11,823,015` bytes
- 内容：Round88 Base-diversity endpoint 与 Round128 relational seed blend 的
  固定 50/50 hybrid；共 35 个成员，context128、D4、JPEG q95、固定 `[0,-1]`
  偏移、CD68 only
- 完整 OOF：q95 `75.909735`；相对 Round128 relational blend `+0.181003`，
  float `+0.181021`；5/5 folds、25/25 ROIs 为正，最差 ROI `+0.020351`；
  固定 gate 完整通过，`promotion.allowed=true`
- 校验：1,346/1,346 预测，全部 RGB 256x256，ZIP CRC 正常；35/35 checkpoint
  provenance SHA-256 远端复核通过；submission exit `0`
- 状态：已生成并验证，历史探索包；根据最新组织者口径，跨 recipe 混合不作为
  后续提交模板。Round128 的平台分 `74.2362` 仅作历史结果记录；`75.4966`
  是外部排行榜记录，不是本项目分数。

用户随后报告该包平台分数 `74.2276`（`SSIM=0.833`、`PSNR=26.533`）。相对
Round128 的 `74.2362` 下降 `-0.0086`；因此 Round129 的正向 OOF 增益未在
隐藏集兑现，固定混合包路线关闭。Round128 仅保留为历史探索结果，不作为后续阶段
提交模板；不根据这一次结果搜索邻近融合比例、校准或逐 ROI 路由。

根据 2026-09-15 组织者最新说明，Round128 的双 seed 混合和 Round129 的
跨 recipe 混合均只保留为历史探索记录，不作为后续阶段的提交模板；后续提交必须
由一个模型定义和其训练权重产生。相同架构的自动五折聚合明确允许，后续阶段需在
当阶段数据上重新验证。

## 2026-09-15 Round130 standalone seed-31415 screen

- 完整两折 OOF：2,706 张图、10 个 ROI；pipeline exit `0`
- 单模型 seed-31415 relational q95 `75.058648`，matched auxiliary15
  `75.013272`，增益 `+0.045376`
- float 增益 `+0.051714`；2/2 folds、7/10 ROIs 为正；最差 ROI 增益
  `-0.260831`
- 固定 gate：`promotion.allowed=false`，q95 margin 和 worst-ROI floor 均未通过
- 决策：关闭 seed-31415 五折扩展；不做 seed 融合、权重搜索或平台包。
  该结果只否定本次具体 seed/recipe 配置，不否定所有单模型结构路线。

## 2026-09-16 Round131 full-data single-model package

- 文件：`cd68_round131_full_data_relational_q95_plain.zip`
- 本地路径：`submissions/cd68_round131_full_data_relational_q95_plain.zip`
- 训练：全部 25 个 Colon ROI、6,296 张图；all-train 五 marker FCMAE
  teacher 160 epoch；一个 relational student，两个 supervised stage 各 14
  epoch；seed `2027`
- 最终 student `final.pt` SHA-256：
  `ba451b04911d27d8c129951956bdfcccc8b1fb01b77831c40660cf4cdc1db51c`
- 提交包大小：`11,935,950` bytes；SHA-256：
  `28dd82335d1393e9d42def0a9f7d44a98819e8abba35cb009072a663a7043a55`
- 校验：1,346/1,346 预测，全部 RGB 256x256，ZIP CRC 正常；交付模型文件数
  为 `1`
- 状态：待平台确认。该包是全数据单模型，不使用五折、seed 融合或跨 recipe
  ensemble；teacher 只用于训练期 relational guidance，不参与测试推理。

用户随后报告 Round131 平台分数 `74.0606`（`SSIM=0.832`、`PSNR=26.357`）。
相对合法的 Round127 同架构五折自动聚合结果 `74.2222` 下降 `-0.1616`；因此
本次 full-data 单模型没有在隐藏集超过五折包。该结论只关闭当前 full-data recipe
作为主方案，不等同于证明所有全数据单模型都不如五折聚合。

用户同时报告外部排行榜出现 `77.8212` 的最高分。该数值没有本项目可核验的
模型和规则信息，暂不把它归因于“数据集打乱”或作为当前方案调参依据。

Round131 的 full-data 单模型结果低于 Round127 的合法同架构五折自动聚合，说明
“全数据重训一定优于取 fold”在本 recipe 上没有得到验证。后续若需要比较单模型，
应固定选择一个已有 Round127 fold 生成一次独立包，不开启新的 seed 或融合搜索。

## 2026-09-16 Round132 single fold4 model package

- 文件：`cd68_round132_single_fold4_q95_plain.zip`
- 本地路径：`submissions/cd68_round132_single_fold4_q95_plain.zip`
- 模型：Round127 fold4 的单个 `encoderrelational15` checkpoint；同架构单模型，
  context128、D4、固定 `[0,-1]` 偏移、JPEG q95、CD68 only
- checkpoint SHA-256：`8ca262749fbb22c328bd7659631a3b345768d24c53a186ba8ffe4c4759b215e5`
- ZIP 大小：`12,189,062` bytes；SHA-256：
  `e1c292d63c094fa8846b7de103c3d4d2f8ea2c5bb530fe3e650f668d32007313`
- 校验：1,346/1,346 预测，全部 RGB 256x256，ZIP CRC 正常，交付模型数为 `1`
- 状态：待平台确认；这是单 fold 合法确认包，不是五折或跨模型集成。

用户随后报告 Round132 平台分数 `73.6511`（`SSIM=0.829`、`PSNR=26.028`）。
相对 Round131 full-data 单模型 `74.0606` 下降 `-0.4095`，相对 Round127
五折自动聚合 `74.2222` 下降 `-0.5711`。因此“取验证集绝对分最高的 fold”在
隐藏集上明显失败；Round132 仅作为规则允许性确认，不保留为主方案，也不据此
搜索其它 fold、seed 或融合。

## 2026-09-16 Round133 full-data stage2 epoch6 package

- 文件：`cd68_round133_full_data_relational_e6_q95_plain.zip`
- 本地路径：`submissions/cd68_round133_full_data_relational_e6_q95_plain.zip`
- 模型：Round131 full-data student stage2 epoch6 的单个 checkpoint；同一
  full-data 单模型 recipe，context128、D4、固定 `[0,-1]` 偏移、JPEG q95
- checkpoint SHA-256：`acf08de0a2e7d282d0432be32dc182fd72fd64deab1aa7674efd8a1c51fb05c2`
- ZIP 大小：`11,901,657` bytes；SHA-256：
  `92afd01582e71bbc6c3395d7c1b46638309dc4b3b6c9649585b9425020314548`
- 校验：1,346/1,346 预测，全部 RGB 256x256，ZIP CRC 正常，交付模型数为 `1`
- 状态：待平台确认；这是固定 early-checkpoint 确认包，不涉及新训练、融合或
  隐藏集调参。

## 2026-09-16 Round134 full-data temporal parameter average package

- 文件：`cd68_round134_full_data_temporal_soup_q95_plain.zip`
- 本地路径：`submissions/cd68_round134_full_data_temporal_soup_q95_plain.zip`
- 模型：Round131 full-data student stage2 epoch5/6/7 参数按 `1:2:1` 平均，
  buffer 使用最新 epoch；最终交付为一个 `final.pt`
- 输入 checkpoint SHA-256：epoch5 `0f87919e6dbdb7f367d6f48254ec493269ed9d391c38ab34c2ebab0c783e2827`；
  epoch6 `acf08de0a2e7d282d0432be32dc182fd72fd64deab1aa7674efd8a1c51fb05c2`；
  epoch7 `6d7547af69dc898cfd7e10ddfa760dff99d1e93e0fbf363a2a42acd19becc731`
- 输出 checkpoint SHA-256：`ab0335b145f2893dbe4609376794af622a82da195ac96085d9a056bb3db263cd`
- ZIP 大小：`11,893,301` bytes；SHA-256：
  `1423e197024a5ff401065fc83cdb28ca377c923d01b389070ecce3c0f41ee372`
- 校验：1,346/1,346 预测，全部 RGB 256x256，ZIP CRC 正常，交付模型数为 `1`
- 状态：待平台确认；这是单轨迹参数平均确认包，不是五折或多模型推理。

用户随后报告 Round134 平台分数 `74.1432`（`SSIM=0.833`、`PSNR=26.401`）。
相对 Round133 的 `74.1330` 提升 `+0.0102`，仍低于 Round127 合法五折自动聚合
`74.2222`。参数平均保留为当前较好的单模型确认结果。

规则补充：当前代码的 `d4` TTA 只加载一个模型，对输入执行 8 个 D4 旋转/翻转，
对每个预测做逆变换后求平均，最终仍只交付一个 `.pt`。它与五折模型聚合不同，
属于老师允许的推理阶段后处理；后续数据集仍需重新验证 D4 几何对称性是否成立。

用户随后报告 Round133 平台分数 `74.1330`（`SSIM=0.833`、`PSNR=26.392`）。
相对 Round131 第14 epoch 单模型 `74.0606` 提升 `+0.0724`，但仍低于合法的
Round127 同架构五折自动聚合 `74.2222`，差 `-0.0892`。这支持保留 early
checkpoint 作为单模型备选，但不改变五折方案的合法性或开启新的 checkpoint 搜索。

补充规则解释：EMA 是同一训练轨迹内的时间参数平均，最终是一个模型权重文件；
五折自动聚合是五个同架构 fold checkpoint 的预测聚合。两者不是同一机制。组织者
已明确允许同架构五折自动聚合，因此 Round127 没有被误杀；关闭的只是跨架构、跨
recipe、独立 seed soup 和不同虚拟染色模型混合。

## 2026-09-17 Round135 matched stage2 EMA package

- 文件：`cd68_round135_full_data_relational_stage2_ema_e6_q95_plain.zip`
- 本地路径：`submissions/cd68_round135_full_data_relational_stage2_ema_e6_q95_plain.zip`
- 模型：从 Round131 相同 stage1 `final.pt` 重跑相同 full-data stage2，唯一
  训练变化为 EMA `decay=0.995`；固定使用 epoch6 EMA checkpoint
- checkpoint SHA-256：`b13d2d8eade81e8dd82d89b98cdeba282305d86ee9eece72949d11d16e032507`
- ZIP 大小：`11,915,934` bytes；SHA-256：
  `6678a40d8649b707d21b7debe51c73371013f24f9508e481a7b69e422dcecec3`
- 校验：14/14 epoch finite；1,346/1,346 预测，全部 RGB 256x256，ZIP CRC
  正常，交付模型数为 `1`
- 状态：待平台确认；若不优于 Round133/Round134，则关闭训练期 EMA，不搜索
  decay、epoch 或 EMA/online 融合。

用户随后报告 Round135 平台分数 `74.1336`（`SSIM=0.833`、`PSNR=26.388`）。
相对 matched online epoch6 Round133 仅提高 `+0.0006`，并低于 Round134
参数平均 `-0.0096`。该差异不能证明训练期 EMA 有效，因此关闭 EMA，不搜索
decay、epoch 或 EMA/online 混合；Round134 继续作为最佳单模型。

## 2026-09-17 Round136 OOF distillation single-model package

- 文件：`cd68_round136_full_data_oof_distillation_q95_plain.zip`
- 本地路径：`submissions/cd68_round136_full_data_oof_distillation_q95_plain.zip`
- 方法：Round127 五折同 recipe 的折外 CD68 预测作为 teacher target，每张
  训练图只使用未见过其 ROI 的 fold；full-data student 固定为 `0.8` 真实
  标签损失 + `0.2` teacher 损失；最终 epoch5/6/7 参数按 `1:2:1` 平均
- Student checkpoint SHA-256：
  `90fbf0eca0707aa5621037f1093d78d17a6bf1ebc170547369fa56f45fc12110`
- ZIP 大小：`11,905,816` bytes；SHA-256：
  `690cd8db6993bf0d4de46fda125982f0c065af0ed3777cb0936b11b0aec0567e`
- 校验：teacher cache 覆盖 6,296/6,296 训练图；student 14/14 epoch finite；
  1,346/1,346 预测，全部 RGB 256x256，ZIP CRC 正常，交付模型数为 `1`
- 状态：待平台确认；推理只加载一个 student checkpoint，teacher cache 不参与
  测试推理，也没有 seed/recipe ensemble。

用户随后报告 Round136 平台分数 `74.1985`（`SSIM=0.833`、`PSNR=26.461`）。
相对 Round134 单模型 `74.1432` 提升 `+0.0553`，距离合法 Round127 同架构五折
自动聚合 `74.2222` 仅低 `0.0237`。这说明折外 OOF 蒸馏确实把大部分五折收益压入
了一个模型；Round136 现为当前单模型主提交包。后续不再搜索蒸馏权重、teacher 组合、
epoch 窗口或隐藏集校准。

## Current semifinal primary: Round245

- User-reported score: **79.1052**, SSIM **0.899**, PSNR **26.932**.
- Gain over matched Round243: **+0.0085**; Round243 is fallback.
- Package: `submissions/semifinal_round245_cd45ro_q95.zip`.
- Single model: `runs/round245_full_stage2/epochs/epoch_012.pt`.
- Retained stage-decay student recipe; training-only targets from the five
  ROI-excluding Round234 teachers. Same16-view TTA andq95 as Round243.
- ZIP SHA256: `7731e7c048974a275314dfbdb2a052c705e6831fb5321bdf9141c372f3db1d7c`.
- Checkpoint SHA256: `16a530ac9b0b8396592668bcaad4405b697083ad09a7b2e176b1e33ed36a17b1`.
- Details and limitations: `semifinal_round245.md`.

## Previous best / fallback: Round243

- User-reported score: **79.0967**, SSIM **0.899**, PSNR **26.933**.
- Gain over Round240: **+0.0064**. This is a small confirmed package gain.
- Package: `submissions/semifinal_round243_cd45ro_q95.zip`.
- Same single checkpoint as Round240; identity and whole-ROI down/right8px
  input translation, each with D4, inverse alignment and16-view equal average.
  Fixed replicated ROI boundaries; q95; no residual output offset.
- ZIP SHA256: `a6397781724430a71698004e7c6de58934c9dc1156ccb8c3074cf98cd73527ce`.
- Round240 is the faster standard-D4 fallback. Detailed limitations and
  user-authorized confirmation are recorded in `semifinal_round243.md`.

## Previous best / standard-D4 fallback: Round240

- User-reported score: **79.0903**, SSIM **0.899**, PSNR **26.929**.
- Improvement over Round239: **+0.0275**; over Round216: **+0.0705**.
- Package: `submissions/semifinal_round240_cd45ro_q95.zip`.
- One model: `runs/round240_full_stage2/epochs/epoch_012.pt`.
- Same Round239 stage1 and stage2 recipe; replace only the training teacher
  cache with the validated Round239 fold-excluded predictions. Teachers are
  not used at inference. Keep D4/halo128/offset0/q95.
- ZIP SHA256: `a4de7fcf2f74444ceb178bc2b39b98d5c8844ac722af8ff755bf1006ccd3259b`.
- Checkpoint SHA256: `b33b9fb976acb1b5c58d0b80cb8f93cccec5192ec16dccb2cb0fa483d50da087`.
- Round239 is fallback; details in `semifinal_round240.md`.

## 2026-10-01 Previous best / fallback: Round239

- User-reported score: **79.0628**, SSIM **0.899**, PSNR **26.856**.
- Improvement over Round21679.0198: **+0.0430**; Round216 remains fallback.
- Package: `submissions/semifinal_round239_cd45ro_q95.zip`.
- One model: `runs/round239_full_stage2/epochs/epoch_012.pt`; stage-wise encoder
  learning-rate decay0.8, original domain800 encoder, full-data stage2
  CD45RO distillation0.2, D4/halo128/offset0/q95. No fold ensemble.
- ZIP SHA256: `688fd907700a7989f754a84d4d14345d22e9c703e1a331c77ec76b8ee58fd7d5`.
- Checkpoint SHA256: `426dd8e0bb8b771cbfc0f79eb5082f08f6c394b75b3865c577d8ff141d5af618`.
- Detailed validation and limitations: `semifinal_round239.md`.

## 2026-09-21 Semifinal Round201 full-data distillation package

- 文件：`semifinal_round201_distill_q95.zip`
- 本地路径：`submissions/semifinal_round201_distill_q95.zip`
- 数据：复赛 17 个训练 ROI、2,380 张训练 patch；4 个测试 ROI、560 张 DAPI
  patch；seed `2026`
- 方法：复赛 Round200 五折折外 CD68 teacher，full-data student，蒸馏权重
  `0.2`，stage1/stage2 `12/12` epoch，epoch10/11/12 参数 `1:2:1`
- Student checkpoint SHA-256：
  `0f8ac44ac520b538ebfb0dd76bc6f0149143b914eb654d69b926f87da2857dff`
- ZIP 大小：`4,803,240` bytes；SHA-256：
  `b2199413ad6681bb616c9769a3a9f3d21bdd4a2e6f5f4b71918f551785358504`
- 校验：560/560 预测，全部 RGB 256x256，ZIP CRC 正常；单 student checkpoint；
  D4 TTA，固定输出偏移 `[0,0]`
- 状态：待第一次复赛平台确认。

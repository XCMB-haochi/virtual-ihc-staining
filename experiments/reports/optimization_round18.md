# 优化记录 Round 18：full-data 容量端点诊断

日期：2026-07-18

## 起点

Round17 full-data base32/base48 raw 50/50 的服务器分为 `71.5953`，显示 `SSIM=0.804`、`PSNR=25.523`。相对旧 full-data raw 50/50 的 `71.7440` 低 `0.1487`；相对当前最佳逐 ROI 组合 `72.0437` 低 `0.4484`，其中 `0.2997` 是旧模型已经通过测试 ROI 端点选择和 ROI029 校准获得的收益，不能归因于 Round17 模型本身。

Round17 双折晋级证据偏弱：四项总体增益平均 `+0.308776`，只比 `+0.30` 门槛高 `0.008776`。H1 的 50/50 相对 base48 端点反而为 `-0.015644/-0.017364`，因此当前服务器失败可能来自 base32 拖累、base48 测试域负迁移，或 full-data epoch 映射整体失效。只看混合包无法区分三者。

## 固定端点

共同设置：CD68、full-data stage2 epoch 12、halo64、D4、JPEG q95、raw，不使用 ROI scale、calibration 或 checkpoint soup。

| 端点 | checkpoint | SHA-256 | 固定输出包 |
|---|---|---|---|
| base32 | `runs/round17_full_unet_base32_stage2/epochs/epoch_012.pt` | `966ba7d8c32c529e50c4f58719d97e061472a812433a80675bedc80b3349934f` | `submissions/cd68_round18_base32e12_context64_d4_raw_q95_plain.zip` |
| base48 | `runs/round17_full_unet_base48_stage2/epochs/epoch_012.pt` | `e4a9e3b3b8aff6d97cc1957dfbffbc124dba3fbac067c9d5f93ca15435f7ddf5` | `submissions/cd68_round18_base48e12_context64_d4_raw_q95_plain.zip` |

两包均从 checkpoint 独立重新推理，不从 50/50 JPEG 反推端点。提交顺序固定为 base32、base48。

## 判定规则

设 Round17 混合分为 `M=71.5953`，两个端点分为 `E32` 与 `E48`。

1. 若 `max(E32,E48) >= M+0.05`，判定 50/50 融合在测试域有害，后续只保留较强端点；
2. 若 `E48 >= E32+0.05`，base32 被判为主要拖累；若 `E32 >= E48+0.05`，base48 被判为测试域负迁移；
3. 若两个端点都不超过 `M+0.05`，当前融合不是主要问题，失败优先归因于 full-data 时钟/训练域外推；下一候选只能按新预注册规则测试更稳健的早期 epoch，不搜索连续融合权重；
4. 在两个端点结果完整之前，不生成 old-best + Round17 单 ROI 包；
5. 不提交带 ROI029 校准的 Round17 raw 端点。旧最佳的 ROI029 log0.75 已有精确 `+0.2435` 证据，而 Round17 raw ROI029 与旧 raw 均值接近，直接替换会把模型差异与已知校准收益混淆。

## 包生成结果

| 顺序 | 端点 | bytes | ZIP SHA-256 | 校验 | 服务器结果 |
|---:|---|---:|---|---|---|
| 1 | base32 e12 | 11,299,942 | `abf3111342f0ce4103d5a11985146a68af3c79254ce1f057e959102e43aa166f` | 1346 个唯一条目，missing/extra/error 0，CRC 正常 | 71.4557；SSIM 0.803；PSNR 25.445 |
| 2 | base48 e12 | 11,211,931 | `e0416ecff44487c7e8ecb340ec682761d8f04ba9b9b27934e95e31d2cc13299e` | 1346 个唯一条目，missing/extra/error 0，CRC 正常 | 71.5729；SSIM 0.804；PSNR 25.524 |

远端与本机文件大小、SHA-256 和 canonical 内部路径均一致。

## 判定

- `E48-E32=+0.1172`，base48 单端点明显强于 base32；
- `M-E48=+0.0224`，50/50 融合仍略强于最强端点；
- `M-E32=+0.1396`；
- 两个端点均未达到 `M+0.05=71.6453`。

因此 50/50 融合不是 Round17 服务器失败的主要原因。base32 单独较弱，但其与 base48 的误差仍有小幅互补；继续搜索全局融合权重的预期空间很小。当前失败优先归因于 full-data epoch/训练域外推，而不是容量选择。Round18 关闭，不生成连续权重、ROI 校准或单 ROI 拼接包；下一轮先审计同构完整推理下的固定 epoch，再决定是否需要新训练。

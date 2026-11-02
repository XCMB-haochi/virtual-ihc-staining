# 优化记录 Round 19：单模型四 marker 服务器排序

日期：2026-07-18

## 目标

组织者已确认初赛最低输出一种 marker，同时输出多种时按各 marker 分数平均。单 CD68 不因输出数量少而扣分，但服务器上哪一种 marker 最容易尚未实测。Round19 只回答 marker 难度排序，不把现有多任务 checkpoint 当作各 marker 的最终上限。

## 固定 checkpoint

- 路径：`runs/round11_h2_unet_multitask_stage2/best_global_mse.pt`；
- SHA-256：`c41758f666b4bd013d925899c816e201700fbb658b9f8926a00b3a4216c35841`；
- base32 residual U-Net，stage2 internal epoch 10；
- 训练 5480 张、验证 816 张；
- marker 顺序：`CD68,CD45RO,HLA-DR,Vimentin`；
- loss 权重：`0.5,1/6,1/6,1/6`；
- checkpoint 按 CD68 global-MSE 选择。

四类从同一次 `--all-markers` 前向得到，分别以 halo64、D4、JPEG q95、raw 打成单-marker ZIP。禁止把四类合并成平均提交，也不使用 ROI scale 或 calibration。

## 提交顺序与判定

固定顺序：CD68 控制、HLA-DR、CD45RO、Vimentin。

1. 以同一 checkpoint 的 CD68 服务器分为控制，比较另外三类的相对差值；
2. 若某个非 CD68 marker 高于 CD68 控制，立即转入该 marker 的 dedicated H2/full-data 训练；
3. 若某个 marker 距 CD68 控制不超过 `0.50`，允许训练一个 dedicated H2 候选，再按完整验证决定是否 full-data；
4. 若三个 marker 均低于 CD68 控制超过 `1.00`，关闭 marker 选择路线，后续只保留一对多作为决赛创新项；
5. 四个结果完整前不根据部分反馈改变模型、后处理或提交顺序。

## 本地先验

同一 checkpoint 的 H2 q95 mean-image 分数为：CD68 `69.9812`、HLA-DR `67.4742`、CD45RO `66.1347`、Vimentin `64.4902`。该先验预测 CD68 最高，但服务器测试域可能改变排序，因此仍保留一次受控实测。

## 包生成结果

| 顺序 | marker | ZIP | bytes | SHA-256 | 校验 | 服务器结果 |
|---:|---|---|---:|---|---|---|
| 1 | CD68 | `cd68_round19_multitask_h2e10_context64_d4_raw_q95_plain.zip` | 11,538,626 | `7e67d4068cd3c493a10e5bdb8f0769c9408af467f737d9767f252df145a4ca77` | 1346 unique，CRC 正常 | 71.3323；SSIM 0.802；PSNR 25.351 |
| 2 | HLA-DR | `hladr_round19_multitask_h2e10_context64_d4_raw_q95_plain.zip` | 8,806,640 | `b4e9983897d50c300a548f46b4cce696de8788e1858a0c12090033abb4cbf1cc` | 1346 unique，CRC 正常 | 67.4934；SSIM 0.779；PSNR 21.582 |
| 3 | CD45RO | `cd45ro_round19_multitask_h2e10_context64_d4_raw_q95_plain.zip` | 10,020,924 | `788ca3867a1c89e5f48a0f4fb70379fc930d79a94e32625479f25566073dff2a` | 1346 unique，CRC 正常 | 68.1347；SSIM 0.782；PSNR 22.347 |
| 4 | Vimentin | `vimentin_round19_multitask_h2e10_context64_d4_raw_q95_plain.zip` | 10,028,704 | `5b1c2be55dad41deeb7b686caf98d8fed5e4b9b601417a9baaa1aae404cef9c7` | 1346 unique，CRC 正常 | 65.7261；SSIM 0.755；PSNR 21.433 |

四包的远端/本机大小、SHA-256、条目数与内部 marker 路径均一致。必须分别提交；不得把四个目录合并成一个取平均的 ZIP。

## 判定

相对同模型 CD68 控制：

- HLA-DR：`-3.8389`；
- CD45RO：`-3.1976`；
- Vimentin：`-5.6062`。

若把四类合并提交，按组织者确认的平均规则，总分为 `(71.3323+67.4934+68.1347+65.7261)/4=68.1716`。三个非 CD68 marker 均低于控制超过 `1.00`，满足关闭条件。服务器排序为 CD68 > CD45RO > HLA-DR > Vimentin；HLA-DR/CD45RO 的相对次序与本地 H2 先验互换，但不影响 CD68 明显最高的结论。

Round19 关闭：不训练其他 marker 的 dedicated full-data 模型，不生成多-marker 平均包。多输出只保留为总决赛创新附加项；初赛追榜继续使用 CD68。

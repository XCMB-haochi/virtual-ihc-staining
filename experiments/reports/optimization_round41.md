# 优化记录 Round 41：官方指标对齐的检查点选择

日期：2026-07-23

## 立项依据

组织方已确认最终分数按每张 RGB uint8 图像分别计算 SSIM 和封顶 50 dB 的
PSNR，再逐图平均。审计发现，Round35 及其后续通用 OOF/测试预测路径仍硬编码
读取 `best_global_mse.pt`；该检查点按全数据像素 MSE 聚合选择，与官方逐图聚合
不一致。训练程序同期已经自动保存 `best_mean_image.pt`，其选择分数正是 q95 下
逐图 SSIM 加逐图 PSNR 的竞赛分数，因此无需重新训练即可进行公平对照。

十个 Round35 有效成员中，八组 global-MSE 与 mean-image 检查点来自同一 epoch。
只有 fold2 不同：base32 为 epoch 13 对 epoch 12，continued-base48 为 epoch 14
对 epoch 13。本轮不搜索 epoch、融合权重或 JPEG 参数，只把十个成员统一切换到
训练时自动选择的 `best_mean_image.pt`。

## 固定协议

- 模型、五折、70% base32 / 30% continued-base48、halo64、D4 和 q95 全部沿用
  Round35；
- baseline 是已存档的 Round35 `best_global_mse.pt` 完整 OOF 报告；
- candidate 从相同 run 读取 `best_mean_image.pt`，重新计算完整五折 OOF；
- OOF 评估显式设置 `test_like_diagnostic=false`，不读取测试图像或其统计量；
- 不使用排行榜反馈、测试 ROI、逐图挑选或校准；
- 只有所有 gate 同时通过，才允许用同一固定检查点规则生成一次测试候选。

机器协议为
`configs/round41_cv5_official_checkpoint_selection.json`。OOF 与最终预测工具共同
从 `inference.checkpoint_filename` 读取检查点文件名，防止验证和部署路径不一致。

## 预注册 gate

相对 Round35 的 `base32_70_base48_30`，必须同时满足：

1. 完整 OOF q95 官方分数至少提高 `+0.02`；
2. 完整 OOF float 官方分数严格为正增益；
3. 唯一实际换 epoch 的 fold2，q95 至少提高 `+0.05`；
4. fold2 的五个 ROI 至少三个为正；
5. fold2 最差 ROI 的 q95 增益不低于 `-0.10`；
6. 其余四折 q95 与 Round35 的绝对差不超过 `1e-6`，用于验证同 epoch
   检查点确实输出一致。

任一条件失败即保留 Round35，不生成测试预测或提交包，也不追加逐 fold/ROI
混合检查点、epoch 搜索或排行榜探针。

## 当前状态

协议、通用检查点选择路径和机器 gate 已实现。本地定向测试为
`53 passed`；Windows 可稳定执行的完整测试集为
`208 passed, 1 skipped, 2 deselected`。两个 deselect 是本机已知会导致
NumPy/MKL 进程级退出的线性拟合用例。Linux 服务器无排除复验为
`211 passed, 63 subtests passed`。

正式运行前逐张量核验了十组 global-MSE / mean-image 检查点。fold0、fold1、
fold3 和 fold4 的八组检查点不仅 epoch 相同，模型权重也完全相同；只有 fold2
的两组权重发生变化：base32 从 epoch 13 切换到 epoch 12，continued-base48
从 epoch 14 切换到 epoch 13。这验证了 gate 中 changed/unchanged fold 的划分。

## 正式 OOF 结果

物理 GPU0 上的完整五折 OOF 用时 `1016.871 s`。报告中的
`test_dapi_roi_means=null`、test-like 集合为空，确认本轮没有读取测试 DAPI 或
使用其统计量。固定 70% base32 / 30% continued-base48 候选相对 Round35 为：

| 指标 | Round35 global-MSE checkpoint | Round41 mean-image checkpoint | 增益 |
|---|---:|---:|---:|
| float 官方分数 | 74.948362 | 74.954771 | +0.006409 |
| q95 官方分数 | 74.869164 | 74.875434 | +0.006270 |

唯一变化的 fold2，q95 增益为 `+0.028257`。五个 fold2 ROI 中只有 ROI000
`+0.198168` 和 ROI008 `+0.114052` 为正；ROI007、ROI014、ROI024 分别为
`-0.015165`、`-0.024395`、`-0.130768`。其余四折最大绝对差仅
`2.56e-13`，与逐张量相等检查一致。

机器 gate 的六项结果依次为：完整 q95 失败、完整 float 通过、fold2 增益失败、
正向 ROI 数失败、最差 ROI 下界失败、未变化折一致性通过。最终
`passed=false`。正式产物为：

- `reports/round41_cv5_official_checkpoint_selection_oof.json`；
- `reports/round41_checkpoint_selection_gate.json`。

## 结论

按官方逐图指标选择检查点的方向在完整 OOF 上是轻微正增益，但幅度只有
`+0.006270`，且 fold2 的 ROI 稳定性不足，不满足预注册的可迁移性要求。本轮
保留 Round35 的 global-MSE 检查点方案，不生成测试预测或提交包，不进行逐折、
逐 ROI 或相邻 epoch 混选，也不使用排行榜探针补救。

# 优化记录 Round 43：固定独立 seed 集成诊断

日期：2026-07-23

## 立项边界

Round42 为隔离 MixStyle 机制，在查看 OOF 前已经固定了 matched online control：

```text
0.35 * Round31 base32
+ 0.35 * Round32 matched online continuation
+ 0.30 * Round34 continued-base48
```

它不是 Round42 的 promotion candidate，但配方、seed、checkpoint、融合权重和
完整 OOF 都在结果前固定。Round42 完成后，该 control 相对 Round35 的官方逐图
分数为：

- q95：`+0.051587691735`；
- float：`+0.051404892135`；
- 五折 q95：`+0.073283, +0.038856, +0.041093, +0.048577, +0.055239`；
- 21/25 ROI 为正，最差 ROI001 为 `-0.045852`。

这组证据比 MixStyle 稳定，但“决定提交”发生在看到 OOF 之后，因此本轮明确记为
初赛阶段的单次事后诊断，不伪装成预注册 promotion。它检验的是独立 seed 集成能
否迁移到隐藏测试集，而不是用排行榜选择 seed 或融合权重。

## 固定候选

直接复用 Round42 协议中的 `round35_online35`，不重新训练。15 个有效成员中，
每折 Round31 base32、Round32 matched online 和 Round34 base48 的权重分别为
`0.07/0.07/0.06`。推理固定 halo64、D4 TTA、JPEG q95；只使用物理 GPU0，避免
当前 GPU1 的高并发负载影响任务完成时间。

本轮只允许生成一次固定测试预测和一个 plain ZIP。禁止：

- 搜索相邻融合权重、额外 seed 或逐折权重；
- 根据测试 DAPI 统计量、排行榜反馈或逐图观察进行校准；
- 将本次线上分数用于同配方的后续 rescue 搜索。

无论线上结果如何，这个流程本身可在复赛或决赛的新训练集上通过独立 seed 训练
和固定平均复用；代价是从 10 个成员增加到 15 个成员，推理成本约增加 50%。

## 当前状态

推理与打包脚本为 `tools/run_round43_submission_pipeline.sh`。物理 GPU0 上的
正式推理已完成，用时 `769.168 s`。预测报告确认候选为
`round35_online35`，15 个成员权重和为 1，输出 1346 张 q95 RGB JPEG。

提交包为：

- `submissions/cd68_round43_online_seed_ensemble_context64_d4_raw_q95_plain.zip`

包大小为 `11,576,964` bytes，SHA256 为
`32ce1525d5fe1970dd5eb29fea38390d2a5407ef00009904977350d2d0430cf4`。
服务器打包检查和本机独立复验均确认：1346 个唯一条目，全部位于
`results/test/CD68/`，与测试 DAPI 文件名完全对应；缺失、多余、CRC、尺寸、
RGB 模式和三通道一致性错误均为 0。完整成员与 checkpoint 哈希见
`reports/round43_online_seed_ensemble_prediction.json`。

## 线上结果与结论

初赛单次提交得分 `73.3050`，显示 `SSIM=0.825`、`PSNR=25.959`。相对 Round35
个人最佳 `73.2716` 提高 `+0.0334`，方向与完整 OOF 的 `+0.051588` 一致，实际
线上增益约为 OOF 增益的 `64.7%`。提交时已知榜首为 `75.4966`，仍相差
`2.1916`。

独立 seed 集成因此被保留为可迁移、可复现的最终集成手段，但其收益量级不能
解释当前榜首差距。按立项边界关闭同类 rescue：不增加第 4 条同质 seed 轨迹，
不根据本次排行榜结果搜索 `0.35/0.35/0.30` 附近权重，也不追加校准或逐 ROI
路由。后续资源用于改变单模型误差结构的机制。

# 优化记录 Round 107：同轨迹局部 checkpoint 平均

## 动机与固定协议

Round103 证明从成熟 UNeXt2 监督 checkpoint 重新开启一个完整余弦周期会退化，
Round106 又排除了跨 ROI 形态复制。Round107 测试更窄的训练后方差机制：不重新
训练模型，对每个 Round55 `officialauxiliary15` fold 的 global-MSE 最佳 stage2
epoch，取其前一、当前、后一 epoch checkpoint，按固定 `1:2:1` 做参数平均。

基线是同一最佳 checkpoint；两者都使用 context128、D4、固定 `[0,-1]` 偏移和
JPEG q95。该规则只使用每折已有训练验证选择，不读取测试数据，不搜索窗口、
权重或 leaderboard 反馈。先筛 fold0/fold2，若通过才扩展剩余三折。

## Gate

候选必须相对基线提高至少 `+0.01` q95，float 为正，两个 fold 均为正，至少
7/10 ROI 为正且最差 ROI 不低于 `-0.12`。失败即关闭当前 SWA/soup 路线。

## 执行状态

协议、checkpoint 平均工具、启动器和测试已提交。远端 Screen 已结束，pipeline
退出码为 `0`；fold0/fold2 均成功生成相邻 epoch 的 `1:2:1` 参数平均
checkpoint，随后完成 context128、D4、固定 `[0,-1]` 偏移和 q95 的 OOF。OOF
覆盖 2706 张图、10 个 ROI，耗时 `1559.905 s`。输入 checkpoint、输出
checkpoint 和 OOF/gate 文件均已复制回本地并核验 SHA-256。

## 结果

| candidate | float official | JPEG-q95 official |
| --- | ---: | ---: |
| baseline context128 D4 | 75.083722 | 75.013272 |
| local checkpoint SWA (1:2:1) | 75.060290 | 74.987202 |

候选 q95 相对基线下降 `-0.026070`，float 下降 `-0.023432`。fold0/fold2
分别为 `-0.051990/-0.001783`，没有正增益 fold；仅 4/10 ROI 为正，最差
ROI024 为 `-0.176361`（ROI020 为 `-0.122709`）。OOF SHA-256 为
`00b2260eed8a69d3aeb66d1117318a0209462298e50929c5b6a0fc4968fc6793`，gate
SHA-256 为 `06e03776c61ab8586bcfc49461f11d7f7d2716d9c4574530262e0b633f10b44`。

## 决策

gate 的总体、float、双折、ROI 数和 ROI floor 检查全部失败。关闭当前
checkpoint SWA/soup 路线，不搜索邻域窗口、平均权重或 epoch 组合，也不生成
提交包。局部参数平均没有改善当前模型在困难 ROI 上的误差；下一步转向训练期
的配对错位鲁棒性和条件特征对齐机制，先做小规模、可迁移的固定 screen。

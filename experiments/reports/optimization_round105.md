# 优化记录 Round 105：训练期 CD68 活动度 hard-mix

## 动机与固定协议

Round104 在现有五折 OOF 上发现，目标活动度与基线像素误差明显相关：目标
活动度与 MSE 的 Pearson 相关为 `0.5856`，最高活动度四分位的平均 MSE
`354.32`，低活动度四分位为 `86.95`。Round105 测试一个不依赖测试标签的
训练期数据效率机制：活动度由训练 CD68 标签计算，前 75% epoch 保持原随机
采样，后 25% epoch 用最高活动度四分位替换 30% 采样位置。所有其他设置保持
Round55 `officialauxiliary15` 不变：fold-local ImageNet-1K FCMAE encoder、
UNeXt2-Tiny、四标记 `0.85/0.05/0.05/0.05`、context64 训练、14+14 epoch、
Uniform-7 hybrid loss、context128+D4 推理和固定 `[0,-1]` 偏移。

基线是同一 Round55 fold0/fold2 checkpoint 的 context128+D4 重放。候选先做
两折 screen；没有使用测试输入统计、测试标签或排行榜反馈。

## 完成审计

远端 pipeline 退出码为 `0`，Screen 和 producer 进程均已结束。smoke 成功验证
训练标签活动度、最高分位样本池、fold-local FCMAE 初始化和可加载 checkpoint。
fold0/fold2 的 stage1、stage2 均各有连续 14 个 epoch 行，状态和 checkpoint
完整；训练峰值显存约 `25.52 GiB`。两折在 GPU1 串行执行，未改变 GPU0 上的
其他任务。OOF 覆盖 2706 张图、10 个固定 ROI，耗时 `1542.450 s`。

候选 stage2 checkpoint SHA-256 为 fold0
`1190b697e94361338743eb23a2c219645ef8827570c21389e575fb5d6b65baf0`、fold2
`965dd94ef6178f8dd6443e8d4177eaceaad9feaa6a5b3e4202afa12efd180162`。

## 结果

| candidate | float official | JPEG-q95 official |
| --- | ---: | ---: |
| baseline context128 D4 | 75.083722 | 75.013272 |
| activity hard-mix | 75.043857 | 74.973171 |

候选 q95 相对基线下降 `-0.040101`，float 下降 `-0.039864`。fold0/2 分别为
`-0.147927/+0.060933`；仅 3/10 ROI 为正，最差 ROI 为 ROI022 的
`-0.273439`。gate 的总体、float、双折、ROI 数和 ROI floor 检查全部失败。

OOF SHA-256 为
`9e22225dd98c795e70df304454d51c82350ccfa3f717233faa98ae1a605f4e75`，gate
SHA-256 为
`952982093844855450c0f7b31cc81acfe5bd0706c2d2be10b22031070fdc6107`。

## 决策

关闭当前目标活动度 hard-mix sampler，不搜索活动度公式、采样比例、启动 epoch、
第二个 sampler 周期或排行榜条件路由。目标活动度是困难样本诊断信号，但简单
重采样会牺牲普通 ROI 的整体结构得分，不能作为当前 endpoint 的可迁移改进。

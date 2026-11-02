# 优化记录 Round 108：当前 UNeXt2 的官方分数直接优化

## 动机与固定协议

现有 UNeXt2 训练使用 `0.7*SSIM + 0.2*L1 + 0.1*MSE` 作为可微近似，检查点
再按 global-MSE 选择。旧 Round5 的 exact-score 试验只发生在已经淘汰的旧
U-Net，不能回答同一目标在 FCMAE-UNeXt2 上是否有益。本轮保留当前官方
FCMAE800 编码器、四级 UNeXt2 解码器、context128 推理、D4 和固定 `[0,-1]`
偏移，只把单输出 CD68 的训练损失改为直接优化官方逐图分数：
`1 - (0.7*SSIM_uniform7 + 0.3*clip(PSNR/50, 0, 1))`。

候选从每折相同的 Round52 fold-local FCMAE800 编码器 fresh 开始，使用与
Round55/52 相同的 seed、数据划分、context64 监督、`14+14` epoch、batch、
优化器、选模和推理协议；唯一模型训练变量是 `loss_profile=exact-score`，
且只训练 CD68 一个输出，避免对辅助标记使用未定义的额外分数。

## Gate

候选相对同一 fold 的官方 FCMAE800 单输出 baseline 必须提高至少 `+0.05`
JPEG-q95，float 为正、两个 fold 均为正、至少 7/10 ROI 为正且最差 ROI 不
低于 `-0.12`。失败即关闭当前 exact-score 目标，不搜索 loss 权重、温度、
epoch、checkpoint、融合或排行榜反馈；通过才允许另行设计完整五折扩展。

## 执行状态

初次 Screen 已完成资源等待和真实 smoke，但 fold0 正式启动在训练器参数校验
阶段退出：单标记 `exact-score` 不允许同时传入 `--marker-loss-weights 1.0`。
这次没有生成正式 checkpoint、OOF 或 gate，故不构成模型结果。协议已修正为
让单标记 CD68 使用训练器默认通道路径，并新增回归测试确认不会再传入该参数。
修正后将使用相同的两折、时钟和 gate 重跑；该实验不读取测试图像或标签，也不
生成初赛提交包。

## 完成审计

修复后的 detached Screen 正常退出，`reports/round108_pipeline.exit=0`，且
Screen 和 producer 进程均已结束。fold0/fold2 的 stage1、stage2 各含连续
14 个 epoch；配置均记录 `loss_profile=exact-score` 和单输出 `CD68`。stage1
加载了对应的 fold-local FCMAE800 encoder，stage2 为 strict checkpoint 初始化。
两折 stage2 选中 epoch 10/6，checkpoint SHA-256 分别为
`46180b4b2a970782e31603cef298096a8b3d59757d9c6b21f9c1e1c336f5f3d8` 和
`c343ea8aa526b6b59309c95142df0e1b83ed7991a1d9038858bce3e144e05689`，与
OOF provenance 一致。训练峰值显存约 `25.37 GiB`。

OOF 覆盖 2706 张图和全部 10 个预注册 ROI，耗时 `1534.042 s`。OOF
SHA-256 为
`172f77428ee57cae32e5b9741c01686de244aaa82495ad686ce4010768487501`，gate
SHA-256 为
`ca0567829bc986b9914251f949cfbc66a939c653fc9ce23089e7ba89b84bcdae`；gate
绑定的 OOF 哈希一致。

## 结果

| candidate | SSIM q95 | PSNR q95 | float official | JPEG-q95 official |
| --- | ---: | ---: | ---: | ---: |
| official FCMAE800 hybrid | 0.840628 | 26.905892 | 75.061378 | 74.987515 |
| exact-score | 0.837977 | 26.969624 | 74.911692 | 74.840191 |

exact-score 的 q95 相对 baseline 下降 `-0.147324`，float 下降 `-0.149686`。
它将平均 PSNR 提高约 `+0.0637 dB`，但 SSIM 下降约 `-0.00265`，净结果明显
为负。fold0/fold2 分别下降 `-0.029641/-0.257593`；仅 5/10 ROI 为正，
ROI008 最差为 `-0.966731`。总体、float、双折、ROI 数和 ROI floor 检查
全部失败，`promotion.allowed=false`。

## 决策

关闭当前 UNeXt2 的纯 exact-score 目标，不搜索 PSNR 截断、SSIM/PSNR 权重、
学习率、epoch 或 checkpoint，也不扩展五折或生成提交包。该结果与旧 U-Net
exact-score 的负结果同方向，说明去掉 L1/MSE 稳定项会以结构质量换取小幅
PSNR，不能解释当前排行榜差距。后续训练目标若继续研究，必须引入新的结构或
表征机制，而不是在官方公式附近做权重搜索。

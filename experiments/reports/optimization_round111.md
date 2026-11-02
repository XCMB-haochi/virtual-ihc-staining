# 优化记录 Round 111：Gaussian pyramid 低频结构约束

## 立项依据

Round108 的 exact-score 目标牺牲 SSIM，Round110 的冻结 FCMAE feature guidance
同时损失 SSIM 和 PSNR。当前训练器已经实现了 Gaussian downsample 后的低频 L1
项。历史更正：Round2 已在旧 residual U-Net 上测试过该项并得到负结果；此前
“从未正式使用”的描述不准确。本轮只回答它在当前模型上的效果：在保持当前
UNeXt2/FCMAE/auxiliary15 配方不变时，低频结构约束能否减少局部噪声并提高官方
SSIM/PSNR 组合。

## 固定协议

- student：Round55 `officialauxiliary15` 的 UNeXt2-Tiny、fold-local
  Round52 FCMAE800、四输出权重 `0.85/0.05/0.05/0.05`；
- 唯一变化：hybrid loss 叠加 `pyramid_weight=0.1`，使用现有固定 3x3
  Gaussian stride-2 下采样后的 L1；
- fold0/fold2、seed2044、context64 训练、`14+14` epoch、batch16、
  context128+D4 推理、global-MSE 选模、JPEG-q95 和 `[0,-1]` 偏移全部不变；
- 先运行两折 OOF，不读取测试标签，不进行权重、尺度、核、epoch、checkpoint
  或融合搜索。

## Gate

候选相对 matched Round55 auxiliary15 必须提高至少 `+0.05` q95，float 为正，
两个 fold 均为正，至少 7/10 ROI 为正，最差 ROI 不低于 `-0.12`。失败即关闭
该低频 loss 路线，不扩展五折、不生成提交包；通过才允许另行注册完整扩展。

## 执行状态

Round111 已通过本地和远端协议加载、runner dry-run、JSON 解析、Bash 语法，
以及 6 个定向测试和 88 个 CV 协议测试。启动前 GPU1 为 `0%` 利用率、
`40422 MB` 空闲，GPU0 有外部高负载而被排除。正式脚本已于 2026-09-05
12:50 在 detached Screen `round111_pyramid_loss` 启动；它会先执行真实 smoke，
再串行训练 fold0/fold2 并运行完整 OOF/gate。结果和依赖证据在完成后补充。

## 完成结果（2026-09-05）

退出码 `0`；两折各完成两个连续 14-epoch 阶段，`pyramid_weight=0.1`，
stage1 为 fold-local FCMAE 初始化、stage2 为 strict 初始化。实际选中 checkpoint
（epoch 7/3）的哈希与 OOF 一致；OOF 覆盖 2706 张图、10 个 ROI，耗时
1565.71 秒。训练每折约 96 分钟，峰值 25.65 GiB。

| candidate | JPEG-q95 | float |
| --- | ---: | ---: |
| auxiliary15 baseline | 75.013272 | 75.083722 |
| pyramid auxiliary15 | 74.929804 | 74.999217 |

q95 差值 `-0.083468`，float `-0.084504`；fold0/2 为
`-0.031440/-0.132219`，仅 2/10 ROI 为正，最差 ROI007 `-0.392563`。
全部性能 gate 失败。本配方不扩展、不提交。该结果不证明所有多尺度表示无效。
下一步改为独立的自监督 Transformer 表示筛查，停止当前损失微调链。
机器证据保存在同目录的 `round111_cv2_unext2_pyramid_loss_oof.json` 和
`round111_cv2_unext2_pyramid_loss_gate.json`；gate 已绑定核验过的 OOF 哈希。

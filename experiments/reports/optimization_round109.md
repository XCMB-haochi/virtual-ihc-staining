# 优化记录 Round 109：跨模态 FCMAE 特征可预测性诊断

## 目的

Round108 关闭了纯官方分数直接优化。近期虚拟 IHC 方法提出用目标侧任务模型
或重建模型的语义特征指导生成器，但在当前 DAPI/CD68 数据上直接实现会新增
teacher、投影头和特征损失，成本较高。本轮先回答其必要前提：训练 ROI 中学到
的 DAPI-to-CD68 特征映射，能否在未见 ROI 上稳定预测目标表征。

## 固定方法

- teacher encoder：Round96 每折严格排除验证 ROI、在 DAPI 与四个 IHC 通道
  上做曝光量匹配 FCMAE 的 Tiny encoder；
- 表征：对 256x256 DAPI/CD68 分别取 stage4 `768x16x16` 特征并空间平均；
- 映射：在每折训练 ROI 上分别标准化输入/目标，固定 Ridge `alpha=10`、
  Cholesky 求解，再只在该折验证 ROI 上计算标准化特征 R2；
- 配对检查：比较正确 DAPI/CD68 对的余弦相似度与同 ROI 内确定性错配对；
- 数据边界：只读取官方 train，fold0/fold2 共覆盖 2706 张未见 ROI 图像，不读取
  test、排行榜或现有生成预测。

## Gate

只有总体加权标准化 R2 至少 `0.05`、两个 fold 的 R2 均非负，并且两个 fold
的正确配对余弦都比同 ROI 错配至少高 `0.01`，才允许设计一次固定的训练期
target-feature guidance screen。失败则关闭当前直接特征蒸馏，不搜索 Ridge、
层级、teacher、特征损失权重或投影维数。

## 状态

协议、分析工具、runner 和合成回归测试已实现。真实 smoke 使用每折 128 个训练
样本和 64 个验证样本，成功验证两个 encoder、双模态特征提取和 Ridge 全链路；
其抽样 fold2 R2 为负，只作为执行验证，没有参与正式判定。

## 正式结果

detached Screen 正常退出，`reports/round109_pipeline.exit=0`，且 producer 进程
均已结束。正式报告覆盖 fold0 的 4987/1309 个训练/验证样本和 fold2 的
4899/1397 个训练/验证样本，共 2706 张未见 ROI 图像。两份 encoder SHA-256
分别为
`fa2d96bfa3196b913f954a15862987c94e45d7cc2e0088edde19be2029c9449c` 和
`ce54525418ccd5ef3dc08085e2fa3614d76aa2b51bffdff57f48e00d52129c72`，与
Round96 记录一致。正式计算耗时 `97.994 s`，报告 SHA-256 为
`8e323c7c32351efaf59fbd58d135a68959cb015524b9eac31e8cdb0dab82df33`。

| fold | 标准化特征 R2 | mapped centered cosine | 正确配对余弦 | 同 ROI 错配余弦 | 配对优势 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fold0 | 0.671613 | 0.812187 | 0.980818 | 0.970311 | 0.010507 |
| fold2 | 0.638097 | 0.795258 | 0.972783 | 0.958942 | 0.013841 |

按两折标准化残差/基线平方和汇总的加权 R2 为 `0.654895`。总体 R2、两个
fold 的非负 R2 和两个 fold 的最小配对优势三项 gate 全部通过，
`feature_guidance_actionable=true`。

## 决策

该结果只证明 target-feature guidance 具有跨未见 ROI 的可学习前提，不证明
它一定提高图像指标。下一轮允许一次固定训练实验：用同一 fold-local Round96
encoder 作为冻结 teacher，对生成 CD68 与真实 CD68 的低分辨率特征施加一个
小权重感知约束；teacher 不进入推理。不得根据本诊断搜索 Ridge、teacher 层、
投影维数或损失权重。候选仍须通过 matched fold0/fold2 图像 OOF gate 才能扩展。

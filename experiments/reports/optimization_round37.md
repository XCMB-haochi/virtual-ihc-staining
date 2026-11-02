# 优化记录 Round 37：cross-fitted residual refiner

日期：2026-07-21

## 立项依据

Round35 的 70% base32 / 30% continued-base48 在平台取得 `73.2716`，但与
榜首 `74.8890` 仍差 `1.6174`。Round32--36 已表明继续训练、EMA、容量微调
和扩大 halo 的收益已衰减到约 `+0.03--+0.12`；继续搜索相邻 epoch、权重、
halo 或排行榜 ROI 探针不具备复赛和决赛可迁移性。

现有 OOF 证据同时表明 base32、base48、full-target、spatial-film 和
thumbnail 模型在不同 ROI 上存在互补误差，但此前只使用全局常数权重。下一项
固定检验：一个只在训练 OOF 预测上学习的统一像素残差模型，能否在未见完整
ROI 上稳定修正固定 Round35 基线。该方法可对新器官按相同五折流程重新训练，
不依赖测试集身份或排行榜反馈。

## 严格 cross-fit 数据流

机器可读协议为 `configs/round37_cv5_crossfit_refiner.json`。底层预测固定来自
Round35 的两个 recipe：

- Round31 `thumbnailcontinue` base32；
- Round34 `thumbnailbase48continue` base48；
- 固定基线为 `0.7 * base32 + 0.3 * base48`。

先按既有五折生成每张训练图的真正 OOF 浮点预测。对外层 fold `f`：

1. refiner 训练集只包含其余四折的 OOF 预测和训练标签；
2. refiner 验证集只包含 fold `f`，其任何标签不进入训练；
3. 底层输入均来自没有训练过对应图像的 base checkpoint；
4. checkpoint 明确记录四个训练 fold、唯一 holdout fold、ROI、协议和缓存哈希；
5. 五个 refiner 分别完成后，才汇总 25 ROI 的完整 OOF。

若晋级测试推理，每个 refiner 只修正同编号 fold 的 base32/base48 测试预测，
最后五折等权平均。这使测试端的单成员输入分布与 OOF 验证一致，不把 refiner
直接套在一个训练时未见的五折平均图上。

禁止使用测试标签、测试 ROI 路由、排行榜贡献探针、逐 ROI 参数、输出校准、
相邻融合权重搜索或 OOF 后改动固定协议。测试 DAPI 只在 gate 判定时用于沿用
既有的输入均值范围定义 test-like 训练 ROI，不参与训练。

## 固定模型与训练

每个样本的五个输入通道依次为：

1. 中心 DAPI；
2. base32 OOF 浮点预测；
3. base48 OOF 浮点预测；
4. 固定 70/30 baseline；
5. `base48 - base32` disagreement。

refiner 使用 width48 的纯卷积残差网络，七个 block 的 dilation 固定为
`1/2/4/8/4/2/1`，共 `295,057` 个参数。输出为：

```text
clip(baseline + 0.25 * tanh(residual), 0, 1)
```

最后一层权重和偏置严格零初始化，首次前向逐值等于固定 baseline。训练固定为
14 epoch、batch 16、workers 4、AdamW、lr `3e-4` cosine、weight decay
`1e-4`、seed `2037 + fold_index`，损失仍为 Uniform-7 hybrid
`0.7 SSIM + 0.2 L1 + 0.1 MSE`。训练增强和选模推理均为 D4，best checkpoint
按 q95 global-MSE competition score 选择。网络宽度、dilation、残差上限、
epoch、loss 和 seed 均不依据 OOF 结果搜索。

## 预注册晋级门槛

direct refined 相对固定 Round35 baseline 必须同时满足：

1. 完整 OOF q95 mean-image/global-MSE 均至少提高 `+0.15`；
2. input-defined test-like q95 两项均至少提高 `+0.15`；
3. 完整和 test-like 的四项 float 增益全部为正；
4. 至少 4/5 folds 的 q95 两项均为正；
5. 任一 test-like ROI 的 q95 指标不得低于 `-0.10`；
6. test-like 双负 ROI 不得超过一个。

全部通过才实现测试预测和唯一提交包。任一失败即关闭当前 refiner，不搜索
残差缩放、模型宽度、dilation、epoch、训练 loss 或 baseline/refined 融合。
后续改用空间对齐的低分辨率上下文特征分支，而不是救援本轮。

## 实现与当前状态

当前已加入：

- float32 OOF base prediction 与 uint8 DAPI/target 的可恢复缓存器；
- 缓存协议、fold、权重、ROI 和底层 checkpoint SHA-256 复用校验；
- zero-init refiner、D4 推理、四折训练/一折验证数据集；
- best/last 原子 checkpoint、断点续训、完整 OOF evaluator 和机器 gate；
- Round37 单 GPU pipeline。

实现提交后，A100/Linux 环境的完整回归为 `173 passed, 63 subtests passed`；
compileall、shell 语法、三个 CLI `--help` 与 `git diff --check` 均通过。
本机相邻旧测试仍会在已知 NumPy/MKL 双 OpenMP 冲突处原生终止，未使用修改
OpenMP 安全检查的规避方案。

真实 OOF 缓存已经生成并通过复用校验：五折分别含
`1309/1145/1397/1214/1231` 张图，共 `6296` 张，占用约 `3.7 GiB`，生成耗时
`652.81 s`。五折缓存内重新计算的固定 baseline 指标均与 Round35 原始指标
逐项相等。GPU0 上的 64-train/32-validation 单 epoch 烟雾测试及断点恢复通过，
峰值 CUDA 内存 `9,518,753,792` bytes；临时验证集的分数只用于验证执行链，
不用于改协议或选择参数。

## 正式训练与 OOF

正式运行基于服务器 commit `b516ba8`，五折均完整训练 14 epoch。按 q95
global-MSE 选中的最佳 epoch 依次为 `3/5/7/8/4`；五折累计训练时间
`6726.72 s`，每折峰值 CUDA 内存均为 `9,518,753,792` bytes。完整 OOF
覆盖 25 个 ROI、6296 张图，评估耗时 `579.05 s`。

机器可读报告为 `reports/round37_cv5_crossfit_refiner_oof.json`，大小
`55,415` bytes，服务器 SHA-256 为
`BB3E740AF002461F854A987A5C0D3C14DA4A19F54159BE9A3DAE33CFEF4B2641`。
q95 绝对分和 refined 相对固定 Round35 baseline 的差值为：

| 范围 | baseline mean/global | refined mean/global | delta mean/global |
|---|---:|---:|---:|
| 完整 OOF | 74.869163 / 74.073733 | 74.867964 / 74.081587 | **-0.001199 / +0.007853** |
| test-like | 72.714860 / 72.101365 | 72.726668 / 72.118198 | **+0.011808 / +0.016833** |

对应 float delta 为完整 OOF `-0.001537/+0.007547`、test-like
`+0.011301/+0.016324`。变化量级接近零，并且完整 OOF 的 mean-image 口径
在 float 与 q95 下均为负。

逐折 q95 delta 为：

| fold | mean-image | global-MSE |
|---|---:|---:|
| fold0 | +0.025379 | +0.052009 |
| fold1 | -0.012348 | -0.001706 |
| fold2 | -0.010656 | -0.007263 |
| fold3 | +0.001101 | +0.000999 |
| fold4 | -0.010625 | -0.008718 |

只有 2/5 folds 双正。九个 test-like ROI 中，ROI004、ROI007、ROI013 和
ROI019 双负；最小单项为 ROI020 mean-image 的 `-0.011788`。正增益主要集中
在 ROI015、ROI022 和 ROI024，没有形成跨 ROI 一致机制。

## Gate 判定与结论

promotion gate 失败：完整和 test-like q95 均远低于 `+0.15`；完整 float
含负项；只有 2/5 folds 双正；test-like 有四个双负 ROI，超过最多一个的
限制。只有 `-0.10` 的单 ROI floor 通过。因此 Round37 不生成、读取或提交
任何测试预测，也不搜索残差缩放、模型宽度、dilation、epoch、loss 或
baseline/refined 融合。

该结果说明局部残差 CNN 在跨折条件下主要学成固定 70/30 baseline 的近似
恒等映射，无法稳定修正不同 ROI 的系统误差。Round37 至此关闭；下一轮按
预注册方向改为与中心 tile 同步变换、保留二维位置的低分辨率邻域特征分支，
不再使用 D4 平均后的全局 context embedding。

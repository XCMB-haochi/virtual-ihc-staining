# 优化记录 Round 20：D4 等变一致性 stage2

日期：2026-07-18

## 立项依据

Round19 的四个单 marker 服务器结果为 CD68 `71.3323`、CD45RO `68.1347`、HLA-DR `67.4934`、Vimentin `65.7261`。多 marker 按组织者规则取平均仅为 `68.1716`，因此初赛继续固定单输出 CD68。

配准审计在 25 个训练 ROI 上发现 CD68 相对 DAPI 约有 `+0.7~1 px` 的固定水平偏移，但已训练模型输出右移 1 px 的 H2 内区理论收益只有约 `+0.039` 分；配准不作为主训练路线。Gaussian-pyramid、exact-score、多任务、单纯增宽和 ConvNeXt 也均已有负结果。

当前 base32 H2 从 no-TTA 到 D4 完整推理可提高约 `+0.45` 分，而原生 context64 模型从 D4 获得的收益明显缩小并最终落后，说明模型的变换响应会影响部署排序。本轮只检验：在其他训练条件完全相同时，显式约束同一图像的 D4 预测一致性，能否提高正式 halo64+D4 输出。

## 固定对照与初始化

- 验证 ROI：`ROI020,ROI022,ROI024`；
- marker：仅 `CD68`；
- 对照：`runs/round11_h2_unet_cd68_stage2/best_global_mse.pt`；
- 对照 SHA-256：`f4c15098ea7479c205eeb419ff5e152b8691bc15f801a7b0ccb30ee4cd5832e3`；
- 候选初始化：`runs/round11_h2_unet_cd68_stage1/best_global_mse.pt`；
- 初始化 SHA-256：`4b02f87d355c91cea9ef687e45b432a7a97695af79e28ef5ec9a8166755572fb`；
- 候选输出：`runs/round20_h2_unet_d4_consistency_stage2/`。

候选从与 Round11 对照完全相同的 stage1 checkpoint 重建 stage2。模型、训练图像顺序、随机 D4 数据增强、seed、optimizer、scheduler 和主监督损失保持不变，不重新训练对照。

## 固定候选

| 项目 | 固定值 |
|---|---|
| 模型 | base32 residual U-Net，单输出，condition none |
| 输入 | context0，256x256 |
| epoch | 14 |
| batch / workers | `8 / 4` |
| optimizer | AdamW，lr `1e-4` cosine 到 0，weight decay `1e-4` |
| 主损失 | Uniform-7 hybrid `0.7 SSIM / 0.2 L1 / 0.1 MSE` |
| 一致性权重 | `0.1` |
| 选模 | CD68 global-MSE，JPEG q95 |
| 正式评估 | halo64，D4，分别记录 q95 与 float |
| 诊断评估 | halo64，no-TTA，分别记录 q95 与 float |

每个 batch 先计算原视图预测，再按全局 batch 序号在 7 个非恒等 D4 变换中确定性循环。第二次前向的预测逆变换回原坐标后，与原视图预测计算同一 Uniform-7 hybrid loss；两个分支都保留梯度。总损失固定为：

```text
supervised_hybrid + 0.1 * d4_consistency_hybrid
```

不搜索一致性权重、变换子集、单边 detach、loss 组成、学习率、epoch、checkpoint soup 或输出校准。新增前向不使用随机数，因而不改变既有 loader generator 和数据增强随机序列。

## Promotion gate

1. 主比较固定为候选与 Round11 对照各自的 `best_global_mse.pt`；
2. halo64+D4+q95 下，候选 mean-image 与 global-MSE 分均至少提高 `+0.15`；
3. ROI020、ROI022、ROI024 的两种 q95 聚合均不得回退超过 `0.05`；
4. D4+float 的两个总体增益必须为正，逐 ROI同样不得回退超过 `0.05`；
5. no-TTA 只诊断一致性是否缩小部署差距，不参与候选选择；
6. H2 任一主 gate 失败立即淘汰，不调整一致性权重；随后进入预先固定的 `dualbranch-swin-s` 结构实验；
7. H2 通过后才在 H1 `ROI015,ROI018,ROI019` 重建匹配 stage1/stage2。H1/H2 四个 q95 总体增益必须全部为正、平均至少 `+0.40`，六个 ROI 均不得回退超过 `0.05`，才允许训练 full-data。

## 启动条件

1. D4 变换/逆变换、零权重 legacy 路径、双分支梯度、确定性调度和参数校验测试通过；
2. 本地定向测试、远端全量测试、compileall、CLI 与 diff check 通过；
3. 代码与本预注册文档提交、推送并同步远端；
4. GPU1 资源检查完成，不停止或干扰其他用户进程；
5. 小样本预检确认额外显存、两次前向、checkpoint metadata 和训练日志正常。

## 工程验证

- 本机定向与相邻测试 `103 passed, 1 skipped, 1 deselected`；deselect 为既有 Windows NumPy/MKL `lstsq` 原生崩溃；
- 远端 Python 3.12 / PyTorch 2.8 / torchvision 0.23 全量测试 `106 passed, 1 skipped, 45 subtests passed`；
- 本机 compileall、CLI 与 `git diff --check` 通过；
- 远端 512 train / 64 val、batch 8 预检单 epoch 用时 `6.86 s`，峰值 PyTorch 显存 `3.606 GiB`；
- 预检记录 `train_supervised_loss=0.132896`、`train_d4_consistency_loss=0.006495`，确认第二前向和独立日志正常；
- 正式 runtime commit 为 `afeb129aaf5ab4751b89408dc880915b6b7b3f34`，物理 GPU1；未停止或修改既有 GPU 进程。

预检和正式任务退出时均可能出现 NFS 上 Python multiprocessing 临时目录的 `EBUSY` 清理噪声；训练主进程退出码为 0，config、14 个 epoch 指标和 checkpoint 均完整。

## H2 训练结果

14 epoch 全部完成，单 epoch 约 `47-50 s`，峰值显存始终为 `3.606 GiB`。候选最佳 global-MSE checkpoint 为 epoch 1：

| 裸 q95 | mean-image | global-MSE | 候选 - 对照 |
|---|---:|---:|---:|
| Round11 对照 epoch 1 | 69.546398 | 68.895627 | - |
| D4 一致性候选 epoch 1 | 69.560886 | 68.908144 | +0.014487 / +0.012517 |

候选 checkpoint SHA-256 为 `32457c887ad45ccef8b2b1aea4700c8c844dca0b96398974bd8e64a1aa6e9448`。epoch 2--14 均未超过 epoch 1，不改选末轮或其他快照。

## 正式评估

完整 halo64+D4 总体结果：

| 输出 | 对照 mean / global | 候选 mean / global | 候选 - 对照 |
|---|---:|---:|---:|
| q95 | 70.012066 / 69.343546 | 70.000950 / 69.331020 | -0.011116 / -0.012526 |
| float | 70.073465 / 69.404339 | 70.062803 / 69.392302 | -0.010662 / -0.012037 |

q95 逐 ROI 增益：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI020 | -0.007227 | -0.010127 |
| ROI022 | -0.011882 | -0.012304 |
| ROI024 | -0.013898 | -0.015155 |

float 的三个 ROI、两种聚合也全部为负，范围为 `-0.007953` 到 `-0.014765`。原始结果保存为：

- `reports/round20_candidate_h2_d4.json`；
- `reports/round20_candidate_h2_none.json`；
- `reports/round20_control_h2_none.json`。

no-TTA q95 下，候选相对对照提高 `+0.014706/+0.012194`，与裸选模增益一致；但 q95 的 D4 增益从对照的 `+0.162384/+0.156160` 降为候选的 `+0.136562/+0.131440`。一致性项小幅改善单视图，却削弱了约 `0.0258/0.0247` 的 TTA 集成收益，最终完整管线反而回退。

## 决策

Round20 淘汰：q95 两个总体指标不仅未达到 `+0.15`，而且均为负；float 和所有逐 ROI 结果也同方向。按预注册规则，不训练 H1/full-data，不搜索一致性权重、detach、变换子集、epoch 或模型融合，也不生成排行榜提交。下一实验固定为 `reports/optimization_round21.md` 的无 GAN `dualbranch-swin-s`。

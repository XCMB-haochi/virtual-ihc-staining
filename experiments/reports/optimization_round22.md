# 优化记录 Round 22：Context64 全区域监督

日期：2026-07-18

## 立项依据

Round12 的原生 context64 模型使用 384x384 DAPI 输入，但训练 target 和 loss 仍只有中心 256x256。它的裸 q95 相对 context0 提高约 `+0.21`，正式 halo64+D4 却回退约 `-0.12`；外圈 64 px 输出从未被直接监督，是训练与部署等变响应不一致的一个具体缺口。

Round20 显式 D4 prediction consistency 和 Round21 轻量 global-Swin 均失败，因此本轮不再增加正则或模型容量。唯一新因素是：训练时同时加载相邻 CD68 tile，把 target 扩为与输入相同的 384x384，让 loss 覆盖完整 context64 区域。验证和最终评分仍只使用官方中心 256x256 target，不改变指标或推理输出。

## 数据与安全边界

- 训练/验证仍按完整 ROI 分开，H2 验证为 `ROI020,ROI022,ROI024`；
- 训练 target halo 只从同一训练 ROI 的相邻 CD68 tile 加载；缺失邻居沿用现有中心 tile reflect fallback；
- 验证 ROI 的任何 CD68 像素都不会进入训练 target context；
- source 与 target 接受完全相同的随机 D4 augmentation；
- `--train-target-context-halo` 只作用于训练 loader，验证 target 固定为原始 256x256；
- 禁止测试标签、test statistics、输出校准和排行榜反馈参与训练。

## H2 固定协议

| 项目 | 固定值 |
|---|---|
| 模型 | base32 residual U-Net，单输出 CD68 |
| seed | `2026` |
| 训练输入 | DAPI context64，384x384 |
| 训练 target | CD68 context64，384x384，全区域 loss |
| 验证输入 / target | DAPI 384x384 / CD68 中心 256x256 |
| batch / workers | `8 / 4` |
| loss | Uniform-7 hybrid `0.7/0.2/0.1` |
| stage1 | fresh，14 epoch，AdamW lr `3e-4` cosine，wd `1e-4` |
| stage2 | stage1 `best_global_mse.pt` strict init，14 epoch，lr `1e-4` cosine |
| 选模 | CD68 global-MSE，JPEG q95 |
| 正式评估 | halo64+D4，分别记录 q95 与 float |

固定输出：

- `runs/round22_h2_unet_context64_fulltarget_stage1/`；
- `runs/round22_h2_unet_context64_fulltarget_stage2/`；
- `runs/_round22_eval/unet_context64_fulltarget_h2.json`。

部署主对照固定为 `reports/unet_cd68_h2.json`。Round12 的 `reports/unet_context64_h2.json` 只用于解释“全区域监督相对中心监督”的差异，不替代 base32 promotion gate。

禁止 D4 consistency、pyramid、exact-score、EMA、ROI weighting、conditioning、辅助 marker、模型增宽、额外 epoch、checkpoint soup、融合和后处理。不得搜索 target halo；本轮唯一值为 64。

## Promotion gate

1. halo64+D4+q95 下，相对 context0 base32 的 mean-image 与 global-MSE 均至少提高 `+0.25`；
2. ROI020、ROI022、ROI024 的两种 q95 聚合均不得回退超过 `0.05`；
3. float 两种总体增益必须为正，逐 ROI同样不得回退超过 `0.05`；
4. 同时记录相对 Round12 center-only context64 的结果，但不以该较弱对照晋级；
5. H2 通过后锁定设置，在 H1 `ROI015,ROI018,ROI019` 从头重建两阶段轨迹；
6. H1/H2 四个 q95 总体增益必须各至少 `+0.15`、平均至少 `+0.40`，六个 ROI 均不得回退超过 `0.05`，才允许 full-data；
7. 任一 H2 gate 失败即停止，不搜索 halo、batch、loss、epoch 或 context 模型融合。

## 启动条件

1. dataset 默认 256 target 路径保持不变，full-context target 的 shape、相邻 tile 内容、augmentation 和参数边界测试通过；
2. GroupDRO 风险 loader 在启用时显式复用相同 target halo，避免训练/风险定义不一致；
3. 本地和远端全量测试、compileall、CLI 与 diff check 通过；
4. 代码与本预注册文档提交、推送并同步服务器；
5. GPU1 512/64、batch 8 预检确认显存、吞吐、384x384 loss 与 checkpoint metadata；
6. 正式输出目录不存在，资源检查确认不干预其他用户任务。

## 工程验证

- 新增 full-context target、legacy target shape、参数边界和 GroupDRO loader 定向测试通过；
- 本机可运行测试集 `105 passed, 1 skipped, 1 deselected`，远端全量测试 `108 passed, 1 skipped, 53 subtests passed`；
- compileall、CLI 和 `git diff --check` 通过；
- 512 train / 64 val、batch 8 预检单 epoch 用时 `7.00 s`，峰值 PyTorch 显存 `4.294 GiB`；
- 预检 config 明确记录 `context_halo=64`、`train_target_context_halo=64`，验证仍为 64 张官方中心 target；
- 正式 runtime commit 为 `ae90cd852ba5327978f2c4c792ba15f42c623729`，物理 GPU1，未停止或修改既有 GPU 进程。

## H2 训练结果

stage1/stage2 均完整训练 14 epoch，单 epoch 约 `50-54 s`；峰值显存分别为 `4.294/4.323 GiB`。

| 阶段 | best epoch | 候选裸 q95 mean / global | context0 base32 | 候选 - base32 |
|---|---:|---:|---:|---:|
| stage1 | 13 | 69.776791 / 69.104411 | 69.334928 / 68.636064 | +0.441863 / +0.468347 |
| stage2 | 1 | 70.019339 / 69.376659 | 69.546398 / 68.895627 | +0.472941 / +0.481032 |

stage1 best checkpoint SHA-256 为 `a1dece102071bd8d13f0784a4b23dcee93042112a3d686d648aab3288676d493`；stage2 best checkpoint SHA-256 为 `f8830f86142f7c5244dc40f769fac73fedf01a47b31955cd036324afdf1f6a16`。两个阶段都明显超过匹配 context0，stage2 相对 Round12 center-only context64 裸分也提高约 `+0.260/+0.269`。

## 正式评估

固定 stage2 epoch 1，运行 halo64+D4：

| 输出 | context0 base32 | full-target context64 | 候选 - base32 | 候选 - Round12 center-only |
|---|---:|---:|---:|---:|
| q95 mean / global | 70.012066 / 69.343546 | 70.185245 / 69.531532 | +0.173179 / +0.187987 | +0.294316 / +0.304882 |
| float mean / global | 70.073465 / 69.404339 | 70.247873 / 69.593583 | +0.174409 / +0.189244 | +0.295042 / +0.305652 |

q95 逐 ROI 相对 context0：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI020 | +0.431472 | +0.507408 |
| ROI022 | +0.041637 | +0.035331 |
| ROI024 | +0.063758 | +0.072560 |

三个 ROI 的 q95 与 float 均为正。全区域监督成功消除了 Round12 在 ROI022/ROI024 的明显回退，并在完整 D4 管线稳定优于 center-only context64。完整结果为 `reports/round22_unet_context64_fulltarget_h2.json`。

## 决策

Round22 按原 promotion gate 关闭：虽然两个总体指标和所有逐 ROI 指标均为正，但 q95 `+0.173179/+0.187987` 未达到预注册的双 `+0.25` 强门槛，因此本轮不启动 H1 或 full-data，也不事后放宽阈值。

该结果不是负实验：它识别出“context 输入只监督中心”是 Round12 失败的一项真实原因。后续只能作为新的预注册候选使用。Round23 固定检验 full-target 与 context0 base32 的 50/50 raw 误差互补，不搜索权重或校准。

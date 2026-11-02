# 优化记录 Round 33：thumbnail-conditioned base48

日期：2026-07-20

## 立项依据

Round32 证明无新机制的 continuation 已接近饱和：相对 Round31 base 的完整
q95 OOF 只增加 `+0.079098/+0.072277`，test-like 只增加
`+0.027104/+0.034288`，且 ROI013 回退超过 `0.20`。固定 `EMA=0.999`
也被 matched online 明确击败。因此不再追加相同 base32 轨迹。

旧 Round14 的无条件 base48 在统一 H2 完整管线上相对 base32 提高
`+0.2077/+0.2411`；Round17 中 base32/base48 固定 50/50 又在两个独立
fold 全部正向，四项 q95 平均 `+0.3088`。其服务器失败发生在另行训练的
full-data epoch 映射，Round18 端点诊断认为容量选择和融合不是主要原因。

当前五折部署不再训练 full-data 模型，而是直接平均五个 fold checkpoint，
已在 Round27--31 连续迁移到隐藏集。这使 base48 成为尚未在当前可靠协议下
使用的最强本地正证据。相比 DiT、diffusion 或更大 Transformer，它的假设
更单一，且旧实测已证明 A100 资源足够。

## 固定候选

机器可读协议为 `configs/round33_cv5_thumbnail_base48.json`，沿用 Round26--32
完全相同的五折：

| recipe | 宽度 | conditioning | 训练 |
|---|---:|---|---:|
| `thumbnailbase32` | 32 | spatial statistics + 5x5 thumbnail | 复用 Round31 |
| `thumbnailbase48` | 48 | spatial statistics + 5x5 thumbnail | fresh stage1 + stage2 |

base48 从头训练，stage1/stage2 各 14 epoch，lr 分别为 `3e-4/1e-4` cosine；
固定 seed `2033`、batch/workers `8/4`、context/target halo `64/64`、
Uniform-7 hybrid loss、q95 global-MSE 选模。conditioning 从 stage1 起启用，
不从 base32 做不严格的宽度转换。

正式候选固定为每折 `50% base32 + 50% base48` 的浮点预测平均，五折再等权，
共 10 个成员、每个权重 `0.1`。同时记录 base48 direct 诊断端点。禁止搜索
base64、融合权重、epoch、seed、loss、conditioning 结构、校准、soup 或
逐 ROI 路由。

## 预注册判定

50/50 primary promotion gate：

1. 相对 Round31 base32，完整 OOF 与 test-like 的四项 q95 增益均至少
   `+0.15`；
2. 四项 float 增益全部为正；
3. 5/5 folds 的 q95 双指标均为正；
4. 任一 test-like ROI 的 q95 指标不得低于 `-0.10`，双指标为负的 ROI
   不得超过一个；
5. base48 direct 相对 base32 的完整/test-like 四项 q95 均为正，证明融合
   收益包含真实容量信号，而不是只依赖平均修复。

若 primary 失败，base48 direct 只有在完整/test-like 四项 q95 均至少
`+0.20`、至少 4/5 folds 双正且任一 test-like ROI 不低于 `-0.15` 时，才
允许生成唯一 diagnostic 包。若 primary 通过，只生成 50/50 包；不同时提交
direct 端点。

## 当前状态

协议、通用 CV model recipe 编排、测试和 A100 两阶段预检均已完成，可启动
正式训练。正式训练、OOF 与提交包均不存在。

## 实现与预检

- CV recipe 新增经过范围校验的 `model_arch`、`base_channels`、`block_type`；
  未声明时仍严格使用旧默认 `unet/32/residual`；
- 本机定向测试 `58 passed`；排除两项 Windows MKL 原生崩溃用例后的完整
  回归为 `153 passed, 1 skipped, 2 deselected`；
- A100/Linux 完整回归为 `156 passed, 53 subtests passed`，pipeline
  `bash -n` 通过；
- 真实路径 dry-run 恰好生成 10 条命令：五折各一条 stage1 和 stage2；
  stage1 无初始化，stage2 只读取同折 stage1 `best_global_mse.pt`；
- 64/32 张 train/val 的 stage1 预检为 `2.995 s`、峰值 `6.695 GiB`；
  stage2 为 `2.791 s`、峰值 `6.772 GiB`；
- stage2 checkpoint 记录 `18,825,185` 参数、base48 residual U-Net、
  spatial-thumbnail conditioning、AMP 和 `initialization.mode=strict`。

两次预检主进程均以 0 退出，checkpoint 完整。退出后的 `pymp-* EBUSY`
仍是 NFS multiprocessing 临时目录清理噪声，不影响训练或原子写盘；预检
目录与 dry-run 临时文件均已清理。

既有完整日志中，Round32 base32 thumbnail 平均 `50.77 s/epoch`，旧 base48
相对同代 base32 约慢 20%。Round33 共 140 个训练 epoch；计入双模型 OOF，
预计总 wall time 约 2 小时 45 分至 3 小时 15 分。

## 正式训练与 OOF

runtime commit `e91c7ec` 上的 10/10 条 run 全部完成，覆盖 25 个 ROI、
6,296 张图。140 个训练 epoch 的累计计时为 `14640.329 s`，OOF 评估耗时
`1292.263 s`，合计约 4 小时 25 分 33 秒。原预估低估约 70--100 分钟；
旧 context0/base48 与 base32 日志没有准确反映后几个 fold 上
spatial-thumbnail base48 的 `100+ s/epoch` 实际吞吐，后续不再沿用该比例。

机器可读报告为 `reports/round33_cv5_thumbnail_base48_oof.json`，大小
`124,488` bytes，本机与服务器 SHA-256 均为
`B08A9C77B6759FCB7BCD6C186BB4470ACB3E8A32160178A05F40BC767FACB4A1`。

base48 两阶段的最佳 epoch：

| fold | stage1 | stage2 |
|---|---:|---:|
| fold0 | 14 | 11 |
| fold1 | 14 | 14 |
| fold2 | 12 | 12 |
| fold3 | 14 | 13 |
| fold4 | 14 | 11 |

stage2 已在每折显著超过自己的 stage1，但最佳点均位于后半程。该事实说明
当前模型不是早期过拟合；它不改变 Round33 gate，但会影响是否另立 continuation
实验。

q95 绝对分如下，每项依次为 mean-image PSNR 与 global-MSE PSNR 口径分数：

| 范围 | base32 | base48 direct | 50/50 |
|---|---:|---:|---:|
| 完整 OOF | 74.745714 / 73.958810 | 74.233179 / 73.440664 | 74.748539 / 73.957587 |
| test-like | 72.551721 / 71.937528 | 72.102657 / 71.512679 | 72.603093 / 71.999505 |

相对 base32 的差值：

| 候选 | 范围 | float mean/global | q95 mean/global |
|---|---|---:|---:|
| base48 direct | 完整 | -0.513820 / -0.519371 | **-0.512534 / -0.518146** |
| base48 direct | test-like | -0.454458 / -0.430186 | **-0.449064 / -0.424848** |
| 50/50 | 完整 | +0.001440 / -0.002621 | **+0.002826 / -0.001223** |
| 50/50 | test-like | +0.047351 / +0.057963 | **+0.051372 / +0.061978** |

base48 direct 的五折 q95 双指标全部为负，范围为约 `-0.42` 至 `-0.63`。
50/50 相对 base32 的逐折 q95 为：

| fold | mean-image | global-MSE |
|---|---:|---:|
| fold0 | +0.061946 | +0.041186 |
| fold1 | -0.013250 | -0.013896 |
| fold2 | +0.048011 | +0.045361 |
| fold3 | -0.037288 | -0.050927 |
| fold4 | -0.056808 | -0.040147 |

只有 2/5 folds 双指标为正。test-like 中 ROI022 改善
`+0.541846/+0.541751`，但 ROI006 回退 `-0.331112/-0.268225`，ROI007
也回退 `-0.144861/-0.175125`，容量互补不稳定。

## Gate 判定与结论

50/50 primary gate 全部失败：完整 OOF 四项近零且包含负值，完整/test-like
q95 均未达到 `+0.15`，仅 2/5 folds 双正，ROI006 低于 `-0.10`，base48
direct 四项 q95 也全部为负。base48 direct diagnostic gate 同样全部失败。
因此 Round33 不生成提交包，不搜索融合权重、base64、epoch、校准或逐 ROI
组合。

本轮严格淘汰的是“fresh base48、固定 14+14 epoch、直接端点或 50/50”的
方案。它证明仅扩大宽度不能在当前预算下超过强 base32。另一方面，base48
stage2 的最佳 epoch 较晚，而 Round31 base32 经历了更多独立 cosine 周期；
因此本轮不能单独证明等训练时钟的 base48 仍会失败。若继续该路线，必须作为
新的固定 continuation 实验，先要求 base48 自身相对 Round33 获得跨五折
增益，再判断与 base32 的融合；不能把追加训练伪装成 Round33 gate 的救援。

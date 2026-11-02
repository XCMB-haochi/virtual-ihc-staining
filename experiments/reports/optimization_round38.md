# 优化记录 Round 38：空间对齐的缩略图特征融合

日期：2026-07-21

## 立项依据

Round37 的严格 cross-fit refiner 未形成可迁移增益：完整 OOF q95 相对
Round35 为 `-0.001199/+0.007853`，test-like 为
`+0.011808/+0.016833`，只有 2/5 folds 双正，四个 test-like ROI 双负。
它已按预注册门槛关闭，不搜索残差缩放、网络宽度或与 baseline 的事后融合。

现有 thumbnail 模型把 5x5 tile 邻域压成两通道 80x80 图，再经过四次
stride-2 卷积、D4 平均和全局平均池化，最终只以 64 维向量做 bottleneck
FiLM。该路径能描述“邻域中有什么”，但主动删除了“它在中心 tile 的哪个
方向和位置”。Round30 的 decoder FiLM 和 Round31 的 ring pooling 均没有
产生 matched structural gain，说明继续扩大全局条件向量不是优先方向。

Round38 检验一个不同、可在后续新数据上按同样五折流程重训的机制：保留
邻域缩略图的二维方向，并把它按真实像素尺度对齐到 U-Net bottleneck。实验
不使用测试标签、排行榜反馈、逐 ROI 路由、输出校准或提交探针。

## 固定结构

新条件模式为 `spatial-thumbnail-aligned-fusion`，完整复用 Round31
`spatial-thumbnail-film` 的统计特征、D4-invariant thumbnail FiLM 和全部
U-Net 权重，另加一条空间分支：

1. 输入仍为 `dapi_mean/presence` 两通道 80x80 邻域缩略图；
2. 使用 width32、stride1 的九层 3x3 context encoder，dilation 固定为
   `1/2/4/8/16/8/4/2/1`，全程保持 80x80；
3. 每个缩略图像素对应原图 16x16 像素，halo64 的 384x384 输入经过四次
   pooling 得到 24x24 bottleneck，恰对应缩略图中心 `[28:52, 28:52]`；
4. context feature 中心裁成 bottleneck 网格，经 zero-initialized 1x1
   projection 后残差相加；
5. 训练时缩略图严格复用源图实际抽到的水平翻转、垂直翻转和旋转；D4 TTA
   也对输入与缩略图执行同一变换，再只逆变换输出。

新分支增加 `91,776` 个参数，base32 总参数从 `8,527,041` 增至
`8,618,817`。1x1 projection 权重与偏置初始化为零，因此转换后的首次前向
与源 Round31 thumbnail checkpoint 逐值相等；新增结构不能靠不同随机初始化
或不同初始输出获得虚假优势。

## 单变量训练协议

机器可读协议为 `configs/round38_cv5_aligned_thumbnail_fusion.json`。五个
`alignedcontext` 均从 Round31 同折 `thumbnailcontinue` best checkpoint
初始化，只训练 stage2；固定 seed `2032`、14 epoch、lr `1e-4` cosine、
batch/workers `8/4`、halo/target halo `64/64`、Uniform-7 hybrid
`0.7 SSIM + 0.2 L1 + 0.1 MSE`、q95 global-MSE 选模和 D4 OOF。

matched control 固定复用 Round32 的同源、同 seed、同 14-epoch
`thumbnailcontinue`，不重训。部署基线固定为 Round35 的
`70% base32 + 30% continued-base48`。primary 只检验一个预先固定的 50/50
融合，展开为 `35% Round31 base32 + 15% Round34 base48 + 50% aligned`。
不搜索 context width、dilation、融合层、projection 初值、epoch、seed、loss、
50% 邻近权重、checkpoint soup 或逐 ROI 模型选择。

## 预注册门槛

机器判定由 `tools/evaluate_cv_gate.py` 读取完整 OOF 报告完成，结果写入独立
gate JSON。首先要求 mechanism gate：aligned direct 相对 Round32 matched
context64 control 必须同时满足：

1. 完整与 test-like q95 mean/global 四项均至少提高 `+0.10`；
2. 完整与 test-like 的四项 float 增益全部严格为正；
3. 至少 4/5 folds 的 q95 两项均为正；
4. 任一 test-like ROI 的 q95 单项不得低于 `-0.10`；
5. test-like 双负 ROI 不得超过一个。

primary gate 要求固定 50/50 融合相对 Round35 的四项 q95 均至少提高
`+0.12`，其余 float、fold 和 ROI 稳定性条件相同，并且 mechanism gate
已经通过。只有 primary 全部通过，才允许它进入测试推理。

若 primary 失败，唯一预注册后备是 aligned direct；它相对 Round35 必须满足
四项 q95 均至少 `+0.18`、四项 float 全正、5/5 folds 双正、单 ROI floor
`-0.12`、最多一个双负 ROI。主候选与 direct 均失败时，本轮不生成测试预测
或提交包。`tools/run_round38_pipeline.sh` 本身严格停在 OOF gate。

## 当前状态

本地实现已完成：条件构造、方向保持 context encoder、同步训练增强、同步 D4
TTA、严格 checkpoint 转换、协议 gate 校验、自动 gate evaluator 和单 GPU
pipeline 均已加入。确定性几何测试证明 halo64 输入降到 24x24 后与 thumbnail
中心裁剪逐格对应；zero-init 前向等价和新增 projection 梯度均有回归测试。

当前本机 focused 回归为：CV/gate `37 passed`，aligned/augmentation/D4
`6 passed`，checkpoint transition `1 passed`；Python compile、五折 dry-run
和 `git diff --check` 通过。

服务器已通过完整 preflight。Linux 全量回归为
`187 passed, 63 subtests passed`，compileall、pipeline `bash -n`、gate CLI 和
五折 dry-run 全部通过；dry-run 只产生五个 `alignedcontext/stage2` 任务，均
从对应 Round31 fold checkpoint 初始化。GPU0 上的 64-train/32-validation、
1 epoch 冒烟基于 commit `77125e1` 正常退出，训练加验证耗时 `3.322 s`，
峰值 CUDA 内存 `4.523520 GiB`，参数量 `8,618,817`。checkpoint 记录的转换为
`spatial-thumbnail-film-to-spatial-thumbnail-aligned-fusion`；训练后新增
projection 的 weight/bias 绝对值和分别为 `2.467700/0.040401`，确认分支已
离开零输出初态。末尾 `pymp-* EBUSY` 是该 NFS 环境既有的 worker 临时目录
清理噪声，进程退出码为 0。临时 smoke run 已核验路径后清理。

## 正式训练与 OOF

正式 pipeline 基于服务器 commit `9702e8d` 在 GPU0 完成，五折均达到固定
14 epoch。训练记录如下：

| fold | best epoch | selection score | 累计 epoch 时间 |
|---|---:|---:|---:|
| fold0 | 11 | 73.409734 | 875.773 s |
| fold1 | 14 | 73.679685 | 880.415 s |
| fold2 | 11 | 73.232092 | 875.990 s |
| fold3 | 11 | 74.162869 | 885.272 s |
| fold4 | 8 | 74.225895 | 882.781 s |

五折累计 epoch 时间为 `4400.230 s`，峰值 CUDA 内存均为
`4.523520 GiB`。完整 OOF 覆盖 25 ROI、6296 张图，评估耗时
`1733.627 s`。绝对 q95 mean-image/global-MSE 分数为：

| candidate | 完整 OOF | test-like OOF |
|---|---:|---:|
| Round35 70/30 baseline | 74.869164 / 74.073735 | 72.714860 / 72.101365 |
| matched context64 control | 74.824812 / 74.031087 | 72.578825 / 71.971816 |
| aligned direct | 74.822239 / 74.038315 | 72.589067 / 71.985616 |
| fixed 50% Round35 + 50% aligned | 74.898152 / 74.108390 | 72.710020 / 72.101838 |

原始 OOF 报告为
`reports/round38_cv5_aligned_thumbnail_fusion_oof.json`，大小 `202,134`
bytes，SHA-256 为
`BD7C11C665011DCD7231842D2B667058065489A246025A9ED07DDD728A4A1F27`。
机器 gate 报告大小 `7,630` bytes，SHA-256 为
`E2ABEAE60D16BA77036237487A20DF98CC4452731FC41421420244DE3B82ACDE`。

## Gate 结果

mechanism gate 中，aligned direct 相对 matched context64 control 的完整
q95 delta 为 `-0.002573/+0.007229`，test-like 为
`+0.010243/+0.013800`；完整 float 仍含 `-0.004066`。虽然 4/5 folds 双正、
没有 test-like 双负 ROI，变化量接近零，远低于固定 `+0.10` 门槛，且 fold4
mean-image 为负。因此空间对齐分支没有形成可辨认的 matched structural gain。

primary 固定融合相对 Round35 的完整 q95 增益为
`+0.028988/+0.034655`，但 test-like 为 `-0.004839/+0.000473`，对应 float
也含负项。尽管 5/5 folds 双正，ROI004、ROI015、ROI022、ROI024 双负，最差
为 ROI015 global-MSE 的 `-0.182591`，同时 mechanism dependency 失败。

aligned direct 相对 Round35 的完整 q95 为
`-0.046925/-0.035419`，test-like 为 `-0.125792/-0.115749`，0/5 folds
双正，七个 test-like ROI 双负，最差单项为 `-0.432771`。direct diagnostic
同样明确失败。

机器结论为 `promotion.allowed=false`。Round38 不生成测试预测、提交包、相邻
融合权重、checkpoint soup 或结构救援实验，并至此关闭。该结果只否定当前
空间分支作为成熟 Round31 U-Net 的 drop-in 增量；它不等价于从独立初始化
训练的新表示体系也失败。后者保留在独立的
`reports/backlog_representation_reset.md` 中，必须另行预注册后才能执行。

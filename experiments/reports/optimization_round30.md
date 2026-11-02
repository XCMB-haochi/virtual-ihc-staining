# 优化记录 Round 30：thumbnail decoder 多尺度 FiLM

日期：2026-07-20

## 启动依据

Round29 direct thumbnail-film 在平台得到 `72.8833`，显示
`SSIM=0.820, PSNR=25.793`，相对 direct spatial-film 提高 `+0.3667`。
完整 OOF 和 test-like 相对同一 spatial base 也分别提高
`+0.259167/+0.274245` 与 `+0.216019/+0.255854`，证明真实 5x5 DAPI
邻域是目前唯一连续在 OOF 和隐藏集兑现的新增结构信号。

现有 thumbnail 条件只在 U-Net bottleneck 注入。四级 encoder skip 仍完全
不受条件控制，并在 decoder 每一级与主分支拼接，可能逐步稀释 bottleneck
中的邻域信息。本轮固定检验：将同一个 D4-invariant 条件 embedding 以
零初始化 FiLM 注入四个 decoder block 后，能否提高结构净收益。

## 固定模型变化

新 condition mode 为 `spatial-thumbnail-multiscale-film`。它完整保留
Round29 的 101 维 spatial statistics、2x80x80 thumbnail、thumbnail CNN
和 bottleneck FiLM，仅新增四个 decoder affine head：

| decoder 输出通道倍数 | 8 | 4 | 2 | 1 |
|---|---:|---:|---:|---:|
| 注入 | channel-wise scale/shift | channel-wise scale/shift | channel-wise scale/shift | channel-wise scale/shift |

每个 head 输入固定为 `256` 维 stats hidden 与 `64` 维 thumbnail embedding
的拼接，共 `320` 维；weight 和 bias 均严格零初始化。Round29 checkpoint
的所有共享 tensor 必须名称和 shape 完全一致，新增 tensor 必须全部位于
`condition_film.decoder_outputs.*`。转换后的首次输出必须与来源模型逐值
一致。四个 head 共新增 `308,160` 个参数，模型总参数由 `8,527,041`
增至 `8,835,201`，仍不是扩大主干容量的实验。

不修改 thumbnail 半径/分辨率、U-Net 宽度、loss、halo、D4、JPEG、训练
target、checkpoint selection 或输出后处理。本轮也不加入 dropout、额外
全局分支、输出 affine、融合或 ROI 路由，因此唯一因素是 decoder 多尺度
条件注入。

## 匹配五折协议

机器可读协议为 `configs/round30_cv5_multiscale_film.json`，沿用 Round29
五折。每折从同一个 Round29 thumbnail best checkpoint 启动两条同预算分支：

| recipe | condition | 初始化 | 训练 |
|---|---|---|---|
| `thumbnailcontinue` | bottleneck only | strict | 14 epoch |
| `multiscalefilm` | bottleneck + four decoders | add-conditioning | 14 epoch |

两路固定 seed `2030`、lr `1e-4`、batch/workers `8/4`、halo64、full-target
监督和 Uniform-7 hybrid loss。原 Round29 `thumbnailbase` 只作不续训的
当前最佳参考。这样可以分离 decoder 注入收益与额外 14 epoch 的收益。

## 预注册判定

正式 promotion gate：

1. 相对 matched continuation，完整 OOF 与 test-like 的四项 q95 增益均
   至少为 `+0.05`；
2. 相对 Round29 thumbnail base，上述四项 q95 增益均至少为 `+0.12`；
3. 至少 4/5 folds 的 q95 双指标为正，且五折平均为正；
4. 相对 continuation，任一 test-like ROI 不得低于 `-0.20`，且双指标为负
   的 test-like ROI 不得超过两个；
5. 完整 OOF 与 test-like 的四项 float 增益均为正。

若 promotion gate 失败，但相对当前 Round29 base 的完整/test-like 四项 q95
均至少提高 `+0.15`，且五折双指标全部为正，只允许生成一次 direct
multiscale 诊断包。两个层级都禁止搜索 decoder 子集、融合比例、epoch、
seed、loss、校准或逐 ROI 路由。

## 当前状态

实现已完成。测试覆盖 condition metadata、旧/new condition 值完全一致、
Round29 thumbnail 模型到 multiscale 模型的共享 tensor 复制、首次输出逐值
一致、四个 decoder head 梯度、checkpoint metadata/strict roundtrip，以及
Round30 五折来源和十条训练命令。

本机完整可运行测试为 `142 passed, 1 skipped, 2 deselected`；两个 deselect
仍是既有 Windows Anaconda/MKL 原生崩溃用例。`compileall`、CLI、协议
dry-run 和 `git diff --check` 均通过；dry-run 恰好展开五条 matched
continuation 与五条 multiscale stage2。

服务器同步 runtime commit `ce16870` 后，全套测试为
`145 passed, 53 subtests passed`。物理 GPU1 启动前利用率 0%，已有其他
用户的 `5,749 MiB` 进程未被停止或修改。fold0 的 64 train / 32 val、
batch/workers `8/4`、1 epoch CUDA 烟测均成功：

| recipe | init | epoch 秒数 | 峰值 PyTorch 显存 |
|---|---|---:|---:|
| `thumbnailcontinue` | strict | 3.007 | 4.389 GiB |
| `multiscalefilm` | thumbnail-to-multiscale | 3.190 | 4.516 GiB |

转换报告只新增四个 decoder Linear 的 weight/bias，共 8 个 tensor。训练后
四个 decoder weight L1 依次为 `29.8740/13.7232/6.2398/2.5944`，确认
四个尺度全部获得更新。小样本单 epoch selection score 为 control
`71.8134`、multiscale `71.7001`；该烟测只验证执行、显存和梯度链，不参与
正式候选选择。进程退出码均为 0，末尾 `pymp-* EBUSY` 仍是既有 NFS
worker finalizer 噪声。

## 正式训练与 OOF 结果

runtime commit `edf8570` 上的十条正式 run 全部完成；每条 run 均产生了
独立的 `best_global_mse.pt`，随后自动执行完整 OOF。机器可读报告为
`reports/round30_cv5_multiscale_film_oof.json`，覆盖 25 个 ROI、6,296 张图，
评估耗时 `1146.802 s`。

multiscale 相对 matched thumbnail continuation 的差值如下。每一项依次为
mean-image PSNR 口径分数与 global-MSE PSNR 口径分数：

| 范围 | 编码 | vs continuation mean/global | vs Round29 base mean/global |
|---|---|---:|---:|
| 完整 OOF | float | +0.002064 / -0.004540 | +0.294751 / +0.266948 |
| 完整 OOF | q95 | **+0.002153 / -0.004446** | **+0.291993 / +0.264316** |
| test-like | float | +0.010161 / +0.007287 | +0.309373 / +0.277145 |
| test-like | q95 | **+0.010617 / +0.007730** | **+0.307018 / +0.274895** |

multiscale 的完整 OOF q95 绝对分为 `74.590587/73.787646`，test-like 为
`72.374155/71.747362`。matched continuation 对应为
`74.588434/73.792092` 与 `72.363539/71.739632`。这说明额外 14 epoch
续训相对 Round29 base 有明显收益，但四级 decoder FiLM 的结构净收益接近
零，且在完整 OOF global-MSE 口径略为负值。

相对 matched continuation 的逐折 q95 差值为：

| fold | mean-image 增益 | global-MSE 增益 |
|---|---:|---:|
| fold0 | +0.029095 | +0.032586 |
| fold1 | -0.010518 | -0.031372 |
| fold2 | +0.000483 | +0.001794 |
| fold3 | -0.011192 | +0.000650 |
| fold4 | +0.000343 | -0.028791 |

只有 fold0 与 fold2 双指标为正。test-like ROI 中，ROI015 回退
`-0.267731/-0.280271`，ROI019 回退 `-0.053332/-0.036280`；最大正增益
来自 ROI013，为 `+0.186132/+0.142659`。相对 Round29 base，五折 q95
双指标则全部为正，范围为 `+0.191506` 至 `+0.400865`。

## Gate 判定

正式 promotion gate 失败：

1. **失败**：相对 continuation 的完整/test-like q95 四项只有
   `-0.004446` 至 `+0.010617`，未达到 `+0.05`；
2. **通过**：相对 Round29 base 的四项 q95 为
   `+0.264316` 至 `+0.307018`，均超过 `+0.12`；
3. **失败**：相对 continuation 只有 2/5 folds 双指标为正；
4. **失败**：ROI015 低于预注册的 `-0.20` 回退限制；
5. **失败**：完整 OOF float global-MSE 口径为 `-0.004540`。

因此不能把 decoder 多尺度注入认定为有效结构改进。Round30 的平均增益
主要来自对 Round29 权重的额外续训。

诊断层 gate 通过：multiscale 相对 Round29 base 的完整/test-like 四项 q95
均超过 `+0.15`，且五折 q95 双指标全部为正。依照预注册规则，只生成一个
不融合、不校准、不作 ROI 路由的 direct multiscale 隐藏集诊断包。

## 单端点诊断包

测试推理只加载五个 multiscale best checkpoint，各权重 `0.2`；1,346 张图
耗时 `206.068 s`。产物为：

- ZIP：`submissions/cd68_round30_cv5_multiscalefilm_context64_d4_raw_q95_plain.zip`；
- ZIP 大小：`11,569,424` bytes；
- ZIP SHA-256：`EEDA4982FA0ADB876D0C4A0202256D27C8C37248FB53C962BF4838E9DD51F13F`；
- prediction provenance SHA-256：`829B75FA78274AC5E2D5229AB8B91EF70C6186DE6E40237A80CB5CF4DE68BC34`；
- OOF report SHA-256：`C31C682198AC8740BB367C2EB260568F886C27B9DEDBE7A46F01C86CD6D49B8C`。

服务器打包检查 expected/actual `1346/1346`、missing/extra 0、图像错误 0、
全部 RGB，独立 `unzip -tq` 无 CRC 错误。本机同步后的三份 SHA-256 与
服务器一致，`python -m zipfile -t` 再次通过。该包只用于判断续训后的
multiscale 固定端点能否超过当前平台最佳 `72.8833`，不改变正式 gate
失败的结论。

## 平台诊断结果与结论

平台返回精确分 `73.0409`，显示 `SSIM=0.822`、`PSNR=25.830`。相对
Round29 direct thumbnail-film 的 `72.8833` 提高 `+0.1576`；显示 SSIM
提高 `+0.002`，PSNR 提高 `+0.037`，两项指标方向一致。相对 Round28
direct spatial-film 的累计增益为 `+0.5243`。

这次隐藏集增益低于 multiscale 相对 Round29 base 的完整 OOF
`+0.291993/+0.264316` 和 test-like `+0.307018/+0.274895`，但符号一致，
证明额外续训后的固定端点仍能迁移，并成为新的平台最佳。相对此前记录的
榜首 `74.8890`，差距缩小到 `1.8481`。

该结果不能证明四级 decoder FiLM 有效。multiscale 与 matched continuation
在 OOF 上基本持平且局部不稳定，本次又没有提交 continuation 隐藏集端点，
因此 `+0.1576` 只能归因于“Round30 完整端点相对 Round29”的联合变化。
结合 matched OOF，最有证据的解释仍是额外 14 epoch 续训贡献了主要收益，
decoder 多尺度注入的独立贡献未知且很可能很小。后续不再沿此结构增加更多
FiLM 层，也不根据本次隐藏反馈搜索 decoder 子集。

# 优化记录 Round 29：真实邻域 DAPI 缩略图条件化

日期：2026-07-19

## 启动依据

Round28 direct spatial-film 在平台得到 `72.5166`，显示
`SSIM=0.816, PSNR=25.651`，相对 Round27 提高 `+0.2286`，但距榜首
`74.8890` 仍差 `2.3724`。这证明由 patch 坐标、局部统计量和邻域统计量
组成的 101 维条件有效，同时也说明只靠这些手工汇总量无法表达邻近 DAPI
中的实际组织形态。

本轮固定测试一个结构性假设：目标 patch 周围真实 DAPI 内容能够补充单 patch
halo64 和统计向量缺失的低频组织上下文。实验不搜索融合比例、seed、epoch、
缩略图尺寸或 ROI 特例，也不以 DiT 或更大 U-Net 替代当前已验证的主干。

## 固定模型设计

`spatial-thumbnail-film` 保留 Round27 的 101 维 spatial-statistics MLP 及其
参数名，并新增一个 DAPI-only 缩略图分支：

- 对每个 256x256 DAPI tile 做确定性的非重叠块均值，得到 16x16 缩略图；
- 按 patch 坐标拼成半径 2 的 5x5 邻域，即 80x80 intensity 通道；
- 增加同尺寸 presence 通道，显式区分缺失 tile 与真实黑色区域；
- 条件总维数为 `101 + 2*80*80 = 12,901`；
- 四层 stride-2 CNN 将缩略图编码成 64 维向量，再映射为 bottleneck FiLM；
- 编码器对 D4 的八种旋转/翻转求平均，保证训练增强和 D4 TTA 下方向一致；
- 新 FiLM 输出层严格零初始化，因此从 Round27 checkpoint 转换后的首次输出
  与原 spatial-film 模型逐 bit 相同。

base32 单输出模型参数量从 `8,399,937` 增至 `8,527,041`，仅新增
`127,104` 个参数。因此该实验检验的是邻域信息，而不是参数规模。

## 匹配对照协议

机器可读协议为 `configs/round29_cv5_thumbnail_film.json`。沿用 Round27 的
五个 ROI fold 和每折 `best_global_mse.pt`，每折训练两条同预算分支：

| recipe | 初始化 | 条件 | 训练 |
|---|---|---|---|
| `spatialcontinue` | Round27 spatial-film stage2 | 101 维统计 | stage2 续训 14 epoch |
| `thumbnailfilm` | 同一 checkpoint | 统计 + 真实邻域缩略图 | stage2 续训 14 epoch |

两条分支统一使用 seed `2029`、batch `8`、lr `1e-4`、halo64、相同 loss、
global-MSE checkpoint selection、D4 和 JPEG q95。旧 Round27 spatial-film
仅作为无额外训练的 `spatialbase` 参考。这样可以把新增上下文的收益与单纯
多训 14 epoch 的收益分开，而不是把续训本身误判为结构提升。

## 预注册推进门槛

在任何测试推理前固定以下 gate。`thumbnailfilm` 只有同时满足全部条件才可
生成主提交候选：

1. 相对 matched `spatialcontinue`，完整 OOF 和 test-like 子集的 q95
   mean-image/global-MSE competition score 均至少提高 `+0.20`；
2. 相对旧 `spatialbase`，上述四项 q95 增益均至少为 `+0.10`；
3. 至少 4/5 folds 的两项 q95 指标均优于 `spatialcontinue`，且五折平均为正；
4. 任一 test-like ROI 相对 `spatialcontinue` 的 q95 回退不得低于 `-0.08`；
5. 完整 OOF 和 test-like 的 float 双口径增益均为正。

不因未过 gate 而搜索缩略图半径、分辨率、融合权重、epoch、seed、逐 ROI
路由或后处理补救。`spatialcontinue` 若要独立晋级，也必须相对
`spatialbase` 通过同类完整 OOF、test-like、fold 和 ROI 稳健性检查。

## 实现与本地验证

训练器支持从 `spatial-stats-film` 到 `spatial-thumbnail-film` 的显式
`--init-add-conditioning` 转换；共享 tensor 必须名称和 shape 完全一致，
新增 tensor 必须全部属于 thumbnail 分支。条件 metadata 固定记录下采样、
邻域、通道、归一化、D4 聚合与初始化方案，预测时严格复核。

单元测试覆盖缩略图块均值、邻域 intensity/presence 布局、metadata、D4
不变性、旧 checkpoint 无损转换、首次输出完全一致、梯度进入新输出层及
Round29 十条命令展开。条件值哈希改为逐向量流式计算，保持旧 SHA-256
语义，同时避免为 12,901 维条件再复制约 310 MiB 矩阵；图像也改为逐张
提取后立即释放，不让旧 spatial-statistics 路径额外保留全部 DAPI。

本机结果为 `137 passed, 1 skipped, 2 deselected`；两个 deselect 是既有
Windows Anaconda/MKL 原生崩溃用例。`compileall`、协议 dry-run 和
`git diff --check` 均通过，dry-run 恰好展开五条 matched continuation 与
五条 thumbnail run。

## 服务器预检

服务器无损快进到 runtime commit `8cc2da0` 后，Linux 全套测试为
`140 passed, 53 subtests passed`。远程 wrapper 同时固定使用
`/workspace/envs/virtual-ihc`，不再继承远端 home 下的 Python。

物理 GPU1 为 A100-PCIE-40GB；启动前已有其他用户的 `5,726 MiB` 空闲进程，
利用率为 0%，该进程未被停止或修改。fold0 的两条 64 train / 32 val、
batch/workers `8/4`、1 epoch CUDA 烟测均成功：

| recipe | init | epoch 秒数 | 峰值 PyTorch 显存 |
|---|---|---:|---:|
| `spatialcontinue` | strict | 6.796 | 4.355 GiB |
| `thumbnailfilm` | spatial-to-thumbnail | 5.038 | 4.388 GiB |

thumbnail checkpoint 的 `thumbnail_output.weight` L1 为 `10.3410`，bias
L1 为 `0.1121`；按 seed `2029` 重建初始模型后，thumbnail encoder 权重
变化 L1 为 `16.2138`。这确认零初始化输出层已打开，梯度随后进入编码器，
而不是只保存了未使用的新参数。进程退出码均为 0；末尾
`/workspace/tmp-meimy/pymp-*` 的 `EBUSY` 是既有 NFS worker
finalizer 噪声，config、metrics 和 checkpoint 完整。

预检未发现显存、速度、初始化或数据条件构建方面的阻塞。

## 正式训练与 OOF 结果

runtime commit `b1aaf41` 上的十条正式 run 全部完成；每条恰好包含 14 个
epoch 和独立 `best_global_mse.pt`。完整 OOF 报告为
`reports/round29_cv5_thumbnail_film_oof.json`，覆盖 25 个 ROI、6,296 张图，
评估耗时 `1128.446 s`。

thumbnail 相对 matched continuation 以及旧 spatial base 的差值如下：

| 范围 | 编码 | vs continuation mean/global | vs spatial base mean/global |
|---|---|---:|---:|
| 完整 OOF | float | +0.063211 / +0.075511 | +0.258843 / +0.273939 |
| 完整 OOF | q95 | **+0.063708 / +0.075955** | **+0.259167 / +0.274245** |
| test-like | float | +0.115161 / +0.118412 | +0.215505 / +0.255412 |
| test-like | q95 | **+0.113933 / +0.117142** | **+0.216019 / +0.255854** |

thumbnail 的完整 OOF q95 绝对分为 `74.298594/73.523330`，test-like 为
`72.067137/71.472467`。相对 continuation 的逐折 q95 结果为：

| fold | mean-image 增益 | global-MSE 增益 |
|---|---:|---:|
| fold0 | +0.132408 | +0.141645 |
| fold1 | +0.013099 | +0.049848 |
| fold2 | +0.121066 | +0.139158 |
| fold3 | -0.027450 | -0.028568 |
| fold4 | +0.062535 | +0.055262 |

四个 fold 双指标为正，但 test-like ROI 并不稳定。相对 continuation，
ROI004 为 `-0.240518/-0.232675`，明显超过预注册的 `-0.08` 回退限制；
ROI007、ROI015 和 ROI022 则分别提高 `+0.430136/+0.407308`、
`+0.659497/+0.665922` 和 `+0.197658/+0.184095`。这表明真实邻域内容有
可学习信号，但收益强烈依赖 ROI，而非稳定的全局改进。

## Gate 判定

预注册 gate 最终为失败：

1. **失败**：完整与 test-like q95 相对 continuation 只有
   `+0.064~+0.117`，未达到四项 `+0.20`；
2. **通过**：相对旧 spatial base 的四项 q95 均超过 `+0.10`；
3. **通过**：4/5 folds 双指标为正且平均为正；
4. **失败**：ROI004 回退 `-0.240518/-0.232675`；
5. **通过**：完整与 test-like 的四项 float 增益均为正。

matched continuation 本身相对旧 base 的完整 q95 为
`+0.195459/+0.198289`，test-like 为 `+0.102085/+0.138713`，但 ROI006、
ROI013、ROI015 存在明显回退，因此同样不作为稳定晋级候选。结论是额外
续训和 thumbnail 都带来平均收益，但 Round29 没有产生通过预注册标准的
正式替代模型；不据此搜索融合权重、缩略图参数、epoch、seed 或逐 ROI 路由。

## 单端点诊断包

尽管未通过正式晋级 gate，thumbnail 相对当前已提交 Round28 spatial base
在五折 q95 双指标全部为正，完整 OOF 为 `+0.259167/+0.274245`，test-like
为 `+0.216019/+0.255854`。考虑到初赛允许不限次数提交，且
`thumbnailfilm` 是实验前已经固定的单端点，本轮额外生成一次明确标记为
**诊断**的 direct 包，用隐藏集反馈判断平均 OOF 增益是否迁移。它不改变
gate 失败结论，也不使用 post-hoc 融合、校准或 ROI 选择。

测试推理只加载五个 thumbnail best checkpoint，各权重 `0.2`；1346 张图
耗时 `200.693 s`。产物为：

- ZIP：`submissions/cd68_round29_cv5_thumbnailfilm_context64_d4_raw_q95_plain.zip`；
- ZIP 大小：`11,561,425` bytes；
- ZIP SHA-256：`0B33ADCBDA5164BBBB80EAC2020597016B4413A18FBB4793BCB7576A08AB331E`；
- prediction provenance SHA-256：`0FC2C5F27D29DD4D8DEB9F51A3C2FF2D63267D835EB3711C24129F3FC2C35FD5`；
- OOF report SHA-256：`CB5E8A231420AA22CE27051940D33708E20B4D3CFD66A6C280F76863C5C4B4DA`。

服务器打包检查 expected/actual `1346/1346`、missing/extra 0、图像错误 0、
全部 RGB，独立 `unzip -tq` 无 CRC 错误。本机同步后再次得到 1346 个唯一
canonical 条目、非法路径 0、CRC error 0、解码/尺寸/灰度 RGB 通道错误 0，
且三份 SHA-256 与服务器一致。

## 平台诊断结果与结论

平台返回精确分 `72.8833`，显示 `SSIM=0.820`、`PSNR=25.793`。相对
Round28 direct spatial-film 的 `72.5166` 提高 `+0.3667`；显示 SSIM 提高
`+0.004`，PSNR 提高 `+0.142`，两类指标同时改善。隐藏集增益高于完整 OOF
相对 spatial base 的 `+0.259167/+0.274245`，也高于 test-like 的
`+0.216019/+0.255854`，因此平均收益成功迁移且没有被 OOF 高估。

这项结果不改变预注册 gate 的失败事实：thumbnail 相对 matched
continuation 的结构净增益不足 `+0.20`，且训练 ROI 中存在明显局部回退。
它说明隐藏测试集的总体分布更有利于 thumbnail，而不能证明 ROI004 类风险
已经消失。诊断包因此从“有平均收益但稳定性不足”更新为当前实际最佳模型，
后续实验应保留真实邻域分支并针对跨 ROI 波动做结构性约束，而不是回退到
纯 spatial-film 或根据本次结果搜索融合比例。

相对此前记录的榜首 `74.8890`，当前仍差 `2.0057`。下一轮仍需要新的模型
改进，不能把这次 `+0.3667` 当作已接近榜首。

# 优化记录 Round 26：ROI 分层 5-fold OOF 集成

日期：2026-07-18

## 立项依据

Round14、15、16 与 25 都出现相同的跨域取舍：ROI020 明显提高，而
ROI024 明显回退。Round22/24 的 fresh full-target 模型在两个独立 fold 的
总体分均提高，固定 50/50 融合还能消除大部分单 ROI 回退，但两折平均增益
约为 `+0.264`，不足以稳健映射到 full-data epoch。Round17 又证明，即使
H1/H2 通过，单个固定 full-data epoch 仍可能在服务器失效。

本轮不再用两个小 fold 选择 full-data 时钟。改用覆盖全部 25 个训练 ROI 的
分层 5-fold OOF：每折分别训练 matched base32 与 full-target，同折固定
50/50 输出平均；OOF 中每张训练图只由未见该 ROI 的模型预测。若通过，测试
阶段平均五折的十个模型，从而避免另行猜测 full-data epoch。

## 固定折分

折分只使用 DAPI ROI 均值，不使用 CD68 标签。将 25 个 ROI 按 DAPI 均值
排序，连续划为五个强度带，再以蛇形顺序向五折各分配一个 ROI：

| fold | validation ROI |
|---|---|
| fold0 | ROI011, ROI012, ROI001, ROI020, ROI022 |
| fold1 | ROI002, ROI009, ROI018, ROI015, ROI005 |
| fold2 | ROI008, ROI000, ROI007, ROI024, ROI014 |
| fold3 | ROI023, ROI017, ROI004, ROI019, ROI010 |
| fold4 | ROI016, ROI021, ROI013, ROI006, ROI003 |

机器可读协议为 `configs/round26_cv5.json`。25 个 ROI 必须恰好出现一次，
任何缺失或重复均 fail closed。

## 固定训练与推理

每折两条 matched 轨迹均使用 CD68、seed 2026、base32 residual U-Net、
Uniform-7 hybrid `0.7/0.2/0.1`、batch/workers `8/4`：

| recipe | 训练 input / target | stage1 | stage2 |
|---|---|---|---|
| base32 | 256 / 256 | fresh 14e, lr 3e-4 | stage1 best 初始化 14e, lr 1e-4 |
| fulltarget | 384 / 384 | fresh 14e, lr 3e-4 | stage1 best 初始化 14e, lr 1e-4 |

两者均按 validation global-MSE、JPEG q95 保存 best，推理统一使用
halo64+D4。主候选固定为同折 base32/fulltarget 浮点输出 50/50 后再编码
q95；不搜索折权重、模型权重、epoch、seed、halo、校准或后处理。

训练器保存每个 epoch，并在 20 个 stage 全部完成后生成完整 OOF。DAPI
test-like 子集完全由输入定义：训练 ROI 的 DAPI 均值落在五个测试 ROI
DAPI 均值最小值与最大值之间；边界在评估时由数据重新计算。

## Promotion gate

主比较是 `base32_fulltarget_50` 相对 matched `base32`：

1. 完整 25-ROI OOF 的 q95 mean-image 与 global-MSE 均至少提高 `+0.20`；
2. DAPI test-like OOF 子集的两项 q95 分也均至少提高 `+0.20`；
3. 五个 fold 的两项 q95 增益均为正；
4. test-like 单 ROI 的两项 q95 回退均不得超过 `0.05`；
5. 完整 OOF 与 test-like 的两项 float 增益均为正；
6. 任一条件失败即关闭，不搜索融合权重、删除困难折或改选 checkpoint。

## 测试候选边界

只有通过全部 gate，才实现测试预测器并生成一个 raw q95 候选：每折先做
base32/fulltarget 50/50，再对五折等权平均，等价于十个 checkpoint 各
`0.1`。首个服务器包不使用 ROI029 校准或既有逐 ROI 端点选择，以便直接
判断 CV 模型本身。若 raw 确有增益，后续与当前最佳包的组合需另立协议。

## 启动条件

1. 协议校验、命令构造、完成态识别和固定加权单元测试通过；
2. 本地可运行测试、compileall、CLI、dry-run 与 diff check 通过；
3. 远端全量测试通过，代码 commit 已推送并通过校验 bundle 同步；
4. GPU1 资源与 `/workspace` 容量检查通过；
5. 正式输出目录均不存在，不干预其他用户进程。

## 执行与工程验证

20 个 stage 全部一次完成，状态文件最终记录 `complete=true`。总训练 wall
time 约 `3.51 h`；base32/full-target 峰值显存约为 `2.003/4.323 GiB`。
退出时出现的 `/workspace/tmp-meimy/pymp-*` `EBUSY` 是已知 NFS
worker finalizer 噪声；每个 run 的 14 条 metrics、best、last 和逐 epoch
checkpoint 均完整，编排器最终明确输出 `COMPLETE round26_cv5: 20 stages`。

- 本机新增定向测试：`8/8` 通过；
- 本机可运行全套：`113 passed, 1 skipped, 2 deselected`，两个 deselect
  均为既有 Windows Anaconda/MKL `lstsq` 原生崩溃；
- 服务器干净环境全量：`116 passed, 53 subtests passed`；
- 本机与服务器 compileall、CLI、protocol dry-run 和 diff check 通过；
- 服务器 commit：`8cc0ac3`，20 条 Linux 训练命令在启动前完整展开核验。

## 完整 OOF 结果

正式 halo64+D4 评估覆盖 25 个 ROI、6296 张图，每张图只由未见其 ROI
的同折 checkpoint 预测。原始报告为 `reports/round26_cv5_oof.json`，评估
耗时 `975.6 s`。

完整 OOF：

| candidate | q95 mean / global | float mean / global |
|---|---:|---:|
| base32 | 73.247188 / 72.474117 | 73.325240 / 72.551311 |
| full-target | 73.406448 / 72.650904 | 73.485747 / 72.729392 |
| fixed 50/50 | **73.466948 / 72.698706** | **73.545141 / 72.776053** |

固定 50/50 相对 matched base32：

| scope | q95 mean / global | float mean / global |
|---|---:|---:|
| 完整 OOF | +0.219760 / +0.224590 | +0.219900 / +0.224742 |
| DAPI test-like | +0.238628 / +0.238596 | +0.239703 / +0.239667 |

DAPI test-like 范围由五个测试 ROI 的输入均值自动得到
`36.504716-39.280706`，对应 9 个训练 ROI、2318 张图：ROI004、ROI006、
ROI007、ROI013、ROI015、ROI019、ROI020、ROI022、ROI024。

五折 q95 增益：

| fold | mean-image | global-MSE |
|---|---:|---:|
| fold0 | +0.164602 | +0.134397 |
| fold1 | +0.253266 | +0.300297 |
| fold2 | +0.241044 | +0.220087 |
| fold3 | +0.143844 | +0.161575 |
| fold4 | +0.297963 | +0.315284 |

test-like 单 ROI q95 增益：

| ROI | mean-image | global-MSE |
|---|---:|---:|
| ROI004 | -0.012746 | +0.052304 |
| ROI006 | +0.513170 | +0.519542 |
| ROI007 | +0.141578 | +0.132490 |
| ROI013 | +0.258397 | +0.258915 |
| ROI015 | +0.441775 | +0.446984 |
| ROI019 | +0.288044 | +0.289554 |
| ROI020 | +0.250508 | +0.245535 |
| ROI022 | +0.198985 | +0.176040 |
| ROI024 | +0.094519 | +0.094763 |

## Gate 判定

全部预注册条件通过：完整 OOF 与 test-like q95 均超过双 `+0.20`；五折
两项 q95 全部为正；test-like 最差项仅 ROI004 mean-image `-0.012746`，
未超过 `-0.05` 边界且其 global-MSE 为正；完整 OOF 与 test-like 的四项
float 增益也全部为正。

因此允许生成唯一 raw q95 测试候选。测试时每折先做 base32/full-target
50/50，再对五折等权平均，等价于十个 stage2 best checkpoint 各 `0.1`。
首包禁止 ROI calibration、旧模型混合、逐测试 ROI 选择或其他后处理。

## 唯一测试候选

gate 通过后新增 `tools/predict_cv_ensemble.py`。服务器干净环境最终测试为
`117 passed, 53 subtests passed`；推理器根据协议加载十颗 stage2 best，
逐颗核验 fold ROI 与 marker 元数据，记录 SHA-256，并确认十个有效权重均为
`0.1`。完整 provenance 为 `reports/round26_cv5_test_prediction.json`。

- 预测图：1346 张 canonical JPEG；
- test 推理耗时：`332.844 s`；
- 推理：十模型 float 等权、halo64、D4、raw、JPEG q95；
- ZIP：`submissions/cd68_round26_cv5_base32_fulltarget50_context64_d4_raw_q95_plain.zip`；
- ZIP SHA-256：`2172AABAECB467E251A3383E6951089F1BA7CF3B9B8B8C622C28819E03858E24`；
- ZIP 大小：`11,330,647` bytes；
- provenance SHA-256：`E5335A57B112067EB184CF717149A472330D27AEA84B2FEB6AE8FF3D361B0D89`。

既有打包器逐图检查得到 expected/actual `1346/1346`、missing/extra 0、
image error 0、全部 RGB。同步回本机后再次独立检查：1346 个 ZIP entry
全部唯一，duplicate 0、CRC error 0、非法路径 0、图像解码/尺寸/通道错误
0，且本地/服务器 SHA-256 完全一致。

## 服务器结果与结论

平台返回精确分 `71.8249`，显示 `SSIM=0.807`、`PSNR=25.565`。相对旧的
full-data raw 50/50 基线 `71.7440` 提高 `+0.0809`，相对 Round17 的
base32/base48 full-data 50/50 `71.5953` 提高 `+0.2296`；但相对经过逐测试
ROI 端点选择与 ROI029 校准的当前最佳 `72.0437` 仍低 `0.2188`。

因此 Round26 的主要结论是：完整 ROI OOF 选择得到的 full-target/CV 集成
确实转化成了隐藏测试 raw 提升，但服务器增益小于本地 matched OOF 中约
`+0.22` 的幅度，仍不足以替代此前依赖榜单反馈的逐 ROI 组合。该结果支持
继续把完整五折 OOF 用作后续结构比较协议，但不支持直接扩大同构 U-Net
集成数量。首包的受控比较已经完成；若后续融合旧模型或加入 ROI029 校准，
必须作为独立实验记录，不能归入本轮 raw 结果。

# 优化记录 Round 24：full-target 固定融合的独立 H1 复现

日期：2026-07-18

## 目标与解释边界

Round23 的 H2 固定 50/50 raw 融合相对 base32 提高 `+0.235475/+0.240439`，三个 ROI 全部为正，但略低于该轮双 `+0.25` 门槛。Round24 不重新使用 H2 选择任何参数，只在独立 H1 `ROI015,ROI018,ROI019` 上复现完全相同的 full-target 训练和固定融合。

本轮不是放宽 Round23 gate。它回答一个新的问题：H2 的次强、均匀正增益能否在未参与选择的完整 ROI fold 上达到足够强度，使双折证据超过 Round17 已失败的 `+0.31` 水平。

## 固定 H1 训练

| 项目 | 固定值 |
|---|---|
| 验证 ROI | `ROI015,ROI018,ROI019` |
| marker / seed | `CD68 / 2026` |
| 模型 | base32 residual U-Net |
| 训练 input / target | context64 384x384 / context64 384x384 全区域监督 |
| 验证 input / target | context64 384x384 / 中心 256x256 |
| batch / workers | `8 / 4` |
| loss | Uniform-7 hybrid `0.7/0.2/0.1` |
| stage1 | fresh，14 epoch，lr `3e-4` cosine，wd `1e-4` |
| stage2 | stage1 `best_global_mse.pt` strict init，14 epoch，lr `1e-4` cosine |
| 选模 | global-MSE，JPEG q95 |
| 推理 | halo64+D4 |

固定输出：

- `runs/round24_h1_unet_context64_fulltarget_stage1/`；
- `runs/round24_h1_unet_context64_fulltarget_stage2/`；
- `runs/_round24_eval/fulltarget_h1.json`；
- `runs/_round24_eval/base32_fulltarget_50_h1_q95.json`；
- q95 通过后才生成 `runs/_round24_eval/base32_fulltarget_50_h1_float.json`。

## 固定 H1 对照与融合

- base32 checkpoint：`runs/round17_h1_unet_base32_stage2/best_global_mse.pt`；
- base32 SHA-256：`aadcdc89ff1b907fe03c38a584ac4229d4dfbcce858b45fb852feb1cf8649ece`；
- base32 q95：`70.374826/69.842100`；
- 浮点融合：`0.5 * base32 + 0.5 * H1 full-target`；
- 只读取 `weight_b_0.5_raw`，禁止 calibration variants。

不搜索训练 halo、target halo、seed、epoch、融合权重、校准、soup、base48 或第三模型。

## Promotion gate

已知 H2 的两项 q95 增益和为 `0.475914`。要让 H1/H2 四项平均至少为 `+0.40`，H1 两项增益之和必须至少：

```text
4 * 0.40 - 0.475914 = 1.124086
```

固定 gate：

1. H1 q95 mean-image 与 global-MSE 增益各至少 `+0.15`，且二者之和至少 `1.124086`；
2. ROI015、ROI018、ROI019 的两种 q95 聚合均不得回退超过 `0.05`；
3. q95 通过后才运行 float，两个总体增益必须为正，逐 ROI不得回退超过 `0.05`；
4. 结合 H2 后，四项 q95 增益必须各为正、平均至少 `+0.40`，六个 ROI均不得回退超过 `0.05`；
5. 任一条件失败即关闭 full-target full-data 路线，不调整权重或回到 H2 搜索。

## Full-data 边界

本轮通过也不直接训练 full-data。必须先按 H1/H2 同 epoch 指标预注册 stage1/stage2 epoch 映射、full-data seed 和唯一 raw 提交包；不得依据服务器分数选择模型、epoch、权重或校准。

## 启动条件

1. Round22 代码与测试保持不变，远端 commit 与 tracked tree 已核验；
2. H1 base32 checkpoint 的 fold、model spec、内部 epoch、marker、conditioning 和 SHA-256 已核验；
3. 本预注册提交、推送并同步服务器；
4. GPU1 资源检查通过，正式输出目录不存在，不干预其他用户进程。

## H1 训练结果

full-target stage1/stage2 均完成 14 epoch，单 epoch 约 `49-52 s`，峰值显存 `4.294/4.323 GiB`。

| 阶段 | best epoch | 裸 q95 mean / global | checkpoint SHA-256 |
|---|---:|---:|---|
| stage1 | 13 | 70.001733 / 69.446512 | `d40e3313a7ef6bccd3a634b7e03e89dedd56f24b8275fcfd7c5d7104c6c52d64` |
| stage2 | 13 | 70.465233 / 69.916972 | `5e44ad1fd4c6b3c977d09e9f6fb82c48349548c7c184170683b4def3b1c82aa9` |

H1 延续了 H2 的正方向，且 stage2 后程继续改善，没有复现 H2 的 epoch1 早停形态。

## H1 正式结果

full-target direct halo64+D4：

| 输出 | base32 | full-target | direct 增益 |
|---|---:|---:|---:|
| q95 mean / global | 70.374826 / 69.842100 | 70.696794 / 70.142887 | +0.321968 / +0.300787 |
| float mean / global | 70.447770 / 69.914658 | 70.772034 / 70.217687 | +0.324264 / +0.303029 |

固定 50/50 raw q95：

| 输出 | base32 | 50/50 raw | 融合增益 |
|---|---:|---:|---:|
| mean-image | 70.374826 | 70.669771 | +0.294946 |
| global-MSE | 69.842100 | 70.126918 | +0.284819 |

逐 ROI q95 增益：

| ROI | direct mean / global | 50/50 mean / global |
|---|---:|---:|
| ROI015 | -0.247809 / -0.272192 | +0.011064 / -0.004376 |
| ROI018 | +0.146430 / +0.126318 | +0.253644 / +0.252605 |
| ROI019 | +0.788105 / +0.773083 | +0.494513 / +0.486055 |

direct 总体更高，但 ROI015 明显回退；固定融合消除了该回退并满足逐 ROI 边界，但总体幅度下降。原始证据：

- `reports/round24_fulltarget_h1.json`；
- `reports/round24_base32_fulltarget_50_h1_q95.json`。

## 决策

Round24 未通过：H1 固定融合两项增益之和为 `0.579765`，低于预注册的 `1.124086`；H1/H2 四项平均增益仅约 `+0.2639`，低于 `+0.40`。因此不运行 blend float，不训练 full-data，不调整融合权重，也不生成排行榜包。

两个独立 fold 证明 full-target 是真实正方向，但从头训练会在个别 ROI 产生域回退，而后验融合只能部分修复。下一项只能改变训练起点而非继续调融合：Round25 从既有强 base32 checkpoint 严格初始化，固定做一次 full-target adaptation，检验是否保留 base 泛化同时吸收完整区域监督。

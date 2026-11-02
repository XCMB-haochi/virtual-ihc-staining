# 优化记录 Round 32：late-continuation EMA

日期：2026-07-20

## 立项依据

Round31 direct thumbnail continuation 在平台得到 `73.1837`，显示
`SSIM=0.823, PSNR=25.911`，相对 Round30 提高 `+0.1428`。隐藏增益与
完整 OOF 的 `+0.157280/+0.166717` 只差 `0.0145~0.0239`，证明当前五折
协议可以可靠判断 continuation。Round31 五折最佳 epoch 为
`7/13/13/13/11`，多数折在后程达到最佳，但 epoch14 已有小幅回落。

本轮不修改模型结构、输入、loss 或推理。唯一问题是：在同一晚期
continuation 轨迹上使用参数指数滑动平均，能否减少末期 mini-batch 更新噪声
并稳定提高五折 best checkpoint。

## 固定 EMA

EMA decay 固定为 `0.999`。EMA 模型在加载 Round31 source checkpoint 后
与 online 模型逐 tensor 相同；每次 optimizer update 后执行：

```text
ema = 0.999 * ema + 0.001 * online
```

每折约有 600 余个 optimizer updates/epoch，14 epoch 后初始 source 在 EMA
中的残余权重低于 `0.001`。验证、best checkpoint、每 epoch snapshot 和最终
OOF 均读取 EMA 权重；optimizer 仍只更新 online 模型。既有训练器已经实现
并测试该路径，本轮只给 CV recipe 增加显式、范围校验后的 `ema_decay` 编排。

Round2 的 `EMA=0.999` 负结果来自短预算早期模型，EMA 当时明显滞后。本轮
从已连续三轮验证的强 checkpoint 启动，测试的是 late-stage 平滑，不能把
两者视为同一训练状态。

## 匹配五折协议

机器可读协议为 `configs/round32_cv5_thumbnail_ema.json`，沿用 Round29-31
五折。每折从同一个 Round31 thumbnail continuation best checkpoint 启动：

| recipe | checkpoint model | 初始化 | 训练 |
|---|---|---|---:|
| `thumbnailcontinue` | online | strict | 14 epoch |
| `thumbnailema` | EMA 0.999 | strict, EMA=online | 14 epoch |

两路固定 seed `2032`、lr `1e-4` cosine、batch/workers `8/4`、halo64、
full-target 监督和 Uniform-7 hybrid loss。Round31 source 只作不续训 base。
两路 online optimizer、数据顺序、augmentation 和更新次数必须相同，唯一
差异是验证及保存时读取 online 还是 EMA 权重。

## 预注册判定

EMA 正式 promotion gate：

1. 相对 matched online continuation，完整 OOF 与 test-like 的四项 q95
   增益均至少为 `+0.04`；
2. 完整 OOF 与 test-like 的四项 float 增益均为正；
3. 至少 4/5 folds 的 q95 双指标为正，且五折平均为正；
4. 相对 online，任一 test-like ROI 不得低于 `-0.10`，且双指标为负的
   test-like ROI 不得超过两个；
5. 相对 Round31 base，完整/test-like 四项 q95 均至少提高 `+0.10`。

独立 diagnostic gate：online 或 EMA 固定端点只有在相对 Round31 base 的
完整/test-like 四项 q95 均至少提高 `+0.10`、五折 q95 双指标全部为正，且
任一 test-like ROI 不低于 `-0.20` 时才可生成包。若两者都通过，只生成四项
q95 增益最小值更高的一个；精确并列时优先 EMA。禁止搜索 decay、epoch、
EMA/online 融合、loss、seed、校准或逐 ROI 路由。

## 实现与预检

- 本机定向测试：`55 passed`；排除两项 Windows MKL 原生崩溃用例后的完整
  回归：`150 passed, 1 skipped, 2 deselected`；
- A100/Linux 完整回归：`153 passed, 53 subtests passed`；shell 语法检查通过；
- 真实路径 dry-run 生成 10 条 run：5 条 matched online、5 条 EMA，EMA 参数
  只出现在 5 条 `thumbnailema` 命令中；
- fold0 EMA 限量预检使用 64/32 张 train/val、1 epoch，训练耗时
  `3.128 s`，峰值显存 `4.417 GiB`；checkpoint 记录
  `ema_decay=0.999`、`initialization.mode=strict`，并且 EMA 权重已相对 source
  发生非零变化（浮点参数 mean absolute change `1.115e-6`）。

预检退出码为 0，checkpoint 完整。Python multiprocessing 在 NFS `TMPDIR`
退出时报告临时目录 `Device or resource busy`；该警告发生在训练与原子写盘完成
之后，不影响产物或正式训练。预检目录及传输临时文件均已清理。

## 当前状态

协议、测试、A100 预检、正式十条 run 与 OOF 均已完成。以下结果按预注册
规则判定；Round32 不生成测试包。

## 正式训练与 OOF

runtime commit `996eabb` 上的 10/10 条 run 全部完成，覆盖 25 个 ROI、
6,296 张训练图。OOF 评估耗时 `1115.600 s`。机器可读报告为
`reports/round32_cv5_thumbnail_ema_oof.json`，大小 `148,492` bytes，
本机与服务器 SHA-256 均为
`B0A4A4EC6D8E862F6D013E25F7C3ECBFF3B0FB46B93CE3C56FB7CB68DFB188F5`。

两路每折按 global-MSE 选中的最佳 epoch 为：

| fold | online | EMA |
|---|---:|---:|
| fold0 | 11 | 1 |
| fold1 | 14 | 8 |
| fold2 | 11 | 7 |
| fold3 | 11 | 9 |
| fold4 | 10 | 1 |

EMA 在两个 fold 的 epoch1 即达到最佳，而 online 多数在 epoch10--14 达到
最佳，说明固定 `0.999` 在这段晚期、短 cosine 轨迹上存在明显滞后。

q95 绝对分如下，每项依次为 mean-image PSNR 与 global-MSE PSNR 口径分数：

| 范围 | Round31 base | matched online | EMA |
|---|---:|---:|---:|
| 完整 OOF | 74.745714 / 73.958810 | 74.824812 / 74.031087 | 74.783771 / 73.991795 |
| test-like | 72.551721 / 71.937528 | 72.578825 / 71.971816 | 72.581122 / 71.961976 |

EMA 相对 matched online 的差值：

| 范围 | float mean/global | q95 mean/global |
|---|---:|---:|
| 完整 OOF | -0.040659 / -0.038873 | **-0.041040 / -0.039292** |
| test-like | +0.003051 / -0.009062 | **+0.002297 / -0.009840** |

逐折 q95 的 EMA - online 为：

| fold | mean-image | global-MSE |
|---|---:|---:|
| fold0 | -0.093246 | -0.100577 |
| fold1 | -0.018661 | -0.030771 |
| fold2 | +0.011184 | +0.012743 |
| fold3 | -0.028869 | -0.033823 |
| fold4 | -0.077612 | -0.043523 |

test-like 中 EMA 相对 online 有五个 ROI 双指标为负；ROI020 最差，为
`-0.192571/-0.207936`。正向变化主要集中在 ROI006、ROI013 与 ROI024，
没有形成跨 ROI 的稳定收益。

## Gate 判定

EMA promotion gate 全部失败：

1. 完整 OOF q95 两项为负，test-like global-MSE 也为负，未达到 `+0.04`；
2. float 同样在完整 OOF 双负、test-like global-MSE 为负；
3. 仅 1/5 folds 的 q95 双指标为正；
4. 五个 test-like ROI 双指标为负，且 ROI020 低于 `-0.10`；
5. EMA 相对 Round31 base 的四项 q95 仅为
   `+0.038057/+0.032986/+0.029401/+0.024449`，未达到 `+0.10`。

matched online 相对 Round31 base 的结果为：

| 范围 | float mean/global | q95 mean/global |
|---|---:|---:|
| 完整 OOF | +0.078756 / +0.071897 | **+0.079098 / +0.072277** |
| test-like | +0.025596 / +0.032760 | **+0.027104 / +0.034288** |

online 的五折 q95 双指标全部为正，但四项增益都低于 diagnostic gate 的
`+0.10`；test-like ROI013 还回退 `-0.207396/-0.226427`，超过 `-0.20`
边界。因此 online diagnostic gate 也失败，不生成低置信度 continuation 包。

## 结论

关闭固定 `EMA=0.999` 路线，不搜索 decay、EMA/online 融合或 EMA epoch。
连续 continuation 的完整 OOF 增益已从 Round31 的约 `+0.16` 衰减到约
`+0.07`，test-like 仅约 `+0.03`，说明当前 checkpoint 已接近这条训练轨迹的
收益上限。后续保留 Round31 endpoint 作为已验证基线，但不再用无新机制的
cosine 重启作为主实验。

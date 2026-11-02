# Context64 Soup 全局提交探针

日期：2026-07-17

## 候选

两个候选都不训练新模型，主成员均为现有 H2 context64 seed2026 late soup：

| 版本 | seed2026 成员 | seed2027 成员 | 各自权重 |
|---|---|---|---:|
| full-data hybrid | context64 H2 late soup，SHA `6BC2026E...C874C30` | full-data continuation e6，SHA `DBDB01C6...BA2F2` | 50% |
| H2 matched | context64 H2 late soup，SHA `6BC2026E...C874C30` | H2 continuation e6，SHA `B045A65B...D7481` | 50% |

推理固定为 `context_halo=64`、`tta=d4`、JPEG quality 95、raw 无 ROI 缩放。

## H2 固定证据

已有报告：`reports/ensemble_context64_soup_late_seed2027e6_h2.json`。该报告严格对应上表的 **H2 matched** 模型对，不对应 full-data hybrid。

| 固定 50/50 raw | 逐图分 | global-MSE 分 |
|---|---:|---:|
| H2 continued 双成员 | 70.2233 | 69.5237 |
| context64 late soup + H2 seed2027 e6 | 70.3767 | 69.6877 |
| 增益 | +0.1535 | +0.1640 |

逐 ROI 增益：

| ROI | 逐图分 | global-MSE 分 |
|---|---:|---:|
| ROI020 | +0.0348 | +0.0654 |
| ROI022 | +0.2740 | +0.2761 |
| ROI024 | +0.1476 | +0.1481 |

三 ROI 和两种聚合均为正，但两个 H2 成员都未使用 ROI020/022/024 训练，context64 soup 的 epoch 组合也在同一 H2 上选择。该结果属于同折选择证据，不是独立验证或服务器分数预测。

full-data hybrid 将 H2 seed2027 替换为 full-data seed2027。后者使用了全部 25 个训练 ROI，理论上数据更充分，但因其见过 H2 验证目标，无法用同一 H2 公平离线计分。因此 full-data hybrid 没有严格对应的离线增益，只作为成员替换探索。

## 提交包与服务器结果

| 版本 | 文件 | SHA-256 | 大小 | 精确分 | SSIM | PSNR | 相对 raw 71.7440 |
|---|---|---|---:|---:|---:|---:|---:|
| full-data hybrid | `cd68_context64late_s2026_50_full_s2027e6_50_context64_d4_raw_q95_plain.zip` | `F4E3B734C2C3A610C672E176599026799CC0999F0404F6BB85746BAA94EE08C4` | 11,246,236 bytes | 71.3833 | 0.802 | 25.363 | -0.3607 |
| H2 matched | `cd68_context64late_s2026_50_h2_s2027e6_50_context64_d4_raw_q95_plain.zip` | `19F4DD8E01D22E8B9612A91D5F9DB3AD3EBFAB3972F977B0A1ABC8F92D967196` | 11,199,966 bytes | 71.3481 | 0.802 | 25.327 | -0.3959 |

两包均有 1346 个唯一 canonical JPEG、CRC 正常、缺失 0、多余 0、图像错误 0；相对当前 raw 50/50 基准均为 1346/1346 张 JPEG 变化。

两包作为 per-ROI 端点探针之后的第 11、12 个独立提交。它们不参与 `reports/per_roi_blend_probes.md` 的 ROI 加和计算。

H2 matched 在本地相对 matched control 为 `+0.1535/+0.1640`，服务器候选却比当前 full-data raw 低 `0.3959`。两者基准并非严格相同，所以不能把差值解释为精确的服务器兑现率或严格的符号反转；但该结果足以否定“当前 H2 context64 soup 可直接迁移到测试集”的假设。full-data hybrid 仍含 H2 context64 成员，它不是真正的 full-data context64 模型，因此也不能证明从全部 25 个训练 ROI 训练的 context64 必然失败。

决策：淘汰这两个 soup，不做 ROI029 校准版本，也不据此租 GPU 推广全量 context64。只有新的容量或大上下文方案在至少两个独立完整 ROI fold 上方向一致，并通过预注册门槛后，才考虑云端扩大训练。

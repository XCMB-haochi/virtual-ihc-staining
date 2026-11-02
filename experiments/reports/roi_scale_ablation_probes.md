# 测试 ROI 校准边际贡献探针

日期：2026-07-16；更新：2026-07-17

## 基准

- 全部 ROI 使用 log0.75：`71.6491`
- 全部 ROI 不缩放：`71.5653`
- 总收益：`+0.0838`

两包使用相同的 40% Uniform-7 seed 2026 + 60% seed 2027、halo64、D4 和 JPEG q95，只改变 ROI 缩放。

## Leave-one-ROI-out 包

每个包保持另外四个 ROI 使用 log0.75，仅将文件名所示 ROI 替换为无缩放输出。

| 关闭缩放的 ROI | 图像数 | SHA-256 | 文件 |
|---|---:|---|---|
| ROI025 | 277 | `FEFA25ABC4224595ABA9D23FC52213E4A907C350646DE4473DB6022D0051F5DC` | `cd68_u7s2_log075_leaveout_disable_ROI025.zip` |
| ROI026 | 322 | `F8D76A813A129F7EAC04B4653516445E330AB4E0183A76AFFCBFAD26F720D6A3` | `cd68_u7s2_log075_leaveout_disable_ROI026.zip` |
| ROI027 | 243 | `628E66C223365500D156A8D84F71A32FDD52D17E7F160093673DC55EE21C122E` | `cd68_u7s2_log075_leaveout_disable_ROI027.zip` |
| ROI028 | 246 | `E77BA0F84F778FCBE609C3A254CAB3554BE7285C2119CA0A6F7413E0D92A896E` | `cd68_u7s2_log075_leaveout_disable_ROI028.zip` |
| ROI029 | 258 | `535BFA7558FBF729B0A9DBFA3923104CD06D89AE0233A68D815BFF3BEAE147DD` | `cd68_u7s2_log075_leaveout_disable_ROI029.zip` |

目录：`submissions/roi_scale_ablation/`。每个 ZIP 均有 1346 个唯一条目并通过 CRC 检查。

## 解释方式

若消融包得分为 `S_i`，则 ROI 的当前边际贡献定义为：

```text
contribution_i = 71.6491 - S_i
```

- `contribution_i > 0`：当前 log0.75 对该 ROI 有益。
- `contribution_i < 0`：关闭该 ROI 缩放反而提高总分，应在组合包中禁用。
- 五个贡献之和若接近 `0.0838`，说明当前评分对按 ROI 替换近似可加；若偏差明显，则存在全局 PSNR 聚合或其他非线性。

第一轮只提交这五包，不同时改变强度。得到五个分数后，再只对正贡献且幅度较大的 ROI 搜索强度，避免将 ROI 选择和强度选择混在一次反馈中。

## Round 2：续训模型固定 50/50 权重

Round 1 使用旧模型。新续训 experimental 与 raw 的融合权重不同，不能直接分解校准贡献，因此 Round 2 先构建固定 50% seed 2026 e5 + 50% seed 2027 e6 的对照。

### 固定权重基准

- raw 已知服务器分：`71.7440`
- log0.75 控制包：`cd68_continue_s2026e5_50_s2027e6_50_context64_d4_log075_q95_control_plain.zip`
- 控制包 SHA-256：`383E9B401D23483F84E9FCAB30D9295C4503B1C6AC2A52E11832F543F1DEB78A`
- 控制包服务器分：`C=71.8597`，显示 `SSIM=0.808`、`PSNR=25.554`

### Leave-one-ROI-out 包

| 关闭缩放的 ROI | 图像数 | 精确分 `L_i` | SSIM | PSNR | `C-L_i` | SHA-256 | 文件 |
|---|---:|---:|---:|---:|---:|---|---|
| ROI025 | 277 | 71.8755 | 0.808 | 25.556 | -0.0158 | `A6AD97323289C9BFEDF598EDDC462CDE0F289D4E69A52DD76633605E04A5738A` | `cd68_continue_50_50_log075_leaveout_disable_ROI025.zip` |
| ROI026 | 322 | 71.9350 | 0.809 | 25.556 | -0.0753 | `C5C070EEE8C10A2D90CB5483F60FD93115EA9425900CECEAAB61C63C9A572E6A` | `cd68_continue_50_50_log075_leaveout_disable_ROI026.zip` |
| ROI027 | 243 | 71.8927 | 0.808 | 25.569 | -0.0330 | `BD9E7ACACA5F024900D6287AC3ECE21559706C9F7A592BF215D7E8472FAF9C94` | `cd68_continue_50_50_log075_leaveout_disable_ROI027.zip` |
| ROI028 | 246 | 71.8636 | 0.808 | 25.556 | -0.0039 | `EB982C494843865A5298AE0F1CF10F2B447213641A5B9BEE502B96C0E5576E90` | `cd68_continue_50_50_log075_leaveout_disable_ROI028.zip` |
| ROI029 | 258 | 71.6162 | 0.805 | 25.396 | +0.2435 | `A279C8F7EDB1FF6BC454B72852C19D14BA0700385D40DA7020040A12603C7446` | `cd68_continue_50_50_log075_leaveout_disable_ROI029.zip` |

控制包位于 `submissions/`，五个消融包位于 `submissions/roi_scale_ablation_round2/`。六个新 ZIP 均有 1346 个唯一条目并通过 CRC 检查。

Round 2 的边际贡献定义为 `C-L_i`。负值表示启用该 ROI 缩放有害，正值表示有益。因此 ROI026 消融包虽然以 `71.9350` 成为消融阶段中间最高提交，其语义是 ROI026 缩放应关闭。ROI025、ROI026、ROI027、ROI028 的贡献均为负，只有 ROI029 为正。

五项贡献之和为 `+0.1155`，整体校准收益为 `C-71.7440=+0.1157`，残差仅 `0.0002`。最终组合只对 ROI029 使用 log0.75：提交前从 raw 路径预测为 `71.9875`，从 control 剔除四项负贡献预测为 `71.9877`。平台实际分为 `71.9875`，命中预测下界，验证了近似可加推导。

### ROI029-only 最终组合包

- 文件：`submissions/cd68_continue_s2026e5_50_s2027e6_50_context64_d4_roi029log075_q95_plain.zip`
- SHA-256：`E0077FF5E4CBCD91FA1949ED9D4D006A208A715762416A65494FA5AF39D77828`
- 大小：`11,290,645` bytes
- ROI025-ROI028：1088 张，逐字节取自 raw 包
- ROI029：258 张，逐字节取自 log0.75 control 包
- 校验：1346 个唯一条目、CRC 正常、逐文件来源错配 0
- 平台结果：`71.9875`，`SSIM=0.809`、`PSNR=25.573`
- 状态：ROI 校准阶段最高提交；相对 raw 提高 `+0.2435`

## Round 3：ROI029 强度曲线

只改变 ROI029 的 log-space strength，ROI025-ROI028 保持 raw：

| ROI029 strength | 精确分 | SSIM | PSNR |
|---:|---:|---:|---:|
| 0 | 71.7440 | 0.807 | 25.414 |
| 0.75 | 71.9875 | 0.809 | 25.573 |
| 1.5 | 71.7563 | 0.808 | 25.351 |

log1.5 比 log0.75 低 `0.2312`。三点二阶差分为 `D=-0.4747`，拟合顶点为 `0.7597`，理论增益仅约 `+0.00004`，因此保留 log0.75，不生成确认包并停止强度搜索。完整推导见 `reports/optimization_round7.md`。

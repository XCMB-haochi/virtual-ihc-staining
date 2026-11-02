# 优化记录 Round 42：MixStyle 跨 ROI 风格泛化

日期：2026-07-23

## 立项依据

Round11--41 已测试多任务、NAFNet、预训练 ConvNeXt、GroupDRO、Swin、
Restormer、EMA、更大上下文、残差精修和全局校准。反复出现的主要问题不是训练
分不足，而是某些 ROI 提高时另一些 ROI 回退。当前成熟训练增强只有 D4 几何
变换，没有显式模拟 DAPI 获取强度和低层纹理统计的域偏移。

[MixStyle 原始论文](https://openreview.net/forum?id=6xHJ37MVxxp)提出在浅层混合
不同训练样本的 instance mean/std，以隐式合成新域；
[作者官方 MIT 仓库](https://github.com/KaiyangZhou/mixstyle-release)在无域标签时
支持随机样本混合，并给出固定默认值 `p=0.5, alpha=0.1`。本轮独立实现该公式，
不复制模型代码，也不引入外部数据或预训练权重。

## 回归适配

分类网络可直接在浅层块后混合，但虚拟染色需要保持逐像素配准。本实现因此在
U-Net 前两次 pooling 后、进入更深主干前执行 MixStyle；对应的高分辨率 skip
在混合前保存，避免直接破坏最浅层定位信息。统计量 detach，归一化与混合以
float32 执行，随后恢复原 dtype。模块无参数、无 buffer，默认和 eval 状态均为
逐值恒等，因此旧 checkpoint 的 state key、参数量和推理结果不变。

训练时每个浅层位置以 `p=0.5` 启用，从对称
`Beta(0.1, 0.1)` 为每个样本采样插值系数，并随机排列 batch 内 style 统计。
不使用 ROI 标签、测试输入、测试统计或排行榜反馈。

## 匹配实验

机器协议为 `configs/round42_cv5_mixstyle_domain_generalization.json`。候选和
Round32 matched online 均从相同的 Round31 五折 checkpoint 初始化，固定：

- seed `2032`、14 epoch、`1e-4` cosine、batch/workers `8/4`；
- base32 residual U-Net、halo64/full-target64、spatial-thumbnail FiLM；
- Uniform-7 hybrid `0.7/0.2/0.1`、global-MSE checkpoint、D4、q95；
- 唯一差异为候选启用两处固定 MixStyle。

本轮只训练五个 `mixstylecontinue` fold。Round31 base32、Round32 matched
online 和 Round34 continued-base48 均复用既有 checkpoint。完整 OOF 固定计算：

1. matched online direct；
2. MixStyle direct；
3. Round35：70% Round31 base32 + 30% base48；
4. matched seed blend：35% Round31 + 35% matched online + 30% base48；
5. MixStyle blend：35% Round31 + 35% MixStyle + 30% base48。

35% 不是从本轮 OOF 搜索得到，而是把 Round35 中 70% 的 base32 权重等分给原
模型和第二轨迹。不会测试其他融合权重。

## 预注册 gate

所有 gate 只使用 25 个训练 ROI 的完整 OOF，配置显式
`test_like_diagnostic=false`，不读取测试 DAPI。

### Direct mechanism

MixStyle direct 相对 matched online 必须同时满足：q95 至少 `+0.05`、float
为正、至少 4/5 folds 和 15/25 ROI 为正、最差 ROI 不低于 `-0.12`。

### Ensemble mechanism

MixStyle blend 相对 matched online blend 必须同时满足：q95 至少 `+0.02`、
float 为正、至少 4/5 folds 和 15/25 ROI 为正、最差 ROI 不低于 `-0.08`，
并依赖 direct mechanism 通过。

### Primary promotion

MixStyle blend 相对 Round35 必须同时满足：q95 至少 `+0.08`、float 为正、
至少 4/5 folds 和 15/25 ROI 为正、最差 ROI 不低于 `-0.08`，并依赖前两层
mechanism gate 通过。只有 primary 全部通过才允许生成一次固定测试候选。

任一依赖失败即不生成测试预测或提交包，并关闭当前 `p/alpha/层位` 配方；不
追加强度扰动、MixStyle 超参数搜索、逐折/ROI 路由或排行榜探针。

## 当前状态

协议、训练路径、完整分布 gate 和单元测试已实现。本地 MixStyle、gate 与 CV
协议定向测试为 `90 passed`；完整可运行测试为 `219 passed, 1 skipped, 2
deselected`。两项 deselect 是 Windows MKL 已知崩溃用例，未由本轮改动引入。

Linux/A100 全量测试为 `220 passed, 2 deselected, 71 subtests passed`，Bash
语法检查和协议 dry-run 均通过。GPU0 烟雾训练使用 64/32 张训练/验证图、1 epoch，
峰值显存 `4.39 GiB`；源与新 checkpoint 的 91 个 state tensor 键完全相同，
初始化报告为 `strict`，训练后 91 个 tensor 均发生更新。烟雾输出已清理。

## 正式 OOF 结果

五个 MixStyle fold 均完成 14 epoch。GPU0 上的完整 D4 OOF 覆盖 25 个 ROI、
6296 张图，用时 `1793.68 s`。官方逐图 q95/float 分数如下：

| 候选 | q95 | float |
|---|---:|---:|
| matched online direct | 74.824812 | 74.905388 |
| MixStyle direct | 74.793239 | 74.874140 |
| Round35 baseline | 74.869164 | 74.948362 |
| matched online blend | 74.920752 | 74.999767 |
| MixStyle blend | 74.919076 | 74.997839 |

MixStyle direct 相对 matched online 为 `-0.031573/-0.031249`，只有 fold2 和
10/25 ROI 的 q95 为正，最差 ROI011 为 `-0.371542`。MixStyle blend 相对
matched online blend 仍为 `-0.001676/-0.001928`，只有 2/5 folds 和 12/25
ROI 为正，最差 ROI011 为 `-0.111220`。因此 direct 和 ensemble mechanism
gate 均失败。

MixStyle blend 相对 Round35 虽为 `+0.049912/+0.049477`，且 5/5 folds、
20/25 ROI 为正，但低于预注册的 `+0.08`，并依赖已经失败的 mechanism gate，
所以 primary 也失败。机器判定为 `promotion.allowed=false`。

## 结论

关闭当前 MixStyle 配方，不搜索概率、Beta 参数、层位或融合权重，也不为
MixStyle 生成测试预测。相对 Round35 的正增益来自额外独立训练轨迹的集成，
而不是 MixStyle：预先固定的 matched online blend 略高于 MixStyle blend。
完整产物为：

- `reports/round42_cv5_mixstyle_domain_generalization_oof.json`；
- `reports/round42_cv5_mixstyle_domain_generalization_gate.json`。

OOF SHA256 为
`8607e787ae43046f954e533e7243478db9ed1b72d9bfe6d5b4c18a72dee9a7a9`。

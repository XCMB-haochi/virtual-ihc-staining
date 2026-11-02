# 虚拟染色开源仓库调研

日期：2026-07-18

## 结论

当前没有一个仓库可以不经适配直接替代本项目。多数虚拟染色工作优化视觉真实性、分布匹配或多蛋白联合生成，而本赛题由配准图像的 SSIM/PSNR 主导。直接引入 GAN、diffusion 或 H&E foundation encoder 风险高；优先借鉴多尺度像素约束、局部/全局双分支和可控的 marker-specific head，并继续以纯回归为对照。

## 第一方仓库

| 优先级 | 仓库 | 可借鉴内容 | 主要限制 |
|---|---|---|---|
| A | [VisCy / Cytoland](https://github.com/mehta-lab/VisCy) | UNeXt2 ConvNeXt V2 编解码、DAPI/phase FCMAE 预训练、2D virtual-staining 配方 | 原论文为细胞显微图像且预训练 800 epoch；需折内验证并适配 IHC 强度分布 |
| A | [HistoPlexer](https://github.com/ratschlab/HistoPlexer) | 多通道生成、Gaussian-pyramid L1、patch contrastive、singleplex/全 marker 接口 | cGAN 主线；CC BY-NC-ND，不直接复制或修改分发代码 |
| A | [VIRVS](https://github.com/casus/virvs) | 同生成器的 U-Net 与 pix2pix 对照；paired 虚拟荧光训练/评估脚本 | 任务不是 IHC；其证据反而支持 SSIM/PSNR 场景先用确定性回归 |
| A | [Pix2pix DualBranch / HEMIT](https://github.com/BianChang/Pix2pix_DualBranch) | 局部卷积分支 + Swin 全局分支；MIT | 原方法带 GAN；本项目应先只移植回归骨干 |
| B | [MAS](https://github.com/ZixiaZ/MAS) | DAPI/自发荧光 masked-autoencoder 预训练、多模态 attention、跨 marker 迁移 | TensorFlow 2.6、无许可证；本项目只有 DAPI 输入 |
| B | [DeepLIIF](https://github.com/nadeemlab/DeepLIIF) | 成熟的 paired 多模态 translation 与多头数据组织 | 原方向为 IHC 到多路 mpIF/分割；Apache 2.0 + Commons Clause |
| B | [UNIStainNet](https://github.com/facevoid/UNIStainNet) | marker embedding、FiLM/SPADE 式条件化 | 依赖 H&E 语义的 UNI encoder；research-only 且无标准许可证 |
| C | [non-aligned-i2i](https://github.com/honkamj/non-aligned-i2i) | paired-but-misaligned 的 deformation-equivariant translation | 当前数据看似严格配准；先做微移位审计再考虑 |
| C | [VirtualMultiplexer](https://github.com/AI4SCR/VirtualMultiplexer) | PatchNCE 与细胞/邻域/全局一致性 | unpaired/CUT 路线偏分布真实性，与像素指标不一致 |

## 对本项目的影响

1. 不把多输出本身当作初赛提分项；组织者已确认多个 marker 取平均，只有新增 marker 单项高于当前均值时才有收益。
2. 先用同一四输出 checkpoint 分别提交四个单-marker ZIP，确认服务器 marker 难度排序；不要把四类放进同一个平均包。
3. 若 CD68 仍最高，下一结构实验优先做 HistoPlexer-lite：保留当前 SSIM/MSE 回归主损失，只加入低权重双尺度 Gaussian-pyramid L1，或实现无 GAN 的 local-conv + global-Swin 双分支。
4. 已有 pyramid 单尺度和四任务共享 U-Net 均未通过 H2，因此任何新实验必须带新的结构因素，不能简单恢复旧配置。
5. 暂不投入 GAN、diffusion、UNI encoder 或无配准 CUT；这些方法的训练目标与排行榜指标不匹配。

## 2026-08-26 复核

- HistoPlexer 官方仓库提交 `b378fc5` 的许可证为 CC BY-NC-ND 4.0，不复制或修改其实现。更重要的是，发布版 trainer 在 `torch.no_grad()` 中提取生成/真实 IHC 特征，随后又以 `torch.tensor(total_asp_loss)` 重建 PatchNCE 汇总，生成器计算图被切断；该发布实现不能作为 PatchNCE 确实训练 generator 的可靠证据。Gaussian-pyramid 项也已被本项目早期 matched 实验否定。因此不启动 HistoPlexer-lite loss 实验。
- [ImplicitStainer](https://arxiv.org/abs/2505.09831) 尚未发布代码。论文方法使用无下采样的全分辨率 CNN+Swin 特征、3x3 局部窗口、坐标编码和逐像素 MLP。其局部输出机制与 Round49 的半分辨率48通道、PixelShuffle、3x3宽头有较大重叠，而论文坐标编码消融差异很小；Round49直接模型已下降约0.31，因此不把轻量坐标MLP包装成独立新结构。
- ImplicitStainer 补充消融显示，去掉感知损失时 texture metrics 的 PSNR/SSIM 反而最高；感知项主要改善 FID等分布指标。这与本赛题的逐图SSIM/PSNR目标不一致，进一步否定VGG/LPIPS/PatchNCE作为当前优先训练loss。
- Meta [ConvNeXt-V2官方实现](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py)在block残差分支使用逐样本DropPath，且官方Tiny fine-tuning命令固定最大率0.2并按深度线性增加。Round74因此测试该明确、无推理成本的encoder正则化机制，而不是继续搜索学习率或感知loss。

## 2026-09-01 空间蛋白基础模型复核

- [VirTues](https://github.com/bunnelab/virtues)提供了公开模型和完全遮蔽目标
  channel 的 marker-reconstruction 接口，说明跨 marker 表征是合理研究方向；但其
  官方输入词表不含 DAPI，示例依赖其它蛋白通道，不能原样用于本赛题的 DAPI-only
  推理。
- VirTues 预训练覆盖大量公开空间蛋白数据。现有审计只能确认 IMMUcan IMC panel
  含 Histone H3、HLA-DR、CD68、CD45RO 和核通道但不含 Vimentin，不能排除其它
  VirTues 数据集与比赛隐藏测试来源重叠。规则禁止预训练数据与测试数据重叠，因此
  在来源未被排除前不下载或使用 VirTues 权重。
- 公开的同赛题仓库
  [Virtual-Staining-IHC-Image-Generation](https://github.com/qingfeng123456-cyber/Virtual-Staining-IHC-Image-Generation)
  提供 NAF/频率/多路结构代码，但没有可核验的 checkpoint、完整 grouped OOF 或
  平台结果；其单次 5-ROI 数字不能与本项目 25-ROI OOF 比较。仓库只能作为结构
  线索，不能作为复制其复杂模型的正证据。
- Round96 因此采用更窄且可审计的迁移：只用官方训练集，在每个外层 fold 内将
  DAPI、CD68、CD45RO、HLA-DR、Vimentin 分别作为独立灰度图做同一个 FCMAE
  掩码重建，并只迁移 encoder。它不使用外部空间蛋白权重，也不让预训练看到
  validation ROI 的任一 target channel。

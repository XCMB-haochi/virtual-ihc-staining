# 优化记录 Round 110：冻结多染色 FCMAE 特征引导

## 立项依据

Round109 在未见 ROI 上得到加权标准化特征 R2 `0.654895`，fold0/fold2 分别
为 `0.671613/0.638097`；正确 DAPI/CD68 配对相对同 ROI 错配的余弦优势也均
超过固定门槛。这证明 DAPI 包含可跨 ROI 预测的 CD68 teacher 表征，允许一次
真正的训练期 feature-guidance screen。

公开的双对齐多任务特征引导工作使用辅助任务模型缓解配对图像错位与任务差异。
本轮不复制其 GAN、聚类或主动匹配系统，而在已验证的确定性回归管线中测试一个
更窄、可归因的机制：用官方训练数据训练出的 fold-local 多染色 FCMAE encoder
作为冻结 perceptual teacher。

## 固定机制

- student：与 Round55 `officialauxiliary15` 相同的 UNeXt2-Tiny、官方
  FCMAE800 初始化、四输出及 `0.85/0.05/0.05/0.05` marker 权重；
- teacher：同折 Round96 五模态 FCMAE encoder，预训练时对 DAPI 和四个 IHC
  目录都排除了该折验证 ROI，参数冻结且始终 eval；
- 特征：只取生成 CD68 与真实 CD68 中心 256x256 的 stage4 `768x16x16`
  feature，沿 channel 做 L2 normalization 后计算逐位置 cosine distance；
- 权重：固定 `0.05`，叠加在原 Uniform-7 hybrid loss 上；
- 成本边界：每个随机 batch 只取前 4 个样本计算 teacher loss。DataLoader 已
  shuffle，因此样本随 epoch 变化；student 仍保持 batch16，不引入梯度累积
  或数据顺序差异，teacher 不进入推理和 checkpoint model state。

两折均从相同 Round52 fold-local encoder fresh 开始，保持 seed2044、context64
全区域监督、`14+14` epoch、学习率、选模、context128+D4、q95 和固定
`[0,-1]` 偏移。没有 teacher 层、crop、batch 子集、loss 权重、epoch 或融合搜索。

## Gate

相对历史 matched Round55 auxiliary15，候选必须提高至少 `+0.05` q95、float
为正、两个 fold 均为正、至少 7/10 ROI 为正且最差 ROI 不低于 `-0.12`。
失败即关闭该直接 perceptual feature-guidance 路线，不扩展五折或生成提交；
通过才允许原样扩展。

## 执行与完成审计

实现通过本地及远端共 138 项 feature-guidance、训练安全和 CV 协议定向测试，
同时通过 Python 编译、runner dry-run、shell 语法和 Round109 依赖哈希检查。
权威源码提交为 `0376b02`。服务器 checkout 的 Git 对象库仍处于此前已知损坏
状态，训练器记录的旧 `git_commit` 不代表本轮代码；正式运行额外绑定下列远端
文件 SHA-256：

- `src/ihc_feature_guidance.py`：
  `7037433b8a510f30a65db40a118775dff1fa6b12d3932948ec50a34428636de9`；
- `tools/train_unet.py`：
  `bb0785c21b43888f64172bfa480a3ca33a5645ac0e655a4bce54f1f39097d39f`；
- `tools/run_cv_training.py`：
  `dc261eb6bb9cb0b1fac644620fcb01ca8068976792c24b8dc66ab0b316f0444a`；
- `tools/cv_protocol.py`：
  `2f866d6150acfd6b0a2aa4cf04d858aec616136eca0c4308c8601f042ea32f34`；
- protocol：
  `b66081c19e9026dd8fb98da69a0feaca77da2474355c27e05230d6aa8a4efabb`；
- runner：
  `ae18c7eacabccd0fe74eda661999a48511ec706ac63c726510d4f0fa0fce0639`。

真实 fold0 smoke 使用 batch16、64 个训练样本和 32 个验证样本，teacher hash
与 fold exclusion 校验通过。首轮 feature loss 为 `0.271811`，总损失
`0.395568`、原 hybrid loss `0.381977`，数值满足
`total = supervised + 0.05*feature`；四次 optimizer step 完成并写出可加载
checkpoint。峰值显存 `26.583 GiB`，低于正式启动门槛 `30,000 MB`。

detached Screen `round110_feature_guidance` 于 2026-09-05 启动。调度器在 GPU0
持续被外部任务占用时将两折串行安排到 GPU1，未干预其他进程。fold0 于
10:03:38 完成，fold2 于 11:47:42 完成；两折各有 stage1/stage2 连续 14 个
epoch，状态文件为 `complete=true`，并生成可加载的 `best_global_mse.pt`。
训练峰值显存约 `26.70 GiB`。日志中的 `pymp-*` 临时目录清理告警发生在
训练完成之后，未改变 fold 状态或 pipeline 退出码。

OOF 覆盖预注册的 10 个 ROI、2706 张验证图像，完整评估耗时 `1554.261 s`。
候选 checkpoint SHA-256 为 fold0
`7554ef404583cc59041bc182a942dc60e5f3f1dcd2b6c877f57bd1659814d965`（选中
epoch 1），fold2
`c5d299009614483917214b337d457dd2d3d36bdd6b9ed14f70a9465ffe1bc0d0`（选中
epoch 9）。Round109 依赖报告、协议和源码哈希与启动记录一致；pipeline
退出码为 `0`，Screen 及 producer 进程均已结束。

OOF SHA-256 为
`a765b737a173078b3ad666fba3f30dca6c7d74a77b3fe1f2f15e9a2b6d32fea6`，gate
SHA-256 为
`962da05239fc0b4c541fe66ec5fc3e2386ffecb2d35d03b0c25ef8dec0741c7f`；gate
绑定的 OOF 哈希一致。

## 结果

| candidate | float official | JPEG-q95 official |
| --- | ---: | ---: |
| official auxiliary15 | 75.083722 | 75.013272 |
| frozen FCMAE feature guidance | 74.965745 | 74.888558 |

候选相对基线的 q95 差值为 `-0.124715`，float 差值为 `-0.117977`。fold0/2
分别下降 `-0.210676/-0.044168`，仅 3/10 ROI 为正，最差 ROI020 为
`-0.484634`。SSIM q95 从 `0.840527` 降至 `0.839258`，PSNR 也从
`26.9607` 降至 `26.9008`。总体、float、双折、ROI 数和 ROI floor 检查
全部失败，`promotion.allowed=false`。

## 决策

关闭冻结多染色 FCMAE feature-guidance 及其直接 perceptual 路线，不搜索
teacher 层、损失权重、crop、子批大小、epoch、checkpoint 或五折扩展，也不
生成提交包。Round109 的可预测性只证明表征映射存在，不能证明把该表征作为
训练约束会改善图像指标；本轮结果明确否定了当前固定用法。

# 复赛代码运行说明

队伍：通信速通生物  
结果：Round245，Score 79.1052，SSIM 0.899，PSNR 26.932；输出 CD45RO。

## 环境

Linux、Python 3.12.13、PyTorch 2.8.0+cu128；训练与推理使用 NVIDIA CUDA GPU。
原训练运行在 NVIDIA A100 40GB。所有命令在本目录执行。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## 输入数据

使用该阶段官方数据，保留图片名称和 ROI 坐标：

```text
/path/to/liver/
├── train/
│   ├── DAPI/
│   ├── CD68/
│   ├── CD45RO/
│   ├── HLA-DR/
│   └── Vimentin/
└── test/
    └── DAPI/
```

训练集2380张/17个ROI；测试输入560张/4个ROI；每张256×256。
`configs/recipe.json` 保存五折 ROI 划分和固定参数。

## 加载模型并推理

最终模型在根文件夹的 `AIC-2026-51865112-模型文件/model.pt`。
选物理GPU1时设置 `CUDA_VISIBLE_DEVICES=1`，程序内部使用 `cuda:0`。

```bash
CUDA_VISIBLE_DEVICES=1 python infer.py \
  --input-dir /path/to/liver/test/DAPI \
  --checkpoint ../AIC-2026-51865112-模型文件/model.pt \
  --output-dir /path/to/output \
  --work-dir /path/to/temp --device cuda:0
```

输出：`/path/to/output/results/test/CD45RO/*.jpg`，文件名与DAPI输入一致。
输出目录使用新路径。推理无需读取 `train/` 或训练初始化权重。

推理依次执行：完整ROI拼接 → 原视角及右下各8像素平移视角 →
每个视角8种旋转/翻转 → 逆变换与ROI坐标恢复 → 16视角等权平均 →
保存RGB、256×256、JPEG quality95。ROI平移边界使用边缘复制。
每个视角重算DAPI图块统计和5×5邻域缩略图。最终只加载上述一个checkpoint。

## 完整训练

`assets/public_encoder.pt` 是训练初始化文件，不用于测试推理。
源权重为公开 ConvNeXt V2 Tiny FCMAE ImageNet-1K，包含本次实验固定的单通道输入层初始化。

```bash
# 查看完整命令列表
python train.py --dataset /path/to/liver --work /path/to/training --plan

# 单GPU按依赖顺序执行完整训练
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset /path/to/liver --work /path/to/training --device cuda:0
```

执行顺序：

1. 每折排除对应验证ROI，完成800轮DAPI域内FCMAE预训练。
2. 每折教师训练14+14轮，使用第二阶段epoch12生成该折CD45RO目标。
3. 全训练集完成800轮DAPI域内FCMAE预训练。
4. 汇集五折目标，每张训练图仅对应一个未使用其ROI训练的教师。
5. 全量学生训练14+14轮，第二阶段使用0.2权重的CD45RO蒸馏损失。

最终训练输出：`/path/to/training/student/stage2/epochs/epoch_012.pt`。
训练日志：`/path/to/training/logs/`；每阶段参数和指标在对应目录。

```bash
# 跳过已完成且文件哈希匹配的阶段
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset /path/to/liver --work /path/to/training --device cuda:0 --resume-completed
```

域内预训练每25轮保存 `latest.pt`，`--resume-completed` 自动续接有断点的预训练。
监督训练每轮保存模型与 `last.pt`；训练入口跳过已有完成记录的监督阶段。

## 固定训练参数

| 项目 | 参数 |
|---|---|
| seed | 2026 |
| 编码器 | ConvNeXt V2 Tiny，stage blocks 3/3/9/3，dims 96/192/384/768，stride16 |
| 解码器 | PixelShuffle上采样，DAPI统计与邻域缩略图FiLM条件 |
| 域内预训练 | DAPI，800轮，batch8×累积4，AdamW lr2e-4，warmup10，mask0.5 |
| 监督训练 | 每阶段14轮，batch4×累积4，AdamW，lr3e-4/1e-4，cosine至0 |
| Marker顺序/权重 | CD68/CD45RO/HLA-DR/Vimentin：0.05/0.85/0.05/0.05 |
| 图像损失 | 0.7×(1−Uniform7 SSIM)+0.2×L1+0.1×MSE |
| 训练增强 | DAPI、标签和教师目标同步翻转及90°旋转 |
| 教师条件 | 固定屏蔽11项全ROI描述，保留局部统计与缩略图 |
| 学生条件 | 完整统计与缩略图；encoder stage decay0.8 |
| 学生第二阶段 | 0.8×标签损失+0.2×CD45RO教师损失 |
| halo | 训练64；推理128 |
| 最终权重 | 学生第二阶段epoch12，不做checkpoint平均 |

`pretrain.py`、`train_stage.py`、`build_targets.py` 分别可单独执行，使用 `--help` 查看参数。
`src/` 包含网络、图像读取、空间条件和训练损失。

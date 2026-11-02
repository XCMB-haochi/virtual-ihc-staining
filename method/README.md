# 半决赛代码运行说明

队伍：通信速通生物  
结果：Round327，加权分74.0677；Colon73.1963、Liver78.7011、Stomach72.8684。  
输出Marker：Colon/CD68、Liver/CD45RO、Stomach/CD68。

## 环境

Windows、Python3.13.5、PyTorch2.8.0+cu128、NVIDIA RTX5090。
以下命令在本目录执行，训练使用单张CUDA GPU。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## 输入数据

使用半决赛官方数据，保留图片名称和目录。三个器官目录名分别以colon、liver、stomach开头，大小写均可。

```text
dataset/
├── colon/train/{DAPI,CD68,CD45RO,HLA-DR,Vimentin}/
├── colon/test/DAPI/
├── liver/train/{DAPI,CD68,CD45RO,HLA-DR,Vimentin}/
├── liver/test/DAPI/
├── stomach/train/{DAPI,CD68,CD45RO,HLA-DR,Vimentin}/
└── stomach/test/DAPI/
```

训练图数分别为6296、2380、1680；测试图数为1346、560、280。每张256×256。
训练划分按器官内完整ROI进行：seed2026打乱ROI，前20%向上取整作为验证集。

## 加载模型并推理

最终模型：`checkpoints/model.pt`。三个任务加载同一权重。

```powershell
python infer.py --input-dir D:/data/colon/test/DAPI --marker CD68 --checkpoint checkpoints/model.pt --output-dir D:/output/results/colon/test/CD68
python infer.py --input-dir D:/data/liver/test/DAPI --marker CD45RO --checkpoint checkpoints/model.pt --output-dir D:/output/results/liver/test/CD45RO
python infer.py --input-dir D:/data/stomach/test/DAPI --marker CD68 --checkpoint checkpoints/model.pt --output-dir D:/output/results/stomach/test/CD68
```

输出目录使用新路径。每次读取完整单张DAPI，转为单通道并除以255；指定目标marker，FP32单次前向，四舍五入为8位RGB并保存JPEG quality95。文件名与输入一致。

## 完整训练

`assets/convnextv2_tiny_1k_224_fcmae.pt`为公开训练初始化权重，来源见`THIRD_PARTY.md`。

```powershell
python train.py --dataset D:/data --work D:/training --plan
python train.py --dataset D:/data --work D:/training
```

执行顺序：

1. 在固定划分的8084张训练DAPI上完成800轮域内FCMAE预训练。
2. 在上述训练ROI内划分五组，逐组排除其监督标签，各训练一个17轮教师；五个教师共用该域内编码器。
3. 每张训练图由未使用其ROI监督标签的一个教师生成目标；外层验证ROI按固定顺序分配给一个教师，汇集全量10356张训练目标。
4. 从固定域内编码器和随机任务层重新训练全量学生：第一阶段14轮，第二阶段12轮，输出`D:/training/student/final.pt`。

最终轮次26及一致性权重0.4由Round326固定ROI验证确定。
可通过`--stage pretrain/teachers/targets/student`单独执行对应步骤。重新执行相同命令会续接已保存的训练状态；工作目录和数据路径保持不变。

## 固定参数

| 项目 | 参数 |
|---|---|
| seed | 2026 |
| 编码器/解码器 | ConvNeXt V2 Tiny / PixelShuffle解码器，marker FiLM条件 |
| 域内预训练 | 800轮，batch1×累积32，mask0.5，AdamW lr2e-4，warmup10，weight decay0.05 |
| 监督训练 | batch1×累积16，AdamW lr3e-4/1e-4，两个阶段cosine周期各14轮，weight decay1e-4，clip1 |
| 编码器学习率 | 从深到浅按0.8递减 |
| 器官损失权重 | Colon0.1、Liver0.2、Stomach0.7，按各器官样本数换算逐图权重 |
| 图像损失 | 0.7×(1−Uniform7 SSIM)+0.2×L1+0.1×MSE |
| 训练增强 | 输入、标签和教师目标同步90°旋转、翻转 |
| 学生第二阶段 | 0.8×标签损失+0.2×教师损失+0.4×几何一致性损失 |
| 训练精度 | BF16前向，FP32损失 |

`src/`包含数据读取、网络、损失及训练过程；`configs/recipe.json`保存固定配置。

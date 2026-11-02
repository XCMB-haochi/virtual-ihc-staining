# 优化记录 Round 34：base48 fixed continuation

日期：2026-07-20

## 立项依据

Round33 的 fresh base48 在固定 14+14 epoch 后，相对成熟 Round31 base32 的
完整 q95 OOF 仍低 `-0.512534/-0.518146`，50/50 也没有通过 gate，因此
Round33 已关闭且不生成提交包。

但“容量无效”与“训练时钟不匹配”不能混为同一结论。Round33 stage2 相对
stage1 的每折裸 global-MSE selection score 分别提高约
`+0.378/+0.350/+0.549/+0.464/+0.335`；stage2 最佳 epoch 为
`11/14/12/13/11`，全部位于后半程。成熟 base32 在形成 Round31 endpoint
前还经历了多次独立 14-epoch cosine continuation。base48 当前少至少一个
同类周期。

本轮只回答：给 base48 增加一个固定、同配方 14-epoch continuation 后，
能否获得跨五折增益，并让固定跨容量集成真正超过 base32。它是最后一次
base48 训练时钟校验，不是事后修改 Round33 gate。

## 固定协议

机器可读协议为 `configs/round34_cv5_thumbnail_base48_continue.json`，沿用
相同五折。每折 base48 从 Round33 同折 stage2 `best_global_mse.pt` 严格
初始化，固定 seed `2034`、14 epoch、lr `1e-4` cosine、batch/workers
`8/4`、context/target halo `64/64`、spatial-thumbnail conditioning、
Uniform-7 hybrid loss 和 q95 global-MSE 选模。

Round31 base32 checkpoint 不训练。正式 primary 为每折 base32 与 continued
base48 各 50%，五折等权后共 10 个成员、每个权重 `0.1`。同时记录 continued
base48 direct。禁止搜索 epoch、额外 continuation、base64、融合权重、loss、
seed、校准、soup、conditioning 或逐 ROI 路由。

## 预注册判定

primary promotion gate：

1. continued base48 相对 Round33 base48 的完整/test-like 四项 q95 均至少
   提高 `+0.15`，且四项 float 均为正；
2. 50/50 相对 Round31 base32 的完整/test-like 四项 q95 均至少提高
   `+0.12`；
3. 50/50 的四项 float 增益均为正；
4. 至少 4/5 folds 的 50/50 q95 双指标为正，且五折平均为正；
5. 任一 test-like ROI 的 50/50 q95 指标不得低于 `-0.12`，双指标为负的
   ROI 不得超过一个。

若 primary 失败，continued base48 direct 只有在相对 base32 的完整/test-like
四项 q95 均至少 `+0.15`、至少 4/5 folds 双正、任一 test-like ROI 不低于
`-0.15` 时，才允许生成唯一 diagnostic 包。两个 gate 都失败时彻底关闭
base48 路线，不追加第二个 continuation。

## 当前状态

协议与单卡路径测试已完成。正式训练、OOF 与提交包均不存在。服务器两张
A100 分别仍有约 30/27 GiB 空闲，而本模型单进程预检峰值为 6.8 GiB；按用户
授权，本轮使用 `tools/run_round34_dual_gpu_pipeline.sh` 并行训练，不结束、
暂停或修改其他用户进程。物理 GPU0 固定运行 fold0/2/4，GPU1 固定运行
fold1/3；两个子任务使用互不相同的 run 目录与状态文件。两边均成功后，单次
全协议扫描只验证 5/5 run 完整并写 canonical status，随后在 GPU1 执行 OOF。

## 双卡实现与预检

- `run_cv_training.py --folds` 对空集合、重复和未知 fold fail closed，并按
  protocol 顺序执行；子集状态文件包含 fold tag，不会并发覆盖；
- 本机定向测试 `25 passed`；排除两项 Windows MKL 原生崩溃用例后的完整
  回归为 `156 passed, 1 skipped, 2 deselected`；
- A100/Linux 完整回归为 `159 passed, 53 subtests passed`，单卡与双卡
  pipeline 的 `bash -n` 均通过；
- 服务器 dry-run 的 GPU0/GPU1 子集分别为 3/2 条，合并后恰好覆盖
  fold0--fold4 各一次；
- 使用 Round33 strong checkpoint 的 64/32 张 train/val continuation 预检
  在两张物理卡上均成功：GPU0 `3.344 s`，GPU1 `3.171 s`；两者峰值均为
  `6.772 GiB`，参数量 `18,825,185`，并记录正确的
  `CUDA_VISIBLE_DEVICES=0/1` 与 `initialization.mode=strict`。

预检产物已核验并清理。退出时的 `pymp-* EBUSY` 仍是 NFS worker 临时目录
清理噪声，不影响 checkpoint、metrics 或退出码。按 Round33 对应五折的实际
stage2 时间，GPU0 三折累计约 74 分钟，GPU1 两折约 50 分钟，OOF 约 22
分钟；无争抢时预计约 1 小时 40 分，当前两卡均有其他低占用任务时保守预计
2--3 小时。

## 正式训练与 OOF

runtime commit `6c15372` 上的 5/5 条 continuation 全部完成。两张卡累计
70 个训练 epoch 的计时和为 `6778.507 s`；按 GPU0 三折与 GPU1 两折的
实际分配，训练关键路径约 70 分钟。canonical 状态写入后 OOF 耗时
`1279.054 s`，双卡启动至全部结束约 1 小时 37 分，明显短于保守预估。

机器可读报告为
`reports/round34_cv5_thumbnail_base48_continue_oof.json`，大小
`124,755` bytes，本机与服务器 SHA-256 均为
`87BC495F38F4B9FE051A8DE152619358E52B08AC4FC33138AE0CC6A12744A836`。

continued base48 每折最佳 epoch 为 `7/13/14/14/3`。fold2/3 仍在末轮达到
最佳，但本轮已预注册为最后一次 base48 continuation，不据此追加周期。

q95 绝对分如下，每项依次为 mean-image PSNR 与 global-MSE PSNR 口径分数：

| 范围 | base32 | continued base48 | 50/50 |
|---|---:|---:|---:|
| 完整 OOF | 74.745714 / 73.958810 | 74.466847 / 73.655465 | 74.850750 / 74.050232 |
| test-like | 72.551721 / 71.937528 | 72.381746 / 71.783040 | 72.717498 / 72.107225 |

continued base48 相对 Round33 base48 的增益：

| 范围 | float mean/global | q95 mean/global |
|---|---:|---:|
| 完整 OOF | +0.231206 / +0.212282 | **+0.233668 / +0.214801** |
| test-like | +0.278780 / +0.270023 | **+0.279089 / +0.270360** |

说明额外训练时钟假设成立，但 continued direct 仍比成熟 base32 低：完整 q95
`-0.278867/-0.303345`，test-like q95 `-0.169975/-0.154488`。

50/50 相对 base32 的增益：

| 范围 | float mean/global | q95 mean/global |
|---|---:|---:|
| 完整 OOF | +0.102317 / +0.088686 | **+0.105036 / +0.091423** |
| test-like | +0.161397 / +0.165318 | **+0.165777 / +0.169697** |

五折 50/50 q95 均双指标为正：

| fold | mean-image | global-MSE |
|---|---:|---:|
| fold0 | +0.115528 | +0.095781 |
| fold1 | +0.075128 | +0.083472 |
| fold2 | +0.157444 | +0.154610 |
| fold3 | +0.091526 | +0.082250 |
| fold4 | +0.075545 | +0.034166 |

但 test-like 中 ROI006、ROI007、ROI013 三个 ROI 双指标为负；ROI006 最差，
为 `-0.130726/-0.138480`，超过 `-0.12` 边界。正向收益主要来自 ROI004、
ROI015、ROI020、ROI022 和 ROI024。

## Gate 判定

primary gate：

1. **通过**：continued base48 相对 Round33 base48 的完整/test-like 四项
   q95 均超过 `+0.15`，四项 float 也为正；
2. **失败**：50/50 完整 q95 仅 `+0.105036/+0.091423`，未达到双
   `+0.12`；
3. **通过**：50/50 四项 float 增益全部为正；
4. **通过**：5/5 folds 的 50/50 q95 双指标为正；
5. **失败**：三个 test-like ROI 双负，且 ROI006 低于 `-0.12`。

continued direct diagnostic gate 失败，因为相对 base32 的完整/test-like 四项
q95 仍全部为负。故 Round34 不生成提交包，不追加第二个 base48 continuation，
base48 训练路线到此关闭。

## 固定收缩后续

50/50 的五折全正和 test-like 正增益表明 base48 有互补性，只是弱端点权重
过高。由 base32、base48 和 50/50 三个 global-MSE PSNR 精确反解二次误差
曲线，base48 的 MSE 最优权重为：

| 范围 | float | q95 |
|---|---:|---:|
| 完整 OOF | 0.275518 | 0.275549 |
| test-like | 0.383994 | 0.384090 |

编码前后解析结果几乎相同，说明该信号不是 JPEG 噪声。Round35 因此只验证
固定、取整后的 `30% base48 + 70% base32`；不训练、不做权重网格，也不读取
排行榜反馈调权。Round34 的 promotion 失败结论保持不变。

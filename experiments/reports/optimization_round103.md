# 优化记录 Round 103：UNeXt2 auxiliary15 固定低学习率续训

## 动机与假设

Round102 的 stride-phase 推理自集成没有增益，当前剩余的低风险主线转回训练
时钟。Round55 的 `officialauxiliary15` 在每折 stage2 末尾已经把余弦学习率降到
零，但 Round5、Round31 的同器官证据显示，从成熟 checkpoint 重新开启一个固定
低学习率余弦周期仍可能带来可迁移收益。Round32 又显示连续重启的收益会衰减，
因此本轮只测试一个周期，不搜索 epoch、学习率、权重、折特定路由或提交反馈。

## 预注册协议

- 基线：Round55 fold0/fold2 的 `officialauxiliary15` stage2 最佳 checkpoint，
  SHA-256 分别为 `7e6c49129b1013596f486449bdf77c403471240a438476d1e85fdcfb24f99fd6`
  和 `f1c6d2beaebf19da8821b43c4fad3603596f191aaff76c08e12627a41d9f3752`。
- 候选：从每个相同 checkpoint 开始，以同一 UNeXt2-Tiny、四标记
  `0.85/0.05/0.05/0.05` hybrid loss、context64 训练、batch16、workers4、seed2044
  和 `1e-4 -> 0` 的新 14 epoch 余弦周期训练；优化器和 scheduler 重新初始化。
- 推理：两者均为 context128、D4、固定 `[0,-1]` source offset、JPEG q95，
  只评估 CD68；两折验证覆盖十个固定 ROI、2706 张图。
- 基线分数绑定 Round102 标准 context128 OOF（报告 SHA-256
  `f69478acdea97bcfec2d192bd49ce82974358c19499f561641bc10434feb95b8`，q95
  `75.0132724440166`），防止 checkpoint 或评估路径漂移。

## Gate

候选必须相对基线提高至少 `+0.03` q95，float 为正，两个 fold 均为正，至少
7/10 ROI 为正，且最差 ROI 不低于 `-0.12`。失败即关闭“无新机制重复续训”路线，
不搜索第二个周期；通过才按完全相同协议扩展剩余三折。

## 完成审计与结果

远端 pipeline 退出码为 `0`，Screen 和所有 producer 进程均已结束。fold0 和
fold2 的状态文件均为 `complete=true`，各有连续 14 个 epoch 指标行、四个
checkpoint 产物和可加载的 `best_global_mse.pt`。训练峰值显存均约 `25.64 GiB`；
由于 GPU0 的可用显存低于安全阈值，两折按调度器在 GPU1 串行完成，未干扰其他
任务。OOF 评估覆盖十个固定 ROI、2706 张图，耗时 `1528.216 s`。

候选 checkpoint 的 SHA-256 为：fold0
`b1288aac9b2c6736a9fc618bd1cc74659c5938f13efbc30eb5b96851c176bd28`，fold2
`fd8442d7945df93dec383fd4c874a233ba7516e6661b0ec8d3151d5982ed1c09`。两者的
初始化元数据均为对 Round55 checkpoint 的严格同架构加载。

| candidate | float official | JPEG-q95 official |
| --- | ---: | ---: |
| baseline context128 D4 | 75.083722 | 75.013272 |
| one-cycle continuation | 75.041346 | 74.968910 |

候选相对基线的 q95 差值为 `-0.044362`，float 差值为 `-0.042375`。fold0/2
分别为 `-0.122672/+0.029014`；仅 4/10 ROI 为正，最差 ROI 为 ROI020 的
`-0.345122`，其次 ROI022 为 `-0.199151`。所有预注册 gate 均失败，保存的
gate SHA-256 为 `f383120ed436b143447a7e4762aab172195561c9b6dceae63321fecc2c5a1413`；
OOF SHA-256 为 `921e8cccd266523e0d532a4407e336be2a4c72487989defc538fc62a2e94e115`。

Round103 关闭“从成熟 UNeXt2 监督 checkpoint 无新机制重复开启余弦周期”路线，
不测试第二周期、学习率网格或 endpoint 权重。该结果也说明 Round5/Round31 的
续训收益不能直接外推到当前 auxiliary15/context128 UNeXt2。

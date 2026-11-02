# Round302:800-epoch DAPI pretraining and fixed-marker supervision

Pretraining remains the original launched trajectory: public ConvNeXtV2 Tiny
initialization, seed2026, current grouped Train DAPI only,8084 images per complete
epoch,800 epochs,6,467,200 exposures and202400 optimizer updates. Batch1,
accumulation32 (last20 normalized by20), mask0.5, AdamW2e-4, warmup10, cosine1e-6,
weight decay0.05, BF16/FP32 masked MSE, D4 train augmentation. Same held-out ROIs
as Round300/301. No Val/Test images or marker labels. Checkpoints every10 epochs;
encoder snapshots every200; only the completed800 endpoint feeds supervision.
The process, model definitions, pretraining source and checkpoints are preserved.

October26 organizer clarification: each organ's supervised marker set must match
validation/submission. A single shared model/checkpoint may use one different
marker per organ. R05/R06 allow target marker IDs; no organ ID enters the model.
Future Train/Val-based updates/submissions are allowed after a first Test score.

Existing Round301 four-marker validation at128 gives mixed organ targets75.4045,
uniform CD68=74.4428, all-four mean71.1198. The mixed-minus-CD68 difference0.9618
comes entirely from Liver/CD45RO77.1070 versus Liver/CD6872.2982, weighted0.2.
These are historical diagnostic scores, not validation of the new label-limited
training policies. Round300/301 checkpoints are not fallback single-marker models.

User authorized this bounded comparison October26. After the800-epoch audit,
run the two entries of configs/round302_supervised.json sequentially:

1. mixed: Colon/CD68, Liver/CD45RO, Stomach/CD68.
2. cd68: all three organs/CD68.

Each independently initializes the unchanged SingleMarkerStainer from seed2026
and the same public weights plus the same completed domain encoder. Keep192
cycles,256 updates/cycle, accumulation8, batch1, AdamW3e-4/cosine1e-6, weight
decay1e-4, clip1, organ probabilities0.1/0.2/0.7, paired D4 and JPEG95. Loss is
0.7 Gaussian11 SSIM +0.2 L1 +0.1 MSE. Read only DAPI and that organ's declared
label. Sample one fixed target per organ. Validate every16 cycles on identical
grouped Val; select one checkpoint by the fixed target weighted score. No Test
inference, TTA, output correction, teacher or inherited competition checkpoint.

First run a separate32-task CUDA smoke for each policy after pretraining ends;
then reseed and start both real trajectories afresh. Verify the supervised label
counts, identical splits/settings and completed optimizer updates. Comparison is
written to runs/round302_supervised/comparison.json, with per-organ/marker/ROI
scores and one selected candidate. This is one grouped split with only3 Stomach
Val ROIs; report that uncertainty. No further automatic training chain.

Measured pretraining ~267s/epoch (~59h total). Original pretrain projection is
unchanged. Two192-cycle supervised jobs budget8-10h total, no repeat pretraining.
Keep time for later all-Train supervision, inference and packaging before
November1 20:00 Beijing. No performance gain is assumed.

October28 recovery: pretraining completed800 epochs,6467200 exposures and202400
optimizer updates; final encoder passed finite-weight and Train/Val split checks.
Encoder SHA256:17ef9ef244a3a4af6a004b3ddcbc27c0c870a14d0f41f5a6d6f98554b64a1cb9.
Original launcher exited0xC000013A before supervision; termination cause unknown.
Task AIC-Round302-Supervised runs tools/run_round302_supervised.ps1 directly,
preserving the registered two-candidate protocol and completed encoder. No
pretraining restart or training-setting change. User reports pipeline completion.

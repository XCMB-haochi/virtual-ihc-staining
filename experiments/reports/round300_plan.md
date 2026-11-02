# Round300: single-image multi-organ baseline

Stage: October semifinal; submission opens October26 and closes November1 20:00
Beijing time. Source rules: docs/注意事项/半决赛注意事项.md.

Question: establish grouped validation for one shared Colon/Liver/Stomach model
under the new single-image, single-marker, no-TTA rules. No Test predictions or
formal submissions are made in this round. Do not select settings from Test scores.

Use only this stage's supplied Train. Per organ, sort ROI identifiers, shuffle
with Python Random(2026), and reserve ceil(20% ROIs) for validation. Never feed ROI
identifiers, organ labels, filenames or adjacent images to the model. All paired
images must match filenames and be256x256. No external input padding or resizing.

Architecture: shared ConvNeXtV2 Tiny dense encoder, shared PixelShuffle decoder,
marker-index FiLM on its four encoder feature scales, and one single-channel
output head. No independent organ networks or multiple candidate output images.
Initialize only from the documented official Tiny FCMAE weights, hash checked.
No inherited competition weights, domain pretraining, distillation, EMA or
offline checkpoint averaging in this first baseline.

Fixed schedule:48 epochs x256 updates x8 gradient accumulation, image batch1.
AdamW lr3e-4 cosine to1e-6, wd1e-4, clip1; BF16 training, FP32 loss/evaluation.
Seed2026. Sample organs0.1/0.2/0.7, then a Train image uniformly and marker
uniformly. Apply only paired full-image D4 during training. Loss0.7 SSIM +0.2 L1
+0.1 MSE, SSIM Gaussian11 sigma1.5 population valid. These unprovided metric
details remain assumptions. Evaluate every4 epochs, batch1, without transforms.
Fixed RGB JPEG95 roundtrip before validation scoring, no prediction corrections.
Mean per-image score=70 SSIM +0.6 min(PSNR,50). Record every marker and ROI.
Select a single checkpoint and per-organ marker by weighted validation score;
small ROI validation and marker selection uncertainty must be reported.

Budget: smoke first to measure throughput and extrapolate full validation.
Actual budget must be assessed before launch. Preserve a runnable baseline and
allow time for final training/inference before November1; do not launch blind
five-fold or800-epoch pretraining chains. First formal Test score freezes the
algorithm, checkpoint, marker choices and saving scheme under R13/R22.

Compute: Windows RTX5090 with project-local .venv, PyTorch2.8 CUDA12.8.
Run a hidden current-user one-off Scheduled Task with logs and per-epoch resumable state. SSH cleans up direct Start-Process and WMI children on this host.
User reports completion; no continuous monitoring. A100 remains unreachable;
Round137 is not inspected or modified by this launch.

Readiness: source tests passed for single-image/output contract, disjoint grouped
splits and Gaussian SSIM agreement with skimage. Remote data extracted53994
entries. Train/Val counts: Colon5004/1292, Liver1820/560, Stomach1260/420.
The machine's pre-existing torch cannot initialize CUDA. Install the official
CUDA12.8 wheel into .venv; the original pip CDN stalled, so the detached bootstrap
uses the official download.pytorch.org endpoint with resumable download. It then
verifies the public initialization hash, runs a real-data smoke and launches
training only if the extrapolated budget is<=24h and free memory suffices.
Until reports/round300_bootstrap_status.json says training-round300, this is
environment preparation, not an active training claim. Failures are recorded
and stop the chain. No unattended hyperparameter search or Test inference.

Launch verified: Scheduled Task AIC-Round300 is Running after SSH disconnect,
phase download-cuda-wheel. It uses the existing public console session, no stored
password and no recurring trigger. Keep that Windows session logged in and the
computer awake. Full training has not started at this verification.

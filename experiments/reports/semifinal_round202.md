# Round202: Metric and Marker Audit

Registered 2026-09-22 after Round201 scored 70.6241 (SSIM 0.771,
PSNR 27.769). The organizer now specifies Gaussian 11x11 SSIM and permits
choosing another output marker. Sigma, covariance normalization, border
handling and remaining score details have not been confirmed.

Evaluate the same five Round200 `best_global_mse.pt` checkpoints on their
held-out ROIs, once per image with all four output channels. Fix D4, halo128,
offset [0,0], q95, and the original fold partition. Record per-image averaged
PSNR, pooled-MSE PSNR, and both uniform7 and Gaussian11 SSIM by marker,
fold, ROI and overall. Gaussian11 provisionally follows skimage's Wang
settings: sigma1.5, population covariance, valid window. Retain the old
70*SSIM+0.6*clipped-PSNR score provisionally and label assumptions explicitly.
Do not select another checkpoint or submit a package in this diagnostic.

First use the existing rounded, lossless CD68 OOF teacher cache for a CPU
metric-only comparison. Then use original model outputs for all four
markers on GPU1. The latter is one evaluation pass, not the previous
duplicate `baselinecopy` forward pass. Neither uses test labels or a
full-data student as an OOF model. Marker rankings remain conditional on
CD68-selected checkpoints and CD68-heavy training weights.

The existing selected-epoch, no-TTA uniform7 validation logs already show
CD45RO above CD68 in four of five folds (three gains exceed 6.6 points).
This is sufficient to reopen marker choice, but not to predict a hidden
score or establish the best separately trained marker model. Preserve ROI
grouping; a wide fold spread alone does not justify patch-random validation.

Audit findings: Round201 omitted domain FCMAE and relational guidance,
changed stage1 distillation and the cosine clock. The public encoder
packager also left its random input stem unseeded; future packaging now
sets seed2026, while the existing initialization artifact stays fixed.
Training validation currently hardcodes uniform7 independently of loss
mode. Fix metric selection explicitly before any Gaussian-scored training;
do not reinterpret historical scores as Gaussian scores.

Decision after results: prioritize a marker change if paired new-metric OOF
supports it; establish a correctly scored primary-marker baseline before
testing same-stage domain pretraining or relational guidance. No automatic
Colon gate or epoch schedule is inherited. Do not attribute the entire
platform gap to any one finding before matched evidence exists.

Artifacts: `round202_cached_cd68_metrics.json`,
`round202_all_marker_metrics.json`; detached runner
`tools/run_round202_marker_audit.sh`. The long GPU evaluation queues on
physical GPU1 only and is audited after the user's completion notice.

## CPU result

All 2,380 cached CD68 OOF predictions were re-scored at q95. Uniform7:
SSIM 0.77865853, score 71.85358263; Gaussian11: SSIM 0.79452136,
score 72.96398051. The old score reproduces the existing OOF report to
less than 1e-6. With unchanged provisional score weights, the metric change
increases this fixed prediction set by 1.1104 points; it is not evidence
that the formula switch directly caused the platform drop. The comparison
does not measure the full-data student's performance on unseen ROIs.

Four metric/coverage tests passed locally and remotely. A real CPU D4
forward through fold0 passed using complete ROI conditioning and the
expected four-channel model, with matching 256x256 prediction/target crops.
Linux shell syntax was verified after normalizing runner line endings.
Source was committed and pushed. The user authorized concurrent GPU1
execution on 2026-09-22, allowing for the other task's 19.x GiB peak.
A real batch2 D4 CUDA forward measured 871 MiB peak allocated and 1,048 MiB
peak reserved by PyTorch (excluding CUDA context). The detached
`round202_marker_audit` runner now checks memory only, without an idle
compute requirement. GPU0 is not used.

## Completed all-marker audit

Exit 0; elapsed 1,084 seconds; 2,380 images per marker and complete five-fold
and 17-ROI coverage. Gaussian11 mean scores: CD68 72.96398,
CD45RO 77.26542, HLA-DR 72.34671, Vimentin 73.27303. CD45RO gains 4.30144
over CD68, with all five fold means positive and 13/17 ROI means positive.
Its fold scores are 75.7015,79.6601,79.2812,75.7231,76.6728. Its largest
ROI loss relative to CD68 is ROI072 (-9.879); do not claim universal
superiority. No final-model platform result is inferred from OOF.

The worker log contains NFS temporary-directory cleanup errors, but the
complete metric report and coverage assertions succeeded. Subsequent
training uses workers0 to avoid this cleanup path. Round203 creates a
same-final-model CD45RO confirmation package; Round204 trains a CD45RO
primary baseline with explicit Gaussian validation and common fold clocks.

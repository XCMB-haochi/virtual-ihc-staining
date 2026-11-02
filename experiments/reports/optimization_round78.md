# Round 78: Target-Intensity-Balanced L1

## Motivation

Round77 closes Cytoland composition. The next portable bottleneck is the
supervised objective rather than another encoder or ensemble weight. A fixed
analysis of the leakage-safe Round37 cache covers 6,296 OOF images and shows
that target values at least 64 occupy only `16.36%` of pixels, while error rises
sharply with target intensity:

| target uint8 | pixel share | MAE | RMSE | prediction minus target |
| --- | ---: | ---: | ---: | ---: |
| 0--15 | 22.28% | 2.72 | 5.26 | +1.50 |
| 16--31 | 19.44% | 6.62 | 9.06 | +3.33 |
| 32--63 | 41.92% | 7.87 | 10.30 | +0.30 |
| 64--95 | 13.30% | 13.59 | 17.35 | -9.92 |
| 96--127 | 2.06% | 31.57 | 36.88 | -28.03 |
| 128--191 | 0.86% | 52.08 | 60.86 | -48.73 |
| 192--255 | 0.15% | 93.21 | 104.69 | -92.59 |

The per-image fraction of target pixels at least 64 correlates `0.602` with
MAE and `0.466` with MSE. This indicates systematic underprediction of sparse,
strong IHC signal rather than an unsupported aesthetic loss change.

## Fixed Objective

Round78 changes only the existing 20% L1 component. For each image and marker,
the pixel weight is `1 + 2 * target`, then divided by its spatial mean. The
normalization preserves the L1 component's scale while setting a fixed maximum
foreground/background ratio of three. The target is detached from weighting.
The 70% Uniform-7 SSIM and 10% MSE terms remain exactly unchanged.

The strength `2.0` is fixed before training; there is no threshold, exponent,
loss-weight, or strength grid. The mechanism uses only training targets and is
applicable to later organs and markers without leaderboard or test information.

## Matched Screen

`configs/round78_cv2_unext2_target_l1.json` reuses the exact Round52 fold0 and
fold2 official-FCMAE800 encoders by hard link. The candidate starts fresh from
those encoders and preserves seed 2044, data order, `14+14` epochs, effective
batch 16, halo64 supervision, spatial-thumbnail FiLM, optimizer, checkpoint
selection, D4, and JPEG q95.

Direct gain over official-FCMAE800 requires `+0.05` q95, positive float, both
folds, 7/10 ROIs, and a `-0.12` ROI floor. Replacing the official-Tiny share in
the current Base endpoint requires the mechanism plus `+0.04`, both folds,
8/10 ROIs, and a `-0.08` floor. A fixed half-split diversity route may pass
independently under the same deployment checks.

No strength, epoch, checkpoint, loss weight, ensemble weight, fold, or ROI
route will be selected after the result. A passing route must expand unchanged
to five folds before test inference; this screen cannot generate a submission.

## Status

Implementation and prelaunch validation are complete. Focused local validation
passes 120 protocol/training-safety tests and four target-weighted loss tests;
the broader Windows discovery reaches the known NumPy/MKL `lstsq` process abort
and is not bypassed with an unsafe OpenMP override. On clean Linux, all 305
tests and 80 subtests pass, together with Python compilation, shell syntax,
exact four-command dry run, diff checks, hard-link provenance, and the
hash-bound no-side-effect preflight.

Round78 launched from commit `b332008` in detached Screen `round78` at
2026-08-28 01:46:56 +08:00. A one-epoch, 64-train/32-validation smoke on GPU1
verified target L1 strength `2.0`, the exact fold0 FCMAE encoder hash, a complete
metric row, and a loadable selected checkpoint, then removed only its verified
smoke directory. The first post-smoke sample still caught residual utilization,
so the scheduler waited one interval. At 01:53:14 GPU1 was idle with 40,427 MB
free and launched fold0. A later sample measured 98% utilization, 27,555 MB
resident, and about 191 W. GPU0 remains excluded under unrelated load. Fold2,
OOF, and gates subsequently completed successfully. Both fold jobs exited zero;
all four supervised stages contain 14 contiguous metric rows and selected
checkpoints. The unchanged evaluator covered all 2,706 images and ten ROIs in
2,730.14 seconds. All 12 recorded checkpoint hashes independently match their
files. The OOF SHA-256 is
`38a8838c1e2033748b8ead58fcbea70f57ebeb75f568840b38ab2b7b190ca7b5`;
the gate SHA-256 is
`5af2de287659ec254b450f2bf988bf989b3f3f5e5ca51d15a12cca692b5fe77e`.
Independent local gate evaluation reproduces every metric and decision.

## Result

The target-balanced model scores `74.806976` q95 versus `74.847851` for the
matched official-FCMAE800 baseline, a `-0.040874` change. Fold0 loses
`-0.095095`, fold2 gains only `+0.009931`, and just 3/10 ROIs improve; ROI001
loses `-0.279016`. Both SSIM and PSNR are slightly lower.

Full replacement loses `-0.003972` versus the current Base endpoint. The fixed
half-split is essentially tied at `+0.000431` q95, but fold0 is negative and
only 4/10 ROIs improve. Every mechanism and deployment gate fails and
`promotion.allowed=false`.

Round78 closes target-intensity weighting at the fixed strength `2.0`. The
diagnostic error concentration was real, but reweighting the 20% L1 component
does not improve the official metric. No strength, threshold, L1 share,
five-fold expansion, test inference, or submission follows.

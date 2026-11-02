# Round 99: DAPI-Guided High-Frequency Residual

## Motivation

Rounds 94--98 closed texture conditioning, independent multi-marker FCMAE,
larger inference context, and cross-marker logit feedback. The retained error
diagnostic still shows that predictions are substantially less locally detailed
than CD68 targets, while indiscriminate sharpening and global affine correction
have already been rejected. Round99 tests a narrower learned alternative: use
only the high-pass component of the DAPI input to predict a small CD68 logit
residual inside the model.

## Fixed mechanism

`unext2-fcmae-tiny-dapi-detail-residual` keeps the ordinary UNeXt2-Tiny
encoder, decoder, thumbnail FiLM, four-marker output, and auxiliary loss. A
three-convolution head receives `DAPI - mean5x5(DAPI)` with reflection padding
and adds `0.25 * tanh(residual)` to the CD68 logit only. The final projection is
zero-initialized, so the first forward pass is exactly equal to the ordinary
four-output model. The branch is trained end-to-end; no test target, test
statistics, or manual output editing is used.

The fixed comparison is against Round55 `officialauxiliary15`, using the same
fold-local Round52 FCMAE encoders, marker order, `0.85/0.05/0.05/0.05` loss
weights, seed, context, D4, output offset, JPEG quality, and `14+14` schedule.
Only fold0 and fold2 are screened first. A passing candidate would be expanded
unchanged to five folds before any test inference.

## Gate

The direct candidate must gain at least `+0.05` JPEG-q95 official score, have a
positive float gain, improve both folds and at least 7/10 ROIs, and keep the
worst ROI above `-0.12`. No residual-scale, head-width, epoch, checkpoint,
weight, ROI, or leaderboard search is authorized after this screen.

## Completion audit and result

The detached pipeline exited with status `0`; its Screen session and producer
processes are gone. Both folds contain complete stage1/stage2 runs with 14
metric rows per stage, valid selected checkpoints, and the expected fold-local
FCMAE encoder hashes. The OOF report covers all 2,706 images across the ten
registered ROIs, and the machine gate completed without any test inference or
leaderboard input. The known `pymp-*` cleanup warnings occur after successful
stage completion and do not affect the exit status.

At JPEG q95 the matched auxiliary15 baseline scores `75.013272`, while the
detail-residual candidate scores `74.929780`, a delta of `-0.083492`. The float
delta is `-0.084776`. Fold deltas are `-0.052590` and `-0.112448`; only 5/10
ROIs improve, and the minimum ROI delta is `-0.389315`. Every registered gate
check fails (`promotion.allowed=false`). The candidate's mean SSIM is lower
(`0.839251` versus `0.840527`) despite a small mean-image PSNR increase, so the
high-pass signal did not translate into the official combined metric.

Round99 closes both the direct high-pass logit correction and simple bounded
variants of that post-output residual idea. It does not authorize a test
package, scale/width/epoch search, or leaderboard probe.

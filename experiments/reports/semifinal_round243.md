# Round243: one fixed whole-ROI translation TTA confirmation

User reports organizer explicitly permits same-checkpoint fixed integer ROI
translations with inverse alignment and equal-weight averaging alongside D4.
This does not retroactively authorize the historical un-inverted output shift.
Colon Round102 four1-pixel phases lost0.002870 q95. New-stage data/checkpoint
and the fixed8-pixel displacement differ; no high probability of gain assumed.

Fix two input translations: identity and content down/right8px. Apply each to
complete DAPI ROIs before tiling. Recompute all input-derived ROI statistics,
patch statistics and thumbnails on that transformed complete ROI; no original
condition is reused for the shifted branch. Save temporary transformed inputs
losslessly as PNG bytes under original tile filenames, outside official data.
Each branch uses the same checkpoint, D4 and halo128. Stitch each shifted ROI
prediction before applying the inverse translation; average branch outputs0.5
in float, equivalent to16 equally weighted composed geometric views. JPEGq95
only after averaging. No output offset/calibration, adaptive weights or TTA grid.

Finite ROI geometry uses fixed edge replication on forward and inverse moves.
The original bottom/right8-pixel strips are not invertible after forward crop;
inverse boundary predictions are replicated. Explicitly accept this limitation
for this single fixed transform, retain every output pixel and every sample,
and report full-image scores including boundaries. It may erase a small gain.
No hidden test statistic selects padding, displacement or boundary weights.

Use original five Round239 fold weights; replay original D4 baseline against
stored complete OOF to1e-4 score tolerance. Compare the candidate on2380 images
and17 ROIs with Gaussian11 under the same provisional scorer. Fix beforehand:
mean gain>=0.02 and at least3/5 positive folds permits one full-data confirmation.
No later displacement/weight/boundary search if it fails. The full-data model is
Round24079.0903, not the fold models. Automatically infer its560 test images and
package only after validation supports the candidate. Teachers never enter test
inference. No model parameters or BN statistics are updated.

GPU0 only, as requested. Round137 stays stopped; GPU1 is not used. No training
is needed. Estimated30-60min including two passes of fivefold inference and
conditional packaging. Verify coordinate inversion/cross-tile seams, then one
complete-ROI real-data smoke for baseline replay and condition construction.
The smoke checks implementation only; no candidate metric is used to select
settings. Preserve Round240 until a confirmed platform improvement.

## Preflight and launch

Local/Linux geometric test passed, including a cross-tile shift and exact
inverse on the supported interior with explicit finite-border replication.
One complete140-image ROI real-model smoke passed the stored baseline replay
check, transformed-condition construction and finite candidate predictions.
No smoke candidate gain was inspected or used to change settings. Git archive
SHA256 matched and shell syntax passed. Launch detached on physical GPU0 only;
no training, no Round137 recovery, no intermediate polling required.

## Completed decision

Pipeline exit0; complete2380-image five-fold evaluation. Original D4 baseline
replayed78.6864215759, matching the saved Round239 result. Candidate scored
78.6909167338: gain+0.00449516. Fold0..4 changes+0.001258,-0.001094,
+0.020736,+0.007836,-0.004407;3/5 folds and13/17 ROIs improve. Worst ROI076
-0.033877. SSIM rises0.00003073 and capped mean PSNR0.003907dB.

This is a small positive validation change, not evidence of a meaningful
platform gain. It fails the registered+0.02 confirmation condition. The automatic decision did not package or spend a submission; the original
gate result is retained. Subsequent explicit user confirmation is recorded below. Close this fixed transform without displacement/boundary/weight
search. Sixteen views roughly double inference work. The result does not
exhaust translation TTA methods or explain other teams' scores.

Current best stays Round24079.0903, standard D4. Round137 remains stopped;
no new task launched in this completion audit. Evidence: round243_oof.json
and round243_decision.json.

## Explicit user-requested platform confirmation

User requested generating/submitting the unchanged candidate despite the small
OOF gain, given the daily submission allowance. This supersedes the previous
no-package decision for one fixed confirmation, not a retrospective gate pass.
Keep round243_decision.json unchanged. A separate --user-confirmation entry
reuses complete OOF evidence and the same package function, avoiding repeat
validation or any shift/padding/weight changes. Only the Round240 full-data
checkpoint is used. Run on GPU0 and retain Round24079.0903 until actual score.
No automatic upload to the competition platform; deliver a checked local ZIP.

## Submission encoding correction

First confirmation inference completed but produced an abnormally small ZIP.
Audit found prediction values were in0..255 whereas save_grayscale_rgb expects
0..1. That incorrectly saturated outputs and is not a valid submission. The
original OOF used image_metrics directly with0..255 and is unaffected. All
invalid outputs, hash/log/exit files and transformed test cache were archived
under server transfers/round243_invalid_encoding; never submit that archive.

Fixed only the image writer input normalization (divide255), retaining exact
float inference and TTA. New regression confirms byte-identical JPEG encoding
between submission writer and validation encoding; local/Linux tests passed.
Reran unchanged inference on GPU0 to generate valid output. No pixel-level
postprocessing or test-score-based change was introduced: this repairs the
output contract. User has not received or submitted the invalid package.

## Corrected confirmation package ready

Confirmation exit0. Correct ZIP size3245152 bytes, SHA256
 a6397781724430a71698004e7c6de58934c9dc1156ccb8c3074cf98cd73527ce.
Copied locally and checked560 unique expected filenames, RGB256x256 JPEGs,
full decoding, CRC, matching remote hash and exact Round240 model hash.
Mean output intensity16.7254; mean absolute decoded difference versus Round240
0.20724 gray levels, excluding the prior saturation failure. These are output
integrity checks, not evaluation against test labels or parameter selection.
## Platform confirmation

User reported79.0967, SSIM0.899, PSNR26.933: +0.0064 over Round24079.0903,
with displayed PSNR+0.004 and unchanged rounded SSIM. Promote the corrected
Round243 package as the highest-scoring package; retain Round240 standard-D4
as the faster fallback. Both use exactly the same single checkpoint.

This confirms a small gain for this fixed translation TTA on this hidden set,
not a large mechanism gain or broader robustness. Sixteen views roughly double
inference compared to eight-view D4. Original automatic gate remains failed;
the user-authorized bounded confirmation was useful. No displacement, boundary
or weighting search follows from this platform result. Local package hash was
rechecked and matches the corrected artifact manifest.

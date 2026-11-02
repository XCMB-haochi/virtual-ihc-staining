# Round 89: Current Endpoint with Official Per-Image Checkpoint Selection

## Motivation

The organizer scores each output image with per-image SSIM and per-image PSNR,
then averages the image scores. The retained Round85/88 endpoint still loads
best_global_mse.pt, because that protocol was inherited from older rounds. The
repository's current default policy is to save and select best_mean_image.pt,
which uses the confirmed per-image metric during training.

Round41 tested this selector only on the older Round35 U-Net pair. It changed
two of ten members and gained only +0.006270 q95, but that result does not test
the later FCMAE-Tiny/Base endpoint or its context128 inference path. Round89
isolates the selector on the current 25-member endpoint without any new
training, model change, weight change, context change, TTA change, or
output-offset change.

## Fixed protocol

configs/round89_cv5_context128_mean_image_selection.json reuses the exact
Round85 fold assignments, run sources, 25-member Base-diversity weights,
context halo 128, D4 TTA, JPEG quality 95, and output source offset [0,-1].
Its sole change is inference.checkpoint_filename = best_mean_image.pt. The
complete five-fold OOF is compared to the immutable Round85 context128 OOF
using best_global_mse.pt.

The fixed checkpoint-selection gate requires q95 gain at least +0.005, positive
float gain, all four folds whose selected files differ to improve, at least
12/20 ROIs in those folds to improve, and no changed-ROI loss below -0.05.
Fold0 is an unchanged-file control and must remain numerically identical. No
epoch, fold, weight, ROI, calibration, or leaderboard search is allowed.

## Status

The implementation passes local JSON validation, Bash syntax, the existing
87-test protocol suite, and diff checks. The remote preflight verified the
immutable Round85 OOF SHA-256 and the presence of all 25 best_mean_image.pt
files. The complete OOF launched in detached Screen
round89_mean_image_selection at 2026-08-30 18:11:39 (Asia/Shanghai). The
scheduler selected physical GPU1 at 18:11:43 after a fresh low-utilization
sample; GPU0 was left untouched because it was carrying another computation.
The OOF and gate completed on GPU1 in `8028.990 s` (`2 h 13 min 49 s`). The
first exit marker was `1` because the launcher's final postcondition checked a
nonexistent top-level `protocol_name` field in the checkpoint-gate JSON; the
OOF evaluator and gate writer had already completed successfully. The
postcondition was independently replayed against the existing artifacts, all
25 checkpoint SHA-256 values matched, and no OOF process or Screen remained.
After the wrapper fix, its read-only `ROUND89_VERIFY_ONLY=1` path revalidated
the existing artifacts and wrote a corrected exit marker of `0` without
rerunning OOF. The original `1` marker is preserved on the server under
`/workspace/transfers/round89_pre_repair_backup/round89_pipeline.exit.original`.
A passing result would
authorize at most one fixed test inference; a failed result closes this
selector for the current endpoint.

## Result

The candidate uses `best_mean_image.pt` and scores `75.769647` q95 and
`75.847129` float, versus `75.781550` and `75.859508` for the immutable
Round85 `best_global_mse.pt` endpoint. The selector therefore changes q95 by
`-0.011904` and float by `-0.012379`. Fold deltas are `0.000000`, `+0.002043`,
`+0.009387`, `-0.036733`, and `-0.037210`; only 9/20 changed-fold ROIs are
positive and the worst changed ROI is `-0.122426`. The fixed gate fails all
checks except unchanged-fold identity. No test package is authorized.

The candidate OOF SHA-256 is
`014a11f1b2db281b74989f89c58453c39cfc012fdd4c3594449a35bccfe9c1dd`; the
checkpoint-selection gate SHA-256 is
`0481c98a9eb6b0006f8579753c0a7076328a0604b4f99b79c5ab46f95a6ba2de`. The
corrected exit marker is `0`.

This closes the hypothesis that switching the current endpoint to
`best_mean_image.pt` repairs the plateau. The selector remains the correct
training-time metric for future newly trained models, but it is not a rescue
for these already-trained checkpoints.

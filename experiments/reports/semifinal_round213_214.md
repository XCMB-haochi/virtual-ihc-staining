# Round213 error diagnosis and Round214 signal reconstruction screen

The organizer's cell-to-marker comment motivates measurable error questions,
not a claim that a particular architecture or segmentation method is correct.
Pause further relational optimization debugging; its tested model is rejected
and its possible feature degeneration remains open. Distillation remains
untested due to Round212 implementation errors, not a negative method result.

## Round213: actual OOF error distribution

Audit980 cached, rounded, lossless CD45RO predictions from the two Round207
fold-excluded domain models on seven complete validation ROIs. Re-encode q95
identically. Source hashes match the independent Round209 audit manifests;
score exactly reproduces77.08122049. No full-data model, test data or test
labels enter this diagnostic. Masks are fixed intensity/proximity bins, not
manual cell annotations or biological positivity classifications.

- Gray>=128 accounts for0.95% of pixels but71.64% of squared error, with
  mean signed error -162.83 on the uint8 scale.
- Gray>=64 accounts for3.05% of pixels and88.83% of squared error.
- Gray<32 accounts for88.59% of pixels and4.26% of squared error.
- Strong target-gradient regions (3.46% of pixels) account for58.31% of SSE.
- Highest activity quartile scores67.594 versus84.165 for the lowest;
  activity here is the fraction of target pixels>=32, not a cell count.
- An all-black baseline scores14.775, far below77.081. Abundant dark pixels
  do not imply that black output is competitive under this scoring.

Per-ROI highest-MSE examples show missed strong localized/ring-like target
structures despite recovery of some low-intensity tissue layout. These are
deliberately worst examples, not a random visual sample. They do not prove
registration failure, biological identity or recoverability from DAPI.
No registration correction is justified from this audit. Error concentration
alone does not prove that a reweighted objective improves overall SSIM/PSNR.
Artifacts: `round213_error_audit.json`, `round213_error_examples.jpg`.

## Round214: bounded stage2-only test

Historical checks found Colon Round78 already tested intensity-weighted L1
at strength2 in both stages and lost0.04087 overall. Round105 hard sampling
and Round111 low-frequency loss also failed their tested Colon recipes.
This is explicitly not a new invention or evidence that those results were
wrong. Reopening is justified only as a bounded new-stage CD45RO test with
domain pretraining and the measured severe strong-signal underprediction.

Start from each existing Round211 full-batch16 no-guidance stage1 epoch14.
Keep the same four-channel order, CD45RO0.85 loss weight, seed2026, batch16,
optimizer,14-epoch cosine stage2 and fixed epoch12 selection. Change only
the L1 pixel weights to `1+2*target`, normalized by each image/channel mean.
The70% uniform7 SSIM and10% MSE terms stay unchanged. No labels are changed,
no masks are inserted into inference, and no teacher targets or guidance are
used. This stage2-only screen differs from the old two-stage Colon attempt;
do not imply a broad parameter search or compare it as an identical rerun.

Evaluate both folds using D4/halo128/offset0/q95 against the completed
Round211 batch16 controls and Round207 baseline at the same epoch12.
Report per-fold/ROI and all markers. Both-fold and mean improvement supports
further confirmation; mixed signs require magnitude review. Improving only
bright-pixel error is insufficient if total Gaussian SSIM/PSNR worsens.
No automatic full-data run or test package. Close this fixed configuration
if it loses; no strength/threshold/epoch search follows from that loss.

The new command test proves only the stage2 L1 option and artifact path
change. Existing target-weight tests cover normalization, gradients and
unchanged SSIM/MSE. A real stage2 smoke must verify actual optimizer updates
before formal launch. Reuse avoids repeated FCMAE and stage1 work: two
stage2 runs plus D4 should cost roughly1-2h at recent throughput, subject
to load. GPU1 exclusive; GPU0 remains reserved. Round20878.9741 remains
the retained single-model platform reference.

Real128-image stage2 smoke completed with finite loss, target weighting2.0,
actual saved Adam updates and peak allocation25.67GiB. The smoke is only
an execution check, not performance evidence. Original-stage FCMAE and
relation experiments remain preserved; no new teacher training is required.

## Completed

Round214 exit0. Both14-epoch stage2 trajectories have finite losses and
actual saved Adam updates (1718/1592), correct strength2, batch16 and no
guidance/distillation. Matched D4 CD45RO gain is only **+0.002774**:
fold0 **+0.010044**, fold3 **-0.002678**. Four of seven ROIs improve by
small amounts. The +0.155751 relative to Round207 includes the earlier
batch16-control difference and must not be attributed to target weighting.
This tested setting is effectively tied and is not expanded or submitted;
it does not establish that all signal-aware learning is ineffective.
Round20878.9741 remains the platform reference. Next: repair the independent
pixel-distillation experiment rather than search this weighting strength.

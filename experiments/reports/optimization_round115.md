# Round115: Additional-Fold Confirmation Of Round113

## Decision And Budget

Round114 is negative overall and on 8/10 ROIs. Round113 remains borderline
positive: q95 `+0.038374`, float `+0.036817`, 5/10 positive ROIs. The gate review
permits a separately justified confirmation of a borderline result. This round
spends one fold to measure replication rather than automatically rejecting it
or funding all three remaining folds. Round113 took 109-111 minutes per fold;
this avoids about 3.7 hours of training compared with completing all five.

Train the unchanged Round113 output-residual recipe on fold1, the first
remaining fold in the established fold order. The model has not been evaluated
on this fold; other historical models have, so this is additional validation,
not an untouched test set. Fold1 holds out ROI002/009/018/015/005 throughout
FCMAE and supervised training. It uses its own verified Round52 FCMAE encoder,
fresh decoder, official DINO weights, auxiliary15, effective batch16 (`8x2`),
seed2044, 14+14 epochs and the original encoder learning rates. Comparison is
the unchanged fold1 Round55 auxiliary15 checkpoint under context128, D4, q95,
global-MSE selection and offset `[0,-1]`.

## Fixed Interpretation

The confirmation requires q95 `+0.05`, positive float, at least 4/5 positive
ROIs and worst ROI `>= -0.12`. It reports confirmation separately and always
sets `promotion.allowed=false`. A pass supports a subsequent review of the
three-fold evidence and the cost of the two remaining folds; it cannot trigger
test inference or revise Round113's historical gate. A negative result ends
this confirmation; no further fold, seed or weight is selected to rescue it.

Protocol: `configs/round115_fold1_dino_residual_confirmation.json`.
Model, training, inference and loss recipe fields are tested for equality with
Round113. Shared code adds an explicit single-fold confirmation mode and
fold1's existing verified FCMAE source; the model and optimizer are unchanged.

## Alternative Examined

Before selecting confirmation, `tools/diagnose_auxiliary_conflict.py` measured
main/auxiliary gradients on 64 training-only images per fold at the Round55
stage1/stage2 endpoints (16 batch16 measurements). Both gradients were computed
with a common AMP loss scale of 1024 and then unscaled, with finite-value checks.
Only 2/16 full-model and encoder batches had negative dot products; median
cosines were `+0.902748/+0.926835`, and all decoder batches were nonnegative.
The first unscaled-AMP diagnostic was invalidated because gradient underflow
changed its conclusion. This small endpoint sample cannot exclude conflicts
earlier in training, but does not prioritize another expensive routing screen.
Corrected measurements remain in the remote
`reports/round115_auxiliary_conflict_diagnostic_scaled.json`.
The 2026-09-08 audit invalidates the normal-input interpretation of this scaled
diagnostic too: conditions were reconstructed from the 64 sampled images,
rather than complete training ROIs. Its numerical result cannot rule out
conflict during normal training. The new CPU float32 diagnostic uses complete
ROI conditions: 6/16 batches have full-model conflict, median cosine `+0.501234`,
and maximum opposing primary fraction `0.409780`. This supersedes the prior
gradient interpretation but is not a validation gain. See the reassessment
summary; this correction does not affect Round115's formal OOF.
Reference considered: [PCGrad, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

## Validation And Launch

All 101 affected tests passed locally and on Linux. Both real-data smoke stages
exited zero with finite loss, four updates, nonzero trained residual weights,
correct optimizer groups and strict stage2 restoration; peak memory was
15.36 GiB. The fold1 encoder's epoch800 metadata and baseline hash matched.
Screen `round115_residual_confirm` launched training on physical GPU1; the
formal config has 5151 training and 1145 validation images. Source commit was
captured directly from Git and deployed with a hash-verified Git archive.
Receipt: `round115_launch.json`. Training/validation results are pending.
The run is complete; see Result below.

## Result

The pipeline exited `0`. Both 14-epoch stages contain contiguous metrics,
finite losses, valid selected checkpoints and the registered fold1 FCMAE/DINO
initialization. OOF covers 1,145 images from the five confirmation ROIs and the
confirmation gate replay passed its provenance checks.

The candidate q95 is `75.188267` versus `75.296236` for the unchanged fold1
Round55 auxiliary15 baseline, a delta of `-0.107969`. Float delta is
`-0.120972`; only ROI018 improves (`+0.206803`), while ROI009 falls
`-0.298580`. Confirmation is false. Round113's positive two-fold signal does
not replicate on fold1, so the DINO residual branch is closed without training
fold3/fold4 or searching scale, seed, weight, or checkpoint.

Artifacts: `round115_fold1_dino_residual_confirmation_oof.json`,
`round115_fold1_dino_residual_confirmation_gate.json`, and
`round115_pipeline.exit`.

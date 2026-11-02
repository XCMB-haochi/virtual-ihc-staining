# Round114: DINOv2 Bottleneck Feature Fusion

Round113 showed that a DINOv2 output residual is used by the trained model but
missed the fixed two-fold distribution gate. Round114 keeps the validated
UNeXt2-FCMAE-Tiny auxiliary15 training path and injects the DINOv2-S/14 last
feature map into the UNeXt2 bottleneck through a spatially aligned,
zero-initialized `1x1` projection. The zero start preserves the base model's
initial output; DINO is trained at `0.1x` learning rate and the FCMAE encoder
uses the original stage rate.

The screen is fixed at fold0/fold2, fold-local Round52 FCMAE800 initialization,
seed2044, effective batch16 (`8x2`), 14+14 epochs, context64 training,
context128+D4/q95 inference, global-MSE selection, offset `[0,-1]`, and the
Round113 promotion gate. It is a mechanism screen, not a DINO weight or fusion
location search. A failed gate closes this exact recipe without five-fold
expansion or leaderboard probing.

History search found Round112's independent DINO decoder, Round113's output
residual, and Round38's older thumbnail-aligned U-Net continuation. None is the
same pretrained DINO-to-FCMAE bottleneck fusion. This screen changes the fusion
mechanism, not the failed Round113 residual scale or gate. Features can now
influence all four outputs through the shared decoder. Supervised weights are
fresh, not continued from a mature Round55 or Round113 model.

Round113 measured 109-111 minutes per fold and 29 minutes for shared OOF;
rejecting after two folds avoids about 5.5 hours of additional training.
Public DINO weights load only for fresh stage1. Stage2 and inference strictly
restore saved model weights without loading that public weight file again.

## Validation And Launch (2026-09-06)

Local checks passed 181 relevant tests, including coordinate-ramp alignment,
zero-start equivalence, gradients into DINO after a projection update, complete
optimizer grouping and checkpoint restoration without public weights. Ten
focused tests and protocol preflight also passed on Linux. The real two-stage
smoke completed four updates per stage with finite loss, loaded all 196 FCMAE
and 176 DINO tensors, then strictly restored stage1. Peak memory was 14.78 GiB.

Detached Screen `round114_dino_fusion` ran both folds sequentially on GPU1,
then completed one shared OOF/gate pass. Launch receipt: `round114_launch.json`.

The launch argument's full commit identifier was transcribed incorrectly.
The receipt is corrected to the actual source commit after all eight recorded
deployed source files matched Git with line endings normalized. The original
argument is retained in the correction record; training code was unchanged.

## Result

Pipeline exit `0`; exact 14-epoch records in all four stages, finite losses,
strict-loadable selected checkpoints, intended initialization and optimizer
groups, source/checkpoint hashes, 2,706-image/10-ROI OOF and gate replay passed.
Selected epochs were fold0 `13/13` and fold2 `13/1`; trained fusion projections
are nonzero. Training took 100-102 minutes per fold; OOF took 28.9 minutes.

| Metric | Round55 Baseline | Round114 | Change |
| --- | ---: | ---: | ---: |
| q95 | 75.013272 | 74.980477 | -0.032795 |
| float | 75.083722 | 75.054823 | -0.028898 |

Fold0/fold2 q95 deltas are `-0.137222/+0.065053`. Only ROI000 and ROI008
improve (`+0.967260/+0.717095`); the other eight decline, with worst ROI007
`-0.567938`. All performance gates fail. This is negative evidence for this
fixed fusion recipe, not a threshold near miss or a rejection of every DINO
method. No expansion or fusion-parameter search follows.

Artifacts: `round114_cv2_dino_bottleneck_fusion_oof.json`,
`round114_cv2_dino_bottleneck_fusion_gate.json`, `round114_pipeline.exit`.

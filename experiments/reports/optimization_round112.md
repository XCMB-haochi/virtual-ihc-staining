# Round112: Independent DINOv2 Representation Screen

## Rationale

Round108, Round110 and Round111 reject three current loss modifications. The
historical Swin experiment added an unpretrained branch to U-Net; Round39 used
a randomly initialized Restormer. Neither tests a large-scale self-supervised
ViT prior. Round44/46/52 support representation pretraining as a productive
direction, but do not establish that DINOv2 will improve this task.

Use Meta's [DINOv2](https://github.com/facebookresearch/dinov2) ViT-S/14 with
four registers, revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8` and official
LVD142M weights. Source and weights use Apache-2.0. No external microscopy
dataset or competition test images enter training. `tools/prepare_dinov2.py`
prepares the pinned dependency and verifies the public weight digest.

## Fixed Comparison

- Fresh supervised DINOv2 encoder plus bilinear convolutional decoder; intermediate
  layers 3/6/9/12 form the decoder pyramid, with a full-resolution DAPI skip.
- Keep auxiliary15 marker weights, spatial-thumbnail conditioning, seed2044,
  context64 input/target, 14+14 stages, effective batch16, original hybrid loss,
  global-MSE checkpoint selection, context128/D4/q95 and output offset `[0,-1]`.
- Public encoder LR is one-tenth of decoder LR (`3e-5/1e-5` by stage); micro-batch
  8 with two-step accumulation. These are architecture-specific preset choices,
  not tuned parameters. This compares complete representation pipelines; it does
  not isolate architecture from pretraining or claim identical optimization paths.
- Baseline: unchanged same-fold Round55 auxiliary15. Evaluate DINO direct and
  one fixed 50/50 blend in a shared OOF pass. No weight sweep or submission.

Either candidate must gain `+0.05` q95, positive float, both folds positive,
at least 7/10 positive ROIs, and worst ROI >= `-0.12` before expansion is considered.
Failure rejects this recipe, not every Transformer/pretraining method. Any
subsequent revision needs new evidence rather than post-hoc threshold changes.

## Cost And Execution

Round111 measured about 96 minutes per two-stage fold plus 26 minutes OOF.
Two-fold screening avoids three additional folds if the new recipe is weak.
DINOv2 runtime and peak memory are measured by real stage1/stage2 smoke before
launch. The resource scheduler uses both GPUs when independently eligible;
other users' processes are untouched. Best and rolling last checkpoints are
saved every epoch, including optimizer/scheduler state in `last.pt`;
redundant per-epoch snapshot copies are disabled.

The launch receipt records the synchronized local source commit and smoke
results, because the existing remote Git metadata is not authoritative.

## Launch

Local and remote targeted checks pass: 5 DINOv2 integration/CLI tests, 2
resource-scheduler tests, 45 training-safety tests and 88 CV-protocol tests.
Real stage1/stage2 smoke completes four optimizer updates each, loads all 176
public encoder tensors then strict-restores the stage1 checkpoint, and verifies
finite loss/parameters and changed encoder weights. Peak memory is 4.97 GiB
for 28,580,676 parameters. Four-image context128+D4/q95 inference also passes.
The initial CLI whitelist omission was corrected before formal training.

Detached Screen `round112_dinov2` started fold0 on physical GPU1; the actual
trainer config confirms 4987 training images, 1309 validation images and
effective batch16. The scheduler checks both GPUs at subsequent boundaries;
training admission is 12,000 MiB free, above measured peak plus 4 GiB margin.
Source commit: `76684bcb40794c7fa37c4566bf9f3c48dd6167d0`. The compact receipt
is `round112_launch.json`. This records startup, not a completed experiment.

## Completion And Result

The detached Screen exited cleanly with `round112_pipeline.exit=0`; no DINOv2
training producer remained. Both folds have `complete=true`, two 14-epoch
stages, finite metrics, and valid `best_global_mse.pt`/`last.pt` artifacts.
OOF covers all 2706 images in the 10 registered validation ROIs and took
1260.20 seconds. The DINOv2 public weight SHA-256 is
`f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`.
Selected DINO checkpoints are fold0 epoch 12,
`36e275f2276bcf7f851af0f7bd482a026527c40889d8a2464c5bb5bb86a7e203`, and
fold2 epoch 11,
`df84a063f2d47df283e00e8d1a7a2e29483998e598bf5efd8b4ed2e1579ffb4f`.
Independent server audit rehashed both candidate and matched baseline files
and confirmed the recorded epochs.

| candidate | float official | JPEG-q95 official |
| --- | ---: | ---: |
| Round55 auxiliary15 baseline | 75.083722 | 75.013272 |
| DINOv2-S/14 replacement | 73.612041 | 73.537388 |
| fixed 50/50 baseline+DINOv2 | 74.949803 | 74.878722 |

The direct DINO candidate loses `-1.475885` q95 and `-1.471681` float; both
folds and all 10 ROIs are negative. The fixed half blend loses `-0.134550`
q95 and `-0.133919` float, with both folds negative and only one positive ROI.
All registered gates fail and `promotion.allowed=false`. No five-fold
expansion, test inference, or submission follows.

This closes the specific DINOv2 full-backbone replacement and fixed blend.
The result does not establish that DINOv2 features are useless: the tested
decoder had to invent a spatial pyramid from same-resolution token maps, and
the replacement discarded the validated FCMAE UNeXt2 representation. A
separate zero-start residual test is therefore registered as Round113; it is a
changed representation and will not tune this failed result.

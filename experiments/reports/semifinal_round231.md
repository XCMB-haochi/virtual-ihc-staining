# Round231: current-stage encoder stochastic-depth generalization screen

Round229 found training fit above grouped OOF and persistent bright error;
Round230 showed the same model and hybrid objective can fit selected TRAIN
structures near-perfectly. Neither proves overfitting is the sole bottleneck,
but they motivate testing regularization before increasing capacity or
replacing the objective. Retain the best Round21679.0198 submission.

Historical search: Colon Round74 tested encoder DropPath max0.2 and was
effectively tied (+0.000044), with mixed ROI changes. This is not a novel
method or a proven win. New-stage CD45RO,17 rather than25 training ROIs,
domain pretraining and the new fitting diagnostic justify one fixed transfer
test; no rate or depth sweep follows from its result.

Change only encoder DropPath: rates linearly increase to0.2 across the18
ConvNeXtV2 blocks, per sample, training only. Decoder rate remains0 and
inference is deterministic with all blocks active. Keep seed2026,4x4
effective16, immutable corresponding domain800 encoders, spatial thumbnail,
CD45RO0.85/auxiliary0.05 weights, uniform7 hybrid loss,14+14 cosine periods,
stage1 epoch14 and stage2 epoch12. Fresh supervised weights; no continuation,
pixel distillation, relation guidance, occupancy task or altered labels.

Train all five folds directly: encoders are available and recent supervision
costs about1.2-1.3h/fold. All-fold work adds about4 GPU-hours over a two-fold
screen, but provides complete17-ROI evidence in one parallel run, avoiding
repeated fold0/fold3-only selection. Compare fixed D4/halo128/offset0/q95
against Round217 across2380 images. Examine all fold/ROI changes; no automatic
full-data training, package, submission or model-family rejection.

Tests cover exact matched commands, linear DropPath schedule, identical eval
outputs and stochastic training behavior. A two-stage real-data smoke checks
actual optimizer steps and checkpoint restoration with DropPath enabled.
Allocate independent fold queues from the launch-time GPU load and available
memory; each run needs approximately7GiB. Do not alter others' processes.

Launch allocation: GPU1 has competing compute, so assign it only the shorter
fold4; GPU0 handles folds0-3 and final OOF. The longer idle-card queue is
about5h, allowing5-6h overall rather than assuming two equally fast cards.
Real two-stage smoke passed with7/8 actual Adam updates and6.99GiB peak
allocation; the second stage strictly restored its own DropPath checkpoint.
These checks validate execution, not performance. No dropout-rate grid follows.

## Completed

Exit0; ten14-epoch stage trajectories verified, finite losses, saved Adam
updates and configured encoder DropPath0.2. Complete2380-image D4 comparison
gains only+0.033027 for CD45RO. Fold changes:+0.146908,-0.134773,-0.036136,
+0.278492,-0.120125. Eight of17ROIs improve; worst ROI072-0.331762. This
is weak/mixed evidence, not a broadly positive regularizer. No full-data
confirmation or rate search follows. Round21679.0198 remains best.

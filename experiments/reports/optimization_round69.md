# Round 69: Resumed Five-Fold Base Diversity Promotion

## Decision

Round67 produced a valid fixed submission candidate, but its complete-OOF gain
is only `+0.050199`. The strongest unfinished independent mechanism remains the
Round62 ConvNeXt-V2 Base diversity split: `+0.102825` over two folds, positive
in both folds and 9/10 ROIs, with a worst ROI delta of `-0.027625`. Round63's
five-fold expansion was interrupted by a server reboot rather than rejected.
Round69 resumes that preregistered expansion without changing architecture,
weights, epochs, folds, checkpoint selection, inference, or gates.

## Recovery Boundary

Fold1 and fold3 each have an epoch-125 `latest.pt` with model, optimizer, AMP
scaler, CPU RNG, and CUDA RNG state. Their logs extend to epochs 136 and 139,
but these rows postdate the last atomic checkpoint. Resume therefore truncates
both logs to epoch 125 and deterministically replays from epoch 126. Fold4
starts fresh. The inherited artifacts are bound by these SHA-256 values:

- fold1 config:
  `cd861f525cebeac50a84070a8eba33ac2dbc3f2f67f5adaead9b790cf2bd4146`;
- fold1 epoch-125 checkpoint:
  `f7ff4aacbaeaf0c278c063423b7b50bb74b3d9c8c3909288add42b50e332c2f8`;
- fold3 config:
  `07346f1d343d1a14c1585014ed69200d137e15856925e07028d0fa62b476b6ce`;
- fold3 epoch-125 checkpoint:
  `538cb4fb3a1e8a61b03ae570bca1f4eb44104aa5844595b8fb9ca7f20758e80c`.

The first preflight writes a fixed resume manifest. Later relaunches accept
only structurally complete 25-epoch checkpoints from 125 through 800, the same
immutable configs, continuous metrics from epoch one, and no more than 24
post-checkpoint metric rows. This preserves repeated reboot recovery without
pretending uncheckpointed epochs are durable.

## Fixed Gate

The launcher binds the passing Round62 OOF SHA-256
`6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe`
and gate SHA-256
`821cd877c82e415c09e159f2d4b9c020b755c92f96e069b4a4fb23a431223a56`.
Complete promotion still requires `+0.04` q95, positive float gain, at least
4/5 positive folds and 20/25 positive ROIs, and a worst ROI delta of at least
`-0.08`. Test inference is blocked until complete OOF passes.

## Scheduling

At the measured 121--123 seconds per Base FCMAE epoch, fold1 and fold3 each
have about 22.9 hours of durable pretraining remaining; fresh fold4 needs about
27.1 hours. Supervised training and OOF add several hours. A single GPU would
take roughly 73 hours before the final evaluation. The scheduler rechecks each
unassigned physical GPU every five minutes and launches the next fold only
below 30% utilization with at least 27 GB free, so a later-idle second GPU can
reduce wall time without interfering with unrelated jobs.

## Status

Implementation is complete. All 93 focused local protocol, FCMAE resume, and
UNeXt2 tests pass, together with Python compilation, launcher shell syntax, and
diff checks. On Linux, all 284 tests pass. Exact dry runs confirm `--resume`
for fold1/fold3, a fresh fold4, and six fixed supervised stages. The corrected
artifact-bound preflight loads both epoch-125 checkpoints, validates all state
and provenance, and atomically writes the fixed resume manifest. No GPU process,
OOF, gate, or exit marker was created by preflight. Launch remains pending.

Round69 launched from commit `17481b8` in detached Screen `round69` at
2026-08-22 22:47:04 +08:00. Five launch-time samples showed physical GPU0 at
100% utilization with only 8,743 MB free and GPU1 idle with 40,424 MB free.
The scheduler therefore assigned fold1 to GPU1 and left GPU0 untouched. Resume
immediately truncated fold1 metrics from 136 rows to the durable epoch 125, as
preregistered. Initial verification measured 94% GPU1 utilization and 24,095 MB
resident, confirming that checkpoint state loaded and computation started.
Fold3 and fold4 remain queued for dynamically eligible GPUs. Completion remains
unverified until all three FCMAE runs, six supervised stages, complete OOF,
checkpoint provenance, and the fixed gate are audited.

## Completion Audit

The scheduler started fold3 on GPU0 once it became eligible, completed fold1
and fold3 successfully, then ran fresh fold4 on GPU1. The pipeline exited `0`.
All three Base FCMAE directories contain 800 contiguous metric rows, epoch-800
completion markers, and final encoders. Fold1/fold3 resumed from their bound
epoch-125 states; fold4 records the Round69 launch commit. All six supervised
stages contain 14 metric rows and bind the correct epoch-800 encoder hashes.

The complete OOF covers all 6,296 images and 25 ROIs in 8,865.39 seconds. All
25 referenced checkpoint hashes match the server files, while the historical
`t70_o50` metrics and four source-model provenance records exactly replay
Round64. The OOF SHA-256 is
`cf3b9af9ae5b2da3757107bcec5b8d1aa6b013d0e2181369df6cc9ee9e2b9c53`;
the gate SHA-256 is
`fb5994f347e440ddf69d4c16f821ef09286264bb2d92e72c4c0e68d5112f9d77`.
Independent local gate evaluation exactly reproduces the saved report.

## Result

At JPEG q95, `t70_base_split` scores `75.644105` versus `75.560424` for
`t70_o50`, a gain of `+0.083681`; float gain is `+0.082940`. Every fold is
positive (`+0.033545` to `+0.105069`), exactly 20/25 ROIs improve, and the
worst ROI is `-0.070786`, above the fixed `-0.08` floor. Every preregistered
gate check passes and `promotion.allowed=true`.

The Base model alone remains worse than official Tiny (`75.015311` versus
`75.220946`), so this does not support direct capacity superiority. The effect
is stable ensemble diversity. Base replacement scores `75.640309`, slightly
below the selected half-split. No test inference was produced by Round69; the
next bounded step may combine this independently passing endpoint with the
Round67 auxiliary/all-train-DAPI endpoint before choosing a final package.

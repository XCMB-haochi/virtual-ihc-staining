# Round 72: High-Frequency Correction Oracle

## Motivation

Round71 verifies a positive Base-diversity platform gain, but only `23.9%` of
its complete-OOF increment transfers to the hidden set. Capacity and ensemble
diversity therefore do not address the main remaining domain shift. Prior
analysis found that Round35 predictions have only about 72% of the target's
local standard deviation overall and about 62% on dense images, but the target-
aware sharpening oracle proposed during the independent review never finished.

Round72 completes that missing diagnostic before any new architecture or loss
is trained. It uses the existing leakage-safe Round37 cache of 6,296 paired OOF
targets and Round35 predictions. This cache is older than the current Base
candidate, so the result is a mechanism screen: a closed oracle rejects the
direction cheaply; an open oracle requires replication on a current-model cache
before training.

## Fixed Diagnostic

The high-frequency basis is prediction minus its 7x7 uniform local mean. Four
methods are evaluated under the exact float and JPEG-q95 official metric:

- the unchanged Round35 70/30 baseline;
- a target-free cross-fitted global unsharp coefficient, fitted on four folds
  by MSE projection and applied to the fifth;
- a target-aware per-image MSE-optimal coefficient in `[0,1]`, used only as an
  upper bound;
- a target-aware local variance oracle that sharpens only windows whose target
  variance exceeds official SSIM `C2=58.5225`, matching target/prediction local
  variance with scale capped at 2x.

The cache root and all five fold manifests are bound by SHA-256 in
`configs/round72_frequency_oracle.json`. No test DAPI, test prediction,
leaderboard feedback, model training, or submission packaging is used.

## Decision Rules

Cross-fitted fixed sharpening is actionable only at `+0.03` q95, 4/5 positive
folds, 18/25 positive ROIs, and a `-0.05` ROI floor. A training-level frequency
mechanism is actionable only if at least one target-aware oracle reaches
`+0.15` q95 with at least 4/5 positive folds. Passing the latter authorizes only
building a current-model cache and designing a matched frequency-loss screen;
it never authorizes oracle post-processing or test inference.

## Status

Implementation is complete. All 77 focused local frequency-oracle and protocol
tests pass, together with Python compilation, launcher syntax, and diff checks.
The evaluator provides a hash-bound no-side-effect preflight and a formally
non-actionable per-fold image limit for real-cache smoke testing. Linux
validation, smoke, and formal launch remain pending.

On Linux, all 291 tests pass. The no-side-effect preflight validates the bound
6,296-image, five-fold cache and all manifest hashes. Real-cache smoke runs at
2 and 20 images per fold complete all methods; both record `formal=false`, keep
all actionable flags false, and are removed after audit. The 100-image timing
smoke takes 5.56 seconds after cache validation, projecting roughly six minutes
for the full analysis. Formal output and exit artifacts remain absent.

## Result

The formal pipeline exits `0`, covers all 6,296 cached OOF images, and finishes
in 283.02 seconds. The report SHA-256 is
`29cf7b6668f578ff131b5fdee55ea390abb424d4fe4e8ca8cc1b1692a70d0aeb`.
The baseline exactly reproduces Round35 at
`74.869164` JPEG q95.

Cross-fitted global unsharp fitting selects deployment alpha `0.0`; every
fold-specific coefficient is also ineffective. Its q95 delta is `-0.002511`,
with 0/5 positive folds, 2/25 positive ROIs, and a worst ROI delta near
`-0.02`. It fails every actionable check except the ROI floor.

The target-aware per-image MSE oracle improves mean PSNR slightly but lowers
official q95 by `-0.044627` because SSIM falls. Half of images select alpha
zero; the 75th-percentile alpha is `0.5805`, but image-specific sharpening is
still negative on every fold and improves only 7/25 ROIs. Optimizing pixel MSE
inside this unsharp family is therefore not an SSIM-compatible hidden reserve.

The target-aware dense local variance oracle is the strongest variant, but its
q95 gain is only `+0.005168`. Four folds are positive, yet only 9/25 ROIs
improve and the worst ROI falls by about `-0.06`. Its mean gain rises from
negative on sparse deciles to only about `+0.06` on the densest target decile.
This is far below the preregistered `+0.15` training-oracle threshold.

Both `crossfit_actionable` and `frequency_training_actionable` are false, and
test inference remains forbidden. Round72 closes fixed sharpening, adaptive
contrast correction, and a new high-frequency loss as the next score path.
The result also completes the previously unfinished independent-review oracle;
no current-model cache is needed for a mechanism whose generous target-aware
upper bound is this small.

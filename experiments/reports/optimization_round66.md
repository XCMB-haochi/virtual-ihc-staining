# Round 66: Five-Fold All-Train-DAPI Diversity Promotion

## Decision

Round65 passes its preregistered independent diversity gate. At JPEG q95,
`t70_all_dapi_split` gains `+0.042966` over `t70_o50`, is positive in both
screen folds and 9/10 ROIs, and has a worst ROI delta of only `-0.011572`.
Direct replacement does not pass. Round66 therefore expands only the fixed
half-split diversity candidate to the three missing folds before any test
inference.

## Fixed Protocol

The shared encoder is the exact Round65 epoch-630 artifact with SHA-256
`f4a8c93c616310ce4a981f9b1fe6830ba82bf78ff91bdbe52d471cf1c0164554`.
It was pretrained once on all 6,296 official training DAPI patches from 25
ROIs, with no IHC target from any held-out fold. Existing fold0 and fold2
supervised checkpoints are reused byte-for-byte. Fold1, fold3, and fold4 use
the unchanged seed-2044, 14+14 UNeXt2 recipe, loss, context, checkpoint
selection, D4 inference, and JPEG q95 settings from Round65.

The promotion config deliberately retains protocol name
`round65_cv2_unext2_all_dapi_fcmae`. This is the established expansion pattern
used by prior rounds: it reuses the immutable shared encoder and the two
audited screen-fold run paths while the distinct Round66 config, reports, and
exit marker preserve the complete-OOF experiment boundary.

## Gates

The deployment gate compares the same fixed weights:
`0.21*base32 + 0.09*base48 + 0.35*FCMAE800 + 0.175*officialFCMAE800 +
0.175*all-DAPI-FCMAE630` against `t70_o50`. It requires at least `+0.04` q95,
positive float gain, at least 4/5 positive folds, at least 20/25 positive ROIs,
and a worst ROI delta of at least `-0.08`. Direct and replacement gates remain
diagnostics; they do not control the independent diversity route.

The launcher is bound to the Round65 gate SHA-256
`7db9be9bf5d34fb4dfb4fb0bf7a8d80a5d5fc7954de9be2caf15994e8d457daf`,
screen OOF SHA-256
`56a91690be0438569499f903ff0797ad233131c1af28b5c78c3e063477453316`,
and complete baseline OOF SHA-256
`e0a137d30383555ed930bf7ab03f27f101c76f82f3bba9a879bc6777b032a8ed`.
It schedules missing folds from live utilization and free-memory samples and
does not alter unrelated GPU jobs.

## Interpretation Limit

Complete OOF can establish whether the diversity gain is stable across the 25
official training ROIs, but it remains semi-transductive because the shared
self-supervised encoder saw the held-out ROIs' DAPI inputs. It never saw their
IHC labels or any test DAPI. A passing complete gate is necessary but does not
prove transfer to unseen test ROIs; no submission will be created solely from
the two-fold result.

## Status

Implementation is complete. The 72 focused local protocol tests, Python
compilation, launcher syntax, and six-command dry run pass. On Linux, all 283
tests pass and the dry run resolves fold1/fold3/fold4 to the fixed shared
encoder and their label-isolated ROI splits. The launcher's no-side-effect
preflight passes every bound gate, OOF, complete-baseline, shared-encoder,
completion-marker, scope, and existing fold0/fold2 checkpoint check.

Recent matched runs take about 160 seconds per supervised epoch, so one 14+14
fold is roughly 75 minutes on an uncontended A100. With two eligible GPUs, the
three missing folds should take about 2.5 hours; complete five-fold OOF and gate
evaluation add roughly 1--1.5 hours. Contention can extend this because the
scheduler waits rather than launching onto an ineligible GPU.

Round66 launched from commit `34639d3` in detached Screen `round66` at
2026-08-22 14:03:13 +08:00. Three-sample scheduler checks found both physical
GPUs at 0% utilization with 40,427 MB free, then assigned fold1 to GPU0 and
fold3 to GPU1. Initial five-second verification showed sustained 98--100%
utilization on both cards with 27,555 MB resident per process and the expected
physical CUDA mappings. Fold4 remains queued for the first eligible card; no
completion claim or OOF result has yet been made.

## Result

The pipeline exited `0`. Fold1 and fold3 completed in parallel, after which
fold4 ran on GPU1; all six new supervised stages contain 14 metric rows. The
complete OOF covers all 6,296 images and 25 ROIs. Its SHA-256 is
`4eee928e2f034fd0b76151359196eb36705b13606cd9b79f3688fd37db243e03`.
All 25 referenced checkpoint hashes match their server files, and the complete
`t70_o50` baseline, fold metrics, ROI metrics, and historical checkpoint
provenance exactly replay Round64. Independent local gate evaluation exactly
reproduces the saved gate report.

At JPEG q95, `t70_all_dapi_split` scores `75.593088` versus `75.560424` for
`t70_o50`, a gain of `+0.032664`. The gain is positive in float and all five
folds, with fold deltas from `+0.012308` to `+0.063338`. It improves 19/25
ROIs and has a worst ROI delta of `-0.032497`. The candidate therefore fails
two preregistered checks: overall gain is below `+0.04`, and positive ROIs are
one short of the required 20. `promotion.allowed=false`.

The direct all-DAPI model gains `+0.047973` over official FCMAE but improves
only 3/5 folds and 15/25 ROIs, with a worst ROI delta of `-0.313610`. Full
replacement gains only `+0.021060`. These failures reinforce that the useful
effect is limited ensemble diversity, not a stable superior representation.
No test inference or submission was created from Round66.

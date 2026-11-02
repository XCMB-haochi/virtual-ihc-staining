# Round 64: Bounded Auxiliary Weight Calibration

## Motivation

Round63 was interrupted by a server reboot early in Base FCMAE pretraining and
would still require roughly two days before complete OOF. Round64 returns to a
short-cycle question that can be answered with existing complete five-fold
checkpoints and no training.

Round55 evaluated three fixed shares of the official-initialized `0.35` slot:
zero auxiliary, one-half auxiliary, and full auxiliary. On the original
fold0/fold2 screen, a quadratic through those three fixed endpoints places the
interior maximum near `0.67`; the complete five-fold endpoint scores have the
same concave ordering. Round64 therefore registers one fixed two-thirds
candidate rather than a dense post-hoc grid.

## Fixed Protocol

`configs/round64_cv5_auxiliary_weight_calibration.json` reuses the exact six
five-fold endpoint families already audited in Round55. The retained baseline
is Round53 `t70_o50`: base32 `0.21`, base48 `0.09`, scratch FCMAE800 `0.35`,
and official FCMAE800 `0.35`.

The primary candidate replaces two-thirds of the official `0.35` slot, giving
official FCMAE800 `0.1166667` and the auxiliary-marker model `0.2333333`. A
secondary shrinkage candidate uses official `0.1166667`, matched control
`0.0583333`, and auxiliary `0.175`. The old half, full, and matched-control
routes remain in the report only as exact evaluation-drift checks.

The primary route must first improve the old half split by at least `+0.001`
q95 with positive float gain, 4/5 positive folds, 13/25 positive ROIs, and a
`-0.03` ROI floor. Deployment then requires at least `+0.04` q95 over
`t70_o50`, positive float gain, all five folds, 18/25 ROIs, and a `-0.05` ROI
floor. The shrinkage fallback requires `+0.03`, all five folds, 20/25 ROIs,
and a `-0.04` floor. These choices are fixed before the new candidate is
evaluated.

This is a one-dimensional Colon ensemble calibration, not an independent
architecture result. Its validation procedure is reusable, but its selected
weight must be re-estimated on later-stage organs.

## Status

The detached pipeline exited zero and evaluated all 6,296 images from all 25
ROIs in 5,262.807 seconds. The four retained Round55 candidates reproduce
their source overall, fold, and ROI metrics exactly, and the complete
checkpoint-provenance object is identical. Independent rehashing verifies all
30 referenced checkpoints. The OOF SHA-256 is
`e0a137d30383555ed930bf7ab03f27f101c76f82f3bba9a879bc6777b032a8ed`;
the gate SHA-256 is
`cbc0de770426bbeecae846ddd4fc653fbc95509b5425aa6633f44f2d84ba8ea7`,
and the gate binds the exact OOF hash.

## Result

| Candidate | q95 score | Delta vs `t70_o50` |
| --- | ---: | ---: |
| `t70_o50` | 75.560424 | baseline |
| `t70_auxiliary_half` | 75.598598 | +0.038174 |
| `t70_auxiliary_full` | 75.589785 | +0.029361 |
| `t70_control_half` | 75.583123 | +0.022699 |
| `t70_auxiliary_two_thirds` | 75.600858 | +0.040434 |
| `t70_auxiliary_control_shrink` | 75.601807 | +0.041383 |

The primary two-thirds route improves the old half split by `+0.002260` q95,
with 4/5 positive folds, 14/25 positive ROIs, and a `-0.019471` worst ROI. Its
advance gate passes. Against `t70_o50`, it improves all five folds and 18/25
ROIs, but ROI018 loses `-0.052987`, missing the preregistered `-0.05` floor by
`0.002987`. The deployment gate therefore fails.

The control-shrink fallback has the highest mean OOF score and improves all
five folds, but only 18/25 ROIs improve versus the required 20/25 and its
`-0.049284` worst ROI misses the `-0.04` floor. Its deployment gate also
fails. The formal result is `promotion.allowed=false`; no Round64 test
prediction or leaderboard submission is permitted.

Round64 confirms that auxiliary-marker diversity has a shallow optimum near
the registered two-thirds share, but the maximum complete-OOF gain remains
about `+0.04`. Further calibration of these same endpoints is unlikely to
close the leaderboard gap and risks validation overfitting, so this weight
surface is closed. The deployed Round53 `t70_o50` ensemble remains unchanged.

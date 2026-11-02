# Round 101: Cross-Fitted Per-Image Ensemble Routing

## Purpose

Rounds 99--100 show that two learned DAPI detail branches do not improve the
current model. Round101 asks whether the existing base32/base48 prediction
diversity contains a larger recoverable gain that can be selected automatically
from public input information, rather than by a fixed global weight.

The diagnostic uses the leakage-safe Round37 cache containing five-fold OOF
DAPI, targets, and base32/base48 predictions for all 6,296 training images.
For each image it evaluates an oracle 0.1-spaced convex-weight grid. A Ridge
gate uses only DAPI intensity/gradient statistics and prediction/disagreement
statistics. For each held-out fold, the gate is fitted on the other four folds;
no target from the evaluated fold and no test image enters its fit.

## Completion and result

The detached diagnostic exited `0`, its Screen and producer process are gone,
and the report covers all five folds and 6,296 images.

| method | JPEG-q95 official score | delta vs fixed 70/30 |
| --- | ---: | ---: |
| fixed base32/base48 70/30 | 74.869164 | baseline |
| per-image 0.1-grid oracle | 75.227702 | +0.358538 |
| five-fold cross-fit Ridge routing | 74.857716 | -0.011448 |

The oracle confirms that image-level complementarity exists, but the automatic
gate loses score on folds 0, 1, 3, and 4 and gains only about `+0.0011` on
fold2. Public DAPI and prediction-disagreement summaries therefore do not
identify the target-dependent best weight reliably.

## Boundary

Close per-image routing for these endpoint families. Do not add feature grids,
classifiers, nonlinear routers, ROI identities, test statistics, or leaderboard
feedback. The oracle is not deployable evidence. A future route would require a
new representation whose uncertainty is intrinsically calibrated, not a more
flexible selector trained on the same OOF targets.

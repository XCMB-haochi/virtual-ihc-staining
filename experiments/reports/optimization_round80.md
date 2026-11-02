# Round 80: Fixed One-Pixel Output Translation

## Motivation

Repository research previously listed micro-registration audit as a prerequisite
before considering misalignment-robust image translation. Rounds 1--79 did not
complete that audit. Round80 measures whether a small, systematic spatial
offset remains after model inference.

The leakage-safe Round37 cache contains 6,296 true OOF predictions and targets
from all 25 training ROIs. A fixed `[-2,2] x [-2,2]` integer translation audit
was performed without test data. For every held-out fold, total MSE on the
other four folds independently selected the same prediction source offset:
`(row=0, col=-1)`. Edge pixels are replicated. The resulting leave-one-fold-out
official q95 gains are:

| fold | fixed shift gain |
| --- | ---: |
| fold0 | +0.140970 |
| fold1 | +0.068399 |
| fold2 | +0.109096 |
| fold3 | +0.095841 |
| fold4 | +0.166009 |

Complete q95 rises from `74.869164` to `74.986058`, a `+0.116894` gain. A
target-aware per-image best-shift oracle reaches only `75.036941`, indicating
that most recoverable translation benefit is the same global one-pixel offset,
not image-specific selection.

## Fixed Current-Model Replication

Round80 applies exactly source offset `[0,-1]` to the current two-fold
`t70_base_split` output after fixed ensembling and before JPEG encoding. In
array terms, output column `x` samples unshifted prediction column `x-1`, with
edge replication. This shifts prediction content one pixel right. The baseline
and shifted candidates have identical models and weights.

The transform is implemented in the shared OOF and test-prediction paths, and
the prediction report records the offset. No test image, test label,
leaderboard feedback, ROI route, subpixel shift, padding mode, shift magnitude,
or neighboring direction is searched after the result.

The shifted candidate must gain at least `+0.05` q95 and positive float score,
improve both folds and 8/10 ROIs, and keep its worst ROI at or above `-0.05`.
A passing result authorizes an unchanged complete five-fold evaluation using
the already existing Base endpoint checkpoints. It does not directly authorize
test inference from this two-fold result.

## Status

Implementation and prelaunch validation are complete. The focused local suite
passes 81 tests. On Linux, all 308 tests and 85 subtests pass, together with
Python compilation, shell syntax, diff checks, and the hash-bound no-side-effect
preflight. Tests cover protocol offset validation, exact edge-padding behavior,
zero-offset identity, and the shared OOF/test-prediction transform path.

Round80 launched from commit `8dcf8c4` in detached Screen `round80` at
2026-08-28 20:29:59 +08:00. The preflight rehashed the Round62 OOF report and
all ten referenced two-fold Base endpoint checkpoints. Formal evaluation runs
on GPU1; a post-start sample measured 100% utilization, 2,933 MB resident, and
about 254 W. GPU0 remains unused. Completion, OOF, and gate status are pending.

The pipeline exited zero and evaluated all 2,706 images from ten fixed ROIs in
2,139.44 seconds. All ten referenced checkpoint hashes independently match the
files. The OOF SHA-256 is
`29ba58e2c86db1caf9a87a05943a0c730d3753e9f6e465f61f47c2e78ce2b893`;
the gate SHA-256 is
`a72b75b738286e4ebd8db6c3f968169b3b12de1cc67f6d67e71406992cf49709`.
Independent local gate evaluation reproduces every metric and decision.

## Result

The fixed shift raises current Base-ensemble q95 from `75.283381` to
`75.400287`, a `+0.116906` gain; float gain is `+0.116948`. SSIM rises from
`0.842804` to `0.844180` and mean-image PSNR from `27.145205` to `27.179465`
dB. Both folds improve (`+0.132644/+0.102160`) and all 10 ROIs are positive.
The smallest ROI gain is still `+0.005307`.

Every preregistered gate passes and `promotion.allowed=true`. The magnitude is
essentially identical to the independent old-model five-fold diagnostic
`+0.116894`, which supports a data-registration mechanism rather than model-
specific error. Round81 may evaluate the unchanged shift on the existing full
five-fold Base endpoint; no adjacent direction, magnitude, padding, subpixel,
or ROI-specific shift is authorized.

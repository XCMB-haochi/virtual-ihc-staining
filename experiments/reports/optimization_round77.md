# Round 77: Cytoland Encoder Composition

## Decision

Round76 separates two effects. Direct Cytoland encoder initialization gains
`+1.252620` q95 over scratch UNeXt2 with both folds and all 10 ROIs positive.
Transferring its compatible decoder stages loses `-0.086995`, so the decoder
branch is closed. The encoder endpoint remains only `0.139740` below official
FCMAE800 and may contain complementary microscopy-domain errors.

Round77 performs one fixed composition check without training. It preserves
the current two-fold Base endpoint and divides only its `0.175` official-Tiny
share equally between official-FCMAE800 and Cytoland encoder-only. The weights
are therefore base32 `0.21`, base48 `0.09`, scratch FCMAE800 `0.35`, official
Tiny `0.0875`, official Base `0.175`, and Cytoland encoder `0.0875`.

No replacement candidate, adjacent weight, continuous grid, ROI route,
calibration, checkpoint, or leaderboard feedback is used. The fixed split must
gain at least `+0.005` q95 and positive float score over `t70_base_split`,
improve both folds and at least 7/10 ROIs, and keep its worst ROI at or above
`-0.04`. Failure closes this composition. Passing authorizes only an unchanged
five-fold expansion of the encoder-only recipe before test inference.

## Audit Boundary

The launcher binds Round76 OOF SHA-256
`1f7223566a654f730212fa23ac1436e20888cb9cf8f4fc9720c449af8e580312`
and gate SHA-256
`d34690c569d4e6b5a350ea228f37c9034aebbb1373a864deacba27d3ed56ac13`.
It requires the direct encoder gate to pass and the decoder gate to fail, then
re-hashes all 16 Round76 checkpoint files before evaluation. The two-fold run
cannot create test predictions or a submission.

## Status

Implementation and validation are complete. The focused local protocol suite
passes 76 tests. On Linux, the same suite passes 76 tests and eight subtests,
together with shell syntax and diff checks. The no-side-effect preflight
validates both bound Round76 report hashes, the passing encoder mechanism, the
failed decoder effect, exact source scores, complete 2,706-image/10-ROI
coverage, and all 16 source checkpoint hashes.

Round77 launched from commit `37f7719` in detached Screen `round77` at
2026-08-28 00:17:26 +08:00. Three launch samples found GPU1 at 0% utilization
with 40,427 MB free, so the fixed six-model OOF evaluation was assigned there.
A later sample measured 100% utilization, 2,955 MB resident, and about 235 W.
GPU0 remained unused.

The pipeline exited zero and evaluated all 2,706 images from the ten fixed
ROIs in 2,640.72 seconds. All 12 referenced checkpoint hashes independently
match their files. The OOF SHA-256 is
`89e68968495512e522db3215ca74ca298f5133198dd71c7d629a5c85f6945539`;
the gate SHA-256 is
`cd2eccbaa9cf5c14f5a4cf4cd24cbccf941bdec75de32072a3d45f6f5c191dbd`.
Independent local gate evaluation reproduces every metric and decision.

## Result

The fixed split scores `75.270254` q95 versus `75.283381` for the retained
Base endpoint, a `-0.013127` change; float score is also negative by
`-0.012937`. Fold0 loses `-0.031666`, while fold2 gains only `+0.004243`.
Only ROI000 and ROI008 improve, and the worst ROI loses `-0.059347`, below the
fixed `-0.04` floor. Every gate check fails except dependency.

Round77 closes Cytoland encoder composition. No adjacent weight, replacement,
five-fold expansion, test inference, or submission follows. Cytoland encoder
initialization remains a strong scratch-to-domain prior, but it does not add
useful diversity to the mature official-FCMAE/Base ensemble.

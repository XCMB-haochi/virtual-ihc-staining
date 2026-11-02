# Round 81: Complete Five-Fold Output Translation

## Authorization

Round80 validates source offset `[0,-1]` on the current two-fold Base endpoint:
`+0.116906` q95, positive float, both folds, all 10 ROIs, and a positive worst
ROI. This independently reproduces the old-model five-fold diagnostic
`+0.116894` and authorizes an unchanged complete five-fold evaluation.

Round81 uses the existing 25 Base-endpoint checkpoints and identical weights.
It changes no model, fold, checkpoint, TTA, JPEG setting, padding mode, or
offset. Baseline and shifted candidates differ only by the shared candidate
output source offset `[0,-1]`.

## Gates

The complete shifted candidate must gain at least `+0.05` q95 and positive
float score over `t70_base_split`, improve all 5 folds and at least 20/25 ROIs,
and keep the worst ROI at or above `-0.05`. Passing authorizes fixed test
inference and packaging with the same offset. Failure closes the shift without
direction, magnitude, padding, subpixel, or ROI-specific search.

The launcher binds Round80 OOF SHA-256
`29ba58e2c86db1caf9a87a05943a0c730d3753e9f6e465f61f47c2e78ce2b893`
and gate SHA-256
`a72b75b738286e4ebd8db6c3f968169b3b12de1cc67f6d67e71406992cf49709`.
It requires the exact passing candidate and every two-fold gate check, then
re-hashes the 25 complete-OOF checkpoint files before formal evaluation.

## Status

Implementation and prelaunch validation are complete. The focused local suite
passes 82 tests. On Linux, the same suite passes 82 tests and 13 subtests,
together with shell syntax and diff checks. The no-side-effect preflight binds
the two Round80 reports, requires its exact passing promotion, validates the
Round70 complete-OOF hash and score, and re-hashes all 25 required checkpoint
files.

Round81 launched from commit `f1e1fa4` in detached Screen `round81` at
2026-08-28 21:17:00 +08:00. Formal five-fold evaluation runs on GPU1; a
post-start sample measured 99% utilization, 2,933 MB resident, and about 223 W.
GPU0 remains unused. Completion, OOF, and gate status are pending.

The pipeline exited zero and evaluated all 6,296 images from 25 ROIs in
5,579.04 seconds. All 25 referenced checkpoint hashes independently match the
files. The OOF SHA-256 is
`c2418b53d443c026cc9589a95f48438f6e50069d3f64fb9bc8e81ca66243def7`;
the gate SHA-256 is
`aee7ecd340a1f242db1e7cca0e7afdd87bcfed96ec4d767d2a722f22fe2b96cf`.
Independent local gate evaluation reproduces every metric and decision.

## Result

The fixed shift raises complete q95 from `75.644105` to `75.755123`, a
`+0.111018` gain; float gain is `+0.111359`. All five folds improve, with
deltas from `+0.067463` to `+0.160704`, and 21/25 ROIs are positive. However,
ROI005 loses `-0.242646`, ROI006 `-0.116665`, ROI010 `-0.063080`, and ROI019
`-0.008193`. The fixed `-0.05` ROI floor therefore fails and
`promotion.allowed=false`; every other gate check passes.

A same-ROI mosaic boundary diagnostic replaces edge replication with the
adjacent tile's final prediction column. On the independent Round37 cache this
reduces q95 by `-0.001691` relative to edge padding and improves only 1/25
ROIs, so seam filling does not explain or repair the floor failure. That branch
is closed.

Round81 does not authorize normal promotion. Because the preliminary platform
allows inexpensive diagnostics, and the exact mechanism independently gains
about `+0.117` on an old five-fold model family plus `+0.111` on the current
family with 5/5 positive folds, one explicitly retrospective hidden-set
confirmation may be packaged without changing the offset. It must preserve
the failed gate record and cannot authorize direction, magnitude, padding,
subpixel, ROI, or leaderboard-guided follow-up tuning.

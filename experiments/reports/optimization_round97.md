# Round 97: Full Neighbor-Ring Context at Inference

## Motivation

Round96 rejects exposure-matched independent multi-marker FCMAE. The direct
representation loses `-0.039674` q95, its two folds disagree, and its safest
fixed ensemble split gains only `+0.012357`, below every promotion threshold.
No marker-ratio, duration, or composition search follows that result.

The retained Base-diversity endpoint has one bounded, structurally meaningful
inference question left. Context128 improved complete OOF by `+0.026427` and
the fixed hidden-set package by `+0.0214`, but its 512x512 view contains only
half of each directly adjacent 256x256 tile. A halo of 256 yields a 768x768
view: the center patch plus one complete neighboring tile in every direction.
This tests the organizer-approved same-ROI DAPI context mechanism at a natural
geometry boundary, rather than starting a halo grid.

## Fixed Protocol

`configs/round97_cv5_context256_inference.json` reuses the exact 25 checkpoint
files, fold assignments, Base-split weights, D4 TTA, JPEG quality 95, and fixed
output offset `[0,-1]` from Round85. No model is retrained or reselected. The
only prediction change is source context halo 256; batch size is reduced from
2 to 1 to bound inference memory without changing any prediction.

The immutable comparator is the complete Round85 context128 OOF with SHA-256
`c09c4da98d87c4cc9254de0b90f3d0b5cd41d38da2415bb2c2e0aca181de615f`.
Promotion requires at least `+0.02` q95, positive float gain, at least 4/5
positive folds, at least 20/25 positive ROIs, and no ROI below `-0.05`.
Checkpoint provenance must match Round85 exactly. No context192/320, batch,
checkpoint, ensemble-weight, offset, calibration, ROI-route, or leaderboard
search is authorized from this screen.

## Completion Audit

The detached pipeline exited with code `0`; its Screen session and producer
processes are gone. The synchronized Linux checkout passed all 338 tests,
Python/JSON/Bash syntax checks, the immutable Round85 gate replay, and the
no-side-effect preflight. The 768x768 smoke passed with peak allocation
`1.718 GiB`. Complete OOF covers 6,296 images across all 25 ROIs and took
`19,802.832 s`.

All 25 candidate checkpoint records were present. Their SHA-256 values matched
the recorded Round85 checkpoints exactly; only the inference halo changed.
The copied candidate OOF has SHA-256
`71d7ef9b2470a6e8c316262802a149899c5101257aa31aa808a2a39c0335f0b3`, and the
gate has SHA-256
`780370050e7948d86f4b404415f1456023ceb553d699fe4b59f8b60837e65a53`.
Independent local gate replay matched the server decision and all semantic
fields exactly.

## Result

The fixed `context256` candidate reached q95 `75.791985` versus the immutable
Round85 context128 baseline `75.781550`, a delta of `+0.010434`. The float
metric delta was `+0.010271` (`75.869779` versus `75.859508`). Fold deltas were
`-0.021372 / +0.025781 / +0.028493 / +0.015338 / +0.004653`; 4/5 folds were
positive. Only 16/25 ROIs improved, with ROI018 at `-0.085058` as the worst
case.

The candidate therefore fails the pre-registered q95 (`+0.02`), positive-ROI
(`20/25`), and ROI-floor (`-0.05`) gates. `promotion.allowed=false`; there is
no submission or leaderboard probe. The result is weak evidence that a larger
same-ROI halo can help some folds, but its ROI instability does not support a
portable context-scale route.

## Boundary

Close the `context256` branch. Do not search additional halos, context-derived
weights, offsets, or ROI-specific rescue rules. Retain the small context-scale
observation only as background for a future dataset-specific grouped-OOF
screen.

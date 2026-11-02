# Round 67: Auxiliary and All-Train-DAPI Composition

## Motivation

Round64 auxiliary calibration and Round66 all-train-DAPI expansion both yield
small, stable, but individually insufficient complete-OOF diversity gains.
Their training mechanisms differ: auxiliary supervision changes the staining
objective, while shared all-train DAPI changes self-supervised representation.
Round67 tests whether these independently established endpoints compose before
spending roughly three GPU-days on the unfinished Base promotion.

## Fixed Candidates

The retained base32, base48, and scratch-FCMAE weights remain `0.21`, `0.09`,
and `0.35`, leaving `0.35` for official Tiny-derived endpoints. Three discrete
hypotheses are preregistered without a continuous weight grid:

- `tri_aux23_residual_half`: retain Round64's 2/3 auxiliary allocation, then
  split its residual official weight equally with all-train DAPI (`0.05833`,
  `0.23333`, `0.05833`).
- `tri_auxhalf_residual_half`: allocate half to auxiliary, then split the
  residual equally between official and all-train DAPI (`0.0875`, `0.175`,
  `0.0875`).
- `tri_no_official_aux2_dapi1`: remove the official endpoint and divide its
  pool 2:1 between auxiliary and all-train DAPI (`0`, `0.23333`, `0.11667`).

Each candidate must gain at least `+0.05` q95 and positive float score over
`t70_o50`, improve all 5 folds and at least 20/25 ROIs, and keep the worst ROI
at or above `-0.05`. No weight will be added or changed after evaluation.

## Audit Boundary

The launcher binds the complete Round64 OOF SHA-256
`e0a137d30383555ed930bf7ab03f27f101c76f82f3bba9a879bc6777b032a8ed`
and Round66 OOF SHA-256
`4eee928e2f034fd0b76151359196eb36705b13606cd9b79f3688fd37db243e03`.
All six model families use existing five-fold checkpoints. Round67 performs no
training, test inference, leaderboard calibration, or ROI-specific routing.

## Status

Implementation is complete. All 70 focused local protocol tests, Python
compilation, JSON validation, launcher shell syntax, and diff checks pass. On
Linux, all 284 tests pass. All 30 referenced five-fold checkpoint files match
the SHA-256 provenance recorded in the audited Round64 and Round66 OOF reports,
and no Round67 output or exit artifact exists before launch.

Round67 launched from commit `a0c71a8` in detached Screen `round67` at
2026-08-22 19:06:29 +08:00. Five launch-time samples showed physical GPU0
occupied by an unrelated task at 100% utilization and GPU1 idle with 17 MB
resident, so evaluation was assigned only to GPU1. Initial verification showed
GPU1 at 69--99% utilization with the expected physical mapping. Completion and
gate status remain unverified until the exit marker, complete OOF, checkpoint
provenance, and saved gate are audited.

## Result

The pipeline exited `0` and evaluated all 6,296 images from all 25 ROIs. The
OOF SHA-256 is
`8551eac7ca9f57bd4de716ba9dc98cacc161ca3a249d1d3aee614a1a3a6301c2`;
the gate SHA-256 is
`cd918b3333b9e33267e81449cc6a8c399bd383ae06ac39812a72c460b1f6dedc`.
All 30 checkpoint hashes match the server files. The `t70_o50` overall, fold,
and ROI metrics exactly replay both source OOF reports, and an independent
local gate evaluation exactly reproduces the saved gate.

`tri_auxhalf_residual_half` passes every preregistered check. Its JPEG-q95 OOF
score is `75.610623` versus `75.560424` for `t70_o50`, a `+0.050199` gain;
float gain is `+0.049604`. All five folds improve, with deltas from `+0.019427`
to `+0.071933`. It improves 21/25 ROIs and the worst ROI delta is `-0.038894`,
above the fixed `-0.05` floor. Promotion selects this candidate with weights
base32 `0.21`, base48 `0.09`, scratch FCMAE `0.35`, official FCMAE `0.0875`,
auxiliary `0.175`, and all-train-DAPI FCMAE `0.0875`.

The other two candidates fail their fixed gates. `tri_aux23_residual_half`
gains `+0.047533`, improves 18/25 ROIs, and reaches a `-0.051866` floor.
`tri_no_official_aux2_dapi1` gains `+0.049981`, improves 17/25 ROIs, and has a
`-0.061346` floor. They are not eligible despite near-equal aggregate scores.
Round68 may generate only the selected fixed candidate; no additional weight
search or leaderboard-derived calibration is authorized.

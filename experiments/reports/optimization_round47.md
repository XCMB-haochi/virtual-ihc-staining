# Round 47: FCMAE800 Blend-Weight Audit

## Motivation

Round46 fixed the FCMAE800 contribution at 20% before its five-fold OOF and
platform evaluation. An independent review proposed that this weight may be
too low. The committed Round46 complete-OOF report gives q95 scores of
`74.869213` at 0% FCMAE800, `75.117710` at 20%, and `75.132343` at 100%.
A quadratic through only these three aggregate points peaks near weight
`0.607` and predicts roughly `+0.20` over the submitted 20% blend, but three
points do not establish that the actual JPEG/SSIM score curve is quadratic.

This round tests that numerical claim directly. It changes no checkpoint,
training data, architecture, inference transform, calibration, or JPEG
setting. It is an OOF-only model-selection diagnostic, not a leaderboard
probe.

## Fixed Grid And Decision Rule

The FCMAE800 weights are fixed before evaluation at
`0.0/0.2/0.4/0.6/0.8/1.0`. The remaining weight uses the unchanged Round35
70/30 base32/base48 mixture. All candidates are evaluated in one inference
pass over the same complete five-fold OOF universe.

The current 20% blend is retained unless a candidate meets every complete-OOF
condition below:

- q95 gain over the 20% blend is at least `+0.08`;
- float gain is positive;
- at least 4/5 folds and 18/25 ROIs improve;
- the worst ROI q95 change is no lower than `-0.08`.

The preregistered primary is the 60% blend implied by the three-point fit.
The 40%, 80%, and 100% candidates diagnose curve shape and have identical
distribution gates. Promotion order prefers 60%, then 40%, 80%, and 100%; a
later platform package is not part of this pipeline and requires separate
review of the full OOF distribution. Test-DAPI-derived `test-like` subsets are
disabled and cannot determine the weight.

## Review Conclusions Retained And Rejected

The review correctly identified that the existing endpoints leave open a
larger FCMAE800 contribution and that direct evaluation is cheaper than new
training. Its claimed gain remains an interpolation hypothesis until this
grid finishes.

The proposed labelled-data scale curve is deferred. Reducing ROI count while
holding an epoch schedule fixed confounds sample count, optimizer updates,
subset composition, and validation variance, so four single-subset runs would
not isolate the value of augmentation or auxiliary supervision. A later scale
study would need nested subsets, update-matched budgets, and repeated subset
draws.

Longer supervised training is also not launched by itself: Round46 stage-2
validation peaks at epochs 10, 11, 14, 11, and 6 across the five folds, so
extra epochs require a separately controlled regularization change.

## Execution

At preregistration time physical GPU0 had active compute near 46% while GPU1
was idle with about 34.7 GB free. OOF evaluation is one serial process, so it
is assigned to GPU1 rather than split across both GPUs. Future fold-independent
training will be scheduled from fresh telemetry and can allocate more folds to
the less-contended GPU.

Expected outputs:

- `reports/round47_cv5_fcmae800_blend_grid_oof.json`
- `reports/round47_cv5_fcmae800_blend_grid_gate.json`

The OOF pipeline itself generates no test prediction or submission archive;
promotion is decided before a separate inference pipeline may run.

## OOF Result

The run completed all five folds and 6,296 validation images in `2572.111 s`.
The measured q95 curve is:

| FCMAE800 weight | q95 | gain vs. 20% | float | SSIM | PSNR |
|---:|---:|---:|---:|---:|---:|
| 0% | 74.869213 | -0.248498 | 74.948397 | 0.839529 | 26.836953 |
| 20% | 75.117710 | - | 75.196639 | 0.841777 | 26.988899 |
| 40% | 75.270404 | +0.152693 | 75.348870 | 0.843114 | 27.087349 |
| 60% | **75.324015** | **+0.206304** | **75.402853** | **0.843531** | **27.128057** |
| 80% | 75.278318 | +0.160607 | 75.357217 | 0.843040 | 27.109191 |
| 100% | 75.132343 | +0.014632 | 75.211470 | 0.841619 | 27.031677 |

The directly evaluated maximum is the preregistered 60% candidate, agreeing
with the earlier three-point interpolation without relying on it. Relative to
the current 20% blend, 60% gains q95 `+0.206304` and float `+0.206214`. All
5/5 folds and 25/25 ROIs improve. Fold gains are
`+0.238491/+0.126345/+0.293113/+0.191077/+0.162954`; the worst ROI gain is
still positive at `+0.033632` and the best is `+0.375208`.

As an additional stability check, each fold was held out in turn and the grid
weight was selected using only the other four folds, weighted by their image
counts. All five selections independently chose 60%. Applying each selection
to its untouched fold gives a combined image-weighted gain of `+0.206304` over
20%, so neither the selected weight nor its estimated gain depends on one
fold's validation metric.

Every preregistered 60% distribution check passes. The 80% and 100% points
show why unrestricted extrapolation is unsafe: 80% has four negative ROIs and
100% is positive in only 2/5 folds and 14/25 ROIs. The approved artifacts are:

- OOF SHA-256:
  `395fb19592239631a34c18535f5fb95d638d9bc9cc1a9148963b7871be6cbf0d`
- gate SHA-256:
  `2d653b259fedec3530d343cc815553316c2b718e661a4947548218e91105a930`
- promotion: `blend_w060_full -> blend_w060`

One fixed 60% test inference and package is therefore authorized. No adjacent
weight, per-ROI route, calibration, or test-derived selection follows this
result.

## Submission Package

The fixed 60% candidate completed test inference in `1624.951 s`. Its 15
members comprise five folds each of FCMAE800, Round35 base32, and Round35
base48, with total recipe weights `0.60/0.28/0.12`; all member weights sum to
one. The prediction report records CD68, 1,346 test images, halo64, D4, and
JPEG q95.

Packaging and independent local inspection both found 1,346 unique canonical
RGB JPEG entries, with no missing, extra, malformed, incorrectly sized, or
duplicate images. ZIP CRC validation passed on the server and after transfer.

- Prediction report:
  `reports/round47_cv5_blend_w060_test_prediction.json`
- Submission:
  `submissions/cd68_round47_cv5_blend_w060_context64_d4_raw_q95_plain.zip`
- ZIP size: `11,414,129` bytes
- ZIP SHA-256:
  `ce0372f83d4c6fe55daeddb40d52a61c0c28af811b8ca8c1508409f7532fb97a`
- Prediction report SHA-256:
  `e75e7a3b209187f96c0a9e1962dc6d0f177f25039e396937d653ec2bc535d0bf`

## Platform Result

The fixed 60% package scored `73.6745`, with displayed `SSIM=0.828` and
`PSNR=26.189`. Relative to the Round46 20% package, this is a platform gain of
`+0.1995`, compared with the preregistered complete-OOF q95 gain of
`+0.206304`. The hidden-set realization of the OOF increment is therefore
approximately `96.7%`.

This is also `+0.4029` over Round35. Relative to the last user-reported
leaderboard maximum of `75.4966`, the remaining gap is `1.8221`; that maximum
was not independently refreshed for this report. The close OOF/platform
agreement supports the fixed five-fold selection protocol, but it does not
authorize adjacent blend weights, per-ROI routing, calibration, or any other
test-derived search.

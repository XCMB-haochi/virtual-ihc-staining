# Round 91: Fixed Round90 Preliminary Diagnostic

## Purpose

Round90 produced a strong aggregate OOF gain for the fixed auxiliary/all-train-
DAPI endpoint with context128 inference, but its preregistered distribution gate
failed because ROI005 regressed beyond the fixed floor. This round performs one
explicitly labelled preliminary confirmation of that unchanged endpoint. It is
not a gate override, weight search, ROI route, or post-hoc calibration.

## Fixed endpoint

The script `tools/run_round91_round90_diagnostic_submission.sh` binds the exact
Round90 protocol, OOF SHA-256, gate SHA-256, all 30 checkpoint hashes, member
weights, D4 TTA, context halo 128, output offset `[0,-1]`, and JPEG quality 95.
No test labels or test-derived output adjustment is used. Platform feedback is
confirmation only and must not be used to choose a follow-up parameter.

The diagnostic is justified only by the fixed Round90 evidence: q95 gain
`+0.138170`, positive float gain `+0.138618`, 5/5 positive folds, and 22/25
positive ROIs. The failed ROI floor remains recorded and unchanged.

## Completion audit

The hash-bound preflight passed on the storage-scoped server. Screen
`round91_round90_diagnostic` launched at `2026-08-30 23:46:22` (Asia/Shanghai)
and selected physical GPU1 at `23:46:36`; GPU0 was left untouched under
unrelated load. The process exited with status `0`.

The prediction report records the fixed candidate, context halo `128`, D4 TTA,
output offset `[0, -1]`, all `1346` test images, and all `30` fixed ensemble
members. The packaged archive contains exactly `1346` files under
`results/test/CD68/`; `zipfile.testzip()` passed and every inspected image is
256x256 with a valid image mode. Archive SHA-256:
`a648e7dc431e82155c01c2c25ad7b207d4700035c7de814bf174f74570119b09`.

This is only a fixed preliminary confirmation package. It does not promote the
Round90 candidate, change the failed ROI-floor gate, or use a leaderboard score
as a tuning signal. Any platform score is external evidence to record after
submission, not a protocol selection input.

## Platform confirmation

The submitted package received `74.0287` with `SSIM=0.831` and
`PSNR=26.400`. This is `-0.0028` versus the previously recorded `74.0315`
submission, so the fixed Round90 endpoint was not promoted by this confirmation.
The difference is within the expected submission/evaluation noise scale and is
not evidence for any new weight, calibration, or test-specific adjustment.

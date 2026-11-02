# Round129: Fixed Base + Relational Hybrid OOF

The current personal best is Round128's fixed 50/50 blend of the two
relational seeds (`74.2362` on the platform). Round88's Base-diversity ensemble
is a materially different endpoint (`74.0315`), so it is useful to test whether
their errors are complementary. This round evaluates one predeclared candidate:
50% Round128 relational seed blend plus 50% Round88 Base-diversity endpoint.

The shared five-fold OOF includes the two direct endpoints and the fixed hybrid
in one pass. The hybrid's effective recipe weights are Base members at half of
their Round88 weights and relational seed2044/seed2027 at `0.25` each. No other
blend ratio, calibration, ROI routing or platform feedback is used. The fixed
gate compares hybrid against the relational seed blend and requires nonnegative
q95 gain, positive float, at least 4/5 positive folds, at least 18/25 positive
ROIs and worst ROI no lower than `-0.12`.

This is an inference-only packaging check. GPU1 is the only eligible device;
the existing best, Round124, Round127 and Round128 packages remain untouched.

The merged protocol passed local and storage-scoped Linux validation. The fixed
OOF runner is detached as Screen `round129_hybrid_oof` on GPU1 only; it waits
for an empty GPU1, evaluates all five folds and all three candidates in one
pass, then writes the fixed gate. No training or hidden-set inference occurs
in this phase.

The first package invocation was accidentally attached to the interactive
session and was interrupted before completion; it left 74 JPEGs and an exit
status `120`, with no ZIP or prediction report. Those partial files were moved
to `transfers/round129_runtime/interrupted/round129_foreground_20260914/` for
provenance. The package was then relaunched as detached Screen
`round129_hybrid_submission_retry` on GPU1 only. The retry started with a clean
output directory and completed with exit status `0`.

## Completion and package audit (2026-09-15)

- OOF pipeline exit: `0`; submission/inference exit: `0`; no Round129 producer
  remains on GPU1
- Test prediction report: `reports/round129_base_relational_hybrid_test_prediction.json`
  (35 members, context halo `128`, D4, fixed output offset `[0,-1]`, 1,346 test
  images; SHA-256 `f40bf27efd3b3ce127225e2ac03f352aa94a6728771d90b7bf787e1356b9ab71`)
- Package: `cd68_round129_base_relational_hybrid_q95_plain.zip`, 11,823,015 bytes
  (SHA-256 `a898db9aad5a41be17dd2184b49a04dcb04eea64a9b4dfa026e66689bc4a5043`)
- Independent local audit: 1,346/1,346 expected entries, all under
  `results/test/CD68/`; every image is readable RGB `256x256`; ZIP CRC test
  passed; remote checkpoint verification passed for all 35 unique members
- Measured hidden-set inference and packaging elapsed time: about `11,691 s`
  (`3 h 15 min`) on GPU1
- Status: fixed gate passed (`promotion.allowed=true`), with OOF q95 gain
  `+0.181003` and float gain `+0.181021` against the fixed Round128 relational
  blend baseline. The package is ready for one manual platform confirmation;
  Round128 remains the verified personal-best score until that confirmation is
  reported. No weight, calibration, ROI-routing, or leaderboard search is
  registered from this result.

## Platform confirmation (2026-09-15)

The user reported platform score `74.2276`, with displayed `SSIM=0.833` and
`PSNR=26.533`. This is `-0.0086` versus the Round128 package (`74.2362`),
despite the positive OOF delta. The hybrid therefore fails hidden-set
confirmation and is closed as a submission route; Round128 remains the primary
package. The small PSNR increase with unchanged rounded SSIM is not evidence for
another blend ratio, calibration, or ROI-specific correction.

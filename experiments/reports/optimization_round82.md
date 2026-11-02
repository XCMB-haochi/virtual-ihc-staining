# Round 82: Fixed-Shift Preliminary Diagnostic

## Authorization Boundary

Round81 does not pass its full distribution gate because four ROIs regress and
ROI005 violates the floor. It nevertheless provides unusually strong repeated
evidence for one fixed mechanism:

- old-model complete OOF: `+0.116894`, 5/5 folds positive;
- current two-fold Base OOF: `+0.116906`, 2/2 folds and 10/10 ROIs positive;
- current complete Base OOF: `+0.111018`, 5/5 folds and 21/25 ROIs positive.

Round82 is therefore an explicitly retrospective preliminary-stage diagnostic,
not a promotion override. It performs one test inference with the unchanged
Base endpoint and source offset `[0,-1]`. No direction, magnitude, padding,
subpixel shift, calibration, ensemble weight, ROI route, or checkpoint is
changed or searched. The platform result confirms hidden transfer only and
will not select a follow-up shift candidate.

## Fixed Inference

The candidate uses the existing 25 `t70_base_split` members, halo64, D4 TTA,
JPEG q95, and weights base32 `0.21`, base48 `0.09`, scratch FCMAE800 `0.35`,
official Tiny FCMAE800 `0.175`, and official Base FCMAE800 `0.175`. After
ensembling, output column `x` samples column `x-1` with edge replication.

The launcher binds Round81 OOF SHA-256
`c2418b53d443c026cc9589a95f48438f6e50069d3f64fb9bc8e81ca66243def7`
and gate SHA-256
`aee7ecd340a1f242db1e7cca0e7afdd87bcfed96ec4d767d2a722f22fe2b96cf`.
It requires the exact failed promotion, all checks except ROI floor to pass,
`+0.10` aggregate gain, all five folds positive, and 21 positive ROIs. Final
validation requires 1,346 RGB outputs, 25 rehashed members, report offset
`[0,-1]`, exact archive paths, and valid ZIP CRCs.

## Status

Implementation and prelaunch validation are complete. The focused local suite
passes 83 tests. On Linux, the same suite passes 83 tests and 13 subtests,
together with shell syntax and diff checks. The no-side-effect preflight binds
both Round81 report hashes, preserves its failed promotion and exact check
pattern, verifies `+0.10` aggregate evidence, 5 positive folds, 21 positive
ROIs, and re-hashes all 25 checkpoint files.

Round82 launched from commit `c4ff937` in detached Screen `round82` at
2026-08-28 23:20:16 +08:00. Fixed 25-member test inference runs on GPU1; a
post-start sample measured 100% utilization, 12,827 MB resident, and about
223 W. GPU0 remains unused. Completion, report, image coverage, ZIP integrity,
and artifact hashes subsequently completed successfully.

## Completion Audit

The detached pipeline exited zero after 5,661.89 seconds and produced all
1,346 test JPEGs. The prediction report records candidate
`t70_base_split_shift_xm1`, source offset `[0,-1]`, halo64, D4, JPEG q95, 25
unique members, and an effective weight sum of one. All 25 checkpoint hashes
still match their files.

Server and local validation independently confirm exact agreement with the
official test DAPI filenames, 1,346 archive entries under
`results/test/CD68/`, `256x256 RGB` images with three identical channels, and
valid ZIP CRCs. The server prediction-directory filename/content manifest has
SHA-256
`2b1b1ab5364d1ef0ab1b676a649a2737961290224bfc5ae4f3b069bb49ac58d2`.
The ZIP entry-path/content manifest has SHA-256
`1148d89ecf5fbbfc2647a55310af857eab9dd15c7a6533f97b04ef9c32128485`.

The local and server artifacts agree:

- prediction report SHA-256:
  `aad68a6ce4d482e007e4879de0567c8bb39d98e516f01bd8ea097699b7bd8513`;
- submission size: `11,745,982` bytes;
- submission SHA-256:
  `f461a8c64f4511025e01f2e089924e08a113893777186c5731f8311597d122a0`.

The validated diagnostic package is
`submissions/cd68_round82_cv5_base_split_shift_xm1_context64_d4_q95_plain.zip`.
Its platform score is pending. That single hidden-set result will be recorded
as confirmation only and will not authorize another shift or padding probe.

## Platform Result

The preliminary platform returned `74.0101`, with displayed `SSIM=0.831` and
`PSNR=26.418`. The unchanged 25-member Round71 package scored `73.9228`,
`SSIM=0.830`, and `PSNR=26.388`, so the fixed shift realizes a `+0.0873`
hidden-set gain, with displayed changes of `+0.001` SSIM and `+0.030 dB` PSNR.

Relative to the complete-OOF gain `+0.111018`, hidden transfer is about
`78.6%`. This makes Round82 the highest verified preliminary package in the
repository. The positive transfer confirms the preregistered `[0,-1]`
registration mechanism; per the authorization boundary, it does not permit
another offset, padding, subpixel, ROI, calibration, or leaderboard-guided
probe.

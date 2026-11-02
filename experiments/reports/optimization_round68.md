# Round 68: Fixed Composition Submission

## Authorization

Round67 promoted exactly one complete-OOF candidate:
`tri_auxhalf_residual_half`. It scores `75.610623` at JPEG q95 versus
`75.560424` for the retained Round53 candidate, improves all five folds and
21/25 ROIs, and keeps its worst ROI at `-0.038894`. No further weight search or
leaderboard-derived calibration is used.

## Fixed Inference

The submission candidate uses base32 `0.21`, base48 `0.09`, scratch FCMAE800
`0.35`, official FCMAE800 `0.0875`, auxiliary `0.175`, and all-train-DAPI
FCMAE630 `0.0875`. Across five folds this loads 30 fixed checkpoints. Inference
remains halo64, D4 TTA, batch size 2, CD68 only, and JPEG q95.

The launcher binds approved Round67 OOF SHA-256
`8551eac7ca9f57bd4de716ba9dc98cacc161ca3a249d1d3aee614a1a3a6301c2`
and gate SHA-256
`cd918b3333b9e33267e81449cc6a8c399bd383ae06ac39812a72c460b1f6dedc`.
It requires the exact promotion object and every selected gate check to pass,
refuses to overwrite any output, and validates prediction count, member count,
weight sum, ZIP paths, and CRC after packaging.

## Status

Implementation is complete. All 70 focused local protocol tests, Python
compilation, launcher shell syntax, and diff checks pass. Linux validation,
artifact preflight, and launch remain pending.

On Linux, all 284 tests pass. All 30 referenced checkpoint files match the
SHA-256 provenance in the approved Round67 OOF, launcher syntax passes, and the
prediction directory, prediction report, submission ZIP, and exit marker are
confirmed absent before launch.

Round68 launched from commit `4d61888` in detached Screen `round68` at
2026-08-22 20:38:47 +08:00. Five launch-time samples showed physical GPU0
occupied by an unrelated task at 88--100% utilization and GPU1 idle with 17 MB
resident, so inference was assigned only to GPU1. The live process has the
fixed `tri_auxhalf_residual_half` command and entered model loading normally;
completion, prediction coverage, provenance, and ZIP integrity remain
unverified until the exit marker is present.

## Completion Audit

The pipeline exited `0` after `4,401.32` seconds and produced all 1,346 test
JPEGs from the fixed 30-member candidate. The prediction report records halo64,
D4, JPEG q95, CD68, 30 unique fold/recipe members, and an effective weight sum
of one. All 30 checkpoint hashes still match the server files.

Independent local validation confirms exactly 1,346 archive entries under
`results/test/CD68/`, exact agreement with the official test DAPI filenames,
256x256 RGB images with three identical channels, and valid CRCs. A canonical
filename/content manifest computed from the local ZIP matches the server
prediction directory at SHA-256
`c9f5cea80e8e6f4b91b0dab3ec61899caedffae82424c64175ed3b245f8a396d`.
The local and server artifact hashes agree:

- prediction report SHA-256:
  `531700145a8a39d660c3243233910c507c363ff7833f2a7c70d6beeb6575ec3d`;
- submission size: `11,393,638` bytes;
- submission SHA-256:
  `6da0fcadd9bf793a3e74cbd97630919d552143aad7265b2cf219c743b6c00f24`.

The validated package is
`submissions/cd68_round68_cv5_auxhalf_dapi_context64_d4_raw_q95_plain.zip`.
Platform score remains unknown until the user submits it; no score or transfer
claim is inferred from OOF alone.

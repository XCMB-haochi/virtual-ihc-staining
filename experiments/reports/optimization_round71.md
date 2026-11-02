# Round 71: Fixed Base Diversity Submission

## Authorization

Round70 closes the bounded weight interpolation and retains exactly one fixed
candidate: `t70_base_split`. Its complete JPEG-q95 OOF is `75.644105`, a
`+0.083681` gain over `t70_o50`, with all five folds and 20/25 ROIs positive
and a worst ROI delta of `-0.070786`. No interpolation improves it.

Following the repository feedback-efficiency policy, Round71 proceeds directly
to preliminary-stage test inference after provenance checks. It adds no OOF,
weight search, calibration, or leaderboard-dependent branch.

## Fixed Inference

The candidate uses base32 `0.21`, base48 `0.09`, scratch FCMAE800 `0.35`,
official Tiny FCMAE800 `0.175`, and official Base FCMAE800 `0.175`. Across five
folds this loads 25 fixed checkpoints. Inference remains halo64, D4 TTA, batch
size 2, CD68 only, and JPEG q95.

The launcher binds Round70 OOF SHA-256
`e3e854ecf193e95fe0b1d9b721a87361e98c3ac3b0c11684f302d22f18eec372`
and gate SHA-256
`7cd5de3e89be461134056072279efe848a85bde7b98c781945883ab44ffc6bad`.
It requires the exact retained-Base promotion and every gate check, refuses to
overwrite output, and validates test coverage, 25-member provenance, weight
sum, ZIP paths, and CRC after packaging.

## Status

Implementation is complete. All 71 focused local protocol tests, Python
compilation, launcher syntax, and diff checks pass. Linux validation, 25-member
artifact preflight, and launch remain pending.

On Linux, all 285 tests pass. The approved Round70 OOF/gate hashes, exact
retained-Base promotion, and every gate check validate. All 25 selected member
files match their recorded SHA-256 provenance, and the prediction directory,
report, submission ZIP, and exit marker are absent before launch.

Round71 launched from commit `560351b` in detached Screen `round71` at
2026-08-25 17:50:34 +08:00. Five launch-time samples showed GPU0 occupied by
an unrelated task at 39--47% utilization and GPU1 idle, so inference was
assigned to GPU1. The live process records the exact `t70_base_split` candidate
and entered sequential 25-model loading; GPU1 memory rose from about 2.0 to
3.0 GB while initialization remained CPU/storage-bound. Completion,
prediction coverage, provenance, and ZIP integrity remain unverified.

## Completion Audit

The pipeline exited `0` after `4,846.08` seconds and produced all 1,346 test
JPEGs from the fixed 25-member Base-diversity candidate. The prediction report
records halo64, D4, JPEG q95, CD68, 25 unique fold/recipe members, and an
effective weight sum of one. All 25 checkpoint hashes still match the server
files.

Independent local validation confirms exactly 1,346 archive entries under
`results/test/CD68/`, exact agreement with the official test DAPI filenames,
256x256 RGB images with three identical channels, and valid CRCs. A canonical
filename/content manifest computed from the local ZIP matches the server
prediction directory at SHA-256
`9c3d0b7fec1441f59ecf206c2e65eaee842970a440a389ac5c3b15bd4ba617e3`.
The local and server artifact hashes agree:

- prediction report SHA-256:
  `19c3c10aa7dbb9b5a5f2293074459026f39a90329e5c8d9cd63f5dde41834c6b`;
- submission size: `11,385,557` bytes;
- submission SHA-256:
  `59bc90793d0fd1baeebd9d41cc8073a1e52f79a1b9d361c9f1d141d72d48b76c`.

The validated package is
`submissions/cd68_round71_cv5_base_split_context64_d4_raw_q95_plain.zip`.

## Platform Result

The preliminary platform returned `73.9228`, with displayed `SSIM=0.830` and
`PSNR=26.388`. The retained Round53 package scored `73.9028`, `SSIM=0.830`,
and `PSNR=26.350`, so the realized gain is `+0.0200`, driven visibly by a
`+0.038` PSNR increase while displayed SSIM is unchanged at three decimals.

The corresponding complete-OOF gain is `+0.083681`, so only about `23.9%` of
the measured increment transferred to the hidden test set. The positive sign
supports Base as useful ensemble diversity, but the substantially weaker
magnitude shows that grouped train-ROI OOF still overestimates this endpoint's
cross-domain benefit. Round71 becomes the highest verified package in this
repository, but it does not justify further Base/Tiny weight interpolation,
calibration, or leaderboard-guided probing. Subsequent work should target a
new transferable mechanism or stronger domain-shift robustness.

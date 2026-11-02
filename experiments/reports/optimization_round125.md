# Round125: Fixed Round124 Hidden-Set Confirmation Package

Round124 is a complete five-fold result with q95 `75.506276` versus `75.411811`
for its matched auxiliary15 ensemble. The fixed candidate gains `+0.094465`
overall and is positive on four of five folds, but fold4 loses `-0.209331`,
only 17/25 ROIs are positive, and the preregistered worst-ROI floor fails at
`-0.586862`. It is therefore not promoted or blended by hand.

Because the preliminary stage has no elimination and the previous personal-best
package is preserved, one bounded platform confirmation has positive information value:
test the exact complete Round124 candidate once without changing any weights,
offsets, ROI handling, or checkpoint selection. The package is diagnostic and
must be compared with the previous personal best using external platform evidence.

Fixed package: five equal fold members of `encoderrelationalrobust15`,
context128, D4, JPEG q95, output source offset `[0,-1]`, CD68 only. The package
contains exactly the 1,346 test DAPI filenames under `results/test/CD68/`.
The script binds the Round124 OOF and gate hashes, checks all five candidate
checkpoint hashes, verifies the prediction report, and runs ZIP CRC validation.
It refuses to overwrite any existing prediction, report, ZIP or status file.

Only GPU1 is used. This is inference and packaging only; no training follows
from the Round124 gate failure. The previous personal-best submission is untouched.

The remote preflight matched protocol SHA-256
`cd430f9f3a5382aaf60545942aea821d9ddd67c9c3a8dfc6f6446395b351ba19`, OOF SHA-256
`bab13d29cef41e96244543142a6d9ec11de0f7affc308829b413fbb381ee3393`, and gate
SHA-256 `24ee940fe0dd8a4838aa7ee753fec2c360ad44a132174cb215e8fcb67fd14bf7`.
No output directory, report, ZIP or status marker existed before launch. Screen
`round124_submission` was detached on 2026-09-13 with GPU1 free and 40.4 GiB
free memory. The task will wait for a genuinely empty GPU1, run test inference,
package the ZIP, test its CRC, and write `reports/round124_submission.exit`.

## Completion

The task exited `0`. It generated 1,346 predictions in 1,719.684 seconds and
validated every prediction as a 256x256 RGB JPEG with equal channels. The ZIP
contains exactly 1,346 entries under `results/test/CD68/`; `zipfile.testzip()`
returned no error. Independent local verification matched the remote package
byte-for-byte:

- file: `submissions/cd68_round124_cv5_relational_q95_plain.zip`;
- size: `11,923,090` bytes;
- SHA-256: `44ce2f04b41f4e351d9a25a9edabc98eb6e44ebddc6f184024a7d39e05fbd089`;
- prediction report: `reports/round124_cv5_relational_test_prediction.json`;
- exit marker: `reports/round124_submission.exit` = `0`.

The package is ready for one manual platform confirmation. Record the returned
score separately; do not use it to tune weights, offsets or ROI-specific rules.

## Platform result (user reported, 2026-09-13)

The package received `74.1630`, with displayed `SSIM=0.832` and `PSNR=26.503`.
This is `+0.1315` over the earlier Round88 context128 package (`74.0315`) and
became the verified personal best at that point, but remains below the recorded
external leaderboard high `75.4966` by `1.3336`. The hidden result is consistent
with a modest transferable benefit from the relational representation, while
the OOF absolute score (`75.506276`) overstates the attainable platform score.
No exact hidden baseline for the Round124 matched auxiliary15 candidate exists,
so an OOF-to-hidden gain ratio must not be inferred from this one submission.

Round124 therefore became the primary personal-best package at that point; it
was later superseded by Round127 and Round128. Do not tune the relation weight
or fold4 from this score, and do not infer an OOF-to-platform calibration factor
without a matched hidden baseline.

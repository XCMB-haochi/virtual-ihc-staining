# Round 88: Fixed Context128 Diagnostic Submission

## Purpose

Round85 completed a five-fold, context128 inference-only evaluation using the
unchanged 25-member endpoint. Its JPEG-q95 OOF gain was +0.026427, its float
gain was +0.026644, and every fold improved, but only 18/25 ROIs improved; the
preregistered distribution gate therefore failed. Rounds86--87 separately
showed that retraining with context128 is negative and is closed.

This round is one fixed preliminary-stage confirmation package, retained only
because the aggregate and fold evidence from Round85 is positive and the
platform currently permits repeated preliminary submissions. It does not
change any checkpoint, weight, TTA, output offset, or context setting, and its
result must not be used to tune another candidate. It is not a promoted route
and is not evidence for later-stage data.

## Fixed recipe

The script tools/run_round88_context128_diagnostic_submission.sh binds the
immutable Round85 OOF and gate SHA-256 values, all 25 checkpoint hashes, and
the exact protocol configs/round85_cv5_context128_inference.json. It emits
only CD68, uses context halo 128, D4 TTA, JPEG quality 95, the fixed output
source offset [0,-1], and the fixed Base-split weights
0.21/0.09/0.35/0.175/0.175. It expects all 1,346 test DAPI inputs and
validates the final ZIP layout and CRC.

The detached runner samples both physical GPUs and waits for one with low
sustained utilization and sufficient free memory. It does not stop or alter
other users' processes. The package is a diagnostic confirmation of a
predeclared endpoint, not a leaderboard-driven search.

## Execution and result

The script was fast-forwarded to the storage-scoped server checkout and passed
the remote preflight. It was launched in detached Screen session
round88_context128_diagnostic at 2026-08-30 14:13:06 (Asia/Shanghai). The
scheduler sampled both GPUs every five minutes, requiring three low-utilization
samples (maximum 20 percent) and at least 24,000 MiB free memory. GPU1 met the
condition at 14:44:12; GPU0 remained contended. No unrelated process was
interrupted.

The prediction process completed with exit marker `reports/round88_submission.exit`
equal to `0`. The fixed endpoint produced 1,346 CD68 images from 25 members in
`10,006.807848439` seconds (`2 h 46 min 47 s`). The prediction report records
context halo `128`, D4 TTA, JPEG quality `95`, output source offset `[0,-1]`,
and member-weight sum `1.0`. Its SHA-256 is
`1a9bb26981e7f99fe39bd2ddb73a0684ae07f6511db556ec8557089546e873d0`.

The package is
`submissions/cd68_round88_cv5_context128_diagnostic_q95_plain.zip`,
11,740,002 bytes, with SHA-256
`68acbee0c23319897fa32d99c2c39a421c6ffd6ab92d98353aca0e948912777f`.
An independent audit found 1,346/1,346 expected filenames, 256x256 RGB
outputs with equal channels, 1,346 ZIP entries under
`results/test/CD68/`, and a successful ZIP CRC check. The package is ready for
manual platform submission; no leaderboard result is inferred until the
organizer's scorer returns one.

This remains a confirmatory diagnostic, not a promoted model: Round85's OOF
gate was positive in aggregate and all five folds but failed its 18/25 positive
ROI requirement, while Rounds86--87 closed context128 supervised training.
Do not tune weights or offsets from a platform score for this package.

## Platform result

The preliminary platform returned `74.0315`, with displayed `SSIM=0.831` and
`PSNR=26.435`. Relative to the Round82 context64 package (`74.0101`), the
fixed context128 inference view gains `+0.0214`. Relative to the Round85
context128 OOF q95 gain of `+0.026427`, the observed hidden transfer is about
`81.0%`. This is a small positive confirmation of the inference mechanism, not
evidence that the failed context128 training branch should be reopened.

The result remains below the current leaderboard high reported by the user;
no leaderboard-derived adjustment is authorized. The next controlled action is
Round89, which tests official per-image checkpoint selection on the same fixed
endpoint without training.

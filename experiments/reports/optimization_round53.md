# Round 53: Bounded Official-FCMAE Weight Surface

## Motivation

Round52 promoted a fixed 30% scratch / 30% official-initialized FCMAE split,
improving the retained Round47 ensemble by `+0.205827` OOF q95. The old
scratch-only surface had favored a total FCMAE weight near 60%, but the new
official-initialized member changed both accuracy and error diversity. Round53
therefore evaluates one bounded, preregistered surface without training or
leaderboard feedback.

## Fixed Protocol

`configs/round53_cv5_official_fcmae_weight_surface.json` reuses the exact 20
approved Round52 checkpoints. The grid crosses total FCMAE weight
`50/60/70%` with official-initialized share `1/3, 1/2, 2/3`; the remaining
weight retains the established 70:30 base32/base48 ratio. The Round52 equal
split at 60% total weight is the baseline.

Promotion requires at least `+0.02` JPEG-q95 OOF gain, positive float gain,
4/5 positive folds, 20/25 positive ROIs, and a worst ROI delta of at least
`-0.04`. Candidate priority was fixed before evaluation, with the nearest
60%-total alternatives considered before wider total-weight changes.

## Completion Audit

The detached pipeline exited zero and evaluated all 6,296 images from all 25
ROIs in `3,985.351 s`. Its Round52 baseline is exactly identical at the
overall, fold, ROI, and checkpoint-provenance levels. All 20 distinct
checkpoint files were independently rehashed and match the OOF report.

- OOF SHA-256: `7a859f3f2eb7add5de1f46918a1452511ea5982cf02b585690005e06df9ee002`;
- gate SHA-256: `5d358788363aa6992f8a4242a12cac956771b0dea0e9271cc654e546f81c5047`;
- the gate binds the exact OOF SHA-256.

## Results

| Candidate | JPEG-q95 OOF | Delta vs t60_o50 | Gate result |
| --- | ---: | ---: | --- |
| `t70_o67` | 75.568790 | +0.038948 | Reject: ROI floor |
| `t70_o50` | 75.560424 | +0.030582 | Pass |
| `t60_o67` | 75.543818 | +0.013975 | Reject |
| `t60_o50` | 75.529842 | baseline | Retained baseline |
| `t70_o33` | 75.515319 | -0.014524 | Reject |
| `t50_o67` | 75.493376 | -0.036466 | Reject |
| `t60_o33` | 75.488582 | -0.041260 | Reject |
| `t50_o50` | 75.476115 | -0.053728 | Reject |
| `t50_o33` | 75.439619 | -0.090224 | Reject |

`t70_o50` improves all five folds by `+0.042499`, `+0.014105`, `+0.042347`,
`+0.021388`, and `+0.028952`. Twenty-two of 25 ROIs improve. The three
non-positive ROIs are ROI004 `-0.002997`, ROI009 `-0.016835`, and ROI023
`-0.012545`, all above the fixed floor. Its q95 components are SSIM
`0.845704` and PSNR `27.268554`; float score is `75.638565`.

Although `t70_o67` has another `+0.008366` mean OOF advantage, ROI023 drops
`-0.121698`, so it fails the preregistered distribution floor. Selecting it
would override the stability rule after seeing results and is therefore not
allowed.

## Decision

Promote `t70_o50`: base32 `0.21`, base48 `0.09`, scratch FCMAE800 `0.35`,
and official-initialized FCMAE800 `0.35`. This is a small but independently
measured improvement over Round52. These weights remain Colon-specific and
must be re-evaluated when later-stage organ data arrive.

## Submission Artifact

`tools/run_round53_submission_pipeline.sh` binds the exact approved OOF and
gate hashes before allowing test inference. The fixed 20-member prediction
completed in `2,559.959 s` and produced all 1,346 expected JPEGs.

Independent post-run validation confirmed 20/20 checkpoint hashes, exact
member weights, unique input/prediction filename coverage, RGB 256x256 images
with identical grayscale channels, and byte-for-byte equality between each
prediction and its ZIP payload. The archive contains exactly 1,346 files
under `results/test/CD68/` and passes its CRC test.

- prediction report SHA-256:
  `bb22914b2554e5ffd06ee106e9b0eec71ffb17727b8d08cf3fe3cc8502498390`;
- submission size: `11,390,002` bytes;
- submission SHA-256:
  `de031b11caa61baa26c0a5d21ae6e86e502ca0885ac84aa066206d9fb325ab28`.

The package scored `73.9028` on the preliminary platform, with displayed
`SSIM=0.830` and `PSNR=26.350`. Relative to Round47's `73.6745`, the platform
gain is `+0.2283`; the corresponding complete-OOF gain is `+0.236409`, so
`96.57%` of the measured OOF increment transferred. The OOF-based point
estimate was only `0.0081` above the realized score. This close transfer
supports the grouped-OOF selection procedure, but the last user-reported
leader score of `75.4966` would still leave a `1.5938` gap and has not been
rechecked in this round.

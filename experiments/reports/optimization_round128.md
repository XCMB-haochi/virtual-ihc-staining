# Round128: Fixed 50/50 Relational Seed Blend OOF

Round127 seed2027 produced a hidden score of `74.2222`, higher than Round124
seed2044's `74.1630`, and its two-fold OOF gate passed strongly. The complete
Round127 gate later missed only the worst-ROI floor because ROI013 fell
`-0.252999`; all five folds were positive. Discarding the relation route would
therefore throw away a reproducible aggregate signal.

This round evaluates one fixed diversity hypothesis: average the corresponding
Round124 seed2044 and Round127 seed2027 predictions at `50/50`. The ratio is
chosen before this OOF and is not fitted to ROI013, the hidden scores, or any
test statistic. Both members use the same model, conditioning, context128+D4,
JPEG q95 and output offset `[0,-1]`; only the supervised seed and resulting
checkpoint differ. No training is performed.

The complete five-fold OOF reports both direct seeds and the fixed blend in one
shared pass. The prospective blend gate requires nonnegative q95 gain versus
seed2027, positive float, at least 4/5 positive folds, at least 18/25 positive
ROIs, and a worst-ROI delta no lower than `-0.12`. A gate failure closes this
blend packaging check; it does not invalidate the already observed seed2027
hidden benefit. No adjacent blend ratios, ROI routing, calibration or
leaderboard probes are allowed.

Only GPU1 is eligible. Expected cost is inference-only, roughly 45-90 minutes;
the then-current personal-best Round127 package and Round124 package remain intact.

The protocol passed local and remote loader validation, and all ten source
checkpoint hashes were checked before launch. The OOF job is detached as Screen
`round128_seed_blend_oof` on GPU1 only. It waits for GPU1 utilization to fall
below 30% with at least 12 GiB free, then runs the shared five-fold OOF and the
fixed blend gate. GPU0 is excluded.

The blend OOF completed with exit `0`. Against seed2027, the fixed blend gains
q95 `+0.102593`, is positive on all five folds and 21/25 ROIs, but its worst
ROI016 delta is `-0.452199`, so the prospective variance-reduction gate fails.
Against the original auxiliary15 reference, the same blend is approximately
`+0.316922` overall q95, 24/25 ROIs positive, and all folds positive; this makes
it a useful fixed diagnostic despite the failed risk gate. No adjacent ratios
or ROI rescue are opened. One separate blend package confirmation is permitted
because the ratio was fixed before OOF and preliminary submissions are still
being used for confirmation; the current best and seed2027 packages remain
untouched.

The separate package script passed shell syntax and remote preflight and is
detached as Screen `round128_seed_blend_submission` on GPU1. It waits for an
empty GPU1, then writes a new ZIP without touching earlier packages.

## Submission package completion

The package task exited `0`. It generated the fixed ten-member, equal-seed
ensemble over all 1,346 test inputs in 3,407.584 seconds. Remote validation and
a second local audit confirmed 1,346 unique 256x256 RGB JPEGs with equal
channels, exact `results/test/CD68/` archive layout, member-weight sum `1.0`,
and a clean ZIP CRC check.

- file: `submissions/cd68_round128_relational_seed_blend_q95_plain.zip`;
- size: `11,937,352` bytes;
- SHA-256: `18e638a23f280277a38b948f5724ca13ee6e4f53ae28f38b968981ded047da95`;
- prediction report: `reports/round128_relational_seed_blend_test_prediction.json`;
- exit marker: `reports/round128_submission.exit` = `0`.

This package is ready for one manual platform confirmation. Its result decides
whether the fixed seed blend is retained; do not add seeds or search ratios
before that external evidence is available.

## Platform result and current status

The user reported `74.2362`, with displayed `SSIM=0.833` and `PSNR=26.519`.
This is `+0.0140` over Round127, `+0.0732` over Round124, and `+0.2047` over
Round88. Round128 is therefore the current verified personal-best submission.
The recorded external leaderboard high is `75.4966`, not a project submission;
the remaining leaderboard gap is `1.2604`.

The failed risk gate remains useful as a warning about ROI variability, but it
does not revoke the observed personal-best platform result. Keep Round128 as
the primary package. Do not search adjacent blend ratios from the small hidden
increment over Round127.

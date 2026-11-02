# Round94: Texture Thumbnail Screen

## Purpose

Round93 completed a fixed full-label endpoint diagnostic, but it cannot provide
honest held-out OOF evidence. Round94 therefore tests one portable structural
change on the same organ: retain the existing 5x5 DAPI neighborhood and add
fixed per-block contrast and gradient maps to the mean/presence thumbnail.
The main UNeXt2-FCMAE-Tiny encoder, loss, schedule, D4 aggregation, output
offset, and JPEG protocol remain unchanged.

## Pre-registered Protocol

`configs/round94_cv2_unext2_texture_thumbnail.json` uses the fixed Round55
official auxiliary model as its two-fold baseline (`fold0` and `fold2`). The
candidate uses four thumbnail channels:

- DAPI block mean;
- DAPI block standard deviation;
- DAPI block mean absolute gradient;
- neighborhood presence.

The maps use 16x16 non-overlapping source blocks, a 5x5 neighborhood, fixed
uint8/255 normalization, D4-invariant mean aggregation, context128 inference,
fixed `[0,-1]` output-source offset, and q95 JPEG evaluation. Candidate
training is fresh at the supervised stage and initializes only the compatible
fold-local official FCMAE encoder; it does not continue a mature supervised
checkpoint. The two-fold gate requires a q95 gain of at least `0.02`, positive
float gain, both folds positive, no more than two jointly negative test-like
ROIs, and a worst ROI delta of at least `-0.15`. No leaderboard result is used
to alter these settings.

## Launch Audit

The implementation was committed as `979d0a5` and pushed to the repository.
Local and remote targeted suites passed (`174 passed, 57 subtests passed`). The
remote protocol and fixed-baseline preflight passed, including the Round55 OOF
and gate hashes and fold-local FCMAE encoder hashes:

- fold0: `50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9`;
- fold2: `c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7`.

At launch on 2026-08-31, GPU0 had sustained 82--100% compute utilization,
whereas GPU1 was idle with about 40 GB free. The scheduler consequently
selected GPU1 and retained live utilization checks; it will use GPU0 only if
the measured straggler tradeoff becomes favorable. The detached Screen session
is `round94_texture`; the pipeline status file is
`reports/round94_pipeline.exit` and is intentionally absent while running.

## Recovery after first attempt

The first detached attempt did not satisfy the completion contract. Fold0
completed both stages, but fold2 was launched on GPU0 after a low-utilization
sample and later received `SIGKILL` during stage1 after epoch 6. Its log has no
CUDA out-of-memory exception; the available stage1 artifacts are preserved at
`/workspace/virtual_ihc_staining_generation/transfers/round94-fold2-stage1-interrupted-sigkill/`.
The pipeline consequently wrote `reports/round94_pipeline.exit=1`, produced no
OOF or gate report, and the partial fold cannot be used for comparison.

After confirming GPU1 was idle, the exact same protocol, seed, encoder hash,
and output paths were relaunched for fold2 in Screen `round94_fold2_retry` at
2026-08-31 16:04:32 (Asia/Shanghai), using physical GPU1. The retry is a
resource-recovery rerun, not a change to the registered texture hypothesis.

The retry completed both fold2 stages successfully. Fold0 and fold2 each now
contain exactly 14 metric rows (`epoch` 1--14), `best_global_mse.pt`,
`best_mean_image.pt`, `best.pt`, and `last.pt`; both fold status files report
their two stages complete. The selected `best_global_mse.pt` hashes are:

- fold0 stage1: `8cb837bb06254cd209b0bf5270597b4f1758c0fc040e09a7f2b0003e3649588f`;
- fold0 stage2: `0d1ce37609ca19225a75065bc7c245e78d373b87247ef0294429989b516ea6c3`;
- fold2 stage1: `ff5df4f3d163b11ce969eaf9bf286176c7d06dcb5c361ad83739eeafbd1b699c`;
- fold2 stage2: `c126199f46795a98e888e8238a4238c7e34b7cdf4aab89ba02c7f481b606c1bc`.

The fold-local FCMAE initialization hashes remain the registered values
`50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9` (fold0)
and `c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7`
(fold2). The original `round94_pipeline.exit=1` records only the interrupted
first attempt; it is not overwritten. A separate fixed recovery evaluator was
launched in Screen `round94_eval` on GPU1 at 2026-08-31 18:33:59. It completed
with `reports/round94_recovery_eval.exit=0`, producing 2,706 OOF images across
the ten screen ROIs. The OOF SHA-256 is
`5b49c94a5b32475bda4897e6ec228ad3788acd390346b65251cb5ab70beb7cc5`, and the
gate SHA-256 is
`9f4339edb983952b1769cc6fb4a6d03a511b70ad8c862592d5d50819428d6d9f`.

The texture mechanism gate passed. Relative to the fixed
`officialauxiliary15_context128` baseline, overall q95 deltas are
`+0.074746/+0.080017` (mean-image/global-MSE), test-like deltas are
`+0.117814/+0.123330`, and float deltas are positive. Both folds are positive;
the fold deltas are `+0.002875/+0.006504` (fold0) and
`+0.142090/+0.148915` (fold2). The four test-like ROIs are all positive, with
the minimum at `+0.058338`; the full ten-ROI diagnostic also records negative
ROI001, ROI011, and ROI014 deltas, so no ROI-specific rescue is allowed.

This is sufficient for the pre-registered next step: one unchanged complete
five-fold expansion. The two existing screen candidate folds will be reused
unchanged, and only folds1, 3, and 4 will be trained under a new five-fold
protocol. No weight, epoch, calibration, test submission, or leaderboard probe
is authorized by this screen result alone.

## Interpretation Boundary

This is a two-fold mechanism screen, not a submission candidate. After the
user reports completion, audit all four stage runs, checkpoint hashes, OOF
coverage, and gate JSON. A failed gate closes the texture branch. A passing
gate permits one unchanged five-fold expansion only after reviewing per-fold
and per-ROI deltas; it does not authorize a weight or epoch search.

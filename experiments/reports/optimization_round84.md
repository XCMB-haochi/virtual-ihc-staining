# Round 84: Independent Official FCMAE Seed

## Motivation

Round83 closed the registration-aligned-target training branch. The strongest
remaining portable diversity mechanism with direct evidence is an independently
seeded supervised trajectory: earlier matched seed ensembles produced useful
complementarity, while this screen keeps the architecture and initialization
fixed. It changes only the supervised random seed from `2044` to `2027`.

## Matched Screen

`configs/round84_cv2_official_fcmae_independent_seed.json` keeps the Round52
official ConvNeXtV2 FCMAE-Tiny encoder, halo64 source/target context, thumbnail
FiLM, Uniform-7 hybrid loss, D4, JPEG q95, and `14+14` downstream epochs. The
fold0 and fold2 encoder files are hard-linked from the audited Round52
pretraining outputs and are hash-checked before training. The new downstream
trajectory uses seed `2027`; no new FCMAE pretraining is run. All candidates
use the already validated fixed output source offset `[0,-1]`.

The screen compares the independent model directly with the shifted official
seed-2044 model, then tests fixed replacement and half-split blends in the
current Base-diversity endpoint. Thresholds are fixed before execution: `+0.05`
q95 for the direct mechanism and `+0.04` for each endpoint blend, positive
float and both folds, at least 7/10 or 8/10 positive ROIs, and ROI floors of
`-0.12` or `-0.08`. A passing blend would expand unchanged to five folds before
any test inference. The result is intended as transferable seed-diversity
evidence, not a leaderboard-tuned weight search.

## Execution and Result

The protocol and launcher passed Linux compilation, shell syntax, the complete
test suite, exact dry-run, source-hash preflight, and a real-data smoke. The
formal detached Screen used GPU1 for fold0 while GPU0 was initially occupied;
when GPU0 became eligible, fold2 was assigned there. Fold0 ran from
`14:09:10` to `15:34:40` and fold2 from `15:14:40` to `16:49:40` on
2026-08-29. The scheduler then evaluated OOF on GPU0 after a fresh resource
check. The pipeline exit marker is `0`; the Screen session is gone.

Each of the four stage directories has exactly 14 consecutive metric rows and
a nonempty `best_global_mse.pt`. Stage-1 selected epochs are 12 and 14 for
fold0/fold2; stage-2 selected epochs are 5 and 14. The fold-local FCMAE
initialization hashes are `50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9`
and `c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7`.
The selected stage-2 checkpoint hashes are
`0f87ec6af41cf28e5423099739b1e0278b8d78048ceed885c02b333481b97cbd` and
`3f7411e711f642d9dd7df1d043fd7093faf03b1ebe3676a930085094a02e3a50`.

OOF covers all 2,706 images from the fixed ten ROIs and took `3611.846 s`.
At JPEG q95 the independent seed alone scores `74.992777` versus `74.959362`
for the shifted official seed-2044 model (`+0.033415`); its float gain is
`+0.032830`. Fold gains are `-0.031144/+0.093908`, only 5/10 ROIs improve,
and the worst ROI is `-0.482496`, so the direct mechanism gate fails.

In the current Base-diversity endpoint, replacing the official member gains
`+0.019183` q95 and splitting its weight gains `+0.014583`. Both blend gains
are positive on both screen folds and satisfy their ROI floors, but both miss
the preregistered `+0.04` threshold and improve only 7/10 ROIs. The gate is
therefore `promotion.allowed=false`; no five-fold expansion, test inference,
submission, or nearby seed/weight search follows.

The recurring `pymp-*` NFS temporary-directory cleanup tracebacks appear only
after each successful training completion and do not change the zero pipeline
exit or artifact checks. Formal artifacts are
`round84_cv2_official_fcmae_independent_seed_oof.json`,
`round84_cv2_official_fcmae_independent_seed_gate.json`, and
`round84_scheduler.log`. Their SHA-256 values are respectively
`b6c47d91e10518a83fe04e4a4c1f176dfc7cfdccdab9dc4e6907b2cd14340665` and
`6071c2a51481ec92be323f632903c704ef5266fd3542b508cb408827f67c3549`.

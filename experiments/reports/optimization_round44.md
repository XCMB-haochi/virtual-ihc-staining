# Optimization Round 44: UNeXt2 and fold-local DAPI FCMAE

Date: 2026-07-23

## Motivation

Round43's fixed independent-seed ensemble scored `73.3050`, only `+0.0334`
above Round35. The known leader is `75.4966`, leaving a `2.1916` gap. This
closes additional homogeneous seed stacking, nearby blend weights, calibration,
and leaderboard-guided routing as useful next steps.

The next experiment changes representation learning rather than continuing the
same U-Net trajectory. The [Cytoland paper](https://www.nature.com/articles/s42256-025-01046-2)
uses an asymmetric U-Net with ConvNeXt V2 blocks and fold-transferable FCMAE
pretraining for virtual staining. Its official
[VisCy repository](https://github.com/mehta-lab/VisCy) is BSD-3-Clause. The
project-local implementation also follows the dense masking rules in the
official [ConvNeXt V2 repository](https://github.com/facebookresearch/ConvNeXt-V2)
and introduces no runtime dependency beyond PyTorch.

Round15 already tested an ImageNet-1K ConvNeXt-Tiny encoder with a generic
decoder. It improved the aggregate H2 score slightly but regressed ROI024 by
about `0.19-0.28`. Round44 is not a retry of that recipe: it uses ConvNeXt V2
GRN blocks in both encoder and decoder, a stride-2 microscopy translation stem,
pixel-shuffle reconstruction, the mature halo64 and thumbnail condition, and
self-supervision on official DAPI images.

## Fixed model and pretraining

The architecture is `unext2-fcmae-tiny`:

- four ConvNeXt V2 encoder stages with blocks `3/3/9/3` and channels
  `96/192/384/768`;
- a non-overlapping stride-2 DAPI stem and total encoder stride 16;
- three pixel-shuffle decoder stages with two ConvNeXt V2 blocks per stage;
- the existing D4-invariant spatial-thumbnail FiLM at the bottleneck;
- a sigmoid CD68 output with a trainable channel bias;
- context64 input and full halo64 supervision, matching the mature pipeline.

Each validation fold gets its own DAPI FCMAE checkpoint. Pretraining excludes
all DAPI patches from that fold's validation ROIs, so no held-out input or label
is used. The fixed pretraining recipe is 100 epochs, batch 32, mask ratio 0.5,
mask patch 16, AdamW at `2e-4`, ten warmup epochs, cosine decay to `1e-6`, and
weight decay `0.05`. Only encoder tensors transfer. The downstream checkpoint
contains the complete model and never needs the FCMAE artifact during restore
or inference.

The Cytoland publication used 800 FCMAE epochs for its 2D experiments. The
100-epoch budget here is a resource-bounded screen, so a negative result rejects
this fixed competition recipe but does not prove that arbitrarily longer
pretraining is ineffective.

## Matched two-fold screen

The machine-readable protocol is
`configs/round44_cv2_unext2_fcmae.json`. Fold0 and fold2 were fixed before any
Round44 result and cover ten standard CV ROIs. Both downstream candidates use
the same seed, data order, loss, augmentation, conditioning, effective batch 16,
and `14+14` epoch schedule:

| Candidate | Initialization | Downstream training |
|---|---|---:|
| `unext2scratch` | random | `14+14` epochs |
| `unext2fcmae` | fold-local DAPI FCMAE encoder | `14+14` epochs |

Round35 remains the deployment baseline. Two fixed 20% diversity blends are
evaluated without a weight search:

- `56% base32 + 24% base48 + 20% unext2scratch`;
- `56% base32 + 24% base48 + 20% unext2fcmae`.

All formal comparisons use complete fold0/fold2 OOF, halo64, D4, and JPEG q95.
Test-like subsets, test DAPI statistics, leaderboard feedback, output
calibration, per-ROI routing, and manual prediction selection are disabled.

## Preregistered promotion

FCMAE first has to beat matched scratch by at least `+0.05` q95 overall, remain
positive in float, improve both folds and at least 7/10 ROIs, and keep the worst
ROI above `-0.12`. Pretrained candidates depend on this mechanism gate.

A direct UNeXt2 model may replace Round35 only with at least `+0.20` q95; a
fixed 20% blend may replace it with at least `+0.08`. Either route additionally
requires positive float, 2/2 positive folds, at least 7/10 positive ROIs, and a
worst-ROI floor of `-0.12` direct or `-0.08` blended. Promotion priority is
pretrained direct, pretrained blend, scratch direct, then scratch blend. The
first passing gate is the only candidate eligible for five-fold expansion.

No Round44 test prediction or submission package is allowed from this two-fold
screen. A passing route must first be repeated across all five folds.

## Implementation status

The model, fold-local resumable pretrainer, strict encoder transfer, CV runner,
protocol, distribution gates, and detached pipeline are implemented. Targeted
local tests pass `56/56`. The complete locally runnable suite passes
`225 passed, 1 skipped, 2 deselected`; the two deselections are the existing
Windows Anaconda/MKL `lstsq` abort cases. Python compilation, protocol
validation, both dry-runs, and `git diff --check` pass. Linux regression passes
`228 tests` plus `71 subtests`.

GPU0 smoke tests confirm FCMAE batch 32 at 256x256 peaks at `22.28 GiB` and the
complete conditioned downstream path at 384x384 works. A no-write throughput
benchmark measured batch 4/8/16 peaks of `6.74/12.91/25.27 GiB`; batch 16 gives
the best throughput while retaining about 9 GiB total-card headroom under the
current fixed GPU0 load. The formal effective batch is therefore fixed at 16
before any Round44 OOF result.

The formal pipeline started at 2026-07-23 17:11 CST from runtime commit
`266f219` in detached Screen `round44_unext2`, using physical GPU0 only. The
first fold-local FCMAE stage reached 99% GPU utilization with about 28.9 GiB
total card memory in use, consistent with the smoke-test budget. Training,
OOF, and gate evaluation completed normally. The status records all eight
downstream stages complete, both fold-local pretrainers reached epoch 100, and
the OOF report hash is
`6a50062c34fb05ad01eac49fbd9a908ba2d6ebf47993f16e1e90a2148bbd7601`.

## Two-fold result

| Candidate | q95 score | Float score | Delta vs. comparison |
|---|---:|---:|---:|
| Round35 | 74.416270 | 74.493883 | - |
| UNeXt2 scratch | 73.455490 | 73.527544 | -0.960780 q95 vs. Round35 |
| UNeXt2 FCMAE | 74.216594 | 74.291676 | +0.761103 q95 vs. scratch |
| Round35 + 20% scratch | 74.463107 | 74.539716 | +0.046837 q95 vs. Round35 |
| Round35 + 20% FCMAE | 74.571356 | 74.648579 | +0.155086 q95 vs. Round35 |

The fresh architecture is not a direct replacement for Round35. Its useful
effect is structural diversity: FCMAE recovers most of the scratch deficit and
the fixed 20% blend improves both evaluated folds and all 10/10 ROIs, with a
worst ROI gain of `+0.012287`. The preregistered pretrained-ensemble gate
therefore passes. The direct pretrained model, scratch model, and scratch blend
all fail their respective gates.

## Five-fold expansion

The passing route expands without changing the model, seed, schedule,
checkpoint selector, inference, or 20% weight. The extension protocol is
`configs/round44_cv5_unext2_fcmae_promotion.json`. It deliberately retains the
Round44 run namespace so the completed fold0/fold2 artifacts are reused exactly
and trains only fold1/fold3/fold4.

Before observing those three folds, the final promotion gate is fixed at
least `+0.08` complete q95, positive float, at least 4/5 positive folds, at
least 17/25 positive ROIs, and a worst-ROI q95 delta no lower than `-0.08`.
The pipeline verifies the original two-fold gate before starting. It stops
after full OOF and the machine gate; test inference remains disallowed until
that final gate passes.

The five-fold extension started at 2026-07-24 09:43 CST from runtime commit
`c031633` in detached Screen `round44_cv5_fcmae`, using physical GPU0 only.
The first launch used a login shell and exited before training because that
shell replaced the storage-scoped Python environment; it created no checkpoint.
The corrected non-login Screen preserves Python 3.12, PyTorch 2.8, and CUDA
12.8, and the first FCMAE stage allocated about 25.6 GiB total GPU memory.
All five fold-local FCMAE runs reached epoch 100 and all six newly required
downstream stages completed. The full OOF evaluation covers all 6,296 images
from all 25 ROIs.

## Five-fold result

| Candidate | q95 score | Float score | q95 delta vs. Round35 |
|---|---:|---:|---:|
| Round35 | 74.869213 | 74.948397 | - |
| UNeXt2 FCMAE | 74.622352 | 74.697753 | -0.246861 |
| Round35 + 20% FCMAE | 75.009599 | 75.087934 | +0.140386 |

The direct UNeXt2 model remains weaker than Round35, but the fixed blend is
positive in all 5 folds and 24/25 ROIs. Its only negative ROI is ROI016 at
`-0.050022`; the best ROI gain is `+0.302753`. The complete-distribution gate
passes every check: q95 `+0.140386 >= +0.08`, positive float `+0.139537`,
5/5 positive folds, 24/25 positive ROIs, and a worst-ROI change above `-0.08`.
The approved OOF report SHA-256 is
`5cb9cbb7bec92cda07868f066b683921f8cf0007f7bbf8e432a9bb8553894025`.

The gate promotes only `round35_fcmae20`. Test inference and packaging are
implemented by `tools/run_round44_submission_pipeline.sh`, which verifies both
the gate decision and the approved OOF hash before loading the fixed 15-member
ensemble. No leaderboard feedback, calibration, ROI routing, or adjacent
blend-weight search is used.

## Test prediction and package

The promoted candidate was run on all 1,346 test DAPI images with the fixed
15-member ensemble, context halo 64, D4 TTA, and JPEG quality 95. Inference
and packaging completed in `1623.926 s`; the package validator found 1,346
RGB images with no missing, extra, size, channel, or CRC errors.

- Prediction report:
  `reports/round44_cv5_round35_fcmae20_test_prediction.json`
- Submission package:
  `submissions/cd68_round44_cv5_round35_fcmae20_context64_d4_raw_q95_plain.zip`
- Package size: `11,511,546` bytes
- Prediction report SHA-256:
  `8776be6ab73169165bda842ba5332a28fa84769ee7f2d3e11c32511e300036b3`
- Package SHA-256:
  `c12618ce4151b8f31694f50fa9196798a3adba707a5a6ebe64412ac441043bf7`

## Platform result

The platform scored the fixed package `73.3871`, displaying
`SSIM=0.825` and `PSNR=26.005`. This is `+0.0821` over Round43's `73.3050`
and the current personal best. The hidden-set gain is about 58% of the full
OOF q95 gain (`+0.0821 / +0.140386`). The direction is therefore validated,
but the conversion is not strong enough to justify post-hoc blend-weight
search. The previously reported leaderboard high of `75.4966` remains
`2.1095` ahead.

# Round 59: Update-Matched Labelled-ROI Scale Study

## Motivation

Round58 rejects longer official-initialized Tiny FCMAE pretraining: the
reconstruction loss continues to fall through epoch 1200, but downstream q95
falls `0.077758` versus the epoch-800 model. Round54, Round56, and Round57 also
close public supervised-22K initialization and both asymmetric auxiliary-loss
schedules. Continuing to change nearby epochs or blend weights has weak
mechanistic support.

The remaining evidence instead points to data efficiency. Spatial context and
fold-local microscopy FCMAE are the only mechanisms that produced large stable
gains, while the complete-OOF auxiliary-marker experiment retained a small
positive effect. Before implementing another multi-marker architecture,
Round59 measures whether supervised performance is still rising materially at
the current 20 training ROIs.

## Fixed Design

`configs/round59_fold0_label_scale.json` fixes fold0 validation to the same five
ROIs used by Round52. The full-data reference is the unchanged Round52
official-FCMAE800 checkpoint, SHA-256
`d1c391350c82d3a094d28040a60836311bf65e5b910647bee2d6a913fd4d9ec0`.
Every subset model starts fresh from the same fold0 FCMAE800 encoder, SHA-256
`50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9`.

The 20 eligible training ROIs are sorted by patch count and divided into five
consecutive four-ROI strata. Draws `a`, `b`, and `c` rotate the selection within
each stratum. Within each draw, the 5-ROI set is a strict subset of the 10-ROI
set, which is a strict subset of the 15-ROI set. This separates scale from one
arbitrary ROI composition while retaining comparable patch-count coverage.

The 20-ROI reference uses 14 epochs per supervised stage. Subset epochs are 19
for 15 ROIs, 28 for 10 ROIs, and 56 for 5 ROIs. With batch 16, the fold0 plans
span 4,200--4,522 optimizer updates per stage versus 4,368 for the reference;
the maximum mismatch is below 4%. To avoid giving smaller subsets more model
selection opportunities merely because they have more epochs, every subset is
validated and checkpoint-selected at 14 evenly spaced training-progress
points. Diagnostic runs retain only last and best checkpoints rather than
per-epoch weights. Both supervised stages use the unchanged learning rates,
seed, loss, halo64 thumbnail FiLM, global-MSE checkpoint selection, and
official-FCMAE800 encoder. Evaluation uses the complete fold0 validation set,
D4 TTA, and JPEG q95 official scoring.

This is a diagnostic, not model promotion. It uses no test image, leaderboard
feedback, calibration, or post-result subset selection. No subset checkpoint
can replace the deployed model.

## Decision Rule

The first phase trains all three nested draws on fold0 and reports the mean,
population standard deviation, and range at each ROI count.

- If at least two of three draw curves are strictly increasing from 5 to 20
  ROIs and full20 exceeds the mean 15-ROI score by at least `0.08`, labelled-data
  efficiency remains a material bottleneck. The next score experiment should
  improve use of available labels, led by marker-specific decoding or gradient
  isolation for the positive Round55 auxiliary signal.
- If at most one curve is monotonic or the full20-minus-mean15 gap is below
  `0.03`, labelled-data scale is not a useful explanation at the current end of
  the curve. The next experiment should change the representation or objective
  rather than add regularization around the same supervised model.
- Any intermediate result is ambiguous and triggers the same frozen study on
  fold2 before choosing a structural direction.

The study does not authorize test inference or a platform submission.

## Status

Implementation is committed through `2335fb2` and synchronized to the GPU
server. Local protocol and summary tests pass. The authoritative Linux run
passes all 265 tests, Python compilation, shell syntax checking, and a fold0
`a5` dry run; the generated schedule contains 56 epochs, the registered five
training ROIs, and validation epochs `4,8,...,56`.

Two isolated GPU1 smoke runs pass. The first loads the registered FCMAE800
encoder with the explicit five-ROI subset and produces a metric row plus all
required checkpoints. Its four persistent DataLoader workers emit non-fatal
multiprocessing temporary-directory cleanup warnings after the successful
process exit. The second run verifies sparse validation at epochs 2 and 4:
epochs 1 and 3 contain no validation, both registered epochs do, and no
per-epoch checkpoint directory is created.

Round52 fold0 measured about 76 minutes for one complete two-stage recipe.
Nine update-matched recipes plus ten full D4 evaluations are expected to take
roughly 13--15 hours on one uncontended A100.

The final check found 10 TB free, physical GPU0 occupied by another user's
work, and physical GPU1 idle with about 40 GB free. Round59 launched on physical
GPU1 at 2026-08-18 14:51 CST in detached Screen session `round59`. Initial
verification found the pipeline process and full20 baseline evaluator alive,
with the registered FCMAE encoder hash intact. Completion was unaudited at
that point.

At 2026-08-18 19:48 CST, the original pipeline had completed full20, `a15`,
`a10`, and `a5`, and was training `b15` stage1 on physical GPU1. A fresh
resource check found physical GPU0 idle with about 40 GB free. Detached Screen
session `round59-gpu0-tail` therefore started a training-only tail queue on
GPU0 in the order `c15`, `c10`, `c5`; the original pipeline continues `b15`,
`b10`, and `b5`, then performs all c-series evaluations after recognizing the
completed training runs. The two live training PIDs were verified against
different physical GPU UUIDs. `reports/round59_gpu0_tail.exit` records the tail
worker exit status. This partition does not interrupt the original pipeline or
change any recipe, seed, checkpoint, validation schedule, or evaluation.

The original pipeline later encountered `c10` stage2 while the GPU0 worker was
still writing that directory, correctly refused to overwrite the incomplete
run, and exited with status 1. The GPU0 tail worker completed with status 0. A
separate completion worker then verified the finished c10/c5 stages, evaluated
both missing recipes, generated the registered summary, and exited with status
0. The final audit found no live Round59 process, both GPUs released, all 18
trained stages at their exact registered epoch counts with nonempty
`best_global_mse.pt` checkpoints, and all 10 evaluations covering the same
1,309 fold0 images. The summary artifact SHA-256 is
`0252e7abbf8eac236375ee3a004da3f1976245dc3078cd82ab65839ca817143f`.

## Result

| Training ROIs | Draws | Mean q95 | Population SD | Gap to full20 |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 3 | 72.961672 | 0.472073 | 2.064767 |
| 10 | 3 | 73.979437 | 0.200164 | 1.047002 |
| 15 | 3 | 74.578788 | 0.254410 | 0.447651 |
| 20 | 1 | 75.026439 | 0.000000 | 0.000000 |

All three fixed draw curves are strictly increasing from 5 through 20 ROIs.
The full20-minus-mean15 gap is `0.447651`, well above the preregistered `0.08`
threshold. Round59 therefore gives a decisive data-efficiency result rather
than an ambiguous result requiring fold2 replication.

The next score experiment should improve how the full labelled set is used,
not continue nearby epoch or blend probes. The leading preregistered direction
is to revisit the small positive Round55 auxiliary-marker signal with
marker-specific decoding or gradient isolation, using a matched full-data
control and the existing complete-ROI validation gates. Round59 itself remains
a diagnostic and does not authorize test inference or a platform submission.

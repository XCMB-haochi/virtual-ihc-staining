# Round 60: Marker-Specific Decoder Isolation

## Motivation

Round59 shows that supervised performance is still strongly data-limited at
the current 20 training ROIs: all three nested labelled-data curves are
monotonic, and full20 exceeds the mean 15-ROI score by `0.447651` q95. Round55
also finds a small repeatable benefit from aligned auxiliary markers, but its
four outputs share one decoder. Round56 and Round57 reject both asymmetric
loss schedules, so moving auxiliary supervision between stages does not remove
the remaining instability.

Round60 tests whether auxiliary labels can improve the shared representation
without directly competing for CD68 decoder parameters. Each marker receives
an independent complete PixelShuffle decoder; only the FCMAE encoder and ROI
conditioning are shared. This is a transferable architecture test rather than
a leaderboard or loss-weight probe.

## Fixed Screen

`configs/round60_cv2_unext2_marker_decoders.json` retains the Round55 fold0 and
fold2 ROIs, seed 2044, fresh fold-local official-FCMAE800 encoder initialization,
14+14 supervised epochs, effective batch 16, halo64 spatial-thumbnail FiLM, Uniform-7
hybrid loss, global-MSE checkpoint selection, D4 inference, and JPEG q95
evaluation. The two new recipes have identical architecture, initialization,
data, and optimization:

- `markerdecodercontrol`: weights `1/0/0/0`;
- `markerdecoderauxiliary15`: weights `0.85/0.05/0.05/0.05`.

The primary decoder is constructed before the three auxiliary decoders, using
the same RNG path as the established single-output model. The matched control
therefore measures architecture and numerical effects, while the control-to-
auxiliary comparison isolates gradients supplied by the auxiliary labels.

## Preregistered Gates

The isolated auxiliary mechanism must gain `+0.08` q95 over its matched
control, improve both folds and at least 7/10 ROIs, and keep the worst ROI above
`-0.12`. The new direct model must also beat the shared-decoder Round55
auxiliary model by `+0.05` under the same distribution requirements.

The fixed 0.175 deployment split must improve the matched isolated-control
split by `+0.02`, improve the prior Round55 auxiliary split by `+0.01`, and
reach `+0.06` over `t70_o50`, with both folds, at least 8/10 positive ROIs for
the deployment comparisons, and the stated ROI floors. A replacement route
requires `+0.08` over `t70_o50`. No decoder depth, auxiliary weight, epoch,
checkpoint, ensemble weight, or ROI-specific choice will be changed after the
screen result.

A passing route must be expanded unchanged to all five folds before test
inference. Failure closes full marker-specific decoding at this scale; it does
not authorize a platform submission.

## Status

Local implementation validation passes 15 UNeXt2/FCMAE tests, 63 CV protocol
tests, four distribution-gate tests, and 42 training-safety tests. Python
compilation, JSON parsing, shell syntax, `git diff --check`, and the exact
eight-job dry run also pass. The dry run contains only the two fixed recipes,
two folds, and two supervised stages. The server passes all 268 tests and the
same static checks.

A real batch-16 smoke reaches 39.17 GiB during the fourth decoder and exits
with CUDA OOM before its first optimizer step. The fixed recipes therefore use
micro-batch 8 with two-step gradient accumulation. UNeXt2 uses per-sample
normalization, so this preserves effective batch 16, optimizer-step count, and
learning-rate schedule for both matched recipes. The replacement smoke trains
64 real images, validates 32, writes four optimizer steps plus a valid selected
checkpoint, and peaks at 20.513 GiB for 41,659,700 parameters.

The formal pipeline launched both folds at commit `9a0d81c`, with fold0 on
physical GPU0 and fold2 on physical GPU1. Both cards were idle over the fixed
five-sample utilization check and had about 40.4 GB free immediately before
launch.

## Formal Outcome

The pipeline exited zero. All eight supervised stages contain 14 consecutive
metric rows, 14 epoch checkpoints, and nonempty selected checkpoints. Stage 1
loads the registered fold-local FCMAE800 encoder; Stage 2 strict-loads its
matching Stage-1 checkpoint. Every run records effective batch 16, seed 2044,
the registered marker weights, and runtime commit `9a0d81c`. Fold0 executes
4,368 optimizer updates per stage and fold2 executes 4,298. Peak allocation is
20.668 GiB.

The OOF evaluation covers all 2,706 images from the fixed ten validation ROIs
and takes 3,274.692 seconds. The selected Stage-2 epochs are 13 and 7 for the
fold0 control and auxiliary models, and 5 and 10 for their fold2 counterparts.
All four selected checkpoint hashes were independently recomputed and match
the OOF provenance.

| Candidate | q95 | Float | q95 delta vs `t70_o50` |
| --- | ---: | ---: | ---: |
| `officialauxiliary15` | 74.873621 | 74.943107 | n/a |
| `markerdecodercontrol` | 74.871848 | 74.941513 | n/a |
| `markerdecoderauxiliary15` | 74.747183 | 74.818278 | n/a |
| `t70_o50` | 75.180556 | 75.255122 | baseline |
| `t70_official_auxiliary_split` | 75.221576 | 75.295013 | +0.041020 |
| `t70_markerdecoder_control_split` | 75.212051 | 75.285649 | +0.031495 |
| `t70_markerdecoder_auxiliary_split` | 75.209259 | 75.282668 | +0.028703 |
| `t70_markerdecoder_auxiliary_replace` | 75.191981 | 75.264848 | +0.011425 |

The isolated-decoder control is effectively tied with the Round55 shared-
decoder auxiliary model (`-0.001774` q95), which argues against micro-batch
accumulation or decoder construction causing a material baseline shift. Adding
the fixed 15% auxiliary loss to the isolated model instead loses `-0.124665`
q95 and `-0.123235` float versus its matched control. Both folds regress
(`-0.088270`, `-0.158766`), only ROI022 and ROI024 improve, and ROI008 falls
`-0.451944`.

The new fixed split loses `-0.012318` q95 to the accepted Round55 auxiliary
split. It improves only fold2 and ROI024 in that comparison. Although it stays
`+0.028703` above `t70_o50`, it misses the `+0.06` threshold, reaches only 6/10
positive ROIs, and fails its mechanism dependencies. The replacement route
gains only `+0.011425` and also fails its fold, ROI, floor, and dependency
checks. Every gate fails and `promotion.allowed=false`; no five-fold expansion
or test inference is authorized.

Round60 rejects full marker-specific decoding at this scale. Decoder isolation
itself is neutral, while the auxiliary gradients that remain in the shared
encoder are harmful under the fixed 15% objective. This closes the hypothesis
that Round55's small gain was being suppressed by output-decoder competition.
The accepted Round55 deployment split and the current submission remain
unchanged.

The OOF SHA-256 is
`4d5005ece10677b5951bcf3103cdf6a744ec8c5a05bd08b703602c26e51dea04`;
the gate SHA-256 is
`1b5f24bda2447017c3d93fe989adb5bf3e5ef3e559924c37dcdeaa282def5275`.

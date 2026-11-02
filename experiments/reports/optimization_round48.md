# Round 48: Whole-ROI DAPI Context on FCMAE800 UNeXt2

## Motivation

Round47's fixed 60% FCMAE800 blend scored `73.6745`, improving Round46 by
`+0.1995`. Its preregistered complete-OOF estimate was `+0.206304`, so the
hidden set realized approximately `96.7%` of the expected increment. This
supports the current ROI-fold model-selection protocol, but the last reported
leaderboard maximum of `75.4966` remains `1.8221` higher.

The existing condition contains 101 patch/ROI statistics and a real DAPI
thumbnail from the target-centered `5x5` tile neighborhood. Training and test
ROIs are almost all `16x21` patch grids, so that thumbnail covers at most
25/336, or about `7.4%`, of the ROI bounding grid. The 101 statistics retain
normalized position, occupancy, ROI-level mean/std summaries, and local
neighborhood summaries, but they do not retain the whole ROI's spatial layout.
Round27's label-free proxy found that ROI geometry and global DAPI statistics
were the strongest context signal. The complete low-frequency DAPI layout is
therefore a distinct, still untested input-only representation.

ImageNet-1K/21K initialization is deferred rather than rejected. The current
UNeXt2 grayscale stride-2 stem does not directly match public RGB stride-4
weights, and the old ImageNet-1K ConvNeXt screen had only a small aggregate
gain with a material ROI024 regression. Whole-ROI DAPI context is in-domain,
explicitly allowed by the organizer, and requires no external-weight mapping,
so it has higher expected value for this round.

## Fixed Representation

Condition mode `spatial-thumbnail-whole-roi-film` retains every existing
spatial-statistics and local-thumbnail value. It appends a deterministic
`4x32x32` whole-ROI map:

- tile-level normalized DAPI mean;
- tile-level normalized DAPI standard deviation;
- patch-presence mask;
- current-target-patch mask.

The ROI bounding grid is resized by nearest-neighbor sampling with preserved
aspect ratio and centered zero letterboxing. A four-layer stride-2 CNN encodes
the map to 64 features. The map and target marker receive the same eight D4
transforms before embedding average, so the condition remains orientation
invariant while retaining target-relative organization. A new linear head
maps the embedding to bottleneck FiLM scale/shift and is initialized to exact
zero.

The old local model and new whole-ROI model have identical shared tensors under
the same random seed, and the candidate's first output is exactly equal when
the appended head is zero. The new head still receives a nonzero gradient on
the first update. The experiment therefore adds information without obtaining
an initialization advantage.

## Fair Two-Fold Screen

The machine-readable protocol is
`configs/round48_cv2_unext2_whole_roi.json`. It reuses Round46 fold0/fold2's
exact fold-local 800-epoch DAPI FCMAE encoders by SHA-identical hard links. The
candidate starts a fresh downstream `14+14` epoch trajectory from that encoder;
it does not continue a Round46 supervised checkpoint. Seed, data order, loss,
batch, learning rates, halo64 supervision, checkpoint selector, D4, and JPEG
q95 match Round46.

The fixed candidates are:

- direct FCMAE800 control and direct whole-ROI candidate;
- the Round47 two-fold analogue: 28% base32, 12% base48, 60% FCMAE800;
- a 60% whole-ROI replacement with the same 40% U-Net base;
- one fixed diversity split: 28% base32, 12% base48, 30% FCMAE800, and 30%
  whole-ROI.

No neighboring context size, map resolution, channel subset, blend weight,
seed, epoch, loss, calibration, or per-ROI route is searched.

## Preregistered Gates

The direct mechanism must gain at least `+0.05` q95 over FCMAE800, remain
positive in float, improve both folds and at least 7/10 ROIs, and keep the
worst ROI above `-0.12`.

The 60% replacement additionally requires the mechanism gate, at least `+0.08`
q95 over the Round47 analogue, positive float, both folds, at least 8/10 ROIs,
and a worst ROI above `-0.08`. The independent fixed 30%/30% diversity route
uses the same distribution checks with a `+0.06` q95 threshold; it need not
beat FCMAE800 directly because structurally diverse models can improve an
ensemble while remaining weaker as standalone models. Replacement has
promotion priority over diversity.

This two-fold screen cannot generate a test package. A passing route must be
expanded to all five folds without changing its recipe, weight, or gate before
any test inference is considered.

## Execution and Result

The implementation, protocol, gate, and detached pipeline are complete. Unit
tests cover whole-ROI layout and target marking, fixed metadata and rebuild,
UNeXt2 condition shape, shared same-seed initialization, exact zero-head
forward equivalence, first-step gradient flow, strict checkpoint restore, and
the fixed protocol weights. The complete locally runnable regression is
`231 passed, 1 skipped, 2 deselected`; the two deselections are the existing
Windows NumPy/MKL `lstsq` process aborts. Python compilation and the four-run
fold0/fold2 dry-run also pass.

A real build over all 6,296 training DAPI images completed in `11.899 s`. It
produced 6,296 unique 16,997-dimensional conditions with SHA-256
`6aaee53cd43051d20e0df886b29bc317bff7986a864513cc6d4daba4db72c2db`.
Server commit `4fc46d2` passes Linux compilation, pipeline syntax validation,
the exact four-command dry-run, and the full regression suite:
`234 passed, 71 subtests passed`. A physical-GPU1 smoke run then loaded the
real fold0 FCMAE800 encoder from epoch 800 with SHA-256
`cfd93eaf4fb67b41b44b3bc829b5edd6a5d6ce253a6903599b0b2feae2a4aa04`.
The 31,907,085-parameter candidate completed one 64-train/32-validation epoch
at batch 16 in `17.046 s`, with peak PyTorch GPU memory of `25.281 GiB`.
The zero-initialized whole-ROI output weight and bias reached L1 sums of
`38.022930` and `0.613129`, confirming that the new path opened during real
training. The process exited zero; trailing `pymp-* EBUSY` messages were the
known NFS worker-finalizer noise. The temporary smoke run was removed after
verification.

The prelaunch five-second sample found physical GPU0 at `42--63%` utilization
with `25.061 GiB` free, below the candidate's measured `25.281 GiB` PyTorch
peak before accounting for CUDA overhead. Physical GPU1 remained at 0%
utilization with `34.692 GiB` free. Round48 is therefore fixed to serial
fold0/fold2 execution on GPU1 for this launch; placing one fold on GPU0 would
risk both OOM and a contended straggler. The pipeline's serial mode changes
only scheduling and retains the same per-fold commands, artifacts, OOF, and
gate.

The detached pipeline exited `0`. All four training stages completed exactly
14 epochs and produced their expected checkpoints. The recorded stage wall
times were `2282.969`, `2285.947`, `2276.162`, and `2275.532` seconds. The
formal OOF evaluation then covered all 2,706 images from the fixed 10-ROI
subset in `1399.705` seconds.

Checkpoint selection also behaved as specified. The actual maxima under the
q95 global-MSE selector were:

| fold | stage 1 best | stage 2 best used by OOF | stage 2 SHA-256 |
|---|---:|---:|---|
| fold0 | epoch 13, `73.572085` | epoch 13, `73.653606` | `b8be79dff308c41568e5c8861d0da84dcfbdac241d1d82de1621dd6ecbfb4f3d` |
| fold2 | epoch 13, `72.835787` | epoch 6, `72.813878` | `505ddb213508e0774ad8adc6cb23f41d38cea1c3d0083d2008a2ce60871f1709` |

The fold2 regression was already present in stage 1 and stage 2 peaked early;
it is not an artifact of selecting the last epoch. Formal q95 results are:

| comparison | baseline | candidate | delta | positive folds | positive ROIs | worst ROI |
|---|---:|---:|---:|---:|---:|---:|
| direct mechanism | `74.822793` | `74.301069` | `-0.521724` | 0/2 | 3/10 | `-1.875965` |
| 60% replacement | `74.960051` | `74.730805` | `-0.229246` | 1/2 | 5/10 | `-1.026607` |
| 30%/30% diversity split | `74.960051` | `74.944894` | `-0.015157` | 1/2 | 6/10 | `-0.425120` |

The corresponding float deltas are `-0.521953`, `-0.229945`, and
`-0.016072`. The direct fold deltas are `-0.044917/-0.968496`; replacement
is `+0.080718/-0.519684`; and the diversity split is
`+0.148331/-0.168346`. Thus all three routes fail their aggregate,
cross-fold, positive-ROI, and worst-ROI checks. Promotion is formally false.

Round48 is closed without five-fold expansion, test inference, or a platform
submission. The near-neutral diversity split only dilutes a substantially
weaker member and is not evidence for searching its blend weight. Changing
the whole-ROI map resolution, channel subset, or weight would be a local
search around a mechanism that failed both aggregate metrics and the held-out
distribution checks, so those follow-ups are not authorized by this result.

The tracked formal artifacts are
`reports/round48_cv2_unext2_whole_roi_oof.json` with SHA-256
`7ab2c34027ed330ebbb03d02f641d7d918222479ac7981e24abcfc8f66544ebb`
and `reports/round48_cv2_unext2_whole_roi_gate.json` with SHA-256
`d7d5051d924f7f05c179b528ec651c21b7478ec0bacbac7b89df6616f4648f03`.

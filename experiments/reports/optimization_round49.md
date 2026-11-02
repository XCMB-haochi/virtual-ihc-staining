# Round 49: Wide-Head UNeXt2 Screen

## Motivation

Round48's whole-ROI conditioning failed its direct and ensemble gates. Its
direct model regressed FCMAE800 by `-0.521724` q95, and the fixed 30%/30%
diversity split remained `-0.015157` below the Round47 analogue. The failure
was cross-fold and included a worst-ROI regression of `-1.875965`, so local
map-size, channel, or blend searches are not justified.

The next structural bottleneck is in the current UNeXt2 decoder. Its final
half-resolution stage contracts from 192 channels directly to
`4 * out_channels`, which is only four channels for CD68, then applies a final
PixelShuffle to obtain one output channel. The official VisCy UNeXt2 computes
48 final decoder channels for a one-channel, one-slice output under its default
head settings, followed by a separate PixelToVoxel head. See the
[UNeXt2 assembly](https://github.com/mehta-lab/VisCy/blob/main/packages/viscy-models/src/viscy_models/unet/unext2.py)
and
[PixelToVoxelHead](https://github.com/mehta-lab/VisCy/blob/main/packages/viscy-models/src/viscy_models/components/heads.py).

This round tests whether the existing four-channel contraction discards
fine-scale information. It is an official-inspired two-dimensional adaptation,
not a claim of reproducing VisCy's three-dimensional head.

## Fixed Architecture Change

The encoder, FCMAE initialization, thumbnail FiLM, loss, and data path remain
unchanged. The candidate changes only the last decoder stage and output head:

- the half-resolution decoder ends at 48 channels instead of four;
- PixelShuffle produces 12 full-resolution feature channels;
- a 3x3 convolution maps 12 channels to 16 channels;
- GELU and a 1x1 convolution produce the CD68 logit;
- the existing learned output bias and sigmoid remain unchanged.

Under the same seed, construction deliberately follows the legacy decoder RNG
path before creating the shared FiLM branch. The old and candidate encoders and
FiLM tensors therefore start exactly equal. The candidate is trained fresh
from the same fold-local FCMAE800 encoder rather than continued from a
supervised checkpoint; this avoids contamination by the legacy narrow head.

## Fair Two-Fold Screen

The machine-readable protocol is
`configs/round49_cv2_unext2_widehead.json`. It uses the same fold0/fold2 ROI
partition, 800-epoch fold-local DAPI encoders, seed 2044, `14+14` epochs,
batch 16, workers 4, halo64 supervision, hybrid loss, global-MSE selector, D4,
and q95 JPEG path as Round46. The reused encoder files must be SHA-identical
hard links to the Round46 artifacts.

The candidates are fixed before training:

- direct FCMAE800 control and direct wide-head candidate;
- the Round47 two-fold analogue: 28% base32, 12% base48, 60% FCMAE800;
- the same blend with the wide-head model replacing FCMAE800;
- one fixed diversity split: 28% base32, 12% base48, 30% FCMAE800, and 30%
  wide-head.

No decoder width, hidden width, epoch, seed, loss, blend weight, calibration,
or per-ROI route is searched.

## Preregistered Gates

The direct mechanism must gain at least `+0.05` q95 over FCMAE800, remain
positive in float, improve both folds and at least 7/10 ROIs, and keep the
worst ROI above `-0.12`.

The 60% replacement additionally requires the mechanism gate, at least `+0.08`
q95 over the Round47 analogue, positive float, both folds, at least 8/10 ROIs,
and a worst ROI above `-0.08`. The independent 30%/30% diversity route requires
`+0.06` q95 with the same distribution constraints but does not depend on the
direct mechanism. Replacement has promotion priority over diversity.

This screen cannot generate a test package. A passing route must be expanded
unchanged to all five folds and pass the corresponding complete-OOF gate before
test inference.

## Execution Status

The legacy and wide-head models contain 31,746,413 and 31,796,882 parameters,
respectively, so the candidate adds 50,469 parameters at high resolution.
Tests verify output shape and gradient flow, same-seed exact equality of every
encoder and thumbnail-FiLM tensor, strict checkpoint round-trip, FCMAE protocol
acceptance, fixed candidate weights, and legacy checkpoint compatibility.

Local Python compilation, the four-command fold0/fold2 dry-run, Git Bash
syntax validation, and `git diff --check` pass. The complete locally runnable
regression is `234 passed, 1 skipped, 2 deselected`; the two deselections are
the existing Windows NumPy/MKL `lstsq` process-abort cases.

Server commit `04a4b08` passes Linux compilation, pipeline syntax validation,
the exact four-command dry-run, `git diff --check`, and the complete regression
suite: `237 passed, 71 subtests passed`. The real fold0 FCMAE800 encoder loaded
at pretraining epoch 800 with SHA-256
`cfd93eaf4fb67b41b44b3bc829b5edd6a5d6ce253a6903599b0b2feae2a4aa04`.

A physical-GPU1 smoke run completed one 64-train/32-validation epoch at batch
16 in `19.406 s`. The 31,796,882-parameter candidate peaked at `27.946 GiB`
of PyTorch GPU allocation. A same-seed reconstruction of the untrained model
found an L1 change of `41.965307145` across the 28 tensors in the final decoder
stage and output head, confirming that the new path updates in real training.
The process exited zero; trailing `pymp-* EBUSY` messages were the known NFS
worker-finalizer noise. The smoke run was removed after checkpoint validation.

The pre-smoke five-second sample found physical GPU0 at sustained `35--40%`
utilization with `27.501 GiB` free, while physical GPU1 stayed at 0% with
`34.692 GiB` free. Formal scheduling must still use a fresh sample immediately
before launch because other users' loads can change.

## Formal Result

The final prelaunch sample found physical GPU0 at sustained `44--62%`
utilization with `27.501 GiB` free, below the measured candidate peak, while
physical GPU1 remained at 0% with `34.692 GiB` free. The detached pipeline was
therefore run serially on GPU1 to avoid both OOM and a contended straggler. It
exited `0`; all four stages contain exactly 14 metric rows, 14 epoch
checkpoints, and the required best/last checkpoints.

The selected checkpoints are not the final epochs:

| fold | stage 1 best | stage 2 best used by OOF | stage 2 SHA-256 |
|---|---:|---:|---|
| fold0 | epoch 9, `73.368939` | epoch 7, `73.462363` | `bcf5f5871df095bbf04e17ad3973678cf0ef1d40fbd46490344d18b12b15d93c` |
| fold2 | epoch 13, `73.359921` | epoch 3, `73.391394` | `e44a34c5ac42bf58db7573604f59878a47988703cb86d4de2cbb43cfb2763aed` |

The four stage times were `2470.606`, `2471.639`, `2804.029`, and
`3026.493` seconds. Stage-2 peak allocation was `28.065 GiB`. Formal OOF
evaluation covered all 2,706 images from the fixed 10 ROIs in `1630.976`
seconds.

The q95 results are:

| comparison | baseline | candidate | delta | positive folds | positive ROIs | worst ROI |
|---|---:|---:|---:|---:|---:|---:|
| direct mechanism | `74.822793` | `74.511205` | `-0.311588` | 0/2 | 1/10 | `-0.624107` |
| 60% replacement | `74.960051` | `74.801912` | `-0.158139` | 0/2 | 1/10 | `-0.379792` |
| 30%/30% diversity split | `74.960051` | `74.943744` | `-0.016307` | 1/2 | 2/10 | `-0.100903` |

The corresponding float deltas are `-0.311084`, `-0.158859`, and
`-0.016613`. Direct fold deltas are `-0.206800/-0.409775`; replacement is
`-0.092782/-0.219380`; and diversity is `+0.006440/-0.037622`. Only ROI022
improves under direct replacement. The diversity split also improves ROI000,
but it remains negative on eight of ten ROIs.

All preregistered gates fail. Round49 is closed without five-fold expansion,
test inference, or platform submission. The selected early checkpoints show
that the result is not caused by evaluating the last epoch. The near-neutral
diversity blend is produced by dilution of a broadly weaker member, not a
stable complementary mechanism, so adjacent decoder widths and blend weights
are not authorized by this result.

The tracked formal artifacts are
`reports/round49_cv2_unext2_widehead_oof.json` with SHA-256
`47f4e9700f4b7c40d6834e2469ce79f6ff0c203c230268f3ecf4c4eb70f0d4f3`
and `reports/round49_cv2_unext2_widehead_gate.json` with SHA-256
`bbafb942f6de13cf7b302bbf073e3a0ad3832b19a6b1fa318413054af76e648b`.

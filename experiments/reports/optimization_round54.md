# Round 54: Supervised ImageNet-22K Initialization

## Motivation

Round53 raises the platform score to `73.9028`, and `96.57%` of its measured
OOF increment transfers to the platform. The remaining gap motivates another
portable representation test rather than a leaderboard-calibrated probe.
Round54 asks whether broader supervised natural-image pretraining adds useful
features before the unchanged fold-local DAPI FCMAE800 stage.

This experiment is not described as 22K FCMAE. The official artifact is a
ConvNeXt-V2 Tiny supervised ImageNet-22K checkpoint with an ImageNet-1K
classification head and EMA weights. It changes both the source data and the
pretraining objective relative to Round52, so the result can only establish
the value of this complete initialization route.

## Audited Public Artifact

The fixed source is Meta's official
`convnextv2_tiny_22k_224_ema.pt` checkpoint:

- URL: `https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt`;
- license reported by the official repository: `CC-BY-NC-4.0`;
- SHA-256: `853d431aa9363f1b058e3c343d4bf2fca5fe2a4196621c381ddbcd4828290a96`;
- file size: `114,604,362` bytes;
- checkpoint schema: one `model` state with 200 float32 tensors.

The 194 structurally mapped encoder tensors are all present. Thirty-six GRN
gamma/beta tensors require only singleton-dimension reshaping. The incompatible
RGB stride-4 input convolution remains seeded locally, while the classification
`norm` and `head` tensors are explicitly ignored. The transfer therefore keeps
the same `27,860,256 / 27,860,736` target-parameter coverage (`99.998277%`) as
Round52. Loading is strict on the mode, source-key set, tensor shapes and dtypes,
and checkpoint SHA-256.

## Fixed Two-Fold Protocol

`configs/round54_cv2_unext2_supervised22k_fcmae.json` keeps the Round52
fold0/fold2 split and changes only the public initialization before DAPI
FCMAE800. Seed 2044, 800 masked-reconstruction epochs, optimizer schedule,
14+14 supervised epochs, halo64 thumbnail FiLM, Uniform-7 hybrid loss,
global-MSE selection, D4 inference, and JPEG q95 evaluation remain fixed.

The current Round53 two-fold analogue, `t70_o50`, is the deployment baseline:
base32 `0.21`, base48 `0.09`, scratch FCMAE800 `0.35`, and official FCMAE-1K
FCMAE800 `0.35`. Three preregistered checks are evaluated once:

- direct supervised-22K-initialized FCMAE800 versus the Round52
  FCMAE-1K-initialized model;
- replacement of the current `0.35` official FCMAE-1K member by the new model;
- equal division of the total `0.70` FCMAE weight among scratch, FCMAE-1K, and
  supervised-22K initialization routes.

The replacement must gain at least `+0.12` q95 to take priority over the more
diverse route. The equal-third candidate requires `+0.04`. Both deployment
gates additionally require positive float gain, 2/2 positive folds, 8/10
positive ROIs, and a worst ROI delta above `-0.08`. No weight, checkpoint,
schedule, stem conversion, or ROI-specific route will be selected after the
two-fold result. A passing candidate must be expanded unchanged to five folds
and pass a complete-OOF gate before test inference.

## Validation and Completion Audit

The local protocol/UNeXt2 suite passes all 66 focused tests. JSON parsing,
Python compilation, Git Bash syntax, `git diff --check`, and the exact six
training-command dry-run also pass. After synchronizing commit `465cc3f`, the
clean server environment passes the full 250-test suite.

The server artifact independently reproduces the registered SHA-256 and loads
all 194 mapped tensors with 36 GRN reshapes, six skipped source tensors, and
the exact registered parameter coverage. A one-step physical-GPU0 smoke run
used 32 real fold0 training images, batch 32, four workers, and AMP. The step
took `12.480 s` and peaked at `22,041,952,256` allocated bytes
(`20.528 GiB`). It exited zero and wrote `complete=true`, one metric row, and
a loadable 196-tensor final encoder with SHA-256
`73338681cbb8885524aead0d9ce4578d653f738c409907a3e6e57856e2093995`.

Python emitted a multiprocessing temporary-directory cleanup warning after
the completed artifact was written; the process still exited zero. Formal
completion therefore requires both the pipeline exit marker and independent
artifact checks rather than relying on terminal appearance.

The formal detached pipeline exited zero. Both fold-local pretraining runs
contain exactly 800 consecutive metric rows, `complete=true` at epoch 800,
loadable 196-tensor final encoders, and the audited public initialization hash.
Fold0 took `49,163.261 s` (`13.656 h`) and has a last-50 mean reconstruction
loss of `0.001294797`; its final encoder SHA-256 is
`e5b93e03d8fa240264ea62cd8ff8298ca674cf8b6eff54d24ef57cf6a82fffe1`.
Fold2 took `44,457.287 s` (`12.349 h`) and has a last-50 mean loss of
`0.001298528`; its final encoder SHA-256 is
`16da9934187a2107d3c1e6fa559c7f8c63de0cde3be3808c79d979cec90fb9af`.
Both losses are above the matched Round52 FCMAE-1K-initialized values
(`0.001262100` and `0.001256987`).

Every supervised stage contains exactly 14 consecutive metric rows. Fold0
selects stage-1 epoch 10 at `73.527352` global-MSE score and stage-2 epoch 7 at
`73.766972`; fold2 selects stage-1 epoch 13 at `73.277517` and stage-2 epoch 3
at `73.438857`. Stage-1 checkpoints strictly bind the corresponding epoch-800
encoder hashes above.

The OOF evaluation covers all 2,706 images from the fixed 10 held-out ROIs and
took `1,801.129 s`. All 10 referenced checkpoints independently rehash to the
embedded values. The eight reused baseline checkpoints, both fold-level
`t70_o50` metrics, and all 10 ROI-level baseline metrics are exactly identical
to Round53, ruling out evaluation or baseline drift.

- OOF SHA-256: `0ca67bd90b30636a4c4e3900375cf27ff0a4157150f393481eb744509aa3f6e2`;
- gate SHA-256: `d88530c7ab452713cb983b366877cd1a6e06a6049131afd6db6e66f3fa193422`;
- the gate report binds the exact OOF SHA-256.

## Results

| Candidate | JPEG-q95 OOF | Delta vs its gate baseline | Float OOF | Gate result |
| --- | ---: | ---: | ---: | --- |
| FCMAE-1K initialized | 74.847851 | baseline for direct test | 74.921061 | reference |
| supervised-22K initialized | 74.541986 | -0.305865 | 74.613766 | reject |
| `t70_o50` | 75.180556 | deployment baseline | 75.255122 | retained |
| replace official 0.35 | 75.109735 | -0.070821 | 75.182799 | reject |
| equal FCMAE thirds | 75.182175 | +0.001619 | 75.255375 | reject |

The direct supervised-22K model is negative on both folds (`-0.214488` and
`-0.391485`), improves only 3/10 ROIs, and reaches a worst ROI delta of
`-1.006665`. Replacing the current official-initialized member is also negative
on both folds and loses `-0.070821` overall.

The equal-third diversity candidate produces only `+0.001619` q95 and
`+0.000253` float. Fold0 gains `+0.034961`, but fold2 loses `-0.029623`; only
6/10 ROIs improve and ROI008 falls `-0.203108`. It therefore fails the fixed
overall, fold, ROI-count, and ROI-floor checks. `promotion.allowed=false`.

## Decision

Reject the supervised ImageNet-22K-to-1K initialization route. Do not expand
it to five folds, search adjacent ensemble weights, or create a test package.
Retain the Round53 `t70_o50` submission. This result closes the audited
supervised-22K-to-1K EMA route; it does not establish that every possible 21K
or 22K pretraining objective is inferior to FCMAE-1K.

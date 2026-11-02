# Round 57: Primary Warm-Up, Late Auxiliary Refinement

## Motivation

Round56 rejects auxiliary supervision only in Stage 1: removing auxiliary loss
from Stage 2 lowers the direct model by `-0.047139` q95 relative to persistent
auxiliary training, with a `-0.092527` fold2 delta and a `-0.285008` ROI008
delta. The evidence says later auxiliary gradients help, but it does not show
whether auxiliary gradients are also needed during the higher-learning-rate
Stage 1.

Round57 completes the fixed two-stage loss-schedule matrix. It starts from the
exact Round55 primary-only four-output control Stage-1 checkpoint and applies
the established `0.85/0.05/0.05/0.05` loss only during a fresh Stage 2. The
other three cells already exist: primary/primary is the Round55 control,
auxiliary/auxiliary is the Round55 candidate, and auxiliary/primary is Round56.
No core training implementation changes.

## Fixed Screen

`configs/round57_cv2_unext2_primary_warm_auxiliary.json` retains fold0/fold2,
seed 2044, 14 Stage-2 epochs at learning rate `0.0001`, batch 16, halo64
spatial-thumbnail FiLM, Uniform-7 hybrid loss, global-MSE selection, D4
inference, and JPEG q95 evaluation. The exact primary-only Stage-1 sources are:

- fold0: `b6c8155aaa9c0b1a16e9c08c3bfe1b76a25ca959e6bb7f8b961fbfe4a4304d62`;
- fold2: `0dec34a881125829ed6238e1d09585a59654096ed304c53a2e2bb40f64034744`.

The OOF report includes all four schedule cells. The fixed deployment split
keeps `t70_o50` unchanged except that half of its 0.35 official member is
assigned to the late-auxiliary model. Replacement assigns the complete 0.35
share to that model.

## Preregistered Gates

The isolated late-auxiliary mechanism must gain at least `+0.05` q95 over the
matched primary/primary control, have positive float gain, improve both folds
and 7/10 ROIs, and keep the worst ROI above `-0.12`. To establish an advance
rather than merely recover Round55, it must then gain `+0.02` over persistent
auxiliary training under both folds, 7/10 ROIs, and a `-0.10` floor.

The late-auxiliary split must improve the persistent auxiliary split by
`+0.01`, with both folds, 8/10 ROIs, and a `-0.06` floor. Deployment additionally
requires `+0.06` over `t70_o50`, both folds, 8/10 ROIs, and a `-0.08` floor.
Replacement separately requires `+0.08` over `t70_o50` and depends on the
direct schedule advance. No epoch, loss weight, marker subset, ensemble weight,
or ROI-specific route will be searched after the result.

A passing route must be expanded unchanged to all five folds before test
inference. Failure closes the two-stage auxiliary schedule branch and retains
Round53 `t70_o50`.

## Status

Local validation passed on 2026-08-16: the protocol parsed, relevant Python
files compiled, all 58 CV protocol tests passed, the shell runner passed
`bash -n`, and the exact dry-run produced only the two preregistered Stage-2
jobs with the expected fold-local primary-only initialization paths and late
auxiliary loss weights.

Linux validation at commit `16b9447` passed all 255 tests plus shell syntax,
Python compilation, JSON parsing, and the exact two-job dry-run. A real fold0
smoke strict-loaded the four-channel Round55 control Stage-1 checkpoint,
trained 64 images, evaluated 32 images, and exited zero. It wrote one valid
metric row and epoch checkpoint, changed all 287 state tensors, resolved weights
to `0.85/0.05/0.05/0.05`, and used 25.657 GiB peak GPU memory. The post-exit
`pymp-*` NFS cleanup trace was accepted only after these artifacts and the zero
exit were verified; the smoke directory was then removed.

The formal two-fold pipeline was launched at 00:26 CST on 2026-08-16 in
detached Screen `round57`, with fold0 on physical GPU0 and fold2 on physical
GPU1. Both cards were idle over the preceding five-sample check and each formal
process loaded about 27.7 GiB.

## Formal Outcome

The pipeline exited zero and evaluated all 2,706 images from the ten fixed
validation ROIs in 3,305.320 seconds. Both fold status files report
`complete=true`; each Stage-2 run contains 14 consecutive metric rows and 14
epoch checkpoints. Global-MSE selection chose epoch 2 for fold0 and epoch 6
for fold2. Their selected checkpoint SHA-256 values were independently
recomputed as
`7f14b00aeb254002436c45c85f2f0fa45e5357b2f706a77b8c537ba965a424f7`
and
`5d73c1b812cf85693e0a975e1bb9327e343d24acba863a5bd7250ced1b274978`,
matching the OOF provenance records. The trailing `pymp-*` NFS cleanup traces
occurred after each run wrote its valid artifacts and completion marker.

| Candidate | q95 | Float | q95 delta vs `t70_o50` |
| --- | ---: | ---: | ---: |
| `officialauxcontrol` | 74.794670 | 74.866941 | n/a |
| `officialauxiliary15` | 74.873621 | 74.943107 | n/a |
| `officialauxwarmprimary` | 74.826482 | 74.899824 | n/a |
| `officialprimarywarmaux15` | 74.777560 | 74.843937 | n/a |
| `t70_o50` | 75.180556 | 75.255122 | baseline |
| `t70_official_auxiliary_split` | 75.221576 | 75.295013 | +0.041020 |
| `t70_official_auxlate_split` | 75.190237 | 75.263819 | +0.009681 |
| `t70_official_auxlate_replace` | 75.158268 | 75.231041 | -0.022288 |

Late auxiliary refinement loses `-0.017110` q95 and `-0.023004` float to the
matched primary-only control. Fold0 loses `-0.042787`, fold2 gains only
`+0.006949`, and ROI000 loses `-0.387171`. It also loses `-0.096062` q95 to
persistent auxiliary training, with neither fold and only 2/10 ROIs positive.

The fixed diversity split loses `-0.031340` q95 to the persistent auxiliary
split. Although it remains `+0.009681` above `t70_o50`, only fold2 and 7/10 ROIs
improve and the gain is far below the fixed `+0.06` deployment threshold. Every
gate therefore fails, `promotion.allowed=false`, and no five-fold expansion or
test inference is permitted. This completes and rejects the two-stage
auxiliary-schedule matrix; the deployed Round53 `t70_o50` ensemble remains
unchanged.

The OOF report SHA-256 is
`5be604a3556990d1bb7c17adf1c89f10f76a282a603ef207c8ca14f9e702e0a9`;
the gate report SHA-256 is
`363765a33e4b5b0e80be65934984709a75f343049f50e4a1079eb79ed8f2aa76`.

# Round 56: Auxiliary Warm-Up, Primary-Only Refinement

## Motivation

Round55 shows a repeatable but weak benefit from aligned auxiliary markers.
Across complete five-fold OOF, the auxiliary model gains `+0.039799` over its
matched four-output control, and the fixed deployment split improves all five
folds with a safe `-0.033516` worst-ROI delta. It still fails promotion because
the deployment gain is `+0.038174` rather than `+0.04`, only 18/25 ROIs are
positive, and the direct model has several large negative-ROI effects.

One plausible mechanism is persistent gradient interference: CD45RO, HLA-DR,
and Vimentin help shape the representation early, but continuing their 15%
loss through the low-learning-rate refinement stage can move the shared model
away from the primary CD68 objective. Round56 tests a fixed schedule that uses
the exact Round55 auxiliary Stage-1 checkpoint, then trains a fresh Stage 2
with weights `1/0/0/0`. This preserves model shape and all optimizer settings;
only the already-stated role of auxiliary supervision changes from persistent
to warm-up-only.

## Fixed Screen

`configs/round56_cv2_unext2_auxiliary_warm_primary.json` uses the unchanged
fold0/fold2 split, seed 2044, 14-epoch Stage 2, learning rate `0.0001`, batch
16, halo64 spatial-thumbnail FiLM, Uniform-7 hybrid loss, global-MSE selection,
D4 inference, and JPEG q95 evaluation. It initializes from the exact Round55
auxiliary Stage-1 checkpoints:

- fold0: `20b64da74849fbee7a4835bcac0e71a638aa89f11285ec2215dd949a104aa588`;
- fold2: `b41654a55a0364f851e593962a9ba646ab05a47aae66eda19ea1697fbaae78ed`.

The candidate retains all four output channels for strict checkpoint loading,
but only CD68 contributes to Stage-2 loss. The matched schedule comparison is
the Round55 model that starts from the same Stage 1 and keeps auxiliary loss
active. The primary deployment candidate keeps the established `t70_o50`
weights and splits the 0.35 official member into 0.175 official and 0.175
warm-primary members.

## Preregistered Gates

The schedule mechanism requires `+0.03` q95 over persistent auxiliary
training, positive float gain, both folds, 7/10 positive ROIs, and a worst ROI
above `-0.12`. Retaining the original auxiliary benefit requires `+0.05` over
the matched primary-only control under the same distribution checks.

The fixed diversity split must first improve the Round55 auxiliary split by
`+0.01`, with both folds, 8/10 ROIs, and a `-0.06` floor. Deployment then
requires `+0.06` over `t70_o50`, both folds, 8/10 ROIs, and a `-0.08` floor.
The replacement route separately requires `+0.08` over `t70_o50` and depends
on retained auxiliary benefit. No loss-weight grid, checkpoint search,
ensemble-weight search, or ROI-specific selection will follow the result.

A passing route must be expanded unchanged to all five folds before test
inference. Failure retains Round53 `t70_o50` and closes this auxiliary schedule
branch.

## Status

Local validation passed on 2026-08-15: the protocol parsed, relevant Python
files compiled, all 57 CV protocol tests passed, the shell runner passed
`bash -n`, and the exact dry-run produced only the two preregistered Stage-2
jobs with the expected fold-local Round55 initialization paths and primary-only
loss weights.

Linux validation at commit `6031958` passed all 254 tests plus shell syntax,
Python compilation, JSON parsing, and the exact two-job dry-run. A real fold0
smoke strict-loaded the four-channel Round55 Stage-1 checkpoint, trained 64
images, evaluated 32 images, and exited zero. It wrote one valid metric row and
checkpoint, changed all 287 state tensors, retained weights `1/0/0/0`, and used
25.657 GiB peak GPU memory. The post-exit `pymp-*` NFS cleanup trace was accepted
only after these artifacts and the zero exit were verified; the smoke directory
was then removed.

The formal two-fold pipeline was launched at 22:02 CST on 2026-08-15 in detached
Screen `round56`, with fold0 on physical GPU0 and fold2 on physical GPU1. Both
cards were idle over the preceding five-sample check and each formal process
loaded about 27.7 GiB.

## Formal Outcome

The pipeline exited zero. Both folds contain 14 consecutive metric rows and 14
epoch checkpoints; their selected global-MSE checkpoints are epoch 7 for fold0
and epoch 3 for fold2. Both strict-load the preregistered Round55 Stage-1 source,
retain 287 tensors and 31,753,556 values, and change all 287 tensors. Peak
allocation was 25.660 GiB. The OOF pass took 2,816.950 seconds and covers all
2,706 images from the fixed ten ROIs. All 14 referenced checkpoints were
independently rehashed, and every shared Round55 metric and provenance record
is exactly unchanged.

| Candidate | q95 | Float | q95 delta vs `t70_o50` |
| --- | ---: | ---: | ---: |
| `officialauxcontrol` | 74.794670 | 74.866941 | n/a |
| `officialauxiliary15` | 74.873621 | 74.943107 | n/a |
| `officialauxwarmprimary` | 74.826482 | 74.899824 | n/a |
| `t70_o50` | 75.180556 | 75.255122 | baseline |
| `t70_official_auxcontrol_split` | 75.195485 | 75.269552 | +0.014929 |
| `t70_official_auxiliary_split` | 75.221576 | 75.295013 | +0.041020 |
| `t70_official_auxwarm_split` | 75.213379 | 75.287166 | +0.032822 |
| `t70_official_auxwarm_replace` | 75.193057 | 75.267120 | +0.012501 |

Removing auxiliary loss in Stage 2 lowers the direct model by `-0.047139` q95
and `-0.043283` float relative to persistent auxiliary training. Fold0 is
effectively tied at `+0.001300`, but fold2 regresses by `-0.092527`; only 4/10
ROIs improve and ROI008 falls by `-0.285008`. Relative to the matched control,
the warm-primary model remains `+0.031812` q95 overall but is negative on fold2
and reaches `-0.500935` on ROI008.

The deployment split remains positive against `t70_o50` on both folds and 8/10
ROIs, with a worst ROI of `-0.076506`, but it is `-0.008198` below the persistent
auxiliary split and fails its mechanism dependency and `+0.06` deployment-gain
threshold. Replacement gains only `+0.012501`. Consequently every gate fails,
`promotion.allowed=false`, and no five-fold expansion or test inference is
permitted.

This result rejects the fixed auxiliary-Stage-1 / primary-only-Stage-2 schedule:
the later auxiliary gradients are beneficial rather than harmful under this
protocol. It does not determine whether primary-only Stage 1 followed by
auxiliary Stage 2 can retain that benefit with less early representation bias.
The deployed Round53 `t70_o50` ensemble remains unchanged.

The OOF SHA-256 is
`c56ef29798c23ec6cae4c90bbba132116d12e1e772e00267fe35fc95a0c59dee`;
the gate SHA-256 is
`da4893a0c4a4526f8f2d2d6b83a5b08e93ec43a0c230f0df78f6d80cfaf95e8b`.

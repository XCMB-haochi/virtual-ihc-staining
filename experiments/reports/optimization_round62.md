# Round 62: ConvNeXt V2 Base FCMAE Capacity Screen

## Motivation

Round59 shows a strong labelled-data bottleneck, while Round60 and Round61
reject marker-specific decoder and gradient-routing changes. Round58 also shows
that extending Tiny FCMAE from 800 to 1200 epochs lowers reconstruction loss
without improving virtual staining. The remaining structural FCMAE question is
capacity, not another continuation schedule or local output probe.

Round62 therefore changes only the ConvNeXt V2 size from Tiny to Base. Both use
the official ImageNet-1K self-supervised FCMAE initialization, then fresh
fold-local DAPI FCMAE pretraining and fresh supervised training. This is not a
continuation from a mature Tiny staining checkpoint.

## Fixed Screen

The official Base encoder uses depths `[3, 3, 27, 3]`, dimensions
`[128, 256, 512, 1024]`, and 87,685,120 local grayscale-encoder parameters.
The audited official checkpoint has SHA-256
`cd59ca7d4db45a8d7bc2128dc447c0a5aee209d2b4920e79e1da4196afca6405`.
Strict mapping loads 374/376 tensors and 87,684,480/87,685,120 parameters;
only the incompatible RGB input-convolution weight and bias remain locally
initialized.

Fold0 and fold2, seed 2044, 800 FCMAE epochs, 14+14 supervised epochs,
halo64 thumbnail FiLM, Uniform-7 hybrid loss, global-MSE checkpoint selection,
D4 inference, and JPEG q95 scoring remain fixed. FCMAE uses micro-batch 16 with
two-step accumulation; supervised training uses micro-batch 8 with two-step
accumulation. Both preserve the corresponding Tiny effective batches, 32 and
16.

The direct Base model must gain at least `+0.10` q95 over official Tiny800,
improve both folds and at least 7/10 ROIs, and keep the worst ROI above `-0.12`.
Replacing Tiny in retained `t70_o50` additionally requires `+0.06`, 8/10 ROIs,
and a `-0.08` floor. A fixed 17.5%/17.5% Tiny/Base diversity split is allowed
independently of direct superiority, but requires `+0.04`, both folds, 8/10
ROIs, and the same floor. No weight, epoch, or ROI-specific route will be tuned
after this screen.

## Validation Status

The local compatibility suite passes 19 UNeXt2/FCMAE tests and 64 protocol
tests. The Linux suite at commit `1dd4612` passes all 274 tests, and a separate
real checkpoint audit reproduces the registered hash and parameter coverage.

Real A100 smoke tests establish the fixed micro-batches. Base FCMAE batch 16
peaks at 23,496,722,944 allocated bytes (21.88 GiB) and processes a warm
64-image epoch in 1.76 seconds. Base supervised batch 8 with the official
384x384 context path peaks at 24.72 GiB. Both leave over 15 GiB against the
observed 40.4-GiB free-memory state. Extrapolating the FCMAE steady rate gives
roughly 30--33 hours per 800-epoch fold; supervised stages and OOF evaluation
bring the expected parallel two-GPU wall time to about 35--40 hours.

The final server at commit `4ed1a62` passes all 276 tests, Python compilation,
shell syntax, and exact dry-runs containing two FCMAE jobs and four supervised
jobs. Immediately before launch, five one-second samples found both physical
GPUs at 0% utilization with about 40.4 GiB free and no compute processes.

Formal Screen `round62_base` launched both folds in parallel, fold0 on physical
GPU0 and fold2 on physical GPU1. Both immutable pretraining configs record
commit `4ed1a62`, Base, micro-batch 16, two-step accumulation, effective batch
32, 90,852,617 parameters, and the registered official checkpoint hash. The
post-launch sample found both GPUs at 99--100% utilization and about 23.5 GiB
used. Completion remains unverified until the exit marker, all 800 metric rows
per fold, completion/checkpoint provenance, four supervised trajectories, OOF
report, and distribution gate are audited.

The remote pipeline completed with exit status `0`; no Round62 Screen socket or
training process remained after completion. Both folds have exactly 800 FCMAE
metric rows, `complete: true` status records, fold-local epoch-800 encoder
checkpoints, and two 14-epoch supervised stages. The fold-local stage-1
initialization records bind the official Base checkpoint and the epoch-800
encoder hashes; stage 2 is initialized from the corresponding stage-1 best
checkpoint. The OOF evaluator independently rehashed every checkpoint named in
`checkpoint_provenance` with no errors.

The OOF report covers 2 folds, 10 expected ROIs, and 2,706 validation images.
Its SHA-256 is
`6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe`; the gate
report SHA-256 is
`821cd877c82e415c09e159f2d4b9c020b755c92f96e069b4a4fb23a431223a56`.

| Candidate | q95 score | Float score |
| --- | ---: | ---: |
| `officialfcmae800` | 74.8479 | 74.9211 |
| `officialbasefcmae800` | 74.7551 | 74.8234 |
| `t70_o50` | 75.1806 | 75.2551 |
| `t70_base_replace` | 75.2956 | 75.3663 |
| `t70_base_split` | 75.2834 | 75.3554 |

`base_capacity` fails: Base alone is `-0.0927` q95 versus official Tiny,
improves only one of two folds and six of ten ROIs, with the worst ROI at
`-0.9199`. The dependent replacement gate also fails its ROI floor. The fixed
`t70_base_split` diversity gate passes with `+0.1028` q95 and `+0.1003` float
score versus `t70_o50`, positive on both folds and nine of ten ROIs; its only
negative ROI is `-0.0276`. Therefore the formal promotion is limited to this
pre-registered ensemble recipe. This is a validation result only; no test
inference or leaderboard-derived calibration has been performed.

The fold logs contain repeated multiprocessing temporary-directory cleanup
tracebacks (`Errno 16`, resource busy), but the pipeline exit status, complete
status records, checkpoint hashes, OOF report, and gate are all valid. They are
recorded as non-fatal cleanup warnings rather than training failures.

## Next Boundary

Following the two-fold pass, the next admissible step is a fixed five-fold
expansion of `t70_base_split`: train the Base member on fold1, fold3, and fold4,
reuse the audited fold0/fold2 members, then run complete five-fold OOF and the
distribution gate. The ensemble weights, checkpoint rule, TTA, JPEG quality,
and gate thresholds remain unchanged. Test inference is blocked until that
five-fold gate is audited.

# Round 51: Low-Weight Auxiliary-Marker Supervision

## Motivation

Round50 closes strong continuous affine augmentation: the direct candidate
lost `-0.047799` q95 and its independent diversity split gained only
`+0.015077` while failing the ROI floor. The successful mechanisms in the
current stack remain DAPI context and fold-local FCMAE pretraining, so the next
experiment targets label efficiency rather than more capacity or another
post-processing probe.

The official data provide aligned CD45RO, HLA-DR, and Vimentin targets for
every CD68 training patch, and the rules explicitly encourage cross-marker
representation learning. Round11 tested 50% total auxiliary weight on an
obsolete residual U-Net and showed meaningful but unstable ROI changes;
Round13 tested equal-weight multitask pretraining followed by CD68-only
fine-tuning on the same old backbone. Neither experiment answers whether small
auxiliary gradients regularize the current FCMAE800 UNeXt2 trajectory.

Using public test DAPI for self-supervised adaptation is excluded: the written
rules prohibit exploiting test-set information and pretraining/test overlap.
Round51 uses training data only.

## Matched Mechanism Test

The machine-readable protocol is
`configs/round51_cv2_unext2_auxiliary_markers.json`. Both fresh recipes have
the same four outputs in the fixed order `CD68,CD45RO,HLA-DR,Vimentin`, start
from the same fold-local FCMAE800 encoder, and share the exact model shape,
seed, data order, target loading, validation, and training schedule.

- `auxcontrol` uses normalized marker weights `1.00/0/0/0`;
- `auxiliary15` uses `0.85/0.05/0.05/0.05`.

The zero-weight control is necessary: comparing only against the historical
single-output checkpoint would confound auxiliary supervision with the
four-output head shape and its changed random-number consumption. Both routes
select checkpoints by CD68 global-MSE and formal OOF evaluation extracts only
the CD68 channel. Auxiliary outputs do not enter the score.

All other settings match Round46/Round50: fold0/fold2, seed 2044, 14+14
epochs, batch 16, workers 4, halo64 supervision, spatial-thumbnail FiLM,
Uniform-7 hybrid loss, D4 inference, and JPEG q95. The fixed FCMAE800 encoders
are reused byte-for-byte; no pretraining is repeated.

## Fixed Candidates and Gates

Direct auxiliary benefit is measured against the matched four-output control,
not the historical single-output model. It requires at least `+0.05` q95,
positive float gain, 2/2 positive folds, at least 7/10 positive ROIs, and a
worst ROI above `-0.12`.

The fixed 60% replacement must also beat the retained Round47 two-fold
analogue by `+0.08`, improve both folds and at least 8/10 ROIs, and keep every
ROI above `-0.08`. A separate diversity route compares a 30% auxiliary member
against an otherwise identical 30% four-output control member before it may
compete with Round47. The auxiliary-specific diversity effect requires
`+0.03`; deployment still requires `+0.06`, both folds, 8/10 ROIs, and the
same ROI floor.

No marker subset, auxiliary weight, epoch, seed, checkpoint, ensemble weight,
or ROI route is searched. This two-fold pipeline cannot create a test package.
A passing recipe must be expanded unchanged to all five folds and pass the
complete-OOF gate before inference on official test inputs.

## Execution Status

The protocol validator now requires a non-empty unique training-marker list,
the primary protocol marker, and a same-length non-negative loss-weight vector
that sums to one. The common CV runner passes both fields to the existing
multi-marker training path. The evaluator already reconstructs models from
checkpoint marker metadata and selects CD68 by name, so no special inference
path is introduced.

Local validation passes Python compilation, JSON parsing, Git Bash pipeline
syntax, `git diff --check`, the exact eight-command protocol dry-run, and 99
focused protocol/training/UNeXt2 tests. Runtime commit `c8ea5bf` was
fast-forwarded to the server by a verified Git bundle. The clean Linux
environment passes the full `243 tests`, compilation, pipeline syntax, and the
same eight-command dry-run.

A physical-GPU1 smoke run used the real fold0 FCMAE800 encoder (SHA-256
`cfd93eaf4fb67b41b44b3bc829b5edd6a5d6ce253a6903599b0b2feae2a4aa04`),
64 training images, 32 validation images, batch 16, and workers 4. The saved
checkpoint contains the exact four-marker order and normalized
`0.85/0.05/0.05/0.05` weights, records 196 transferred encoder tensors, and
loads successfully. Its one epoch used four optimizer steps, took `25.488 s`,
and peaked at `25.541 GiB`. The process exited zero; trailing `pymp-*` NFS
cleanup errors were the known non-fatal worker-finalizer noise. The verified
smoke directory was removed.

The pre-smoke five-sample resource check found physical GPU0 at sustained
`95--98%` utilization with only about `19.4 GiB` free, while physical GPU1
held at `17--18%` with `32.251 GiB` free. GPU0 cannot fit the measured batch;
GPU1 retains about `6.7 GiB` of headroom. Formal scheduling will use a fresh
sample immediately before launch. If the state remains similar, the eight
stages and final OOF evaluation will run serially on GPU1 in detached Screen.
Scaling the Round50 stage times by the measured smoke-time ratio gives an
approximately eight-hour end-to-end estimate; this is a scheduling estimate,
not an experiment result.

The final five-sample check found physical GPU0 at sustained `96--100%`
utilization with only `15.591 GiB` free, while physical GPU1 remained at
`14--16%` with `32.251 GiB` free. The formal runtime therefore remains commit
`c8ea5bf` and uses serial physical GPU1 for all eight stages and OOF. It
launched at 2026-07-27 16:12:28 CST in detached Screen
`round51_auxmarkers`. Logs and the machine-readable exit marker are written
under `reports/`.

## Formal Outcome

The pipeline exited zero and the training log ends with the explicit
`COMPLETE` marker. All eight fold/recipe/stage runs contain exactly 14 metric
rows, 14 epoch checkpoints, and loadable selected checkpoints from runtime
commit `c8ea5bf`. Stage-2 selected epochs are 13/13 for the fold0/fold2
controls and 9/9 for the auxiliary candidates, so the outcome is not a
last-epoch artifact. Peak allocation was `25.657 GiB`. Training consumed
approximately `7.11 h`; OOF inference took another `2360.366 s`.

Formal OOF covers all 2,706 images from the fixed 10 ROIs. The absolute q95
scores are:

| candidate | q95 | float |
|---|---:|---:|
| FCMAE800 | 74.822793 | 74.898429 |
| four-output control | 74.533785 | 74.611489 |
| auxiliary15 | 74.855063 | 74.925777 |
| Round47 two-fold analogue | 74.960051 | 75.036295 |
| 60% auxiliary replacement | 74.982730 | 75.056210 |
| 30%/30% auxiliary split | 75.015320 | 75.089950 |

Auxiliary supervision beats its matched four-output control by `+0.321278`
q95 and `+0.314288` float. Both folds and 9/10 ROIs improve; the only negative
ROI is ROI000 at `-0.073715`, above the fixed floor. The mechanism gate passes.
Against the historical single-output FCMAE800, however, the direct candidate
gain is only `+0.032271` q95. Most of the matched-control delta repairs the
four-output zero-loss control's `-0.289008` regression rather than establishing
a large new deployable endpoint.

The auxiliary-specific 30% diversity comparison also passes: it gains
`+0.079084` q95 over the matched control split, improves both folds and 8/10
ROIs, and has a worst ROI of `-0.040256`. Against the retained Round47
analogue, the same candidate gains `+0.055269` q95 and `+0.053655` float, with
both folds and 8/10 ROIs positive and a worst ROI of `-0.050540`. It misses
only the preregistered `+0.06` magnitude threshold, by `0.004731`.

The 60% replacement gains only `+0.022678` q95, is negative on fold2, improves
6/10 ROIs, and falls to `-0.188265` on ROI000. Its gate fails. Consequently
`promotion.allowed=false`: no five-fold expansion, test inference, package,
or adjacent auxiliary/blend-weight search follows. The result is retained as
positive evidence for cross-marker supervision, but this fixed shared-decoder
recipe is not promoted.

The tracked OOF report SHA-256 is
`670c777411b6858f0d4eb2987b67b9f2acc46b13577095571d4ac68fee2ca696`;
the gate report SHA-256 is
`8511d41c01ff0e083f1a8b1398e9e139d898c5aeea7626143daf26afcda67b1c`.

# Round 63: Five-Fold Base Diversity Promotion

## Preregistered Decision

Round62 does not support a direct capacity claim: the official-initialized Base
model loses `-0.0927` q95 to the matched Tiny model over fold0/fold2. It does,
however, add stable ensemble diversity. The fixed `t70_base_split` candidate
gains `+0.1028` q95 and `+0.1003` float score over `t70_o50`, improves both
screen folds and nine of ten ROIs, and keeps its worst ROI at `-0.0276`.

Round63 expands only that passing recipe. The protocol retains the Round62 run
namespace so its audited fold0/fold2 Base checkpoints are reused byte-for-byte.
It trains fresh fold-local official-Base FCMAE800 and 14+14 supervised models
for fold1, fold3, and fold4. All architecture, optimizer, batch, loss,
checkpoint-selection, halo64 context, D4, q95, and seed settings remain fixed.

The candidate remains base32 `0.21`, base48 `0.09`, scratch Tiny FCMAE800
`0.35`, official Tiny FCMAE800 `0.175`, and official Base FCMAE800 `0.175`.
No weight, epoch, checkpoint, fold, or ROI route will be selected after seeing
the missing-fold results.

## Complete-OOF Gate

The five-fold gate compares `t70_base_split` against retained `t70_o50` over
all 25 ROIs. Promotion requires at least `+0.04` q95, positive float gain,
positive deltas on at least 4/5 folds and 20/25 ROIs, and a worst-ROI delta no
lower than `-0.08`. The launcher first verifies the exact passing Round62 gate
and binds the two-fold OOF SHA-256
`6047a9b3c5d0016dc12f930162129297cf771e572a93f04704eed4e4ddc452fe`.
Test inference remains blocked until the complete five-fold result and all
checkpoint provenance are audited.

## Validation And Launch

Local protocol tests pass `66/66`, together with Python compilation, JSON
parsing, Bash syntax, and diff checks. The synchronized Linux checkout at
commit `2c98cd7` passes all `277` tests and exact dry-runs containing three
Base FCMAE jobs and six supervised stages. All 20 historical five-fold
checkpoints and the two retained Round62 Base checkpoints are present; none of
the fold1/fold3/fold4 Base output directories existed before launch.

Five consecutive pre-launch samples found both physical GPUs at 0% utilization
with about 40.4 GiB free and no compute processes. Detached Screen
`round63_base_cv5` started fold1 on physical GPU0 and fold3 on physical GPU1.
The launcher assigns fold4 to the first completed GPU after a three-sample
utilization check, falling back to the other GPU at the phase boundary if the
first becomes externally busy. The initial post-launch sample measured
100%/98% utilization and about 23.5 GiB used on GPU0/GPU1. Both immutable
pretraining configs record commit `2c98cd7`, Base, micro-batch 16, two-step
accumulation, and effective batch 32. Completion remains unverified until the
exit marker, three 800-row pretraining trajectories, six supervised stages,
complete OOF, provenance hashes, and the five-fold gate are audited.

## Reboot Interruption And Decision

The GPU server later became unreachable and rebooted. The post-reboot audit
found no Round63 process, a dead `round63_base_cv5` Screen socket, no pipeline
exit marker, and no fold4 pretraining directory. Fold1 contains 136 metric rows
and fold3 contains 139, with their final recorded epochs taking 123.35 and
121.10 seconds respectively. Neither run has a completion marker. No
supervised stage, complete OOF report, distribution gate, or submission was
produced.

Round63 is therefore recorded as interrupted, not failed and not complete.
Although large `latest.pt` files remain for fold1 and fold3, their resumable
epoch and optimizer state have not been promoted or treated as evidence. At
the measured rate, finishing the three missing 800-epoch Base pretraining runs
would still consume roughly two days before supervised training and OOF. The
two-fold evidence estimates only a `+0.1028` q95 diversity gain, so the run is
not resumed in the current short-cycle queue. Its artifacts are retained for a
later explicit resume decision; they must never be mistaken for complete
five-fold evidence.

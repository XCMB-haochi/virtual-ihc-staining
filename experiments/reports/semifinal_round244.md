# Round244: one-epoch decoder-first adaptation

Question: does protecting the domain encoder while the random decoder learns
its initial mapping improve generalization? Domain800 helped all17 ROIs and
stage-wise decay improved both OOF and platform, motivating a training-phase
change rather than another decay factor. This is a hypothesis, not a diagnosed
bug or a high-confidence gain. No previous matched decoder-first experiment
was found; historical decoder-only auxiliary gradients are a different mechanism.

Primary research rationale: https://arxiv.org/abs/2202.10054 examines how joint
adaptation from a random task head can distort pretrained features and studies
head training before fine-tuning. Its classification/linear analysis is not a
staining result; our nonlinear decoder and fixed one-epoch duration are an
untuned adaptation, not an exact reproduction of that paper.

Fixed change: stage1 epoch1 freezes encoder parameters (no gradients/AdamW decay);
only decoder and conditioning update. Epoch2 unfreezes every encoder parameter.
Keep stage1 total14 epochs/cosine schedule, so encoder gets13 epochs rather than
adding training. Stage2 is entirely unfrozen with unchanged14 epochs, own-stage1
endpoint14 and final epoch12. Keep stage-decay0.8,domain800 bytes,seed2026,
4x4 effective16,loss/marker weights and all input context. Both full student and
folds are fresh supervised trajectories. Full stage2 uses unchanged Round240
OOF cache and distillation0.2; fold students never see that cache.

GPU0 only under the latest user allocation; Round137 stays stopped. Train
full candidate then five grouped folds sequentially. Prior folds cost roughly
75-120min for two stages; estimated8-12h on one available GPU, contention
may extend this. Both stages are needed to assess the changed initialization.
Full candidate is computed upfront to eliminate a second training handoff.

Compare complete2380-image D4 OOF with matched Round239 (same teacher recipe).
Preregister mean>=0.03 and at least3 positive folds for automatic confirmation;
report all ROIs. Passing packages the full student with the unchanged16-view
Round243 TTA andq95; no shift search. CV isolates training under standardD4;
interaction with full-data distillation and16-view TTA remains for bounded
platform confirmation. One final checkpoint, no cross-fold inference ensemble.
If negative, no adjacent freeze-duration grid. Retain Round24379.0967.

Tests cover actual first-epoch encoder immutability and decoder updates, second
epoch encoder updates, matched CLI settings and fold/full separation. Real-data
smoke uses two stage1 epochs specifically to exercise freeze/unfreeze and one
stage2 epoch; audit saved first encoder versus original FCMAE tensors exactly.
Formal checks require complete14-epoch logs, correct freeze masks, real updates,
finite tensors, source hashes and no train/val overlap. Code default freeze0
preserves prior training behavior. No changes to test-time model parameters.

## Preflight/resource boundary

Two targeted tests and45 training-safety regressions passed on Linux. Real
full-data smoke completed frozen epoch1 (peak1.76GiB); epoch2 was interrupted
by CUDA OOM after other GPU0 processes increased occupancy to leave about3GiB.
No other process was modified. This is not a failed mechanism result; a full
unfreeze/full-data/fold smoke is still required. Preserve the interrupted smoke
under transfers/round244_interrupted_smoke. The detached queue rechecks GPU0
for12000MiB free and utilization<=30 twice before each training stage. It first
reruns both smoke paths and only then starts formal jobs. Partial formal stages
still require review, not silent restart. Timing excludes resource waiting.

## GPU1 handoff (2026-10-03)

User explicitly reassigned remaining work to GPU1. Full-data stages1/2 had
completed on GPU0 (1933/2082 actual encoder-inclusive minimum steps); fold0
had not started. Stop only the waiting Round244 Screen after checking no
training child exists. Add explicit GPU selection and opt-in completed-stage
reuse with the existing full artifact audit; do not retrain completed stages.
All remaining folds, OOF and in-process16-view test inference use physicalGPU1.
Keep12000MiB free-memory checks, but shared compute is allowed by this request;
remove the low-utilization prerequisite. Resource contention affects timing.
Do not restore historical GPU0-only scheduling or resume Round137.

## Completed validation and explicit package request (2026-10-04)

All twelve stages completed and passed audits; pipeline exit0. Complete2380-image
OOF CD45RO score78.7123627, +0.0259411 versus matched Round23978.6864216.
Fold gains+0.047661,-0.087061,+0.011079,-0.010464,+0.141955;3/5 positive.
The original automatic mean>=0.03 gate failed; preserve that decision unchanged.
User explicitly requested one fixed package and local synchronization despite
this near-threshold result. Use existing full-data epoch12 and unchanged16-view
TTA, zero residual output offset, q95. No retraining or parameter search.
A separate confirmation entry audits completed full stages and packages directly.
Round24379.0967 was retained pending platform confirmation.

## Confirmation package ready

Separate confirmation exit0. Package copied locally:
submissions/semifinal_round244_cd45ro_q95.zip (3250333 bytes).
SHA256 b75f19d0ece9ded51c08a8d1c1e01f4af1fed1cdc467b1b09c90b96ec2f623a0.
Verified560 unique expected names, RGB256x256 JPEG, full image decode, ZIP CRC
and matching remote hash. Mean decoded intensity16.6252 and mean absolute
change versus Round2430.70013 gray levels; these are output-integrity checks,
not test-GT evaluation. Fixed epoch12 model SHA is in round244_hashes.json.
## Platform confirmation (2026-10-04)

User reported79.0940, SSIM0.899, PSNR26.928. Relative to the matched16-view
Round24379.0967, score changes-0.0027 and displayed PSNR-0.005; displayed SSIM
is unchanged. Treat this as effectively tied, not a material regression or
evidence that decoder-first adaptation is universally ineffective. The small
OOF improvement did not produce a higher platform score in this comparison.

Keep Round243 as primary and Round240 as standard-D4 fallback. Close this fixed
one-epoch decoder-first configuration without freezing-duration, epoch or LR
search. Preserve the trained artifacts and original automatic gate result.

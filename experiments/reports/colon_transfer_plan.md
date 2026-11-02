# Colon transfer experiment plan

Status: user selected the complete fresh seed2026 workflow. All six scopes
(full-data plus five folds) retrain DAPI and feature-teacher encoders from the
public checkpoint. Both student arms use2026 and share identical within-scope
initialization. Existing Round241 is not interrupted. Implementation is
Round137; no historical seed2044/2027 trained artifacts enter its dependency graph.

## Questions and fixed controls

1. Does the current stage-wise encoder LR policy transfer to Colon? Train
matched baseline (stage decay1) and candidate (stage decay0.8), both seed2026,
from byte-identical domain encoder and feature teacher within each data scope.
Preserve Colon CD68-primary weights0.85/0.05/0.05/0.05, original relational
loss0.05, batch16 without accumulation substitution,14+14 cosine and halo64.
Use seed2026 for every newly run training stage. Do not bring Liver or Stomach
images, teachers, targets or trained parameters into Colon training.

2. Does the historical offset still help? For each evaluated checkpoint,
compute paired zero/[0,-1] results from the same float D4 predictions before
q95 encoding. This needs no second training run and must report overall,
fold and ROI deltas. Zero offset is the primary development score and the
only new submission-package setting. Shifted predictions are TRAIN/Val-only
diagnostics, not a submission or an input to choosing sample-specific offsets.
This does not reveal the hidden-test difference for the exact Round136 model.

3. Prepare full-data single models after choosing seed scope. Use fixed
stage2 epoch5/6/7 average1:2:1 (the existing Colon temporal rule), never search
windows using hidden scores. Keep single-model architecture and parameter
averaging distinct from forbidden prediction mixtures. Preserve original
Round136 and zero-offset candidate packages separately.

## Validation and handoff

Run all five matched grouped folds from the established25-ROI split, as
explicitly requested for the complete workflow; no two-fold stop gate. No
full-data target cache may enter a fold student: its teachers saw outer-val
ROIs. Report both historical uniform7 and provisional Gaussian11 sigma1.5,
population/valid metrics; no assumption that future Colon scoring stays the
same. Report zero-offset changes in both metric profiles and every fold/ROI; choose
neither a metric nor a subset after seeing results. Keep both baseline and
candidate packages, with any preference grounded in the full grouped evidence.

Full-data baseline/candidate use their own stage1 weights, with one selected
teacher per held-out ROI for training-only CD68 distillation0.2. For a strict
single-change structural comparison, keep the same baseline teacher cache
between the full-data arms; refreshed-candidate teachers would be a separately
identified change. No teacher input participates in inference. Final packets
contain1346 test images if the released Colon input inventory is unchanged.

## Resource and provenance constraints

Historical full-data DAPI630 pretraining measured about70.5s/epoch (~12.3h),
not including feature-teacher160 five-modality pretraining or supervised runs.
Thus a fresh whole-pipeline2026 run cannot be promised overnight. Full fold
pretraining800 is additional and must exclude each validation ROI. Reuse is
valid only with explicit scope and seed provenance. Match original true batch16
for relational comparisons; do not silently change batch semantics for memory.
Launch only after live GPU capacity allows, without altering Round241 or others.
Archive transfers under transfers/. Use stage logs/status and resumable FCMAE
checkpoints; verify scope, hash and final actual optimizer updates.

Round137 implements the complete dependency graph, source manifests, paired
zero/shift evaluation, fold checks and both full-data output packages. All tasks
are automatically scheduled only after Round241 exits, under live resource
checks. Pretraining requires24500MiB free; relational supervision requires32000MiB
and retains true batch16. A busy card is not disturbed. When insufficient memory
is available, status explicitly reports waiting-resources rather than training.
Completed stages have checked output hashes. FCMAE can resume from its saved
optimizer/scaler/RNG checkpoints; partial supervised runs require review and
are never silently restarted or reused. On error, stop launching new tasks,
allow already-running own tasks to finish, then exit nonzero.

Both full-data arms share the fresh baseline-fold teacher cache to isolate the
stage-decay comparison. Per-fold source averages5/6/7 are fixed, rather than
using each fold's historically selected best epoch. This differs from the old
Round136 teacher-selection rule and must not be called an exact2026 replica.
It is a matched2026 experiment with a fixed common endpoint. Final models keep
Colon relational guidance; no Liver artifacts participate. Fold checkpoints
are never mixed at inference. Only TRAIN/Val paired offsets are evaluated.

There are12 pretraining jobs plus24 supervised stages. This is a multi-day
workflow and cannot be promised before October5 on shared GPUs. Colon work
prepares the later combined stage, not an additional current Liver submission.
Full models depend on complete OOF targets, so final distilled packets cannot
be available until the fold work is complete. Initial full-scope pretraining
is prioritized, but this does not remove that dependency.

## Registered launch (2026-10-01)

Use the reserved Colon range1-199: this experiment is Round137. The original
queue was stopped before any Colon training began and replaced with the
correctly numbered queue. No datasets or trained weights were moved; Round241
was not interrupted. All executable/config/test/output paths use Round137.

Local and Linux graph/command tests passed. Server preflight verified6296 Colon
training images,1346 test inputs,complete disjoint25-ROI fold coverage,matching
training marker names and the public initialization SHA256. Archive SHA256
matched. Detached Screen round137_colon_complete is confirmed alive and logged
waiting for Round241 to finish. No Colon training has started at this check;
resource-aware scheduling will begin afterward. This is an actual detached
queue, not a claim that pretraining is already running. Source launch commit
is stored in reports/round137_launch_commit.txt. No performance result exists.

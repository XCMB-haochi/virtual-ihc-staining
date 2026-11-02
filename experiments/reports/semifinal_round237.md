# Round237: corrected source-context screen

Audit found the earlier Screen queue had exited before training: no run or
result artifacts existed. The exact shell failure was not logged. The earlier
runner also reused a full-data stage2 command (validation leakage), declared
840 instead of980 validation images, and compared two-fold overall scores
against five-fold overall scores. None of its proposed results is usable.

Corrected recipe: fresh supervised folds0/3 from their exact fold-excluded
Round207 DAPI800 encoders; both stages exclude the validation ROIs. Use the
non-distilled Round207/Round217 fold baselines. Global Round216 teacher targets
are full-data-student-only and cannot safely validate an outer-fold student.
Change source halo64 to128 only; target halo64, micro4 accumulation4, seed2026,
14+14 cosine, own-stage1 epoch14, stage2 epoch12, marker weights and losses
remain matched. D4/halo128/offset0/q95 evaluation covers the same980 images.
Compute the baseline overall over these seven ROIs, not the full17.

This is a current-stage transfer test, not a known implementation correction.
Colon Round86 found this training change negative (-0.267637 direct q95);
Round36's different architecture had a small positive comparison. The earlier
unqualified claim of positive historical evidence was misleading. Changed organ,
marker and metric justify at most this bounded screen; no high gain is promised.
Both folds must improve to consider expansion; report mean and all ROI changes
without rewriting the criterion. No submission or automatic full-data launch.

Validate fold exclusion, source hashes, actual optimizer updates, own-stage1
restoration and the matched comparison before accepting results. A two-stage
smoke measures actual memory. Schedule from live GPU measurements; retain logs
and write pipeline exit only on termination, never write success at startup.
The baseline two-stage folds previously cost roughly1-1.5h each; halo128 costs
more. Screening two avoids three additional fold runs if rejected. Timing under
shared GPU compute remains uncertain. Best package stays Round21679.0198.

## Corrected execution

Local and Linux regression tests pass, including rejecting full-data/distillation
commands and ensuring the overall comparison uses only the shared ROI universe.
The deployed Git archive SHA256 matches locally. First-stage real-data smoke
performed7 actual Adam updates with peak11.65GiB. GPU1 cannot safely fit this
with current free memory; use physical GPU0 sequentially for folds0 and3.
Both GPUs have competing compute. The smoke took46.6s for128 training images
plus8 validation images; this suggests several hours for the full screen, not
a reliable idle-GPU timing. Stage2 smoke also passed8 actual updates, strict own-stage1 initialization,
fold exclusion checks and11.77GiB peak. Both stage losses were finite.
Launch the corrected Screen job on GPU0; no interim polling is needed.

## Stage-boundary recovery

The corrected run completed fold0 stage1 then exited1 in its audit: val_images
and train_images belong to config.json, not checkpoint args. The smoke skipped
this full-dataset count branch. No stage2 or OOF result existed. Fixed the audit
to read config counts and added a regression with checkpoint args lacking them.
Completed-stage reuse is opt-in and verifies all explicit command options,
ROI exclusion, consecutive finite losses, validation counts, actual optimizer
steps, and last/epoch14 model tensor equality before skipping any work.

Real formal artifact audit passed: fold0 stage1 completed14 epochs,1960 training
and420 validation images,1718 actual updates. Epoch14 SHA256:
3d61b1ea4aba0013dadc010c155932ff37446a42e7e492147c93520e229018f4.
Resume uses this exact checkpoint for stage2 without altering any training
command or rebuilding stage1. Run remaining stages and980-image evaluation
on GPU0, which has sufficient memory. No gain or gate outcome exists yet.

## Evaluation interface recovery

All four stages completed14 epochs with finite losses and actual optimizer
updates1718/1718 for fold0 and1592/1592 for fold3. Evaluation initially failed
because the runner's manifest method string differed from the evaluator's
required fold-excluded-single-recipe-d4-teacher contract. Corrected the producer
metadata; no weights, model recipe or evaluator requirements were relaxed.
Re-audited completed stages and ran evaluation only, retaining the failure log.

## Completed decision

Pipeline exit0 after evaluation interface repair. Complete980-image D4 OOF:
CD45RO mean gain+0.1469568; fold0 -0.0871951, fold3 +0.3225708. Four of seven
ROIs improve; all three fold0 ROIs decline; worst ROI087 -0.1484135. This is
mixed-positive evidence, not a proven useless mechanism. The registered
both-folds-positive expansion condition fails. Do not expand or package;
retain the artifacts and proceed to the bounded existing-weight Round238.
Evidence: round237_d4_oof.json and round237_gains.json.

# Round235: conditional full-data handoff after ROI-condition validation

The user asked to accelerate compliant improvement after a leaderboard change.
The organizer's statement that data were shuffled remains underspecified:
Round233 TRAIN seam continuity does not identify ROI remapping, source/sample
mixing or the Train/Test reassignment scheme. Do not claim the shuffle is
fully understood or infer label corruption. Rank alone supplies no evidence
about other teams' methods.

Round234 was kept unchanged. Before viewing its completed OOF result,
we registered one automated confirmation decision: mean CD45RO gain>=0.05 and
at least3/5 positive folds permits a full-data candidate; otherwise stop
for review. This spends bounded compute, not an automatic promotion, ROI
veto or a new rule for every experiment. Check complete2380-image coverage
and a successful predecessor exit before the decision.

Conditional recipe: reuse the exact full-data DAPI800 encoder, freshly train
stage1 with explicit ROI descriptors masked inside the model, then stage2
from its own epoch14. Retain the original Round216 CD45RO OOF targets and
0.2 pixel distillation, four-channel order/weights,4x4 effective16, seed2026,
14+14 cosine schedules, epoch12 endpoint,D4/halo128/offset0/q95. The condition
ablation plus distillation interaction is not independently validated by
Round234; one fixed platform confirmation will assess this candidate.
Teacher inputs are training-only, model inference remains DAPI-only and
uses one checkpoint. No label edits, output correction or epoch search.

Validate560-image ZIP and hashes; manual submission only. Round21679.0198
is retained. The detached handoff waits for Round234 rather than requiring
the user to report completion before training starts. Use GPU1 only once
the predecessor ends and>=12,000MiB is free. If the gate fails, leave no
new training/package and record the reason. Estimated added cost1.5-2h.

Tests verify command equivalence to the retained recipe except the explicit
model descriptor mask and artifact paths, plus both parts of the decision
rule and missing-fold rejection. Two-stage execution smoke precedes launch.

Smoke passed both stages with7/8 actual Adam updates, strict own-stage1
restore, explicit CD45RO distillation index1 and6.99GiB peak allocation.
This verifies execution, not improvement. Original Round234 workers and
their recipe were not changed by preparing the handoff.

## Completed confirmation candidate (2026-09-30)

Both pipeline exits are0. Round234 mean gain+0.1810213 with4/5 positive
folds met the registered condition. Round235 completed14+14 epochs with
finite training losses and verified actual optimizer updates. Stage2 config
confirms2380 training images,0 validation images, seed2026, effective batch16,
the local-condition architecture, own-stage1 epoch14 initialization, and
CD45RO distillation index1/weight0.2. Selected checkpoint remains epoch12.

Package: `submissions/semifinal_round235_cd45ro_q95.zip`; copied locally and
verified560 unique CD45RO JPEGs, RGB256x256, successful ZIP CRC/image decode
and SHA256 matching the remote manifest. Checkpoint/package hashes are in
`round235_hashes.json`; decision in `round235_decision.json`.

Decision: submit this one fixed candidate for platform confirmation. Round216
79.0198 remains best until confirmation. Do not assume the OOF gain transfers
to this full-data distilled student. Pause follow-on changes pending that
bounded confirmation; no further training launched in this completion audit.

## Platform confirmation (2026-09-30)

The package scored `78.9756` (`SSIM=0.898`, `PSNR=26.831`), which is
`-0.0442` below Round216. This closes the local-condition full-data route for
the current stage. The positive Round234 OOF result remains mechanism evidence,
but did not transfer through the full-data student and is not a basis for
further mask or distillation tuning.

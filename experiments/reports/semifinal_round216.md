# Round216: global OOF teachers and one full-data distillation candidate

The user approved replacing nested student validation with one fixed
full-data candidate and a manual platform confirmation. Reuse unchanged
Round207 fold0/fold3 CD45RO domain teachers; complete fold1/fold2/fold4 with
exactly the same recipe: fold-excluded DAPI FCMAE800,8x4 effective32,
14+14 supervised epochs,4x4 effective16, seed2026, CD45RO-heavy weights,
uniform7 loss, stage1 epoch14 and stage2 epoch12. No relation guidance.

Build exactly2,380 D4/halo128 CD45RO targets, one prediction per image from
the teacher that excluded its whole ROI in pretraining and supervision.
Derive the output channel by marker name via the existing checked predictor.
Verify hashes, exclusion sets, exact filename coverage, counts and storage.
Never use this global cache for outer-fold student validation: it is for
the full-data student only. No independent student OOF gain will be claimed.

Reuse Round208's exact full-data stage1 epoch14 and reset stage2 optimizer.
Preserve all Round208 stage2 settings, including4x4 effective16,14-epoch
cosine period, uniform7 loss and fixed epoch12 endpoint. Add only0.2 pixel
distillation with explicit CD45RO channel routing. Inference remains one
checkpoint,D4,halo128,offset0,q95. Validate one560-image package. No epoch,
weight, teacher-mixture or leaderboard search; submit manually once against
the retained Round20878.9741 reference.

GPU0 was conditionally authorized only if the job could finish within10h.
Live GPU0 free memory was25.3GiB, sufficient for this microbatch recipe,
but existing teacher measurements were5.46-5.90h pretraining plus1.3h
supervision each. With3 teachers and2 GPUs the longer sequential queue is
about14h before final student/inference, so the10h condition was not met.
GPU0 is not used. GPU1 runs the originally authorized replacement, expected
roughly22-25h depending on load, rather than the10 nested teachers.

Tests verify new teacher commands retain the old recipe and the student
differs from Round208 only in output path and explicit distillation options.
Marker-gradient/coverage tests from the preceding repair also pass. Existing
partial Round215 artifacts are preserved, excluded from this experiment.

Real full-data-path execution smoke used128 permitted training images,
Round208's stage1 weight and the exact4x4 student batch configuration. It
completed6 actual Adam updates, saved CD45RO index1 and peaked at6.99GiB.
Training labels were used as synthetic smoke targets only; this verifies
execution, not efficacy. Formal targets come only from the five held-out
teacher predictions and exact coverage is checked before student training.

## Completed

Exit0. Three new domain teachers completed800+14+14 epochs each with finite
losses. Five teacher hashes and held-out ROI metadata verified. Target cache
covers2,380 unique training filenames exactly once. The full-data student
completed14 epochs,2,082 actual Adam updates, correct CD45RO index1 and
weight0.2, zero validation images, microbatch4/accumulation4 and the original
Round208 stage1 initialization hash. Selected epoch12 model is finite.
Summed epoch training times are21.27h; inference/IO add overhead.

Package `submissions/semifinal_round216_cd45ro_q95.zip` has3,200,570 bytes;
local server-hash match, ZIP CRC,560 unique CD45RO RGB256x256 images passed.
Hashes are recorded in `round216_hashes.json`, concise audit in
`round216_completion_audit.json`. Recommend one manual platform confirmation
against78.9741. No independent student OOF improvement is claimed.

## Platform confirmation

User reported **79.0198**, SSIM **0.899**, PSNR **26.833**. This improves
Round208's78.9741 by **+0.0457**. Round216 is the current platform-best
single-model package; preserve its hash and do not alter its checkpoint for
post-hoc search.

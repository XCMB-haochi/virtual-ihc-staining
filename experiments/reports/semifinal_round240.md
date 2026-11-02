# Round240: refresh the full-data distillation targets with validated teachers

Current best is Round23979.0628. Its fold-local teachers improved complete
2380-image grouped OOF by0.133787 (four positive folds), but its full student
still used old Round216 OOF targets. Test one fixed teacher replacement using
all five Round239 epoch12 teachers. No old/new mixing, fold-specific teacher
selection or distillation-weight search. Fold1's negative teacher change is a
known limitation; higher aggregate teacher score does not guarantee a better
student. This is a concrete use of existing positive validation, not an
inference from other teams' scores.

Each TRAIN image receives one CD45RO target only from the model excluding
its whole ROI, using the unchanged D4/halo128/offset0 and lossless uint8 cache
representation. DAPI remains the sole model input. Original marker files are
untouched; teacher predictions supply the same separate training loss as before.
These targets are only for full-data student training, never outer-fold validation.

Reuse the exact Round239 full stage1 epoch14, reset stage2 optimizer as in its
matched control, retain stage decay0.8,14-epoch cosine horizon, endpoint12,
seed2026,micro4 accumulation4,distillation0.2 and CD45RO index1. Change only
the teacher cache and output paths. Final inference loads one student weight
file; no fold predictions or teachers participate. D4 equal weights and q95 fixed.

No additional five-fold training: reuse the completed teacher OOF evidence and
run one stage2 full-data candidate followed automatically by inference and
packaging. This avoids another multi-hour validation ladder. Full-data student
generalization is not independently validated; one bounded platform comparison
will confirm this preregistered candidate. Estimated1-2h plus cache/inference,
subject to live server contention. Retain Round239 pending confirmation.

Verify command equality except cache/output paths, source teacher hashes against
Round239 OOF, excluded ROIs,2380 target coverage,stage1 identity,actual updates,
finite losses and final560-image artifact. This uses the established Round216
cache builder algorithm and unchanged tested training code. No new architecture
or hyperparameter grid. Rule source: current docs/复赛注意事项.txt; training-only
auxiliary targets are not additional inference inputs or edited marker labels.

## Launch (2026-10-01)

Connectivity restored. Deployment archive SHA256 matched the local Git archive;
shell syntax and Linux command-equivalence regression passed. Real-artifact
preflight verified all five teacher hashes against Round239 OOF, fold exclusion
and exact full-data stage1 identity. Launched detached Screen
round240_teacher_refresh on physical GPU0; GPU1 has competing compute.
Pipeline builds targets, trains stage2 and packages the fixed epoch12 without
intermediate intervention. No performance claim until completion/confirmation.

## Completed

Pipeline exit0; all14 stage2 epochs completed with2081 actual optimizer updates.
Fold-excluded teacher targets cover2380 training images; teacher hashes match
the evaluated Round239 OOF sources. Stage1 identity, full-data configuration,
distillation weight0.2/channel1 and fixed epoch12 output passed runner checks.

Package `submissions/semifinal_round240_cd45ro_q95.zip` is copied locally.
Verified560 unique CD45RO RGB256x256 JPEGs, full image decode, ZIP CRC and
SHA256 matching `round240_artifacts.json`:
`a4de7fcf2f74444ceb178bc2b39b98d5c8844ac722af8ff755bf1006ccd3259b`.

## Platform confirmation

User reported79.0903, SSIM0.899, PSNR26.929: +0.0275 over matched Round239
79.0628 (PSNR+0.073; SSIM unchanged at the reported precision), and+0.0705
over Round21679.0198. Promote Round240 as the current primary single-model
package; retain Round239 as fallback. Local ZIP SHA256 matches the manifest.

This fixed comparison supports refreshing the teacher targets for this recipe;
it does not establish a general teacher-selection rule or an independent
student OOF gain. No teacher mixtures, distillation-weight or epoch search
is opened by this result.

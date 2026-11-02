# Round239: stage-wise adaptation of the useful domain encoder

Question: can slower low-level adaptation preserve useful domain800 morphology
while high-level encoder and decoder layers learn the staining mapping? Round217
found domain pretraining +1.03110 across17/17 ROIs; this motivates preserving
features, but does not prove catastrophic forgetting. Colon uniform0.1x encoder
LR and other-backbone LR tests are historical negatives, not positive evidence.
This is a different, modest stage-wise policy on the current organ, not a sweep.

Reference: ConvNeXt V2's original fine-tuning procedure uses layer/group LR decay:
https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md
https://openaccess.thecvf.com/content/CVPR2023/supplemental/Woo_ConvNeXt_V2_Co-Designing_CVPR_2023_supplemental.pdf
Its classification results do not establish a virtual-staining gain. Our fixed
stage factor0.8 is a design choice, not a tuned or optimal value. Encoder
stem/stages0/1/2/3 scales are0.4096/0.512/0.64/0.8/1; decoder LR stays1.
All parameters remain trainable. Keep decay1e-4, 14+14 cosine periods, seed2026,
micro4 accumulation4, domain800 initial bytes, marker weights, hybrid uniform7,
train halo64, own-stage1 epoch14 and stage2 epoch12; D4/halo128/offset0/q95.
No architecture, output correction, new data, teacher, seed or epoch search.

Cost/decision: both GPUs currently have ample memory. Run full-data fresh stage1
and distilled stage2 on GPU0 followed by folds0/3; GPU1 runs folds1/2/4. This
spends an extra full-data run even if validation fails to reduce time to a usable
candidate. Previously each two-stage fold took about1-1.5h; expect3-5h total plus
OOF/inference if contention stays low. No intermediate decision is required.

All five validation models are non-distilled and exclude their complete ROIs,
including encoder pretraining. Compare complete2380-image OOF to Round217.
Before results: mean gain>=0.03 and at least3/5 positive folds authorizes one
fixed platform confirmation; report all ROI changes. This is not a significance
claim or automatic best replacement. When condition passes, package the already
trained full-data distilled student. Its teacher cache remains the exact
Round216 cache, only for full-data training, never CV. Distillation interaction
remains unvalidated by CV and needs bounded platform confirmation.

Tests cover exact parameter partition, decay/default behavior, cosine rates,
fold exclusion and stage initialization. Real-data smoke covers both stages of
one fold and the full-data path, with count checks read from config.json,
actual Adam updates and all six initial group rates checked in saved artifacts.
Use established evaluator manifest contract and package checker. Retain
Round21679.0198 until actual platform improvement. If negative, no adjacent
factor or frozen-layer grid is opened.

## Verified launch preflight

Three new optimizer/command tests and ten existing optimizer/model tests passed
on Linux (new tests also passed locally). Both full-data and fold0 real-data
smokes completed both stages with7/8 actual Adam updates. Full peak6.99GiB,
fold peak6.99GiB. Saved stage rates exactly matched the fixed six-group policy;
strict own-stage1 restore, CD45RO distillation index1 only on full stage2, and
fold exclusions passed. Deployment archive SHA256 was verified. Launch both
physical GPUs with the registered queues and no intermediate polling.

## Completed (2026-10-01)

Pipeline exit0. All twelve stage runs completed14 epochs with finite losses,
nonzero actual optimizer updates, correct six-group rates, fold exclusions,
own-stage1 initialization and full-data-only distillation. Full-data training
took4615s of recorded epoch time. Validation covered all2380 images.

CD45RO Gaussian11 score78.6864216 versus matched Round21778.5526345:
mean+0.1337871. Fold0..4 gains+0.139679,-0.201002,+0.078044,+0.398116,
+0.157938. Positive4/5 folds and12/17 ROIs; worst ROI093 -0.366883.
This passes the registered fixed confirmation rule, not automatic promotion.
The subgroup regression remains material and platform transfer is uncertain.

The full-data single checkpoint at stage2 epoch12 generated
`submissions/semifinal_round239_cd45ro_q95.zip`. Local copy verified560 unique
CD45RO JPEG images, RGB256x256, image decoding, ZIP CRC and SHA256 matching
`round239_hashes.json`. Evidence also includes `round239_oof.json`,
`round239_decision.json` and `round239_training_audit.json`.

## Platform confirmation (2026-10-01)

User reported79.0628, SSIM0.899, PSNR26.856: +0.0430 versus Round21679.0198
(PSNR+0.023; reported SSIM unchanged at three decimals). Promote Round239 as
the current primary single-model package; retain Round216 as fallback. Local
ZIP SHA256 was rechecked against round239_hashes.json and matches.

Both grouped validation and this fixed platform comparison support the tested
stage-decay recipe. They do not prove the hypothesized feature-preservation
mechanism or guarantee transfer to another dataset. No adjacent decay-factor,
epoch or weight grid is authorized by this result.

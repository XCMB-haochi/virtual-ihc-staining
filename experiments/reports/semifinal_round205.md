# Round205: fixed endpoint confirmation and full-data training

Question: does Round204's common CD45RO-primary endpoint improve over the
Round200 CD45RO channel under identical D4, halo128, offset0 and q95?
Evaluate all five stage2 epoch12 checkpoints with the existing all-marker
audit, then compare CD45RO against `round202_all_marker_metrics.json`.
Report all paired fold/ROI changes. If the mean gain is positive, spend one
full-data training run; otherwise stop for review. This is a cost decision,
not a significance test or automatic promotion, and no ROI-count gate is
borrowed from Colon. The endpoint is fixed before D4 evaluation.

Conditional full-data recipe: Round204 architecture, immutable public
encoder artifact, four-channel order and CD45RO-heavy loss weights,
seed2026, microbatch4/accumulation4, 14-epoch cosine periods in both stages,
uniform7 hybrid loss, no pixel distillation or relational guidance.
Stage2 initializes the full-data stage1 epoch14, and the final submitted
model is stage2 epoch12. Keep cosine T_max14; do not shorten it to 12 and
thereby change the selected trajectory. No parameter soup is introduced.
Train on all 2,380 new-stage images and infer using one model, D4, halo128,
offset0, q95. Package 560 CD45RO outputs and validate ZIP CRC/structure.

Full-data epoch counts approximate the grouped-validation clock; more
training ROIs mean more optimizer updates per epoch, so they are not a
proven full-data optimum. Platform performance must be confirmed separately.
Round203 (77.595) remains the reference until then. Do not automatically
submit or use the platform to search epoch windows.

The supervised path and memory footprint were verified in Round204. The
new runner's full-data flags and unchanged cosine periods have a focused
test. GPU1 only; concurrent compute permitted with memory headroom.
Run detached as `round205_cd45_full`; output
`submissions/semifinal_round205_cd45ro_q95.zip` only if the OOF check passes.

## Completed

Pipeline exit 0. Matched CD45RO D4 OOF is 77.52153, SSIM0.876210,
up 0.25611 over Round200's CD45RO channel. Fold deltas:
+0.21305,+0.31709,+0.13272,-0.34006,+0.93139; 11/17 ROIs improve.
ROI072 gains 3.04106 and accounts for much of the mean gain; excluding it
leaves a mean gain of approximately 0.08205. Worst ROI loss is -0.68007.
This supports one platform confirmation, not a universal robustness claim.

Both full-data stages completed 14 epochs on 2,380 images with no validation
images, finite training losses, and peak allocation below 7 GiB. Their epoch
times sum to about 71.4 minutes. The final checkpoint is stage2 epoch12,
seed2026, initialized through full-data stage1 epoch14 as registered.
One CD45RO package was produced (3,217,945 bytes, 560 RGB 256x256 images).
The local copy passed server SHA256 matching, ZIP CRC and every image check.
Package and weight hashes are in `round205_hashes.json`; full OOF and paired
deltas are in `round205_round204_d4_oof.json` and `round205_decision.json`.
Platform confirmation is recorded below.

## Platform confirmation

User reported **77.9434**, SSIM **0.888**, PSNR **26.309**. Round205 improves
Round203 by **0.3484** and is the current confirmed single-model reference.
Keep its package and exact checkpoint; no checkpoint-window search follows.

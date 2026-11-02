# Round324: fresh full-data paired-crop refit

Selection: Round323 best grouped-Val epoch23, score76.2781952 (+0.0109587 vs318).
One fixed full-data candidate; weak/heterogeneous evidence in round323_result.md.
Do not change settings using Test results. Keep319 highest reported artifact.

Use all10356 official Train images from the first supervised update. Rerun
stage1 epochs1-14 from seed2026 task initialization and the existing fixed
Round302800 encoder; no prior supervised checkpoint is loaded. Stage1 uses
original full256 paired D4 hard-label training, organ0.1/0.2/0.7, LR3e-4 cosine14.
Then restore this new stage1 final model/RNG and reset AdamW as in the validated
recipe. Stage2 epochs15-23 uses LR1e-4 cosine horizon14, teacher0.2,
geometry consistency0.2, and probability0.5 paired integer240 crops. Reuse the
same fullTrain teacher target cache as309; original GT unchanged. No validation
evaluation or checkpoint choice during final fitting. No repeated pretraining.

Artifacts: runs/round324_stage1 and runs/round324. Source-split equality,
stage1 initialization/source,9072 stage1 updates,5832 stage2 updates, crop schedule,
finite weights and cross-stage provenance checked before packaging. Smoke runs
all14 short stage1 epochs then one stage2 epoch to test the actual transition;
smoke is isolated and never a submission source.

Package uses complete official256 input, one shared model/checkpoint, one FP32
forward, fixedRGB8 JPEG95,2186 images. No crop/TTA/postprocessing at inference.
Save submissions/semifinal_round324_mixed_q95.zip, atomic ready manifest. Local
AIC-Round324-Download checks SHA256/ZIP CRC/count and writes .part then final.
No platform submission automation or subsequent experiments.

Cost:323 measured614.5sec/grouped epoch. Full stage2 projected~13min/epoch;
fresh14-epoch full stage1~1.5h, nine stage2 epochs~2h, plus setup/inference/delivery.
Estimate4h, allow5h. Starting October31 night leaves time before November1 20:00.
Remote task has8h limit and per-epoch resumable states.
Launch October31 22:37 Beijing after user-authorized resume: both14-epoch small stage1 and cropped stage2 smoke completed; transition/optimizer/provenance checks rerun successfully without repeating completed smoke training. Formal stage1 process/config confirmed full_train=true, smoke=false, initial_checkpoint_sha256=null. Training source revision in round324_launch_commit.txt. Local delivery resumed only after CREATE_NO_WINDOW/hidden-startup subprocess fix passed actual Python/PowerShell no-console tests; no submission package exists yet.

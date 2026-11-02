# Round303: restore the Liver training recipe within the current model

Round302 completed both192-cycle runs with valid weights/steps/target counts.
Mixed CD68/CD45RO/CD68 best75.4728314 at32; uniform CD68 best74.5298766 at48.
Mixed wins by0.9429548; fix that marker policy. Colon73.09535 vs73.33333,
Liver76.64852 vs72.52810 (different targets), Stomach75.47656 vs75.27275.
Mixed improves2/3 Stomach ROIs but loses ROI000 by0.30742; no universal gain.

One grouped-Val comparison, no Test use. Reuse completed Round302800 encoder,
exact split and unchanged single-image conditional architecture. Restore prior
Liver recipe elements:14+14 full Train traversals, fresh AdamW each stage,
stage1 lr3e-4 then stage2 lr1e-4, epoch cosine to0, encoder depth decay0.8,
effective batch16, original CompetitionLoss uniform7 with0.7/0.2/0.1 weights.
Each organ-qualified Train image appears exactly once/epoch (8084 total), not
score-weighted replacement sampling. Stage2 starts from stage1 endpoint14,
not its best validation checkpoint. Evaluate Gaussian11/JPEG95 each epoch;
retain best shared checkpoint over the trajectory, with per-organ/ROI results.

This is a recipe-bundle comparison, not a single-factor ablation or exact
reproduction of the old architecture. Keep batch1/accum16, BF16 and clip1 from
the current baseline; old Liver used micro4/accum4, FP16 and no clipping.
No context, TTA, auxiliary marker labels, old competition weights, distillation,
EMA or checkpoint averaging. The old/current loss coefficients were identical;
the training SSIM implementation differed. No training batching optimization.

Expected28x8084=226352 exposures and28x506=14168 optimizer updates; around
2.5-3h based on current25-27 images/s plus validation/checkpoints. Two-stage
smoke checks transition, target coverage, finite updates and evaluation before
launch. Save resumable optimizer/scheduler/RNG every epoch. Scheduled Task runs
Python directly so a separate PowerShell controller is unnecessary. No automatic
follow-up chain; review Val against Round302 before full-data training.

October28 launch: local/remote schedule and coverage tests passed. Two-stage
real-data smoke completed with finite losses, correct group rates and two
updates per stage. Deployed source SHA verified. Installed matching
torchvision0.23.0+cu128 without changing torch2.8.0+cu128. AIC-Round303 starts
Python directly with --log --resume; logs reports/round303.log/.err. Exact source
commit is recorded in round303_launch_commit.txt. No pretraining rerun.

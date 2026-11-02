# Round314 corrected full-data refit

Reuse Round309's all-Train epoch14 state (10356 images per epoch, 9072 optimizer
updates), restoring model and RNG before the unchanged stage2 optimizer reset.
Check source split, first14 metric rows, recipe and finite checkpoint contents.
Stage2 runs epochs15-26, with0.2 pixel distillation and0.1 training-only geometry
consistency. Reuse exactly Round309's all-Train teacher cache. No new pretraining.
Only stage2 uses consistency, matching the Round313 validation intervention.

The earlier Round314 incorrectly reused grouped-Train Round307 stage1. Its
interrupted artifacts are preserved with _superseded suffix, not submission
candidates. Corrected output keeps Round314 number; packaging requires explicit
full-stage1 provenance. Grouped Val evidence is Round31376.2500301 versus
Round30876.2247099, a small +0.0253202 with all3 Stomach ROIs positive.

Measured stage2 cost is about10min/epoch: approximately2h plus inference and
packaging, not26 newly trained epochs. One durable queue executes corrected314,
packages it, then runs315. Local downloader retrieves314 into submissions.

Launch October30: corrected full-data stage1 audit and real-data two-update
smoke passed;315 matched-config smoke passed. Default0.2 loss equivalence and
0.4 gradients checked. Remote AIC-Round314-Queue started after GPU check
(0% utilization,29787MiB free), executing314 then315 even if314 packaging fails.
Local AIC-Round314-Download restarted; transfer completion is not yet confirmed.
Actual launch revision is in round314_repair_launch_commit.txt.

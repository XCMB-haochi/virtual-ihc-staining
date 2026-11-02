# Round318: stronger training-only geometric consistency

One matched comparison to313: consistency coefficient0.1 to0.2. Keep313 stage2
LR1e-4 (do not inherit317), teacher0.2, organ objective0.1/0.2/0.7,307 stage1/RNG,
14-epoch horizon, same grouped Val/encoder/data/augmentation. One forward per
Val/Test image; no TTA or inference changes. No new pretraining or teachers.

Train/Val evidence:313 added0.1 to308 and gained0.0253202 weighted, all3 Stomach
ROIs improved.315 stronger distillation and316 organ reweighting did not improve
weighted score;317 lower LR lost0.0497464. Thus test whether the existing modest
geometric regularizer has remaining benefit. Search found no previous matched
0.2 consistency test in this stage. This is one bounded increase, not a grid.
Could over-smooth predictions or impose incorrect rotational symmetry; no gain
assumed. Do not use314 platform result to choose the setting.

Evaluate epochs15-28 and report all organs/ROIs and endpoint scores. No automatic
full fit. Smoke must include actual report save/reload, finite optimizer audit,
and equality with313 config except consistency strength. Estimated2-3h on5090.
Launch October30: source archive verified; real GPU smoke passed config/optimizer/report save-reload checks. GPU idle before launch with29826MiB free. AIC-Round318 process and formal config verified; no stderr. Source revision in round318_launch_commit.txt.

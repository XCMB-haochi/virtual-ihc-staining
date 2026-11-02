# Overnight queue completed October29

Remote tasks Round304 and Round305 exited0. Round304 completed27 full-data
epochs,279612 exposures,17496 optimizer updates with finite weights; packaged
03:04:55, downloaded03:05 to local submissions with matching SHA and2186 images.
Round305 completed its decision05:34:27. Serial continuation and local delivery
worked; Round306 was deliberately not selected, not a failed or missing job.

Matched grouped-Val best: mixed Round30376.0908141 vs all-CD68 Round30575.2894713,
difference-0.8013428. All-CD68 best global24(stage2 epoch10).
Colon74.8411146 vs74.8907161(-0.0496015); Liver73.5021435 vs77.5821999(-4.0800564,
different target markers); Stomach75.8641873 vs75.8361465(+0.0280408).
Stomach ROI gains[-0.1391511,-0.0983179,+0.3215914]; gain only1/3 ROIs.
No evidence to replace mixed markers. Preserve all checkpoints but use Round304
full-data mixed package for today's first submission. This does not use platform
feedback to choose the candidate. No full-data validation score is claimed.

Next: bounded grouped-Val organ-loss weighting control Round307, justified by
fixed scoring weights vs the current per-image objective. See its plan; do not
repeat800 pretraining or launch further marker searches.

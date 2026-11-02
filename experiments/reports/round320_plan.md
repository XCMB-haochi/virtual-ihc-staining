# Round320: proportional organ composition per optimizer update

Compare to31876.2672365. Only reorder second-stage samples so each full16-image
accumulation group has2-3 Stomach images. Current random order has6.677% groups
with zero Stomach and27.005% with at most one, despite70% organ objective.
Weighted gradients and clipping can therefore vary with random organ mix;
this is a hypothesis, not a measured gradient-conflict diagnosis.

Preserve exact once-per-epoch coverage,506 updates, organ objective0.1/0.2/0.7,
consistency0.2, teacher0.2, LR1e-4/cosine14, original307 stage1+RNG and encoder.
Preserve each image's paired geometry draw and consistency-view index from
baseline order. Locally seeded interleaving consumes no global RNG. No extra
Stomach repeats, loss renormalization, organ input, inference changes or TTA.
Unlike300/301 replacement sampling or316 increased weight, epoch totals stay
identical. Search found no existing matched interleaving test. Reduced batch
noise may help or reduce useful stochasticity; no improvement assumed.

Run epochs15-28 on same grouped Val, report full trajectory and organ/ROI scores.
Only3 Stomach ROIs; no automatic refit. Expected2-3h on5090. Unit tests cover
coverage, full/grouped counts, deterministic order, RNG preservation and image
geometry mapping. GPU smoke includes optimizer audit, config comparison and
actual report save/reload. If promising, refit uses full309 stage1 as before.
Launch October30: archive verified after transient network retry; local/remote order tests passed. Real-data smoke/config/optimizer/report audit passed. GPU idle29951MiB free before AIC-Round320 launch; formal process and update_order config confirmed, no stderr. Source revision in round320_launch_commit.txt.

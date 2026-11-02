# Round313: training-only geometry consistency for single-pass deployment

Bounded stage2 comparison against Round30876.2247099. Same stage1 checkpoint,
RNG, teacher cache, mixed markers, organ weights, uniform7 loss, optimizer,
14-epoch horizon, BF16, accumulation16 and seed2026. Add fixed0.1 consistency
loss between ordinary prediction and inverse-transformed prediction of one
deterministically selected nonidentity D4 transform. Both paths receive gradients;
no output averaging, no label alteration, no extra image/organ inputs. Organ
weight multiplies the combined training loss. Extra forward applies only during
training; Val/Test remain unchanged one complete image, one output, one pass.

Not an untested family: Round20 had small single-view gains but reduced D4-TTA
benefit; Round92 UNeXt2 D4-TTA results were inconsistent and not expanded.
Current deployment forbids TTA and uses one marker per forward, shared organs
and fixed pixel distillation. These differences motivate one no-TTA experiment,
not a claim of recovering past TTA gains. Strength0.1 is historical, not searched.

Use exact existing paired D4 functions, cycle indices1..7 without consuming
augmentation RNG. Log auxiliary loss separately. Compare best14 Val endpoints
and per-organ/ROI results; only3 Stomach Val ROIs limit confidence. Extra
consistency could over-smooth or interfere with distillation. No automatic full
refit, package, strength sweep or further queue. Estimated2-3h on5090.

Tests cover inverse coordinate mapping, gradients through both predictions,
unchanged inputs and RNG, plus real-data stage2 smoke/config equality audit.
Scheduled Task AIC-Round313 runs tools/round312.py --round 313 --log.

Launch October29: local/remote coordinate/gradient/RNG tests passed. Real GPU
smoke completed two updates with finite weights and nonzero consistency loss
0.0061126; baseline config differs only by registered consistency weight.
Source archive SHA f1d2fb496eab63ab0c7ad08854e06520a5f0cca789e0f1e9482b7bbc87e72b70
verified before extraction; source commit in round313_launch_commit.txt.
5090 idle with ample free memory immediately before launch;6h task limit.

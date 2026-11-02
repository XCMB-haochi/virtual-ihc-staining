# Round323: training-only small integer paired crops

Question: can varying cell positions relative to the sampling grid improve
single-image reconstruction without scale interpolation? Reference318 Val
76.2672365, shared head and original random order. This is an unconfirmed
hypothesis, not a measured attribution of our error to shift sensitivity.

Alternatives reviewed:322 marker-head split tied/slightly lost;317 half LR
lost;315 stronger distillation and316 organ overweighting did not improve
weighted Val. Old50 continuous rotation/scale reduced direct q95 by0.047799;
that does not test this small integer crop without interpolation. Rebuilding
teachers is substantially more expensive; EMA permission remains unconfirmed
and its historical matched benefit was negligible. No clearly stronger,
deadline-ready option was identified. No sweep, encoder retraining or Test
feedback used to choose this setting.

Only stage2 augmentation changes: after existing paired D4, with fixed
probability0.5 select one240x240 window from the current256x256 image, top/left
uniform integers0..16 inclusive. Otherwise keep256. Apply the identical window
to DAPI, unchanged GT and its existing label-excluded teacher prediction.
No interpolation, padding, extra image/context or intensity modification.
Original files/cache remain unchanged. Teacher target is a crop of its original
full-image prediction, not a teacher re-inference on the smaller field; this
training-only context difference is explicit. Lost field/boundary information
and reduced supervised pixels can hurt performance. Probability/size are one
bounded design choice, not validated optimal values.

Rule basis:R03 permits paired training geometry; R08's inference restriction
is preserved by complete official256x256 DAPI input at every validation/test
forward. Crop is never called in evaluation, and no crop or TTA is added to
inference. Same declared marker policy CD68/CD45RO/CD68 and single shared model.

Keep exact307 grouped stage1/RNG,800 encoder,14-epoch stage2 cosine horizon,
LR1e-4, teacher0.2, consistency0.2, organ0.1/0.2/0.7 and each image once/epoch.
Crop RNG uses independent Random(2026+200000*epoch); original ordering/D4 RNG
streams unchanged. Crop counters audited by replay against actual per-epoch
Train order. Complete256 validation at every endpoint15..28. Report all
organs/ROIs and full score trajectory; only3 Stomach ROIs limit confidence.

Expected2-3h on5090; no automatic full refit. If positive, select endpoint from
Train/Val and rerun the complete full-data supervised schedule from the fixed
pretrained initialization (shared full-image stage1, specified crop stage2),
never expand a Val-selected checkpoint to allTrain. Reserve about4h plus
delivery for that possible run before November1 20:00 Beijing.319 remains
highest reported artifact. No extra experiment chain.

Tests: exact coordinate alignment after all8 D4 views, bounds, immutable source,
teacher/GT correspondence, identity path, local RNG replay/isolation, full240
and256 model output shapes. GPU smoke must exercise mixed sizes, finite
optimization, complete-image Val, config equality and decision report reload.
Launch October31 19:32 Beijing: archive verified, local/remote tests passed. Real smoke performed21 cropped/11 uncropped samples with exact schedule audit; config comparison, optimizer audit and decision report reload passed. AIC-Round323 formal process and configuration verified, stderr empty. Source revision is recorded in round323_launch_commit.txt.

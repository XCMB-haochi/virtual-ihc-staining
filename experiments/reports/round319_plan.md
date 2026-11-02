# Round318 result / Round319 full refit

Round318 completed14 stage2 epochs/7084 updates with finite checkpoint audit and
no stderr. Best epoch26 score76.2672365 versus31376.2500301 (+0.0172064).
Colon +0.0029195, Liver -0.0037863, Stomach +0.0252453. Stomach ROI changes:
ROI000 +0.0218207, ROI002 -0.0010622, ROI006 +0.0549774.10/14 matched endpoints
improve; best epoch unchanged. Small one-split evidence, not guaranteed gain.

Proceed with319 full-data candidate: reuse verified309 all-Train stage1 epoch14
and exact309 teacher cache. Train epochs15-26 on10356 images, LR1e-4,
teacher0.2, organ0.1/0.2/0.7, consistency0.2. No pretraining/teacher retraining.
Source split/14 full epoch counts/recipe/RNG audited before fitting. Same
single-image inference/JPEG95 package,2186 images, no TTA/context/correction.
Package requires full-stage1 provenance and correct consistency setting.

Expected~2h training plus inference/transfer on5090. Remote task runs319 then
package in one process; local downloader waits for final manifest, verifies
SHA/CRC/count and places semifinal_round319_mixed_q95.zip in submissions.
Keep314 as current best reported artifact. No further experiment chain or
platform-based parameter choices. Retain318 validations and319 checkpoints.
Launch October30: source archive verified, real-data smoke/audit/provenance writing passed. Formal process and initial hash match full-data309 checkpoint, endpoint26/consistency0.2 verified. GPU idle before launch. Local download task running; transfer not yet completed. Source revision in round319_launch_commit.txt.

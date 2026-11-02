# Round245: fixed better-OOF teacher transfer into the retained student

Primary remains Round24379.0967 (Round240 model with16-view TTA). Round234's
local-condition fold teachers score78.7336558 versus Round23978.6864216,
a +0.0472342 gap,4/5 positive fold differences and11/17 positive ROIs. Fold0
is negative(-0.099168); report the nonuniformity rather than guaranteeing gain.
Round235 directly deployed a local-condition student and failed on platform;
that student configuration stays closed. New evidence after its closure:
Round240 showed a fixed teacher refresh improving the unchanged stage-decay
student by0.0275. This motivates one different teacher-transfer test, not
mask/teacher-weight tuning or resurrection of Round235 as the final model.

Keep the Round240 student architecture, exact Round239 full-stage1 epoch14,
seed2026,stage decay0.8,14-epoch stage2 cosine,endpoint12,micro4 accumulation4,
CD45RO primary weights,uniform7 losses,distillation0.2 and channel1. Only change
the cache to all five Round234 epoch12 teachers. Each training image gets one
prediction from its ROI-excluding teacher; no teacher blends or fold selection.
No additional model is used at inference. Target labels remain untouched.
Use the established lossless uint8 PNG cache and Round24316-view TTA/q95.

No new CV training: the complete existing teacher OOF supports one fixed full
student confirmation. There is no independent OOF estimate of the distilled
student and teacher improvement may fail to transfer. No additional teacher
recipe/weight grid follows if this fails. Expected1-2h plus cache/inference
on GPU1, subject to shared compute, before October5 deadline. Round137 stays
stopped. Use campus direct SSH (aic-a100) per latest user network direction.

Validate command equivalence to Round240 except cache/output paths, teacher
hashes against Round234 OOF, fold exclusion,2380 unique target coverage, exact
stage1 identity,14 finite epochs and actual updates. Output uses corrected
0..255-to0..1 save contract,one final checkpoint and560 checked JPEG outputs.
This reuses tested cache building/training/TTA implementations; no new loss or
network code. Final comparative platform baseline is Round24379.0967.

## Verified launch

Campus direct aic-a100 access succeeded. Local/Linux command-equivalence test,
shell syntax and archive SHA256 verification passed. Real-artifact preflight
verified all five Round234 teacher hashes/fold exclusions and retained full
stage1 identity. Detached Screen round245_teacher_transfer is confirmed alive
on physicalGPU1 with target-cache directory created. GPU1 has shared compute
but about26GiB free versus roughly7GiB training peak; wall time may increase.
Round137 remains stopped. Source and launch record are committed and pushed; the exact Git archive is
deployed. GitHub SSH port443 succeeded after port22 timed out.

## Completed and local package verified

Pipeline exit0,14 stage2 epochs and2082 actual optimizer updates. Teacher
manifest covers2380 training images and all five hashes match the Round234
OOF sources. Stage1 hash matches Round240's original initialization. Runner
checks passed for finite losses/weights, full-data scope, seed2026,distillation
channel1/weight0.2 and fixed epoch12 checkpoint.

Package `submissions/semifinal_round245_cd45ro_q95.zip` is copied locally,
3231288 bytes. SHA256 matches `round245_artifacts.json`:
7731e7c048974a275314dfbdb2a052c705e6831fb5321bdf9141c372f3db1d7c.
All560 unique expected images decode as RGB256x256 JPEG; ZIP CRC passed.
Decoded mean16.5943 and mean absolute difference to Round2430.26648 gray
levels were integrity checks only, not test-label evaluation or tuning.

## Platform confirmation (2026-10-04)

User reported79.1052, SSIM0.899, PSNR26.932: +0.0085 versus matched
Round24379.0967. Displayed PSNR is0.001 lower and rounded SSIM is unchanged;
the displayed precision cannot resolve the exact metric contribution to the
score change. Promote Round245 as primary and retain Round243 as fallback.
Local ZIP SHA256 rechecked against the manifest and matched.

The fixed teacher transfer yielded a small confirmed platform improvement;
it does not imply that the rejected Round235 direct student is now validated,
or that arbitrary teacher selection/mixing will help. No independent student
OOF result or teacher/weight grid is claimed. Round137 remains stopped.

## Final-stage decision (2026-10-04)

User requested one further experiment only if a clearly worthwhile direction
exists, otherwise no adjustments before the semifinal/Liver submission deadline.
Review found no sufficiently supported remaining candidate: the validated
stage-decay/teacher/TTA gains are incorporated; late Gaussian and decoder-first
tests do not improve the retained package. Paired random cropping remains an
untested mechanism with nontrivial conditioning/teacher alignment work, not a
deadline-ready correction. Do not initiate speculative grids or new training.

Freeze Round24579.1052 as the primary Liver result through October5 20:00.
Preserve Round24379.0967 as fallback and all reproducibility artifacts. Local
Round245 ZIP SHA256 was reverified unchanged. Round137 stays stopped by user
direction. Reopen only for a concrete correctness/compliance issue, new material
evidence or explicit user redirection; later-stage data requires fresh planning.

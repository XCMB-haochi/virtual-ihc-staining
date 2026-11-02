# Round208: full-data domain FCMAE800 and CD45RO single model

Round207 two-fold fold-local FCMAE800 improved CD45RO by about +1.4 points
on both tested folds. It produced no submission. Round208 therefore runs
the same domain pretraining on all 2,380 permitted training images, starting
from the immutable public epoch0 encoder, then trains one CD45RO-primary
student with the Round204 recipe: seed2026, effective batch32, 14+14
supervised epochs, uniform7 hybrid loss, and no distillation or guidance.

The final checkpoint is stage2 epoch12, inferred with D4, halo128, offset0,
q95, and packaged as one CD45RO model. This is a full-data candidate, not a
fold ensemble. Train-all FCMAE has no held-out validation, so checkpoint12
is transferred from the grouped clock and must be treated as a practical
candidate rather than an unbiased optimum. Do not submit automatically.

GPU1 only; the runner waits for 16 GiB free memory and permits concurrent
compute. The 800-epoch pretraining is expected to take roughly 13 hours on
the measured fold throughput, followed by about 70 minutes supervised
training, subject to shared GPU load.

## Completed

Exit0; all 800 pretraining and 14+14 supervised epochs present with finite
losses. Every stage used all2,380 current-stage training images, seed2026,
and no held-out validation. Final stage2 epoch12 weight is finite and matches
the recorded SHA256. Pretraining epoch times summed to7.63h; supervision
to76.8min. Package: 3,304,074 bytes, 560 CD45RO RGB256x256 images, verified
locally against server SHA256 with ZIP CRC and decoded image checks.
Platform confirmation: **78.9741**, SSIM **0.898**, PSNR **26.894**, reported
by the user with national semifinal rank1 at that time. This improves
Round205 by **1.0307**. Rank can change. Preserve the exact epoch12 weight
and ZIP identified in `round208_hashes.json` as the current best package.

This restores domain FCMAE only, not the entire Colon recipe: relational
guidance and pixel distillation remain absent. Round209 has now completed
the Round207 matched D4 audit, confirming gains+1.54149/+1.41701. The
candidate was developed on two difficult folds rather than all five;
positive platform confirmation does not erase that generalization limitation.

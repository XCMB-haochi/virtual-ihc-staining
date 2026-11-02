# Round131: Full-Data Single-Model Relational Submission

The organizer confirmed that the final result may use one best fold or a
full-data retraining, provided the delivered result has one model. The choice
here is full-data retraining: a best fold would discard about one fifth of the
Colon training images and its validation score would depend on fold difficulty.

This run freezes the Round127 relational student recipe and trains one student
on all 6,296 labelled images from ROI000-ROI024. It first trains one all-train
five-marker FCMAE teacher for the relational guidance loss, then trains the
student for two fixed 14-epoch stages. The teacher and the existing all-train
DAPI FCMAE initialization are training inputs only; test inference loads only
the student's `final.pt`. No fold checkpoint, seed blend, cross-recipe mixture,
or validation-set selection is used.

Fixed student settings: UNeXt2-FCMAE-Tiny, spatial-thumbnail FiLM, four-marker
loss weights `0.85/0.05/0.05/0.05`, seed `2027`, relational-robust guidance at
stage 4 with weight `0.05`, context64 training, context128+D4 inference,
output offset `[0,-1]`, and JPEG quality 95. The teacher uses all-train DAPI,
CD68, CD45RO, HLA-DR and Vimentin FCMAE pretraining for 160 epochs, initialized
from the permitted official ConvNeXt-V2 FCMAE Tiny checkpoint.

## Launch (2026-09-15)

- protocol: `configs/round131_full_data_relational_single.json`
- source commit: `f73429d007f8ed18476f12b477211e9a4738aa99`
- protocol SHA-256: `af84d60eca81020a611269702e9e11cc4bc290b6c9976d223c6940016d664c74`
- physical device: GPU1 only; GPU0 remains reserved
- detached Screen: `round131_full_data_single`
- status: teacher pretraining started; the pipeline will continue to student
  stages, single-model inference and package validation after the teacher exits

The final completion audit must verify teacher and student exits, finite
artifacts, the single `final.pt` provenance, 1,346 prediction files, ZIP CRC,
and that no producer remains.

## Completion and package audit (2026-09-16)

- Pipeline exit: `0`; no Round131 producer remains on GPU1
- All-train teacher: 160 epochs, all 31,480 marker-image samples, finite
  trajectory; `final_encoder.pt` SHA-256
  `0600e7ef8739e6adb34e9a7d74cb477a96773b84120065fa24bb4c0d538309eb`
- Full-data student stage1: 14 epochs, finite; `final.pt` SHA-256
  `f527448ced1cfa09750b7c1f06c4f6d8368e2b6c2089b13cdc1f7597e323cf87`
- Full-data student stage2: 14 epochs, finite; final model SHA-256
  `ba451b04911d27d8c129951956bdfcccc8b1fb01b77831c40660cf4cdc1db51c`
- Test predictions: 1,346/1,346, all readable RGB `256x256`; ZIP CRC passed
- Package: `cd68_round131_full_data_relational_q95_plain.zip`, 11,935,950 bytes
  SHA-256 `28dd82335d1393e9d42def0a9f7d44a98819e8abba35cb009072a663a7043a55`
- Delivery model count: `1`; inference loaded only the student's stage2
  `final.pt`. The teacher and stage1 checkpoint are training artifacts and are
  not part of the delivered model.

The package is ready for one platform confirmation. No leaderboard result is
yet associated with this single-model package, so it should be compared with
Round127's `74.2222` only after the user submits it.

## Platform confirmation (2026-09-16)

The user reported `74.0606`, with displayed `SSIM=0.832` and `PSNR=26.357`.
This is `-0.1616` versus the legal same-architecture five-fold Round127 result
(`74.2222`). The full-data single-model route therefore did not improve this
recipe on the hidden set. The result does not show that full-data training is
generally inferior; it compares one unselected full-data trajectory and one
five-fold automatic aggregation, with different training-time teachers and no
held-out checkpoint selection in the full-data run. It does show that this
particular full-data recipe should not replace Round127 without further
confirmation.

The user also reported an external leaderboard high of `77.8212`. That value
is recorded as an unverified external result and cannot be attributed to the
organizer's comment about future dataset shuffling without additional evidence.

The full-data model's hidden result is also below the legal Round127 five-fold
automatic aggregation. The next single-model comparison, if made, should use
one already trained Round127 fold and one fixed package; no new seed or blend
search is justified by the `77.8212` observation.

Round132 then tested the single Round127 fold with the highest absolute OOF
score (fold4). Its platform result was `73.6511` (`SSIM=0.829`, `PSNR=26.028`),
which is `-0.4095` below this full-data model and `-0.5711` below the Round127
five-fold automatic aggregation. This rejects best-fold selection as a useful
Colon packaging heuristic for this recipe. It does not imply that every fold
is equally poor, nor does it justify searching the remaining folds after seeing
the hidden result.

Round133 packages the already trained full-data student stage2 epoch6 as a
fixed early-checkpoint confirmation. It uses one checkpoint only and changes no
training or inference setting. Its checkpoint SHA-256 is
`acf08de0a2e7d282d0432be32dc182fd72fd64deab1aa7674efd8a1c51fb05c2`; the
validated ZIP SHA-256 is
`92afd01582e71bbc6c3395d7c1b46638309dc4b3b6c9649585b9425020314548`.

The user reported `74.1330` (`SSIM=0.833`, `PSNR=26.392`) for Round133. This
improves the Round131 epoch14 package by `+0.0724` but remains `-0.0892` below
the legal Round127 same-architecture five-fold aggregation at `74.2222`.

EMA approval does not remove the five-fold option: EMA averages checkpoints
from one training trajectory into one model state, while five-fold aggregation
averages predictions from five same-architecture fold checkpoints. The project
continues to treat Round127 as a legal automatic five-fold result and keeps
Round133 as the strongest confirmed single-model result.

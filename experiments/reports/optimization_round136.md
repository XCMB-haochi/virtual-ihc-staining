# Round136: Leakage-Safe OOF Distillation to One Student

Round127's same-recipe five-fold result (`74.2222`) remains above the strongest
single-model Round134 result (`74.1432`). Round136 tests whether that gap can be
compressed into one model. Each official training patch receives a teacher
target only from the Round127 fold that excluded its complete ROI. The cache is
therefore fold-excluded and covers every training image exactly once.

The full-data student starts from the exact Round131 stage1 checkpoint and
replays the unchanged stage2 relational recipe. Its supervised term is fixed
to `0.8` hard-label loss plus `0.2` teacher-target loss; the existing feature
guidance continues to use the real target. Teacher targets follow the same
training flips and rotations as their paired samples. No test image or platform
score enters target construction.

The teacher uses Round127 D4 predictions before the fixed test-only output
offset. Targets are stored losslessly as uint8 PNG bytes under the original
`.jpg` patch names. The student packages the preregistered epoch5/6/7 parameter
average at `1:2:1`, matching the positive Round134 single-model mechanism. The
final inference graph contains one student checkpoint and D4 TTA.

Distillation weight `0.20` is fixed; no weight, epoch, seed or teacher search is
authorized. If this package does not improve Round134, pixel distillation is
closed for Colon. The mechanism remains portable to a smaller later-stage
dataset provided grouped folds can generate leakage-safe teacher targets; its
extra cost is teacher OOF inference plus one final student stage.

## Validation and launch (2026-09-17)

Local and storage-scoped Linux suites each passed 52 targeted tests. A real
64-image, one-epoch A100 smoke completed four optimizer steps, recorded
`distillation_weight=0.2`, produced a loadable `final.pt`, and peaked at about
27.0 GiB. The smoke also verified that the official 256x256 teacher target
constrains the center of the existing 384x384 supervised output while hard
labels and relational guidance retain their original scopes.

Detached Screen `round136_oof_distillation` was launched on physical GPU1 from
source commit `a3a5c7e`. It is currently building the five-fold leakage-safe
teacher cache, then will train one full-data stage2 student, build the fixed
epoch5/6/7 parameter average, run D4 inference, and validate one submission
archive. GPU0 remains untouched.

## Completion and package audit (2026-09-17)

- Pipeline exit: `0`; no Round136 producer remains on GPU1
- Teacher cache: 6,296/6,296 training images covered exactly once across five
  held-out-ROI folds; manifest SHA-256
  `5e800657e896eddd56993d9714f4a0dce399a0973f1c87f5aad059ae4534fc32`
- Student: 14/14 epochs, all numeric trajectory values finite; distillation
  weight `0.20`; final temporal parameter average uses epochs 5/6/7 at `1:2:1`
- Final single student checkpoint SHA-256:
  `90fbf0eca0707aa5621037f1093d78d17a6bf1ebc170547369fa56f45fc12110`
- Package: `cd68_round136_full_data_oof_distillation_q95_plain.zip`,
  11,905,816 bytes; SHA-256
  `690cd8db6993bf0d4de46fda125982f0c065af0ed3777cb0936b11b0aec0567e`
- Independent audit: 1,346/1,346 `results/test/CD68/` entries, readable RGB
  `256x256`, clean ZIP CRC

The package is ready for one manual platform confirmation against Round134's
`74.1432`. If it does not improve, the pixel-distillation route is closed for
Colon; no distillation-weight, teacher, epoch, or package search follows.

## Platform confirmation (2026-09-17)

The user reported `74.1985` (`SSIM=0.833`, `PSNR=26.461`). This improves the
Round134 single-model package (`74.1432`) by `+0.0553` and is `-0.0237` below
the legal Round127 same-architecture five-fold aggregation (`74.2222`). The
OOF-teacher distillation therefore transfers a substantial part of the fold
benefit into one student checkpoint. Round136 is the current primary
single-model package. No new distillation weight, teacher combination, epoch
window, or hidden-set calibration is opened.

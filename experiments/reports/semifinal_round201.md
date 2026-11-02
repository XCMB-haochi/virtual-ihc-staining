# Round201: Semifinal OOF Distillation Student

Round200 grouped five-fold baseline completed on the 17-ROI semifinal training
set with overall q95 `71.8536`, but fold convergence was heterogeneous. Stage
1 best epochs were approximately `12,12,11,14,10`; stage2 best epochs were
`2,1,14,13,12`. Round201 fixes the common full-data clock near the medians:
12 epochs per stage and an actual epoch10/11/12 parameter average at `1:2:1`.

The five Round200 validation checkpoints generate leakage-safe CD68 teacher
targets: every training image is predicted by the fold that excluded its whole
ROI. A seed-2026 full-data student uses `0.8` hard-label loss and `0.2` teacher
loss, with D4 TTA and output offset `[0,0]`. The final package contains one
student checkpoint. No Colon data, Colon checkpoint, external training data,
or multi-model test inference is used.

The simplified student is not a faithful replica of Colon Round136:
it has no new-stage domain FCMAE or relational guidance, applies pixel
distillation in both stages instead of stage2 only, and changes the cosine
period from 14 to 12. These differences were not isolated by matched
controls. The historical metric is uniform7 SSIM, not Gaussian11.

## Completion and package audit (2026-09-21)

- pipeline exit: `0`; cache, stage1, stage2, parameter average and inference all
  completed; GPU1 producer exited
- final student checkpoint SHA-256:
  `0f8ac44ac520b538ebfb0dd76bc6f0149143b914eb654d69b926f87da2857dff`
- package: `semifinal_round201_distill_q95.zip`, 4,803,240 bytes; SHA-256
  `b2199413ad6681bb616c9769a3a9f3d21bdd4a2e6f5f4b71918f551785358504`
- independent audit: 560/560 `results/test/CD68/` entries, RGB `256x256`,
  clean ZIP CRC
- output offset is `[0,0]`; D4 is the only test-time augmentation; one student
  checkpoint is loaded for inference

## Platform result and audit (2026-09-22)

The user reported **70.6241**, SSIM **0.771**, PSNR **27.769**. Rounded
components are consistent with the previous score weights, without proving
the remaining evaluation implementation. The organizer now specifies
Gaussian11 SSIM. Round202 re-evaluates all four existing OOF output channels
under both metrics before choosing the next primary marker and training.

The original runner requested nonexistent epoch13 after 12 training epochs.
Its averaging command now matches the actual epoch10/11/12 package.
Existing weights and submission outputs are preserved.

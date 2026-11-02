# Round245 submission material preparation

Team AIC-2026-51865112 / 通信速通生物 / 团队成员. Local delivery:
`deliverables/AIC-2026-51865112-通信速通生物-复赛-团队成员/`.
PDF not created, per user instruction. Baidu Netdisk upload is user-managed.

Contents: original560 results/test/CD45RO images, unchanged final model.pt,
reduced independent Python code, exact public-only training initialization,
fixed recipe/folds, environment/commands and file hashes. No original dataset
copy, historical experiment runners, unrelated models or GPU/SSH scripts.
Weights/results are ignored by Git. Source is tracked under tools/submission_round245;
assembly is tools/prepare_round245_submission.py.

Final weight SHA25616a530ac9b0b8396592668bcaad4405b697083ad09a7b2e176b1e33ed36a17b1.
Public initialization SHA25698256dc1aca47cb6e553a1c82d7f343d328e14d2c7a1b1019dc34273669d1578.
Original ZIP SHA2567731e7c048974a275314dfbdb2a052c705e6831fb5321bdf9141c372f3db1d7c.
Highest user-reported score79.1052,SSIM0.899,PSNR26.932.

Validation: strict local original-checkpoint load; original/reduced teacher and
student initialization, scalar loss and post-AdamW parameter hashes match
exactly on deterministic numerical tests. Training graph has19 jobs:6 domain
pretrains,10 teacher stages,1 cache generation,2 student stages. CPU source
compilation and command planning pass. Server GPU smoke passed FCMAE2epochs,
teacher stages1/2 and student stages1/2; supervised actual updates7/8/7/8.
Full standalone inference on560 official test inputs produced560 byte-identical
JPEG files compared to the retained Round245 ZIP, with zero mismatches.
Verification folder transfers/round245_delivery_validation is outside delivery.
Root file manifest validates581 files and560 result images. README contains
execution instructions only; no technical-report PDF or report draft was added.

## Result ZIP delivery update (2026-10-05)

The delivery now uses `results.zip` with exactly 560 `AIC-2026-51865112-赛题结果文档/results/test/cd45ro/*.jpg` entries. ZIP CRC and every image byte were verified against the extracted results and the retained original submission ZIP. The original submission archive remains unchanged. The technical-report PDF is already included in the delivery. Assembly and figure-rebuild scripts now use the compressed results.

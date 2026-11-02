# Round135: Matched Full-Data Stage2 EMA

Round134's fixed temporal parameter average scored `74.1432`, a small `+0.0102`
gain over the online epoch6 Round133 package. This provides narrow evidence that
time smoothing may help the current full-data student. Round135 isolates
training-time EMA without repeating FCMAE or stage1: it starts from the exact
Round131 stage1 `final.pt`, replays the unchanged stage2 recipe and changes only
`ema_decay=0.995`.

The fixed EMA half-life is about 138 optimizer updates, or `0.351` of the
current 394-update epoch. Future stages should preserve that epoch-relative
half-life by calculation if their number of updates changes; no decay grid is
authorized. Epoch6 is fixed before launch because the matched online epoch6 is
the strongest confirmed non-averaged single-model endpoint. The final package
uses one epoch6 EMA checkpoint and D4 TTA, so it remains a one-model result.

The older Round32 `EMA=0.999` result was negative by about `0.04` OOF and showed
lag on its short late-continuation trajectory. That result lowers confidence but
does not duplicate this shorter-memory, matched full-data stage2 test. A
negative Round135 closes training-time EMA; no decay, epoch, or online/EMA blend
search follows.

## Launch (2026-09-16)

The protocol and Linux runner passed JSON and shell validation and were deployed
from commit `dce5c5c`. Detached Screen `round135_stage2_ema` passed artifact
preflight and is waiting for physical GPU1 to be empty with at least 36 GiB
free. It will not use GPU0 or interfere with the currently active GPU1 process.

## Completion (2026-09-17)

- Pipeline exit `0`; no Round135 producer remains
- Stage2 trajectory: 14/14 epochs, all recorded numeric values finite
- Fixed epoch6 EMA checkpoint SHA-256:
  `b13d2d8eade81e8dd82d89b98cdeba282305d86ee9eece72949d11d16e032507`
- Package: `cd68_round135_full_data_relational_stage2_ema_e6_q95_plain.zip`,
  11,915,934 bytes; SHA-256
  `6678a40d8649b707d21b7debe51c73371013f24f9508e481a7b69e422dcecec3`
- Independent audit: 1,346/1,346 RGB 256x256 images, correct archive prefix,
  clean ZIP CRC, one inference checkpoint

The package is ready for one platform comparison against Round133 online
epoch6 (`74.1330`) and Round134 temporal parameter average (`74.1432`). A
negative result closes training-time EMA without further decay or epoch search.

## Platform result

The user reported `74.1336` (`SSIM=0.833`, `PSNR=26.388`). This is only
`+0.0006` over the matched online epoch6 Round133 result and `-0.0096` below
Round134's fixed temporal parameter average. The difference does not establish
a useful EMA gain. Training-time EMA is closed without decay, epoch, or
EMA/online mixture search; Round134 remains the strongest single-model result.

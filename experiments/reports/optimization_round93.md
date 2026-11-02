# Round 93: Full-label fresh UNeXt2 endpoint diagnostic

## Purpose

Round91 confirmed that the fixed Round90 endpoint remains around `74.03`, and
Round92 closed the UNeXt2 D4-consistency branch after its two-fold distribution
gate failed. This round tests a different hypothesis: whether a fresh model
trained on all 25 labelled ROIs can use the available labels more effectively
than the current fold ensemble.

This is a preliminary endpoint diagnostic, not a promotion experiment. A
full-label model has no honest held-out OOF in this protocol. Its training
loss, full-data trajectory, and any platform score must not be used to select a
different epoch, weight, calibration, ROI rule, or later-stage recipe.

## Fixed protocol

The complete protocol is bound in
`configs/round93_full_label_unext2_auxiliary15.json` and executed by
`tools/run_round93_full_label_auxiliary15.sh`:

- all `6296` labelled training patches from `ROI000`--`ROI024`, no validation;
- UNeXt2-FCMAE-Tiny, spatial-thumbnail FiLM, context halo `64` for training;
- four marker outputs with fixed loss weights `0.85/0.05/0.05/0.05`;
- all-train DAPI FCMAE encoder, SHA-256
  `f4a8c93c616310ce4a981f9b1fe6830ba82bf78ff91bdbe52d471cf1c0164554`;
- two fresh 14-epoch stages, learning rates `3e-4` and `1e-4`, seed `2044`;
- stage1 snapshot `epoch 13`, stage2 snapshot `epoch 6`, fixed before launch.

The epoch clocks are the medians of the five Round55 `officialauxiliary15`
best-epoch arrays: stage1 `[13,14,10,13,13]` and stage2 `[7,12,6,4,6]`.
Inference is fixed to context halo `128`, D4 TTA, source offset `[0,-1]`, and
JPEG quality `95`, producing CD68 only.

## Launch audit

Local shell syntax, Python compilation, and the existing 45-case training
safety suite passed. Remote protocol/data preflight passed with train/test
counts `6296/1346`; a two-image predictor smoke using the new offset option
also exited `0`.

At `2026-08-31 11:06:42` (Asia/Shanghai), Screen `round93_full_label` was
launched under the storage-scoped environment. The startup sampler observed
GPU1 at `0%` utilization and `40427` MB free; GPU0 varied from `50%` to `100%`
with only `21998`--`25888` MB free, below the four-output training margin.
The scheduler therefore selected GPU1 and leaves GPU0 untouched. The first
visible phase is the bounded 64-image smoke; formal stage training starts only
after its metadata and checkpoint checks pass.

## Result

The detached pipeline exited with status `0`. Both full-data stages contain
exactly 14 metric rows and all 6296 training images; no validation rows were
created. The stage1 and stage2 selected snapshots are respectively epoch 13
and epoch 6, with SHA-256 values:

- stage1 epoch 13: `fdf1c9f59bf54cf98309ec5c80937a6554b29d700bf3b1e402fe5ff0289c090e`;
- stage2 epoch 6: `26356d9ed842fbfaa02ea0a8c1b3d1af4be47d1f917469dbf0382f46b5aec54f`.

The stage configurations record the expected all-train FCMAE encoder hash,
strict stage2 initialization, four marker outputs, context64 training, and
runtime commit `5681404`. Peak allocated memory was 25.52--25.64 GiB on the
40 GiB A100. Each stage took about 2492--2494 seconds. The repeated
`Errno 16: Device or resource busy` messages are multiprocessing temporary
directory cleanup warnings after the successful stage completion; they did not
alter metrics or checkpoints.

Inference produced all `1346` expected test JPEGs. Every inspected file is
256x256 RGB with identical channels. The submission archive contains exactly
1346 `results/test/CD68/*.jpg` entries and passes `zipfile.testzip()`.
Archive SHA-256 is
`1b577b2ff6dfb1578a24374311e2060716c8a6dd24e61e90de9703d304af9049` and its
size is `11982533` bytes.

This endpoint was submitted once for platform confirmation. The reported
result was `73.9176` with displayed `SSIM=0.831` and `PSNR=26.287`; no rank was
provided with the result. Relative to the earlier fixed Round88 ensemble
confirmation (`74.0315`, `SSIM=0.831`, `PSNR=26.435`), this endpoint is lower by
`0.1139` score points and `0.148` dB, while the displayed SSIM is unchanged.
Because this model has no held-out OOF, the platform result is descriptive only:
it does not justify changing the fixed epoch, blend, calibration, ROI rule, or
the running Round94 protocol. The full-label endpoint is therefore closed as a
non-promoted diagnostic.

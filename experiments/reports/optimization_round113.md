# Round113: Zero-Start DINOv2 Residual Guidance

Round112 showed that replacing the validated UNeXt2-FCMAE representation with
DINOv2-S/14 loses 1.48 q95 points. This round tests a narrower hypothesis:
reuse the UNeXt2-FCMAE auxiliary15 architecture and add a DINOv2-S/14
feature branch only as a bounded CD68 image residual. The residual head's last
projection is zero initialized, so the candidate starts with the same output
as the corresponding freshly initialized base model. Supervised decoder
weights start fresh; the trained Round55 checkpoint is used only as the OOF
comparator. The DINO branch is initialized from the pinned official
weight and trained at the encoder learning-rate multiplier 0.1.

The fixed screen uses fold0/fold2, the Round52 fold-local FCMAE800 encoders,
auxiliary15 marker weights, seed2044, effective batch16 (8x2), context64
training, context128+D4/q95 inference, global-MSE checkpoint selection, and
offset [0,-1]. Only one residual scale (0.15) is tested. The candidate must
gain +0.05 q95, positive float, both folds, at least 7/10 positive ROIs, and
worst ROI at least -0.12; otherwise it closes without a weight search.

The main CNN encoder and decoder use the original stage rate; only DINO uses
`0.1x`. The real two-stage smoke loaded all 196 FCMAE and 176 DINO tensors,
verified finite loss and strict checkpoint restoration. See
`round113_smoke_receipt.json` and `round113_launch.json`. The launch receipt
corrects an incorrectly transcribed commit argument after comparing deployed
sources to Git and preserves the original argument.

Fold0 was reused after the first fold2 attempt ran out of memory before epoch1;
fold2 restarted on GPU1 with the unchanged recipe. Recovery provenance is in
`round113_recovery_launch.json`; raw failure files remain on the server.

The implementation uses DINO block index 11 only. The historical model spec
lists all four intermediate layers; that descriptive field is inaccurate and
is retained unchanged for checkpoint compatibility.

## Result And Knockout Diagnostic

The recovered run completed with exit `0`, complete 2,706-image OOF, and an
audit pass: 14 consecutive records per stage, intended optimizer groups,
rehashed OOF-referenced checkpoints and exact 10-ROI coverage. The candidate
reached q95 `75.051646` versus the fixed baseline
`75.013272` (`+0.038374`) and float `+0.036817`. Fold deltas were
`-0.010961/+0.084600`; five of ten ROIs improved, and the promotion gate
failed its q95, both-fold, ROI-count, and ROI-floor requirements. The branch is
closed for promotion and no test package or weight search follows.

The historical 80-image knockout reported q95 `69.5913 -> 64.1930` when
disabling DINO, but its interpretation was withdrawn in the 2026-09-08 audit:
the evaluator built spatial conditions from only the sampled images, changing
ROI statistics and missing-neighbor masks. It measures the branch under a
different input condition, not under the normal complete-ROI pipeline. The
original JSON is preserved as historical evidence; formal full-ROI OOF and the
later fold1 confirmation are unaffected. See `reassessment_20260908.md`.

Artifacts: `round113_cv2_dino_residual_oof.json`,
`round113_cv2_dino_residual_gate.json`, `round113_pipeline.exit`, and
`round113_dino_residual_knockout.json`. Per-fold training was 109-111 minutes;
shared OOF took 29 minutes.

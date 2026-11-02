# Round117: Current-Tiny GroupDRO Screen

Round116 showed that all-train DAPI pretraining plus auxiliary supervision still
produced a fold-dominated result. Round117 tests a different mechanism: an
epoch-level GroupDRO objective over training ROIs on the current official
FCMAE-Tiny UNeXt2 recipe. The goal is to reduce ROI risk imbalance without
changing the representation, architecture or inference path.

The candidate is single-marker CD68 so the robust objective is isolated from
auxiliary-marker weighting. It uses fold-local Round52 FCMAE800 initialization,
seed2044, `eta=5.0`, effective batch16 (`8x2`), context64 training,
context128+D4/q95 inference, global-MSE selection, offset `[0,-1]`, and 14+14
epochs. The baseline is the unchanged fold-local Round52 official FCMAE800
CD68 checkpoint. No eta, ROI weighting, sampler, epoch or ensemble search is
registered.

Round16 rejected GroupDRO on an older Base48 H2/full-data path; this screen is
not claimed independent of that risk. It asks whether the same robust objective
helps the current FCMAE-Tiny representation, whose recent failures are mainly
fold/ROI instability. The fixed gate requires q95 `+0.05`, positive float,
both folds, 7/10 positive ROIs and worst ROI `>= -0.12`. It will run on one
physical GPU only and cannot produce a submission directly.

## Validation And Launch

The protocol passed local and Linux validation, including GroupDRO argument
propagation and 96 focused tests. The real two-stage smoke loaded all 196
fold-local FCMAE tensors, completed the GroupDRO risk pass, strictly restored
stage2 and peaked at 13.13 GiB. Formal execution is restricted to physical
GPU1; no second GPU will be used. The launch receipt and source commit are
written before training starts.

## Result

The single-GPU pipeline exited `0`. All four stages have 14 contiguous epochs,
finite losses, valid selected checkpoints, and the expected fold-local FCMAE
initialization. The OOF covers 2,706 images and 10 ROIs; checkpoint provenance
hashes and independent gate replay passed.

| Metric | Official FCMAE800 | GroupDRO | Change |
| --- | ---: | ---: | ---: |
| q95 | 74.987515 | 74.837196 | -0.150319 |
| float | 75.061378 | 74.911110 | -0.150268 |

Both folds are negative (`-0.255604/-0.051666`). Only ROI000, ROI020 and ROI024
improve; the worst ROI is ROI022 at `-0.429033`. GroupDRO therefore fails
clearly, not because its gate is marginally too strict. The current-Tiny
GroupDRO route is closed without eta, sampler, ROI-weight, checkpoint or fold
search.

Artifacts: `round117_cv2_unext2_groupdro_oof.json`,
`round117_cv2_unext2_groupdro_gate.json`, and `round117_pipeline.exit`.

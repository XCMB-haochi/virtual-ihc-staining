# Round 102: Aligned Stride-Phase Self-Ensemble

## Motivation

Round101 finds `+0.358538` target-aware per-image blend headroom but no
deployable DAPI-based route. Round102 instead tests a fixed, label-free source
of prediction variance inside one model. UNeXt2 repeatedly downsamples by two
and reconstructs the final pixels with PixelShuffle, so a one-pixel input phase
can change the decoder lattice even after geometric re-alignment.

## Fixed protocol

For each target patch the loader obtains context halo 129. Four 512x512 model
windows start at outer offsets `(0,0)`, `(0,1)`, `(1,0)`, and `(1,1)` relative
to the standard centered window. Each receives the existing D4 self-ensemble.
The four outputs are cropped back to the exact same target coordinates before
averaging. The baseline is the ordinary centered context128 D4 prediction.
Both variants then receive the same fixed `[0,-1]` output-source shift and q95
encoding.

The two fixed Round55 `officialauxiliary15` fold0/fold2 checkpoints are reused
byte-for-byte. No model is trained or reselected; no test input, target,
calibration, ROI routing, or leaderboard feedback is used.

## Gate

The four-phase candidate must gain at least `+0.02` q95, remain positive in
float and both folds, improve at least 7/10 ROIs, and keep the worst ROI above
`-0.05`. Passing permits an unchanged complete five-fold evaluation on the
current endpoint; failure closes translation-phase TTA without testing other
offset sets or phase weights.

## Completion audit and result

The detached evaluation exited with status `0`; its Screen and producer
processes are gone. The one-image-per-fold smoke passed, and the formal report
covers all 2,706 images across the ten registered ROIs. The standard D4 branch
replayed the bound Round55 auxiliary15 baseline exactly, so the comparison did
not suffer from a checkpoint or metric-path drift.

| candidate | float official | JPEG-q95 official |
| --- | ---: | ---: |
| standard context128 D4 | 75.083722 | 75.013272 |
| four-phase context128 D4 | 75.083353 | 75.010403 |

The four-phase delta is `-0.002870` q95 and `-0.000369` float. Fold deltas are
`+0.005609` and `-0.010815`; only 4/10 ROIs improve, and the worst ROI delta
is `-0.087284`. The distribution gate therefore fails every substantive check
and records `promotion.allowed=false`.

The aligned phase average is a valid, reproducible inference path, but it does
not provide useful gain for this model. Close stride-phase TTA and do not
search phase weights, additional offsets, or phase-specific ROI routing.

The OOF SHA-256 is
`f69478acdea97bcfec2d192bd49ce82974358c19499f561641bc10434feb95b8`;
the gate SHA-256 is
`4cbc97378483e5cf177b56bce036e4dc64d2b1afdf95e47410706c8620786807`.

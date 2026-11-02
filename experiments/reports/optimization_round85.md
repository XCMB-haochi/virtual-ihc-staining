# Round 85: Context128 Inference on the Fixed Shifted Ensemble

## Motivation

Round36 found a small, five-fold-consistent benefit from reducing artificial
padding with a context128 model, but that result came from a separately trained
U-Net branch. The current deployed endpoint is a fixed 25-member ensemble
trained with halo64 and corrected by the validated output offset `[0,-1]`.
No one has evaluated that unchanged ensemble with a larger DAPI inference view.
This round tests the portable inference mechanism without retraining or
changing any checkpoint, blend weight, TTA, or output correction.

## Fixed Screen

`configs/round85_cv5_context128_inference.json` reuses exactly the 25 checkpoint
files bound by the complete Round81 OOF report. The loader supplies context128,
each model's center prediction is cropped to the official 256x256 target, and
the fixed `[0,-1]` source offset is applied after the existing Base-diversity
blend. The baseline is the immutable Round81 context64 shifted OOF, whose SHA
is checked before evaluation. A one-batch real-data smoke verifies variable
input-size forward compatibility, D4 output shape, and finite values.

The single gate is fixed before execution: q95 gain at least `+0.02`, positive
float gain, at least 4/5 positive folds, at least 20/25 positive ROIs, and a
worst-ROI q95 change no lower than `-0.05`. The comparison is complete five-fold
OOF; no adjacent halo, weight, checkpoint, or ROI search is authorized. If it
passes, the same fixed recipe is eligible for one test inference; if it fails,
the context128 inference branch closes.

## Execution and Result

The protocol passed local and Linux compilation, the complete regression suite,
shell syntax, and the immutable baseline/checkpoint preflight. A real-data
context128 smoke passed with D4 output shape `(N,256,256)` and finite values.
The detached Screen waited through GPU contention, ran the smoke on GPU1, and
then completed the full five-fold OOF on GPU1. The scheduler log records the
observed utilization and free-memory samples; no unrelated process was
interrupted. The pipeline exit marker is `0` and the Screen session is gone.

The OOF covers all `6,296` training images and `25` ROIs. All 25 checkpoint
provenance records are present, and each context128 report record has the same
checkpoint SHA-256 as its bound Round81 context64 report. The evaluation took
`8,696.611 s`. Candidate OOF SHA-256 is
`c09c4da98d87c4cc9254de0b90f3d0b5cd41d38da2415bb2c2e0aca181de615f`.

Against the immutable fixed-shift context64 ensemble, context128 inference
raises q95 from `75.755123` to `75.781550` (`+0.026427`) and float from
`75.832863` to `75.859508` (`+0.026644`). Every fold is positive:
`+0.002009`, `+0.032717`, `+0.047947`, `+0.037622`, and `+0.011081`.
Eighteen of 25 ROIs improve, and the worst ROI change is `-0.033069`.

The fixed gate passes the overall q95, float, all-fold, and ROI-floor checks,
but fails the `20/25` positive-ROI requirement. The gate therefore records
`promotion.allowed=false`; no context128 test package, adjacent halo test,
weight search, or leaderboard-driven rescue is authorized. The result is still
useful portable evidence: larger DAPI context helps broadly and consistently,
but its current inference-only gain is too small and uneven for formal
promotion.

The saved gate SHA-256 is
`bad6bf7e173f6bdbc803057d9bade6ffcc4678d8193d45bbfeda99abc1b2c821`.

The corresponding candidate OOF SHA-256 is
`c09c4da98d87c4cc9254de0b90f3d0b5cd41d38da2415bb2c2e0aca181de615f`.

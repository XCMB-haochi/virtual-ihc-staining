# Round 55: Official-FCMAE Auxiliary-Marker Interaction

## Motivation

Round54 rejects the audited supervised ImageNet-22K-to-1K initialization
route. The new model loses `-0.305865` q95 to the Round52 FCMAE-1K-initialized
model, and its best fixed diversity route gains only `+0.001619` while failing
fold and ROI stability. Increasing public supervised pretraining data is
therefore not the next experiment.

Round51 provides a smaller but cleaner positive mechanism. Giving aligned
CD45RO, HLA-DR, and Vimentin targets a total 15% loss weight improved a matched
four-output control by `+0.321278` q95 on scratch FCMAE800. Its fixed deployment
blend gained `+0.055269` and missed the preregistered threshold by `0.004731`.
Round52 independently showed that official FCMAE-1K initialization adds stable
ensemble diversity. Their interaction has not been tested.

Round55 asks whether low-weight auxiliary-marker gradients remain useful when
the same UNeXt2 starts from the successful official-FCMAE-initialized,
fold-local DAPI FCMAE800 encoder. This is cheaper and better supported than
immediately scaling to ConvNeXt-V2 Base.

## Matched Protocol

`configs/round55_cv2_unext2_official_auxiliary.json` reuses the exact Round52
fold0/fold2 epoch-800 encoders by verified hard link. It does not repeat DAPI
pretraining. For each fold, two fresh four-output models share the exact model
shape, encoder, seed, data order, target loading, 14+14 schedule, halo64
thumbnail FiLM, Uniform-7 hybrid loss, global-MSE selection, D4 inference, and
JPEG q95 evaluation:

- `officialauxcontrol`: CD68/CD45RO/HLA-DR/Vimentin weights `1/0/0/0`;
- `officialauxiliary15`: weights `0.85/0.05/0.05/0.05`.

The matched control isolates auxiliary gradients from the four-output head and
its changed random-number path. Checkpoint selection and OOF scoring use CD68
only; auxiliary outputs never enter the competition score.

## Fixed Deployment Tests

The current Round53 two-fold analogue `t70_o50` remains the deployment
baseline: base32 `0.21`, base48 `0.09`, scratch FCMAE800 `0.35`, and official
FCMAE800 `0.35`.

The direct matched mechanism requires `+0.05` q95, positive float gain, 2/2
positive folds, 7/10 positive ROIs, and a worst ROI above `-0.12`. The new
model is also compared diagnostically with the historical single-output
official FCMAE800 endpoint.

Two deployment routes are fixed before training:

- replace the current `0.35` official member by `officialauxiliary15`;
- split that member into `0.175` official single-output and `0.175` auxiliary.

Replacement requires `+0.08` over `t70_o50`. The diversity route first
requires at least `+0.02` over the matched-control split, then at least `+0.04`
over `t70_o50`. Deployment additionally requires positive float gain, both
folds, 8/10 positive ROIs, and a worst ROI above `-0.08`.

No marker weight, subset, epoch, checkpoint, ensemble weight, or ROI-specific
route will be searched after the result. A passing candidate must be expanded
unchanged to five folds and pass a complete-OOF gate before test inference.

## Formal Result

The formal pipeline exited zero and evaluated 2,706 images from all ten
preregistered validation ROIs. All eight supervised stages contain 14
consecutive metric rows, their selected checkpoints match the maximum saved
selection score, and all 12 checkpoints referenced by the OOF report match
their recorded SHA-256 hashes. The OOF evaluation took 2,338.920 seconds.

The new auxiliary model scores `74.873621` q95, versus `74.794670` for its
matched four-output control. This is a `+0.078951` mechanism gain with both
folds and 7/10 ROIs positive. The conservative mechanism gate nevertheless
fails because ROI000 loses `-0.217709`, below its `-0.12` floor. Relative to
the historical single-output official-FCMAE model, the auxiliary model gains
`+0.025771` overall, but fold2 loses `-0.007713` and the worst ROI loses
`-0.373871`.

The preregistered diversity route is the successful result:

| Candidate | q95 score | Delta |
| --- | ---: | ---: |
| `t70_o50` | 75.180556 | baseline |
| `t70_official_auxcontrol_split` | 75.195485 | +0.014929 |
| `t70_official_auxiliary_split` | 75.221576 | +0.041020 |
| `t70_official_auxiliary_replace` | 75.213565 | +0.033009 |

The auxiliary split improves its matched-control split by `+0.026091` q95
and the current deployment baseline by `+0.041020` q95 (`+0.039891` float).
Both folds and 8/10 ROIs improve, with a worst-ROI delta of only `-0.023387`.
It therefore passes both the diversity-effect and deployment-diversity gates,
and `promotion.allowed` is true. Replacement does not pass its stricter
`+0.08` threshold.

The OOF report SHA-256 is
`db1cd173c0978116e72fc9f67301b33a666f1b2a9b3b299b68c9f893de73d8d7`;
the gate report SHA-256 is
`8a30f825d5cafa9de1ac3a29cdce61234a849b353455515d3f44b2a292886b54`.

## Five-Fold Promotion Protocol

`configs/round55_cv5_unext2_official_auxiliary_promotion.json` preserves the
Round55 namespace, recipes, seed, schedules, checkpoints, and candidate
weights. It reuses fold0/fold2 and trains only the missing control and
auxiliary recipes for fold1/fold3/fold4. The existing Round52 fold-local
official-FCMAE encoders are reused by verified hard link. Their preregistered
SHA-256 values are:

- fold1: `55f821b76fbc7d257a2aebc02ad41736a3e130f08af6607fa6f698ac6030aaf6`;
- fold3: `6c755b0730cb4c512f09b82331942220a5a145f9475eff4d8f7cb5f35bb9e821`;
- fold4: `8c71c765cb217496bcc8a693030c5910106a39b898f4f28573f10738c03d1d59`.

The complete-OOF gates retain all q95 and ROI-floor thresholds. Fold and ROI
support scale before execution from 2/2 to 4/5 folds, from 7/10 to 18/25 ROIs
for mechanism/diversity-effect checks, and from 8/10 to 20/25 ROIs for the
deployment checks. The auxiliary diversity split still requires `+0.02` over
its matched-control split and `+0.04` over `t70_o50`; no ensemble weight,
marker weight, checkpoint, or ROI-specific route is added.

The six independent missing fold/recipe jobs are assigned three per GPU via
the validated `--recipes` selector. This keeps both GPUs productive without
changing any model or optimization setting. The pipeline refuses to start
unless the exact two-fold promotion decision and OOF hash above are present.

## Five-Fold Result

The promotion pipeline exited zero and evaluated all 6,296 images from all 25
ROIs in 5,422.200 seconds. Each of the 12 newly trained stages contains 14
consecutive metric rows and 14 epoch checkpoints; every selected checkpoint
matches the maximum saved selection score. All six task status files report
`complete=true`, all three reused encoders match their preregistered hashes by
hard link, and all 30 checkpoints referenced by the OOF report independently
match their recorded SHA-256 values.

The retained `t70_o50` baseline is exactly identical to Round53 at the
overall, fold, ROI, and 20-checkpoint provenance levels, ruling out evaluation
drift. The complete-OOF scores are:

| Candidate | q95 score | Delta vs `t70_o50` |
| --- | ---: | ---: |
| `officialfcmae800` | 75.220946 | n/a |
| `officialauxcontrol` | 75.243020 | n/a |
| `officialauxiliary15` | 75.282819 | n/a |
| `t70_o50` | 75.560424 | baseline |
| `t70_official_auxcontrol_split` | 75.583123 | +0.022699 |
| `t70_official_auxiliary_split` | 75.598598 | +0.038174 |
| `t70_official_auxiliary_replace` | 75.589785 | +0.029361 |

The direct auxiliary mechanism gains `+0.039799` over its matched control but
misses its `+0.05` threshold, improves only 14/25 ROIs, and has a worst-ROI
delta of `-0.325307`. The diversity effect is stable across 4/5 folds and
18/25 ROIs with a worst delta of `-0.021161`, but its `+0.015475` gain misses
the fixed `+0.02` threshold.

The fixed deployment split improves all five folds and has a worst-ROI delta
of only `-0.033516`. Its overall `+0.038174` gain nevertheless misses the
`+0.04` threshold by `0.001826`, and 18/25 positive ROIs miss the required
20/25. Its dependency also fails, so `promotion.allowed=false`. Replacement
gains only `+0.029361` and fails its gain, ROI-count, ROI-floor, and dependency
checks. No Round55 candidate is promoted and no test prediction is permitted.

The complete OOF report SHA-256 is
`0f2b52e0a5cd0fa34df5ee597019b7ce7ae53b86cfae1593d387a60a68772056`;
the gate report SHA-256 is
`663ee3016d7baedad0702ae204ee5505e5a4bb6473b42c9ce13089aaf82e3a8d`.

## Status

Implementation and pre-launch validation are complete. The related local
protocol suite passed 107 tests, the synchronized server tree passed all 251
tests, and the exact eight-command training dry run passed on both machines.

A one-epoch fold0 smoke run exercised the four-output auxiliary recipe on the
server. It loaded 196 encoder tensors from the exact Round52 epoch-800 encoder
(`50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9`),
resolved markers to CD68/CD45RO/HLA-DR/Vimentin and loss weights to
`0.85/0.05/0.05/0.05`, and wrote a loadable 287-tensor checkpoint with
31,753,556 parameters. The single epoch completed four optimizer steps in
23.997 seconds with a measured 25.541 GiB peak. Its
`best_global_mse.pt` SHA-256 is
`21c0b1cd92103442c95df515d2654c82b556baa1355590f215e91e425a59272f`.

Round55 passed its two-fold screen but failed its unchanged complete-OOF
promotion gate. The result is retained as evidence that auxiliary supervision
adds a small, mostly stable diversity benefit, but the deployed Round53
`t70_o50` ensemble remains unchanged.

# Round 46: 800-Epoch Fold-Local DAPI FCMAE

## Motivation

Round 44 established the only recent mechanism with a large direct effect:
100-epoch fold-local DAPI FCMAE improved the same UNeXt2 architecture by
`+0.761103` q95 on the fixed two-fold screen. Its 100-epoch reconstruction
loss continued reaching new minima around epochs 94-100 on all five folds.
The Cytoland/UNeXt2 reference recipe uses 800 FCMAE epochs for 2D virtual
staining with batch 32, learning rate `2e-4`, mask ratio 0.5, and a
warmup-cosine schedule. Round 44 matched those main settings except duration.

Sources:

- https://www.nature.com/articles/s42256-025-01046-2
- https://github.com/mehta-lab/VisCy
- https://github.com/facebookresearch/ConvNeXt-V2

## Preregistration

Round 46 changes only fold-local FCMAE duration from 100 to 800 epochs. It
retains seed 2044, batch 32, mask ratio 0.5, learning rate `2e-4`, ten warmup
epochs, cosine decay, weight decay 0.05, UNeXt2 architecture, downstream
14+14-epoch schedule, halo 64, D4 inference, JPEG q95, and the Round 44 folds.
The 800-epoch run starts from scratch so its cosine schedule is defined over
the full duration; it does not resume the 100-epoch endpoint.

The screen covers fixed fold0 and fold2. Direct FCMAE800 must beat FCMAE100 by
at least `+0.10` q95, with positive float gain, both folds positive, at least
7/10 ROIs positive, and no ROI below `-0.12`. The fixed Round35 + 20% FCMAE800
ensemble must beat the corresponding FCMAE100 ensemble by at least `+0.02`
and pass its distribution checks. It must also beat Round35 by at least
`+0.16`, with 8/10 positive ROIs. Failure closes longer FCMAE pretraining;
success permits a separately preregistered five-fold extension. No
intermediate epoch, mask ratio, learning rate, or blend-weight selection is
allowed.

## Two-Fold Result

The direct FCMAE800 model improved over FCMAE100 by q95 `+0.606199` and
float `+0.606753`. Both folds and all 10/10 ROIs improved; the smallest ROI
gain was `+0.059700` and the largest was `+1.345425`. The fixed 20% ensemble
improved over the corresponding FCMAE100 ensemble by q95 `+0.122005`, with
both folds and 9/10 ROIs positive and a worst ROI change of only `-0.015018`.
Relative to Round35, the FCMAE800 ensemble gained q95 `+0.277091`, with both
folds and all 10/10 ROIs positive. Every preregistered gate passed. The OOF
report SHA-256 is
`b6b48529e7c5b10552526ff557bfa63d5cf5b6cecdd65dc3c5097f817bc1d57b`.

The FCMAE loss continued reaching its minimum near the end: epoch 793 on
fold0 and epoch 770 on fold2. A five-fold extension therefore retains the
entire recipe and trains only fold1/fold3/fold4. Its final gates require the
direct longer-pretraining effect, the incremental ensemble effect, and the
primary Round35 comparison to remain positive across the complete 25-ROI
distribution before test inference is allowed.

## Five-Fold Result

FCMAE800 improved over FCMAE100 by q95 `+0.509991` and float `+0.513717`.
All five folds and 22/25 ROIs improved, but ROI009 regressed `-0.305986`,
crossing the preregistered direct-model floor of `-0.15`. Consequently the
machine gate correctly reports `promotion.allowed=false`; this result is not
rewritten or re-thresholded after the fact.

The actual fixed deployment candidate is substantially more stable. The 20%
FCMAE800 ensemble improves over the corresponding FCMAE100 ensemble by q95
`+0.108112`, with 5/5 folds and 23/25 ROIs positive and a worst ROI change of
only `-0.015018`. Relative to Round35 it gains q95 `+0.248498` and float
`+0.248242`; all 5/5 folds and all 25/25 ROIs improve, with the smallest ROI
gain still `+0.121877`. Every substantive check on the fixed ensemble passes;
its formal failure is only the dependency on the direct-model ROI floor.

Because this is the unrestricted preliminary stage and the unchanged final
candidate has unusually strong complete-distribution evidence, one controlled
diagnostic submission is authorized without changing any model, checkpoint,
blend weight, inference setting, or gate result. The submission script must
verify the original failed promotion, the passing substantive ensemble checks,
and OOF SHA-256
`eaf9cafb33136cf66185bc0c9f1c5789fbb27c1d48e85eac8934a9052936d3a7`.
No adjacent weight, ROI route, calibration, or rescue package is permitted
after the platform result.

## Diagnostic Package

The fixed 15-member candidate completed test inference in `1625.354 s` and
produced all 1,346 expected images. Packaging validation found no missing,
extra, malformed, incorrectly sized, or non-RGB images; ZIP CRC validation
also passed. Independent local transfer preserved the exact hashes.

- Prediction report:
  `reports/round46_cv5_round35_fcmae800_20_test_prediction.json`
- Submission:
  `submissions/cd68_round46_cv5_round35_fcmae800_20_context64_d4_raw_q95_plain.zip`
- ZIP size: `11,508,543` bytes
- ZIP SHA-256:
  `43508ed1609675cb18660838361f8114f86af8edfe020cf92db4537e0f07940c`
- Prediction report SHA-256:
  `fbc51cad3fe77cad8ad3964268c014064a518c35158b2c56dd544766b6af204f`

## Platform Result and Review Pause

The controlled diagnostic scored `73.4750`, with displayed `SSIM=0.826` and
`PSNR=26.061`. It improves Round44 by `+0.0879` and Round35 by `+0.2034`,
becoming the current personal best. Relative to the corresponding complete
OOF gains, the hidden-set realization is approximately `81.3%`
(`0.0879 / 0.108112`) over Round44 and `81.9%` (`0.2034 / 0.248498`) over
Round35. The longer FCMAE mechanism therefore transferred substantially
better than the earlier 100-epoch extension.

The last user-reported leaderboard high was `75.4966`, leaving a recorded gap
of `2.0216`; its current status has not been rechecked. No additional
training, inference, calibration, blend-weight search, or submission is
authorized at this point. The repository is paused for an independent Claude
review. After that review, the next decision should be based on identified
implementation/protocol defects or a genuinely new transferable mechanism,
not another leaderboard probe.

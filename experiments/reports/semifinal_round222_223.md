# Round222 signal discrimination audit and Round223 local auxiliary task

## Evidence

Round213 located sparse high-intensity errors; Round214 intensity-weighted
L1 was effectively tied and the Round219/220 texture condition lost in the
Round221 platform confirmation. Historical search found no direct local
signal-occupancy supervision at the DAPI bottleneck. Nuclear intensity or
texture alone should not be equated with marker expression or cell identity.

Round222 uses all2,380 Round216 cached global OOF CD45RO teacher images,
each produced by a model excluding its ROI. Histograms compare scores at
true target>=64 and>=128 pixels versus other pixels, with matched DAPI
brightness as a simple ranking reference. No parameter is fitted and no
test data is used. The thresholds are diagnostic signal levels, not clinical
positivity definitions. Quantized predictions reproduce the existing cache;
this is not a performance estimate for the full-data distilled student.

| Target level | Pixel fraction | Prediction ROC-AUC / AP | DAPI ROC-AUC / AP |
| --- | ---: | ---: | ---: |
| >=64 | 0.02302 | 0.88308 / 0.20461 | 0.79655 / 0.08496 |
| >=128 | 0.00700 | 0.87060 / 0.05978 | 0.82064 / 0.03381 |

At>=128 pixels, mean prediction is29.89 versus14.87 elsewhere: the model
has some localization information but still confuses many bright structures
and underestimates their intensities. AP is not accuracy or a fixed-threshold
precision. This audit does not establish irreducible ambiguity or prove that
a new task will improve the competition score. Artifact: `round222_signal_ranking.json`.

## Fixed experiment

Round223 adds a train-only spatial auxiliary head to the same domain-initialized
UNeXt2 encoder. For each16x16 target block, predict the fraction of CD45RO
pixels at least64/255. Use the stage4 pre-FiLM spatial feature map and a
768->64->4 pointwise head; only CD45RO output index1 receives this auxiliary
loss. Binary cross entropy with logits uses fixed positive weight4, normalized
per image by1+3*mean occupancy, and added weight0.05. No threshold, scale,
weight or epoch sweep. This derives a statistic from unchanged training labels;
it does not rewrite the staining image or provide additional inference inputs.

The last auxiliary projection starts at zero/bias-3 and does not feed the
image decoder. Common model weights and RNG match the base at initialization.
The image objective stays0.7 uniform7 SSIM+0.2L1+0.1MSE with existing marker
weights. No teacher, relation loss, pixel distillation, texture condition,
output shift or postprocessing is combined into this screen. DAPI-only image
inference follows the same generator; the auxiliary head is not called during
prediction. One model definition/checkpoint includes the training head.

Use original folds0/3 and immutable Round207 domain encoders. Fresh supervised
14+14 epochs,4x4 effective16, seed2026, stage1 epoch14 and stage2 epoch12.
Compare D4/halo128/offset0/q95 on980 held-out images against the same Round207
folds. Record auxiliary loss and finite gradients plus normal image metrics.
Improved signal classification alone cannot justify promotion: require useful
whole-image score evidence and inspect all fold/ROI changes. No automatic
full-data training or submission. Round21679.0198 stays the platform best.

The hypothesis is that explicit local signal supervision improves the encoder's
ability to distinguish expression patterns; it is different from directly
upweighting bright reconstruction pixels or copying DAPI edges. It is not
claimed to recover unavailable biological information. If negative, reject
this configuration without declaring the entire task family ineffective.

Validation tests cover exact initial image-output equivalence, strict checkpoint
reload, correct CD45RO auxiliary routing, encoder gradient without decoder
gradient from the auxiliary term, normalized loss and matched commands.
GPU0 only; GPU1 is occupied. Existing microbatch recipe uses about7GiB and
two-fold14+14 work costs about2.5-3h plus D4, subject to workload.

Real128-image two-stage smoke passed with7/8 actual Adam updates, nonzero
auxiliary loss0.2826/0.2798 and peak allocation6.99GiB. Supervised and
auxiliary channels/weights were verified. The unmodified trainer safety
and historical distillation suites also pass. These checks validate execution,
not image-quality improvement.

## Completed screen

Exit0. Both14+14 trajectories and fixed epoch12 D4 evaluation completed.
CD45RO gain+0.246920, with fold0+0.101765 and fold3+0.355786. Six of seven
ROIs improve; ROI087 loses0.047044. Other output markers also gain in mean.
This supports unchanged expansion, not a platform score prediction; the
earlier texture screen showed why the three remaining folds matter.
Round21679.0198 remains the confirmed submission reference.

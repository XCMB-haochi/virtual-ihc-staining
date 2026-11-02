# Round 86: Context128 Training on Official FCMAE-Tiny

## Motivation

Round85 evaluated the unchanged 25-member Base-diversity ensemble with a
context128 DAPI view and gained `+0.026427` q95 over context64. The gain was
positive on all five folds and had a safe ROI floor, but only 18/25 ROIs were
positive. Round36 also found a small benefit when a context128 model was
trained rather than merely evaluated. Round86 tests whether allowing the
downstream UNeXt2 model to learn from the larger view adds information beyond
the inference-only correction.

## Fixed Screen

`configs/round86_cv2_context128_training.json` reuses the exact fold-local
official ImageNet-1K FCMAE-Tiny encoder files from Round52. The new model is a
fresh downstream trajectory with source context halo `128`, target supervision
halo `64`, the unchanged thumbnail-FiLM conditioning, Uniform-7 hybrid loss,
D4, output source offset `[0,-1]`, and `14+14` epochs. A micro-batch of `8`
with two-step accumulation preserves effective batch `16`; no pretraining is
repeated. The historical five-member recipe set is evaluated with context128
as a matched baseline, and fixed replacement/split candidates add the trained
model to the current Base-diversity endpoint.

The direct mechanism gate requires q95 `+0.04`, positive float and both folds,
7/10 positive ROIs, and a `-0.12` ROI floor. Replacement requires `+0.03`,
both folds, 8/10 ROIs, and a `-0.08` floor after the mechanism gate. The fixed
diversity split requires `+0.02` with the same distribution checks. All values
are fixed before training; a passing candidate would expand unchanged to five
folds before any test inference.

## Execution and Result

The protocol passed local and Linux compilation, the complete regression suite,
shell syntax, exact dry-run, baseline OOF hash preflight, and a real-data smoke.
The smoke confirmed source context128, target-supervision halo64, effective
batch16 using `8x2` accumulation, the fold-local FCMAE hash, and a loadable
checkpoint. Both formal folds then ran sequentially on GPU1 because GPU0 was
occupied. Fold0 ran from `22:11:54` to `00:37:58`, fold2 from `00:38:00` to
`02:58:00` (2026-08-29/30). The OOF completed on GPU1; no unrelated process
was interrupted.

All four stage directories contain exactly 14 consecutive metric rows and
selected checkpoints. Stage-1 best epochs are 13 and 12 for fold0/fold2;
stage-2 best epochs are 1 and 9. The stage-1 FCMAE initialization hashes are
`50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9` and
`c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7`. The
selected stage-2 checkpoint hashes are
`557d54af2ecfedaa189da3a8c333744f554e660ea8cc6881c7bee267075540d2` and
`53bf718d7cdf08abd02f3683fe66ceac0fefeb3f3790271e4d8f3b43cfa8ce2c`.

The OOF covers the fixed 2,706 images and 10 ROIs and took `4407.690 s`.
Against the context128 baseline, the direct context128-trained model loses
`-0.267637` q95 and `-0.271552` float, with both folds negative, only 3/10
ROIs positive, and worst ROI `-0.779868`. Replacing the baseline ensemble
member loses `-0.013721` q95; the fixed diversity split loses `-0.000622`.
All three gates fail, so no candidate is promoted and no test package is
created. The result closes context128 supervised training: the inference-only
benefit from Round85 does not survive when the model is trained with the
larger input while keeping center/halo64 supervision.

The pipeline initially wrote exit code `1` because its final replay check
incorrectly compared overall metrics from a 10-ROI report with the 25-ROI
Round85 report. Fold metrics and all 10 shared ROI metrics were exactly
identical; after correcting that postcondition, the completion audit passed
and the exit marker was set to `0`. The runner source now compares only the
shared fold/ROI universe.

Formal artifacts are `round86_cv2_context128_training_oof.json`,
`round86_cv2_context128_training_gate.json`, and `round86_scheduler.log`.
The OOF and gate SHA-256 values are respectively
`29e2d92a5b536be6c7802970a9a7debe524a29b4132f33b267e56cc3ec84cb59` and
`b2dfbe5c84d0ec636d9e859d21faa34214afefb5df5732fa2c39b87049bc67be`.

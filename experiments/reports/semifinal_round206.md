# Round206: Gaussian SSIM loss, matched two-fold screen

The organizer's new Gaussian11 scoring provides a reason to revisit the
training SSIM surrogate. Historical Round4 preferred uniform7 by 0.1366 on
a Colon U-Net holdout scored with uniform7; that result is not evidence
against Gaussian training on new-stage CD45RO scored with Gaussian11.

Change only `--ssim-mode uniform7` to `gaussian11` in the Round204 training
recipe. Preserve architecture, immutable initial encoder bytes, primary
weights, seed2026, microbatch4/accumulation4, stage1 epoch14 initialization,
both 14-epoch cosine schedules and fixed stage2 epoch12 endpoint. Retain
Gaussian11 validation. No distillation, FCMAE domain training, guidance,
epoch search, or test submission is added. The implemented Gaussian loss
uses reflect-padded boundary pixels and is a training surrogate; validation
continues to use valid windows. It is not claimed to be byte-identical to
the unknown official evaluator.

Train fold0 and fold3: both are difficult CD45RO folds; fold0 improved in
Round204 while fold3 declined, so this pair probes the weaker part of the
current recipe rather than its easiest folds. Selection is explicit and
not a representative random sample of all folds. Seven held-out ROIs total.
Round204 measured approximately 101 minutes for fold0 and 86 for fold3
under shared compute. Screening two folds avoids about 4.3 hours of the
previous measured five-fold work if negative; current contention and
Gaussian convolution cost can change those times.

After training, inspect the fixed epoch12 and late 11-14 trajectories, then
evaluate matched D4 at epoch12 before any expansion. Do not change the
endpoint to rescue the result. A positive pooled D4 change with both folds
positive supports expansion; mixed signs require ROI-level review, not an
automatic rejection or threshold relaxation. A nonpositive mean closes
this tested configuration absent a concrete implementation failure.

Final output is a training-screen report and fold checkpoints, not a
submission ensemble. Round205 can be submitted independently while this
experiment runs. Run detached on GPU1 with concurrent-memory headroom;
GPU0 remains reserved.

Validation: command-equivalence test confirms that only the training SSIM
mode and artifact paths differ from Round204. A 128-image CUDA smoke passed
with six actual saved optimizer updates, finite losses/validation metrics
and 6.8731 GiB peak allocation. Smoke metrics do not determine promotion.

## Training completed

Exit 0. Fixed epoch12 Gaussian-validation changes versus Round204:
fold0 -0.139554, fold3 -0.061942; image-weighted mean about -0.09520.
All four late epochs (11-14) are negative in both folds. No expansion is
justified from the training screen. The Round207 pipeline first evaluates
these same epoch12 weights with D4/halo128/q95 on the seven held-out ROIs,
including all four output channels, and saves matched fold changes.
This does not delay the independent domain-pretraining test beyond that
short inference pass. No Round206 submission package is produced.

# Round226: validation-to-full-data clock audit

After the negative Round225 confirmation, review existing validation trajectories
without test labels or tuning against platform numbers. This is a retrospective
development analysis, not an independently preregistered efficacy estimate.
Both texture and auxiliary small-OOF gains failed to transfer; neither platform
outcome alone identifies a cause.

Existing no-D4 Gaussian11 five-fold curves show base stage2 best epoch11
(78.4368), plateau11-14; texture best12 (78.4596); signal auxiliary best3
(78.5699), declining to78.4337 at12 and78.3720 at14. All three stage1
curves select14. Applying the baseline's epoch12 to every mechanism did not
capture this convergence difference. Full-data epochs attempt149 updates
versus114/123 per fold, so equal epochs also imply21-31% more updates.
Equalizing updates is not automatically superior: it would change image
exposure and schedule. No corrective training schedule is assumed here.

Compute the maximum image-weighted same-epoch mean for each base/signal
trajectory (earliest exact tie), and report how this choice changes when
each fold is omitted. Then evaluate one common checkpoint per recipe under
D4/halo128/offset0/q95 over all2,380 images. No D4 epoch grid, new training,
package or platform submission; checkpoint choices use Train/Val only.
Fold omission measures selection stability; it does not make the combined
D4 result an untouched validation estimate.

This answers whether the signal-task early optimum persists under final
inference, before considering any expensive update-matched control. It does
not prove why full-data training lost on the hidden set. Preserve Round216
79.0198 and all previous artifacts. GPU1 is idle and used only for this
roughly35-50min inference diagnostic; no need to occupy both cards.

## Completed

Exit0. Common Train/Val-selected clocks are base11 and signal3. Leaving one
fold out chooses base11/12 and signal3/4. Complete D4 scores: base11
78.560365, signal3 78.731967; paired gain+0.171602 with three folds positive
and two negative. Against the signal model's own epoch12 D4 score78.713805,
early selection gains only+0.018162; two folds decline. Do not treat the
larger no-D4 epoch effect as an equivalent final-inference gain. No new
epoch3 submission is justified from this modest mixed result.

The chronology does not establish that full-data overtraining caused the
platform loss. Round227 tests update-budget sensitivity with matched base
and auxiliary models, without test data or platform-driven epoch selection.

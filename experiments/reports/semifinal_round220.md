# Round220: unchanged texture-conditioning five-fold confirmation

Round219's two-fold CD45RO gain+0.232254 is positive in both folds and six
of seven ROIs, warranting expansion. Reuse exact Round219 fold0/fold3
epoch12 weights. Train only folds1/2/4 from their existing Round216 domain
FCMAE800 encoders; no new pretraining. Keep microbatch4/accumulation4,
seed2026,14+14 cosine periods, stage1 epoch14, stage2 epoch12, all loss
weights and four-channel order, texture condition, D4/halo128/offset0/q95.

Evaluate all2,380 images and17 ROIs against the complete Round217 domain
baseline with identical inference. Report all markers, fold/ROI deltas and
aggregate. No automatic full-data training or submission. The small fold0
gain and historical Colon texture expansion failure make the remaining
folds essential evidence; an improvement must not be attributed to the
already-established domain pretraining. The baseline comparison contains
no pixel distillation and cannot predict the final student's platform gain.

Current user allocation: GPU0 only, GPU1 occupied. GPU0 had25.3GiB free;
the identical recipe's training allocation was approximately7GiB. The
wrapper pins storage-scoped Python and physicalGPU0, with an exit marker.
Expected remaining supervised work about4h plus full OOF, subject to load.
Tests confirm only artifact/source paths change from the two-fold recipe.
No existing experiment output is overwritten.

## Completed

Exit0; all three new folds completed14+14 epochs with finite losses,
correct texture condition and actual saved optimizer updates. Complete D4
CD45RO gain is+0.055169, much smaller than the two-fold+0.232254. Fold
changes:+0.026195,-0.135268,+0.018088,+0.386798,-0.084091. Eleven of17
ROIs improve; worst ROI093 loses0.364982. Excluding fold3 makes the mean
negative (about-0.0469), so the benefit is not robust across folds.

Decision: no automatic replacement of Round216. A single fixed full-data
candidate is nevertheless worthwhile given low remaining training cost,
positive total mean and available platform confirmations. Round221 retains
the same distillation targets and all training/inference settings as Round216,
changing only texture conditioning. Do not extrapolate+0.055 to hidden score.

# Round210: CD45RO relational guidance with a matched batch8 control

Question: does the initial-stage median-normalized relational mechanism
improve CD45RO beyond domain FCMAE? Round208 is confirmed at78.9741.
Round207's late validation curve shows modest epoch-selection headroom;
no epoch or loss-weight sweep is registered. Historical Round121/122 tested
pooled stage4 target relations; Round122's approximately0.20 Colon OOF gain
motivates this transfer test without guaranteeing improvement.

For each original fold0/fold3, train a Tiny FCMAE teacher160 on individual
single-channel DAPI,CD68,CD45RO,HLA-DR,Vimentin images. All five modalities
exclude that fold's validation ROIs. Initialize from the immutable
public-only encoder; seed2026, effective batch32 via8x4 accumulation,
LR2e-4 to1e-6, warmup2, mask0.5, decay0.05, resume checkpoints every10 epochs.
Five-modality160 matches DAPI800 in total image presentations. It does not
concatenate markers as model inputs or use previous-stage data.

Train one fresh control and one relational candidate per fold from the
identical Round207 DAPI800 encoder. Both use full batch8, no accumulation,
the same seed/augmentation, 14+14 cosine periods, stage1 epoch14 and fixed
stage2 epoch12. CD45RO-heavy weights and uniform7 hybrid supervised loss
are unchanged; validation is Gaussian11. Candidate alone adds weight0.05
robust encoder-relational loss: stage4 pooled features, crop256, all8
samples, lower-median pair-distance normalization. Only paired training
CD45RO targets enter the frozen teacher; it is absent from student model
state and inference. DAPI remains the only inference input.

Batch8 is a shared-GPU adaptation, not exact Colon batch16 reproduction:
28 pair distances versus120, and twice the supervised updates per epoch
relative to effective16. The new control isolates guidance; comparison to
Round207 exposes batch effects separately. Keep the rejection of relational
microbatch accumulation: independent small-batch relations do not equal
relations across a larger batch. Frozen-teacher construction now preserves
the student's CPU RNG state; loaded teacher parameters are unchanged.

Evaluate both recipes with D4/halo128/offset0/q95 at epoch12 over980 images
and7 held-out ROIs. Report all markers and paired fold/ROI gains both versus
the new control and Round207. Positive pooled CD45RO improvement with both
folds positive supports confirmation; mixed signs require magnitude/ROI
review, not an automatic single-ROI veto. A benefit versus the new control
alone is insufficient if the candidate loses to the retained baseline.
No automatic full-data training, ensemble or submission follows this screen.

Validation: three new tests cover exclusion/config, paired commands and
teacher RNG preservation; seven existing feature/relational tests pass.
A real five-modality smoke teacher and both student stages completed with
11/9 actual optimizer updates, positive finite guidance loss, correct
CD45RO target channel and no teacher in saved model state. Peak allocation
13.70GiB fits alongside the other task's20GiB peak on GPU1. GPU0 reserved;
memory floor18,000MiB, concurrent compute permitted.

Cost reference: two Round207 DAPI800 runs took11.35h for the same image
presentation budget; four recent supervised runs took4.90h. Budget roughly
16-18h plus contention/I/O for teachers, paired students and D4. Screening
two folds avoids three further teacher runs before evidence exists. Launch
detached as `round210_relational`; retain hashes and compact reports only.

## Resource assumption correction

On 2026-09-25 the user clarified that the previous shared-GPU reservation
had expired: GPU1 is now exclusively available. The launch's assumption
of a continuing20GiB reservation was incorrect. Batch8 remains this run's
experimental setting because fold0 control and relational stages have
already completed and fold3 teacher training has started. Changing only
later students would invalidate the matched design. The18,000MiB launch
floor is a minimum needed by this job, not a reservation for another task.
Future runs use current availability and measured memory; batch16 need
not be excluded. Existing teacher artifacts can be reused if a batch16
confirmation is justified, without repeating teacher pretraining.

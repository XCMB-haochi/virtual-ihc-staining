# Round227: controlled update-budget sensitivity on held-out ROIs

Round226 rules out a large simple epoch-selection rescue under D4. Another
specific discrepancy remains:14 full-data epochs attempt2086 optimizer
updates versus1722/1596 in the two existing validation folds. Test whether
increased supervised exposure changes the auxiliary task's relative gain.
This is a diagnostic experiment, not an automatically better training recipe.

Keep original folds0/3 and immutable fold-local domain800 encoders. For both
the ordinary model and signal-auxiliary model, initialize fresh supervised
students and change only stage lengths/cosine horizons. Compute each horizon
as nearest integer to14*149/ceil(fold_train_images/16); final endpoint as
nearest integer to12*149/ceil(fold_train_images/16). No fitted threshold or
epoch grid: fold0 uses17+17 with endpoint15, fold3 uses18+18 with endpoint16.
Stage2 initializes its own recipe's stage1 final epoch. Cycle update counts
are within2% and selected-stage2 counts within3.3% of the full-data reference.
AMP-skipped updates can differ; record attempted counts separately.

Seed2026,4x4 effective16, image losses, marker weights, auxiliary0.05,
original spatial thumbnail and D4/halo128/offset0/q95 remain unchanged.
No relation loss, distillation, texture feature or pretraining change. Compare
both recipes against each other and each against its original14+14 endpoint
on the same980 images. Record interaction (change in auxiliary-minus-base)
and fold/ROI differences, not just whichever model has the highest mean.

More epochs also mean more repeat exposures/augmentations on existing ROIs;
they do not reproduce the extra diversity of full-data training. Per-epoch
cosine steps are only an approximate update-time match. This experiment
can support or weaken sensitivity to extra training, not uniquely explain
the hidden-set loss or certify a full-data early-stopping rule. Any further
decision uses these development results, not a platform epoch sweep.

No submission or full-data candidate is generated. Keep Round21679.0198.
GPU0 runs the fold0 pair, GPU1 the fold3 pair; memory near7GiB each, verified
by the existing models. Estimated3-4h plus D4, depending on contention. Tests
verify budget arithmetic and paired commands; no new model/loss code.

## Completed

Exit0; eight trajectories contain all17/18 epochs, finite losses and actual
Adam updates2047-2087 per stage. Stage2 sources are each arm's own stage1
final checkpoint; all nominal settings match. D4 covers980 held-out images.
Base score77.271167, signal77.241569. Relative to original budgets, base
gains+0.189946 while signal loses0.086572. Signal-minus-base changes from
+0.246920 to-0.029598 (interaction-0.276518). Both folds show that direction:
base gains+0.064250/+0.284219; signal loses0.052261/0.112305.

This supports budget sensitivity, not a proof of the hidden-set failure's
cause. Changes include cosine horizon, repeat exposures and update count;
new-ROI diversity and full-data pretraining are not recreated. Do not infer
that a full-data17/18-epoch model is justified: these fold budgets already
approximate the existing full-data14-epoch budget. Round228 completes only
the base arm's remaining three folds as validation calibration, with no new
submission or platform epoch search. Best remains Round21679.0198.

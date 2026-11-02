# Round130: Independent Seed-31415 Relational Screen

Round129's fixed Base + relational hybrid improved complete OOF but scored
`74.2276` on the hidden platform, below the Round128 historical package at
`74.2362`. This confirms that the OOF gain from mixing independently trained
models did not transfer reliably. The organizer's latest clarification also
requires one submitted model; independently trained virtual-staining models
must not be mixed. Same-architecture automatic five-fold aggregation remains
allowed.

Round130 therefore tested one standalone seed-31415 relational model recipe.
Only the supervised seed changed from the prior relational screen; the Tiny
UNeXt2 architecture, fold-local teachers, DAPI FCMAE initialization, loss,
14+14 training schedule, inference settings, and fixed gate were unchanged.
The pre-registered two-fold screen used folds 0 and 2 and compared the single
candidate with the matched auxiliary15 control. No seed grid, blend, or hidden
set result was used.

## Completion (2026-09-15)

- Pipeline exit: `0`; GPU1 producer exited; no test package was generated
- OOF: `2,706` validation images across `10` complete ROIs
- q95 score: `75.0586479648` vs control `75.0132724440`, gain `+0.045376`
- Float gain: `+0.051714`; both folds positive; `7/10` ROIs positive
- Worst ROI delta: `-0.260831`, below the fixed `-0.12` floor
- Gate: `promotion.allowed=false`; the q95 margin and worst-ROI checks failed

This seed-31415 screen is closed and does not authorize five-fold expansion.
It is evidence against this seed under the tested relational recipe, not proof
that every single-model relational recipe is ineffective. The current
submission direction must follow the organizer's one-model rule; Round128 and
Round129 remain historical exploratory packages and are not templates for a
later-stage submission.

Artifacts:

- `reports/round130_cv2_encoder_relational_seed31415_oof.json`
- `reports/round130_cv2_encoder_relational_seed31415_gate.json`
- `reports/round130_pipeline.exit` = `0`

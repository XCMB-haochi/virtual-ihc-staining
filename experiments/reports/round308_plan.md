# Round308: queued five-fold label-excluded pixel distillation

User requests preparation now rather than waiting for Round307; defer using
October29 submission slots until today's candidates are compared on Val.
Wait for reports/round307_complete.json, including its optional package stage.
Select Round307 recipe only if its best weighted grouped Val exceeds Round303;
otherwise use Round303. Marker policy remains CD68/CD45RO/CD68 for all models.

Five inner folds partition only the8084 outer-Train images by organ-qualified
ROI, with deterministic seed2026 and image-count balancing. Each teacher is
freshly supervised from the same completed Round302800 encoder, excluding its
assigned inner-fold marker labels and every outer-Val ROI. Keep the selected
recipe and common selected endpoint, both cosine horizons14. No fold-specific
checkpoint selection; each training image receives exactly one prediction from
its label-excluding teacher, with no TTA, context or prediction mixture.

Important difference from historical full fold-local pretraining: the shared
DAPI-only encoder saw all8084 outer-Train DAPI, including inner excluded ROIs.
Thus this is supervised-label-excluded cross-fitting, not complete input-excluded
OOF. The2272 outer-Val images remain excluded from all pretraining and supervised
fitting until the later full refit. Do not repeat800 epochs for each teacher.
Use only the permitted marker per organ; auxiliary marker pixels are not loaded.

Cache fixed lossless uint8 PNG teacher outputs. Student starts from the selected
recipe's exact saved stage1 epoch14 model and RNG; reset stage2 AdamW as in the
baseline. Replay14 second-stage epochs with identical order/geometry/schedule,
changing only hard loss to0.8 hard+0.2 teacher loss. Existing optional organ-loss
weights apply to the whole sum. Original labels remain intact. Teacher targets
get the same paired geometric transform and are only loss targets, not inputs.
Evaluate outer grouped Val every epoch with existing Gaussian11/JPEG95. Compare
best stage2 score to the selected non-distilled baseline; report every organ/ROI.
No platform-based decision, extra ROI veto or distillation-weight grid.

If distilled Val is higher, Round309 refits the selected method on all10356
official Train from the original800 encoder, using the selected27-or-other
global endpoint with both14-epoch horizons. For newly added former Val ROIs,
assign one of the existing teachers by fixed sorted-ROI round robin (no score
selection); those teachers never trained on those labels. Stage1 uses hard loss
only, stage2 uses the fixed distillation loss. No further Val evaluation there.
If distillation loses but Round307 beat303, Round309 refits the chosen307
non-distilled recipe instead. If both new changes lose, retain existing Round304
without redundant refitting. Inference always one checkpoint/model, no teacher.

Estimated post-Round307 cost: five teacher fits sum about4x one grouped fit,
roughly6.5-8.5h at recent throughput; target generation minutes; student stage2
about1-1.5h; optional full fit2.5-3h. Expected evening delivery, not guaranteed.
Retain Round304 immediately available and leave time for upload before midnight.
Remote AIC-Round308 direct Python task owns waiting/training/decision/packaging;
local AIC-Round309-Download reconnects and fetches an eligible final package.
No model heartbeat or live SSH dependency. Seed2026, resumable per-epoch states,
atomic decision/completion manifests, bounded one-control chain only.

Unit tests cover whole-ROI fold exclusion/coverage, unchanged grouping RNG,
0.8/0.2 loss gradients and unmodified labels. Real smoke uses small explicitly
non-ROI-disjoint subsets for execution only: two small teachers, cached targets
and one second-stage student update sequence. It is not evidence of performance.
The production grouping test separately enforces full-ROI exclusion.

Launch October29: local/remote unit tests passed. Real execution smoke completed
two teacher fits,96 unique cached targets, source-stage1 restoration and student
epoch15, followed by finite-state/actual-step/cache-exclusion audits. Re-running
smoke audits reused completed artifacts successfully. Archive SHA
1a8f6432e38e3c4d29e96a5bf6ad1cf43794db8f29803d22aa2e97517e0313d4 verified;
exact source in round308_launch_commit.txt. Remote AIC-Round308 has a15h limit
including its dependency wait; each epoch is resumable. Local
AIC-Round309-Download waits for a package or no-refit decision. Round307 was not
interrupted or reconfigured. No guarantee against hardware/session termination.

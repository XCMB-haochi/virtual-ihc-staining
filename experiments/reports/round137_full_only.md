# Round137 full-data comparison (2026-10-21)

User authorized the reduced Colon comparison while waiting for the final stage.
This supersedes the original Round137 complete five-fold launch plan. Do not
launch the old scheduler for these results.

Resume the full-data DAPI FCMAE trajectory to epoch630 with optimizer, scaler
and RNG state. Restart the incomplete five-modality feature teacher from the
same public initialization for160 epochs; preserve its interrupted log folder.
Both use seed2026 and only Colon data.

Train two full-data students, each14+14 epochs, with encoder stage decay1.0
and0.8. Keep the original relational guidance, true batch16, losses, seed and
shared initialization/feature teacher. No new folds or OOF distillation are run.
Average each trajectory's stage2 epochs5/6/7 with weights1:2:1. Generate both
CD68 test packages with D4, halo128, zero output offset and JPEG95.

Students are isolated under runs/round137_colon/full_only; pretraining uses
the original full-scope directories. Records are under reports/round137_colon/full_only.
No independent validation or claim of improvement over Round136 is available.
Keep the previous best Colon package. Offset diagnostics are deferred with folds.

Run sequentially on physical GPU1, observed idle with40427MiB free; GPU0 is
busy. Estimated budget is2-3 days, subject to shared GPU contention. The exact
launch commit is recorded separately. Do not monitor after launch; user reports
completion. No Liver training or submitted files are modified.

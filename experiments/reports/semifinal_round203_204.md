# Round203/204: CD45RO package and primary-marker baseline

## Evidence and decisions (2026-09-22)

Round202 exited 0 with all 2,380 images, all 17 ROIs and all four channels.
Gaussian11 scores: CD68 72.9640, CD45RO 77.2654, HLA-DR 72.3467,
Vimentin 73.2730. CD45RO beats CD68 in all five folds and 13/17 ROIs;
worst paired ROI change is -9.879 (ROI072). It is the strongest mean-score
candidate, not uniformly superior tissue by tissue. Gaussian11 assumptions
remain sigma1.5, population covariance and valid windows. The organizer
has provided no further metric details.

The user supplied another participant's CD45RO platform result: 78.3845,
SSIM0.893, PSNR26.499. This is contextual evidence, not our score or a
matched architecture comparison. Existing OOF proves that marker choice
can produce a large difference even without restoring omitted mechanisms.
It does not quantify the contribution of missing domain FCMAE or guidance.

Existing uniform7 trajectories show CD45RO stage2 fold optima at epochs
11,14,13,13,12 and an aggregate plateau at 11-14. CD68 fold optima are more
heterogeneous. Taking the median of independently optimal fold epochs is
not a substitute for comparing aggregate same-epoch curves. A single
validation fold would reduce ROI coverage and does not identify a full-data
optimum with certainty. Keep the original grouped folds, without test-driven
repartitioning. Hyperparameter selection on these folds is development
evidence, not an independent unbiased final performance estimate.

## Round203: one fixed confirmation package

Use exactly Round201's final weight and inference settings, changing only
the output channel and archive directory to CD45RO. No weight averaging,
retraining, epoch search, or ensemble is added. Preserve the CD68 package.
Expected 560 RGB 256x256 images under `results/test/CD45RO/`.
Output: `submissions/semifinal_round203_cd45ro_q95.zip`.
The score is unknown until the user submits it; do not predict 78+ from OOF.

## Round204: new grouped primary-marker baseline

- Same 17 ROIs and five folds as Round200; DAPI-only inference input.
- Same architecture, context64 training, four-channel order, original fixed
  public initialization bytes, seed2026, optimizer, 14-epoch cosine stage
  budgets and uniform7 hybrid training loss. The old encoder's random stem
  was originally unseeded; preserving its exact artifact holds that source
  constant and does not silently regenerate it.
- Swap loss weights to 0.05/0.85/0.05/0.05; primary marker CD45RO.
- Validate with Gaussian11 q95, mean-image scoring. New explicit validation
  option defaults to uniform7 so historical runs retain their semantics.
- Use microbatch4 with accumulation4 (effective16) for concurrent GPU1
  memory safety; no relational guidance or distillation in this baseline.
  Workers0 avoids the observed NFS multiprocessing cleanup error.
- Train all five stage1 folds before choosing one common stage1 epoch:
  maximize the image-weighted mean score at the same epoch over all folds;
  earliest exact tie. Initialize each stage2 fold from its own checkpoint
  at that common epoch. Apply the same rule to stage2 curves. Report all
  fold trajectories and the near-optimal region, not just the winning epoch.
- No automated full-data retraining or submission from this round. Review
  the complete curves first, then perform D4 evaluation of the fixed common
  endpoint. A stage-best validation checkpoint is not a five-model submission.

This round establishes a new primary-marker baseline; comparison with
Round200 cannot isolate loss weights from the changed selection clock and
microbatch arithmetic. Domain FCMAE, relational guidance and Gaussian loss
remain separate subsequent experiments. No claim is made that omission of
those mechanisms explains the full hidden-score gap.

Round200 measured about 43.7 minutes per 14+14 fold (3.64 hours across all
five, including per-epoch validation). Full coverage is justified here by
the need for a common clock and demonstrated marker gain; a two-fold screen
would leave that question unanswered. Round204's smaller microbatches and
shared GPU make that timing a reference, not a promised completion time.
Round203 inference precedes training in detached Screen on GPU1 only.

## Validation

45 existing training-safety tests and three new metric/common-clock tests
passed locally. Gaussian validation also passed remotely. The initial
32-image smoke only triggered AMP scale adjustment (no optimizer update),
so it was not accepted as a training check. An expanded 128-image smoke
verified six actual Adam updates in the saved optimizer state, finite
metrics, Gaussian11 validation, and peak allocation 6.8658 GiB. The smoke
helper now verifies real optimizer updates explicitly. No smoke score is
used as model-quality evidence.

## Completion and platform confirmation

Round203 platform result reported by the user: **77.595**, SSIM **0.885**,
PSNR **26.059**. Versus the same final model's CD68 package (70.6241),
changing the output marker improves the actual platform result by 6.9709.
This is direct evidence for marker choice, not evidence about an architecture
improvement. Preserve Round203 as the current platform reference.

Round204 pipeline exit 0; all ten 14-epoch trajectories and all epoch
checkpoints exist. Loss/selection metrics are finite, saved optimizer states
show actual updates, and every stage2 run starts from its own stage1 epoch14.
Maximum allocated memory was 6.9919 GiB. Summed epoch runtime was about
7.40 hours under shared compute and microbatch4.

The registered common-clock rule selects stage1 epoch14 (77.16768) and
stage2 epoch12 (77.47538), using Gaussian11 q95 without D4. Stage2 epochs
11/12/13/14 score 77.46533/77.47538/77.47065/77.47140: a stable plateau.
These scores cannot be directly compared with Round202 D4 OOF. Round205
first evaluates the fixed stage2 epoch12 endpoints with matched D4/halo128,
then conditionally trains one full-data model. No re-partition is justified
by these curves, and no best-epoch median is used.

# Round241: fixed late Gaussian-valid SSIM objective

Retain Round24079.0903. Round206 rejected Gaussian-reflect training across both
stages of the public-initialized recipe (before domain800); do not erase this
negative evidence. The new hypothesis is narrower: retain the domain-adapted
uniform7 first stage and use Gaussian11 only for second-stage optimization,
with complete valid windows. This is a changed training phase/representation,
not proof that the old implementation was buggy or the leaderboard can be
matched exactly. Organizer confirmed Gaussian11 only; sigma1.5, population
covariance and valid windows remain provisional evaluator assumptions.

One new loss option gaussian11_valid; original gaussian11/uniform7 defaults
unchanged. Only SSIM's weighting/window support changes, L1/MSE weights and
all target labels are unchanged. In full-data stage2 the same new SSIM term
applies to the supervised and teacher losses; no independent weight tuning.
Reuse exact Round239 stage1 epoch14 for each fold and full-data student. Same
seed2026, encoder stage-decay0.8,14-epoch cosine,endpoint12,micro4 accumulation4,
marker weights,context64,distillation0.2 full-data-only and Round240 target cache.
No extra data or inference inputs, prediction postprocessing or test adaptation.
D4/halo128/offset0/q95 fixed, one final model.

Full stage2 and all five fold stage2 jobs are computed concurrently across
GPU queues; no stage1 retraining. Compare2380-image OOF against Round239,
which shares each exact initialization. Before results: mean>=0.03 with at
least3/5 positive folds authorizes packaging of the already-trained full-data
student for one fixed confirmation. Report every ROI; no post-hoc threshold
changes. Teacher-loss interaction is not established by non-distilled OOF.

GPU0 initially has low utilization; GPU1 has sustained competing compute.
Give GPU0 full/folds0/2/3 and GPU1 folds1/4, each about7GiB plus headroom.
Prior stage2 took roughly40min idle or60min contended; six stage2 runs should
take roughly3-4h total plus OOF/inference, not a guaranteed finish time.
This avoids the additional stage1 and sequential candidate handoff costs.
No adjacent sigma/window/epoch or loss-weight grid on failure.

Test the differentiable SSIM against the local skimage Gaussian11 evaluator,
channel-weight reductions and gradients; retain training safety regressions.
Real-data smoke covers both full-data teacher training and fold exclusion.

## Preflight and launch

Two new numerical/command tests plus45 training safety tests passed on Linux;
new tests also passed locally. Full-data and fold0 real-data smokes both
completed6 actual optimizer updates, finite weights/losses and correct frozen
initialization identity. Peak allocation6.993/6.992GiB. Teacher-loss channel1
and fold exclusion assertions passed. Archive SHA256 matches local Git export.
Formal launch uses the registered shorter GPU1 queue, with no further polling
until user completion notice. No submission score is predicted by these checks.

## Completed decision

Pipeline exit0; all six stage2 trajectories passed the runner's14-epoch,
finite-weight/loss, initialization identity, fold exclusion and actual-update
checks. Complete2380-image OOF gives Gaussian11 CD45RO78.6833261 versus
matched Round23978.6864216: -0.0030955. Fold0..4 changes are -0.001328,
-0.011948,-0.013080,+0.012296,-0.005684. Only1/5 folds and6/17 ROIs improve.
Worst ROI076 -0.174705; ROI072 improves+0.187857. Overall is effectively tied,
not evidence of a meaningful overall regression or gain.

The registered package condition fails. Close this fixed late Gaussian-valid
configuration without sigma/window/weight search; do not spend a submission.
No package or hashes file was generated, by design. Retain trained artifacts;
the full-data student has no platform score. Round24079.0903 remains primary.
Round137 is already running independently and is not interrupted by this audit.
Evidence: round241_oof.json, round241_decision.json, round241_audit.json.

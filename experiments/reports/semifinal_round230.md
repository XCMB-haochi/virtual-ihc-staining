# Round230: fixed small-set fitting capability and objective probe

Round229 shows substantial strong-signal underprediction even on TRAIN.
Before asserting a capacity limit or adding another module, test whether
the same network can fit a small, known set and whether the objective
changes that ability. This experiment has no held-out performance claim.

Select16 unique training centers, four per activity quartile (fraction of
CD45RO pixels>=64), stable ranking and seed2026. Preserve complete ROI DAPI
conditioning before selecting centers, and the usual64-pixel source/target
halos. Cache these tensors; no sparse-neighbor construction, test inputs,
label edits, augmentation, teacher or output correction. Selection uses
permitted training labels for a diagnostic, not a submission strategy.

Two identical copies of the verified Round208 epoch12 model and fresh AdamW
optimizers: original0.7SSIM/0.2L1/0.1MSE versus pure MSE, with the same four
marker weights0.05/0.85/0.05/0.05. LR1e-4 fixed, decay1e-4,4x4 effective16,
2048 attempted updates with identical deterministic sample permutations.
Record actual updates/AMP skips. Both arms deliberately revisit the same
centers; no new hyperparameter selection or model capacity change.

Measure at0/64/256/1024/2048 updates, no TTA, halo64 input, central256 output,
q95/Gaussian11 and bright-pixel bias/RMSE. These differ from Round229's D4/
halo128 settings, so use each arm's own initial measurement for progression.
Save periodic resumable optimizer/scaler/model state every256 steps, outside
the submission pipeline. No best-checkpoint selection or package.

If reconstruction improves, that demonstrates some fit headroom on these
centers, not predictable biology or unseen-ROI generalization. A differential
MSE benefit motivates a subsequent held-out objective study; lack of fit
within2048 updates does not establish an architecture ceiling or mislabeled
data. Retain Round21679.0198 unchanged. GPU0 hybrid, GPU1 MSE, independently
mapped by the coordinator. Roughly30-50min expected including scoring and
checkpoint IO; load can change this.

Validation: fixed stratified sampling and objective/gradient equivalence tests
passed. Both8-update smoke arms used identical selected filenames, initial
checkpoint hashes and initial evaluation scores. Saved Adam states show6/8
actual updates (AMP startup skips recorded), peak allocations6.88GiB. Smoke
is execution validation only; neither score is a held-out result.

## Completed

Exit0. Both arms used identical16 centers, starting weights and2048 attempted
updates. At the fixed endpoint hybrid reaches seen-patch q95/Gaussian11
98.14656, SSIM0.99749; MSE reaches94.97754, SSIM0.97859. Bright128 bias/RMSE
are -6.705/8.810 for hybrid and -0.408/6.856 for MSE. Both fit the selected
strong structures much better than their common initialization. This refutes
an absolute inability of this network to render those structures, not an
information-theoretic limit or an unseen-ROI performance prediction.

Pure MSE improves strong-intensity fidelity but loses combined score even in
this fitting probe. Do not replace the retained loss or submit either probe.
Generalization/regularization is a more defensible next question than another
capacity increase based solely on the seen-training error. Proceed with one
fixed all-fold DropPath control on current-stage CD45RO data.

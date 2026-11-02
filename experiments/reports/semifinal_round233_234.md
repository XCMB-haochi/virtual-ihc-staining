# Round233 training topology audit and Round234 explicit ROI-condition ablation

Round232 primary-only objective was effectively tied and mixed. Do not
reinterpret this as a reason to search auxiliary weights. Check instead a
dataset assumption that affects context construction and potential shortcuts.

## Round233 evidence

On all17 TRAIN ROIs, average DAPI horizontal/vertical adjacent-tile edge
differences are2.236/2.203, versus18.221/18.441 for seeded random tile edges
within ROI; ordinary interior adjacent-column difference2.238. CD45RO gives
1.914/1.897 versus13.491/13.421 random and1.897 interior. All quantities
are uint8 MAE. This supports physical continuity of TRAIN tile numbering,
not a proof of subpixel DAPI-marker alignment or TEST topology. No training
labels, input names, coordinates or submitted output were changed.

## Round234 mechanism and control

Initial Colon Round27 found geometry/statistics useful; this is a current-stage
ablation of their transfer, not a claim they were always harmful. Retain
physical context halos, per-patch statistics, neighborhood statistics and
the original5x5 DAPI thumbnail. Zero only11 explicit descriptors at indices
9:20: normalized row/column, distances to four ROI edges, ROI occupancy and
four whole-ROI intensity summaries. Do so inside the model identically during
training and inference; no test-specific statistics fitting or postprocessing.

The same network parameters and random initialization are preserved. A distinct
model specification records the mask and strict checkpoint restore reproduces
it. No mature supervised weights are continued. Initialize from existing
fold-local DAPI800 encoders; seed2026, original four-marker weights, uniform7
loss,4x4 batch,14+14 cosine, stage1 epoch14/stage2 epoch12, D4/halo128/offset0/q95.
No occupancy task, texture condition, DropPath, distillation or relation loss.

Complete all five original folds versus Round217. Position and ROI summaries
could encode useful tissue context or spurious dataset correlations; result
direction was unknown at registration. The candidate is not fully position invariant: local
presence maps/available neighbors still reveal boundaries. Positive mean alone
does not prove shortcut learning. Inspect all fold/ROI changes; no automatic
full-data model under this initial protocol or mask-grid search. The subsequent
Round235 protocol registered a bounded full-data confirmation before completed
OOF was available. Preserve Round21679.0198.

Tests verify mask index names, untouched local features and caller inputs,
equal initial weights, invariance to masked descriptors after FiLM weights
become nonzero, strict checkpoint reload, and recipe equivalence. A real
two-stage smoke checks updates and restoration before launch. Both cards
available; GPU0 folds0/3, GPU1 folds1/2/4. About4-5h at recent throughput,
using existing encoders and approximately7GiB per run.

Real two-stage smoke completed7/8 actual Adam updates, strict restore and
peak allocation6.99GiB. Local mask/restore/command tests and the45 existing
training-safety tests pass; these verify execution, not benefit.

## User-requested directory continuity audit

Audited TRAIN DAPI and all four marker directories (2380 images/17 ROIs
each), plus released TEST DAPI (560 images/4 ROIs). No TEST marker folders
are supplied. All directories have complete grids; all training marker
filename sets match DAPI. Every ROI has lower adjacent seam MAE than a
seeded within-ROI random non-neighbor reference in both axes. Mean of
horizontal/vertical MAE, uint8 intensity units:

| Directory | Adjacent seam | Random seam |
| --- | ---: | ---: |
| train/DAPI | 2.22 | 18.27 |
| train/CD45RO | 1.91 | 13.35 |
| train/CD68 | 2.56 | 19.27 |
| train/HLA-DR | 1.93 | 16.37 |
| train/Vimentin | 1.82 | 15.54 |
| test/DAPI | 2.39 | 18.29 |

Adjacency MAE is also close to ordinary interior-neighbor MAE. This supports
within-directory geometric continuity, including the released TEST inputs.
It does not establish cross-marker biological correspondence, exact registration,
or the organizer's shuffle procedure. TEST was inspected only as explicitly
requested for integrity, not used to fit preprocessing or select parameters.
No training/inference/label changes result from this audit. Evidence:
`round233_all_directories_continuity.json`.

## Completed result (2026-09-30)

Pipeline exit0; complete2380-image grouped OOF. CD45RO Gaussian11 score
78.7336558 versus Round21778.5526345, gain+0.1810213. Fold0..4 gains:
+0.040512,-0.049739,+0.151479,+0.409668,+0.252983. Positive4/5 folds and
13/17 ROIs; worst ROI072 -0.181609. This supports the tested descriptor
ablation, but does not establish shortcut causality or explain the organizer's
shuffle. Metric assumptions remain provisional. Evidence: `round234_cv5_oof.json`
and `round234_cv5_gains.json`. Round235's registered confirmation condition
passed; its full-data run and package completed. No automatic best-model promotion.

# Round238: fixed temporal parameter average with direct package handoff

Use existing weights, not another training cycle. Colon Round134's fixed
neighboring-epoch1:2:1 average improved its matched online model by0.0102.
Semifinal Round201 used averaging on a different, incomplete baseline; it did
not isolate averaging on the restored domain800 recipe. Expect at most a small
gain; this is not a proposed explanation for the leaderboard gap.

Before results: fix stage2 epochs11/12/13 and weights1:2:1 around the retained
epoch12 endpoint. No window/weight search, model mixtures or output correction.
Average each original Round207/216 fold independently and evaluate complete
2380-image grouped OOF against Round217 under identical D4/halo128/offset0/q95.
Positive image-weighted mean and at least3/5 positive folds authorize one fixed
full-data confirmation; report all ROI changes. This is a compute/submission
decision, not a claim of statistically established improvement.

If supported, immediately average the same11/12/13 endpoints of the retained
Round216 full-data distilled trajectory, infer560 outputs and generate one
package. It is one model with one weight file, never a fold prediction ensemble.
The full-data distillation interaction is not established by the undistilled OOF;
platform confirmation remains necessary. No full-data retraining is required.
If unsupported, stop without packaging or testing adjacent windows.

Use existing strict same-trajectory metadata/state validation in
average_checkpoints.py. Preflight a real fold checkpoint and run its regression
suite, then launch detached on GPU0 after Round237 evaluation ends. Expected
roughly15-30 minutes for OOF plus conditional inference, depending on contention.
Best remains Round21679.0198.

## Completed

Pipeline exit0. The fixed epoch11/12/13 average improved grouped Gaussian11
CD45RO OOF by `+0.0092295`, with 3/5 positive folds. The largest negative ROI
was ROI072 at `-0.08315`; no window or weight search was performed. The fixed
condition therefore passed only the preregistered confirmation condition.

The Round216 full-data trajectory was averaged with the same `1:2:1` weights,
producing one checkpoint and a 560-image package. ZIP CRC, image decoding,
dimensions, RGB mode and SHA256 passed. Package SHA256 is
`b2183831864bed7bca108ce8da04ccc2a639f6e7040613c8eb8d0c8ec8d3e3f0`.
Submit this once for platform confirmation; Round216 remains the retained
best until the score is known. Do not search neighboring averages.

## Platform confirmation (2026-09-30)

Round238 scored `79.0085` (`SSIM=0.899`, `PSNR=26.824`), which is `-0.0113`
below Round216. The small grouped OOF gain did not transfer to the hidden set.
Close temporal averaging for this stage; retain Round216 as the primary package
and do not search adjacent windows or weights.

Preflight passed six existing average-checkpoint regression tests and an actual
fold0 epoch11/12/13 average with strict reload, unchanged fold exclusions and
seed2026. The deployed Git archive hash matches the local archive.

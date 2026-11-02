# Round 83: Registration-Aligned Target Training

## Motivation

Round82 confirms that fixed output source offset `[0,-1]` transfers to the
hidden set: `+0.0873` over the identical unshifted 25-member ensemble. Current
models are therefore trained against targets whose official coordinate frame
is systematically displaced from the DAPI-driven prediction frame. Applying
the correction only after inference recovers metric score, but the supervised
loss still asks the network to fit misaligned structures and may encourage
blur or compromise localization.

Round83 aligns supervision before training. The loaded 384x384 target context
samples source offset `[0,+1]`, moving target content one pixel left into the
DAPI frame before paired D4 augmentation. The model is selected and deployed
with prediction source offset `[0,-1]`, restoring the official target frame.
The center 256x256 supervised/evaluated region remains inside the 64-pixel
same-ROI target context, so the one-pixel operation does not cross the loaded
context boundary.

## Matched Screen

`configs/round83_cv2_aligned_target_training.json` reuses the exact Round52
fold0/fold2 official-FCMAE800 encoders by hard link. The candidate starts a
fresh downstream trajectory with the same seed, data order, `14+14` epochs,
effective batch 16, halo64 source and target supervision, thumbnail FiLM,
Uniform-7 hybrid loss, optimizer, D4, and JPEG q95. Only training-target
coordinates and corresponding validation output coordinates change.

The direct baseline is the historical official-FCMAE800 checkpoint under the
same fixed output shift. Direct aligned-target gain requires `+0.05` q95,
positive float, both folds, 7/10 ROIs, and a `-0.12` floor. Replacement and a
fixed half-split are compared against the shifted current Base endpoint;
deployment requires `+0.04`, both folds, 8/10 ROIs, and a `-0.08` floor.
Replacement depends on the direct mechanism, while diversity may pass
independently.

The target offset, output offset, context, padding, epoch, checkpoint, loss,
ensemble weight, fold, and ROI route are fixed before training. No adjacent
alignment or leaderboard-guided follow-up is authorized. A passing result must
expand unchanged to five folds before test inference.

## Status

Implementation and prelaunch validation are complete. The focused local suite
passes 128 protocol/training-safety tests plus the target-offset data test. On
Linux, all 314 tests and 89 subtests pass, together with Python compilation,
shell syntax, exact four-command dry run, diff checks, reused encoder hashes,
and the Round80-bound no-side-effect preflight.

Round83 launched from commit `c28af24` in detached Screen `round83` at
2026-08-29 02:16:34 +08:00. A one-epoch, 64-train/32-validation smoke on GPU1
verified training target offset `[0,+1]`, validation output offset `[0,-1]`,
the exact fold0 FCMAE encoder hash, a complete metric row, and a loadable
selected checkpoint, then removed only its verified smoke directory. The
scheduler kept the run on GPU1 after measuring unrelated GPU0 utilization and
completed fold0 at 03:48:32 and fold2 at 05:13:34. The complete OOF evaluation
covered 2,706 images across all 10 validation ROIs in 2,734.953 seconds.

The independent OOF result is negative. Against the shifted official baseline,
the aligned-target model reached q95 `74.996883` versus `74.959362`
(`+0.037521`) and float `75.072695` versus `75.032690` (`+0.040004`). Only
fold2 improved (`+0.154943`); fold0 regressed
`-0.087796`, and only 5/10 ROIs improved. The worst ROI delta was `-0.268250`,
so the direct mechanism gate failed. Replacing or splitting one Base endpoint
member also failed (`+0.021384` and `+0.013464` q95 respectively), despite
passing their ROI-floor checks. No Round83 candidate is promoted, and the
aligned-target branch is closed without adjacent-offset follow-ups.

Artifacts are recorded in `round83_cv2_aligned_target_training_oof.json`,
`round83_cv2_aligned_target_training_gate.json`, and `round83_scheduler.log`.
The OOF and gate SHA-256 values are respectively
`c690db4669f1c1d3b1dbb9b93b643bebf929b01844b48cd959590352ef618bc4` and
`4b6323afb490afaffd6e0af0707fb3b8c9c5653b2cbe0f0b0bbb404606388876`.

Two CPU-only diagnostics on the earlier Round37 cache do not justify a new
branch. A cross-fitted input-feature gate for applying the fixed shift changed
q95 only from `74.986058` to `74.986651` (`+0.000593`) and still left the two
known weak ROIs negative. Replacing edge padding at the left seam with the
same-ROI neighbouring column reduced q95 by `0.001691`. Both checks are
diagnostic evidence only; neither was used for selection or submission.

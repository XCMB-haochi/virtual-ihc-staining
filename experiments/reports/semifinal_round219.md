# Round219:复赛 DAPI texture thumbnail screen

Round218 closed fixed output offsets under the no-postprocessing rule. Round219
tests the existing in-domain texture-thumbnail condition, which augments the
5x5 DAPI spatial thumbnail with fixed block contrast and gradient maps. It is
an input condition computed from complete train/test DAPI ROI structure, not
an output correction or target-derived mask.

Only the condition mode changes from the Round204/208 supervised recipe:
domain FCMAE800 fold encoders, CD45RO-primary weights, seed2026, uniform7
hybrid loss, microbatch4/accumulation4 (effective16),14+14 epochs,
Gaussian11 validation, D4/halo128/offset0/q95.
Screen fold0 and fold3 on GPU1/GPU0 in parallel if both are free. No full
expansion or submission unless both folds show a meaningful matched gain.

Historical Colon Round95 gained only +0.01984 and failed its gate; this is a
new-stage transfer check, not a claim that texture conditioning is proven.

## Launch correction and recovery

The original fold0 job completed both14-epoch stages; fold3 never trained:
the raw SSH launcher omitted the project environment and used system Python
without torch. The earlier statement that both folds were running was not
verified. No model efficacy conclusion follows from that launch failure.
The wrapper now pins the storage-scoped Python, records exit status and
automatically evaluates fixed stage2 epoch12 after training. Recover only
fold3 on GPU0 with the unchanged configuration; evaluate completed fold0 on
GPU1 in parallel. Each D4 result compares against the exact same fold of
Round217's domain baseline. Do not compare a single-fold aggregate against
the full five-fold mean. No submission package is generated.

## Completed matched screen

Both fold wrappers exited0 after recovery. CD45RO D4 changes against the
same-fold domain baseline: fold0+0.026195, fold3+0.386798; image-weighted
mean+0.232254 over980 images. Six of seven ROIs improve; ROI087 loses0.017384.
This justifies unchanged expansion, not promotion to a submission. Preserve
both trained folds and train only folds1/2/4. Round21679.0198 remains the
confirmed reference. q95 is fixed and is not re-optimized.

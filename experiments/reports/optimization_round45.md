# Round 45: DAPI Photometric Augmentation Screen

## Preregistration

Round 44 improved the official result to 73.3871, but only about 58% of its
cross-validation gain transferred to the hidden scorer. Round 45 tests a
generalizable domain-shift mechanism rather than another leaderboard probe:
during training, input DAPI alone is randomly transformed with probability
0.5 using gamma in [0.9, 1.1], gain in [0.9, 1.1], and additive offset in
[-0.02, 0.02]. Targets, validation inputs, inference, and TTA are unchanged.

The new stage-2 model and a same-initialization plain control use the same
seed, schedule, UNeXt2-FCMAE initialization, context halo 64, D4 inference,
and global-MSE checkpoint selection. Only fold0 and fold2 are screened first,
using the fixed Round 44 ROI subset. Promotion requires the direct augmented
model to beat its plain control by q95 >= 0.03 with positive float gain, both
folds positive, and at least 7/10 ROIs positive. The 20% blend must then beat
the FCMAE baseline by q95 >= 0.02 under the same distribution gate. The
primary Round 35 comparison requires q95 >= 0.08. A failed gate closes this
mechanism without five-fold expansion.

No hidden labels, test outputs, or platform scores are used to choose these
settings.

## Result

The two-fold run completed in 1,790 seconds. The direct mechanism failed
decisively: `photo_aug - photo_plain` was q95 `-0.283925` and float
`-0.281141`, with both folds negative and only 1/10 ROIs positive. It is
closed and must not be expanded to five folds. The fixed 20% blend was
`+0.048482` q95 versus the FCMAE-only candidate, but this does not rescue the
mechanism because its required direct gate failed; relative to the Round35
baseline the blend was `-0.151194` q95. The temporary worker-cleanup
`Device or resource busy` messages did not affect completion; both fold logs
reported `COMPLETE` and the OOF/gate reports were produced.

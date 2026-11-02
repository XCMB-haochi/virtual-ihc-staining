# Round 87: Context128 Full-Target Training

## Motivation

Round85 showed that the fixed 25-member ensemble benefits from a context128
DAPI view at inference (`+0.026427` q95), while Round86 showed that training
with context128 but retaining only center/halo64 supervision is strongly
negative. Round22's full-target experiment provides the remaining specific
evidence: supervising the available context can stabilize a larger input view.

## Fixed Screen

`configs/round87_cv2_context128_fulltarget.json` reuses the exact fold-local
official ImageNet-1K FCMAE-Tiny encoders from Round52 and starts a fresh UNeXt2
downstream trajectory with source and target halos both `128`. Validation and
scoring still use the official center `256x256` target. The model, thumbnail
FiLM, Uniform-7 hybrid loss, D4, fixed output source offset `[0,-1]`, seed,
and `14+14` schedule are unchanged. A micro-batch of `8` with two-step
accumulation preserves effective batch16; no FCMAE pretraining is repeated.

The screen evaluates the trained model directly against the context128 official
baseline, then tests fixed replacement and half-split diversity in the current
Base-diversity endpoint. Gates are fixed before execution: direct q95 `+0.04`,
replacement `+0.03`, diversity `+0.02`, positive float and both folds, at least
7/10 or 8/10 positive ROIs, and the registered ROI floors. A passing route
would expand unchanged to five folds before test inference.

## Status

The implementation and Linux screen completed successfully on 2026-08-30.
The runner verified the Round85 context128 baseline hash, hard-linked the two
fold-local FCMAE encoders, performed a real-data smoke, scheduled from live GPU
utilization/free memory, and compared only the shared fold/ROI universe when
replaying the historical context128 baseline. The smoke ran on physical GPU1
at 07:19--07:20 (Asia/Shanghai). Fold0 ran on GPU1 from 07:25:28 to 09:46:30
and fold2 from 09:46:32 to 12:06:32; evaluation started at 12:06:34. Both
training processes exited with status 0, and `reports/round87_pipeline.exit`
is `0`. The smoke also emitted the known NFS `pymp-* EBUSY` cleanup traceback
after its successful check; it did not affect the pipeline exit or result files.

The screen covers 2,706 validation images across the ten registered screen
ROIs. Each fold has exactly 14 metric rows (last epoch 14). Stage-1 selected
epoch 14 for fold0 and epoch 10 for fold2; stage-2 selected epoch 1 and epoch 5,
respectively. The selected stage-2 checkpoint SHA-256 values are:

| fold | checkpoint SHA-256 |
| --- | --- |
| fold0 | `cbceceefd0e21bccafe2681927b03e1c947aeff5cda54572c1325c337cf0cc53` |
| fold2 | `64e23abcb5aae5aac26136a7a1151a9d28f9bda692d8833e5bb786d6ed756ffd` |

The OOF report SHA-256 is
`77b7bcecfd0f2531f53903731dbd0578eaa35087b6df05125b6517b5a19f8117` and the
gate report SHA-256 is
`ec98b793afc52fc273512489065378dbaf2cb508d04fe4adbad6779ba226db5c`.
The candidate OOF took `4624.771 s`.

Against the official context128 inference-only baseline, the direct trained
candidate changed JPEG-q95 from `74.987515` to `74.785846` (`-0.201669`) and
the float score from `75.061378` to `74.855637` (`-0.205741`). Neither fold
was positive and only 2/10 screen ROIs improved; the worst ROI change was
`-0.594291`. The fixed Base-split replacement changed q95 by `-0.008421`
and the diversity split by `+0.001909`, with only one of two folds positive in
each comparison. The distribution gate therefore records
`promotion.allowed=false` for all three routes. An independent local replay
of the gate matched the saved result apart from expected absolute path fields,
and the historical context128 baseline was identical on both shared folds and
all ten shared ROIs.

This closes context128 supervised training (Rounds 86--87). Do not expand its
halo, training target, blend, or epoch schedule from this result. Round85's
fixed inference-only context128 ensemble remains a separate, small positive
diagnostic mechanism; any test package made from it must be labelled
confirmatory and must not be used to tune another candidate.

# Round 74: ConvNeXt-V2 Encoder Stochastic Depth

## Motivation

Round73 shows that reducing the entire FCMAE encoder learning rate to 0.1x is
catastrophic: the encoder must adapt strongly to virtual staining. The official
ConvNeXt-V2 fine-tuning recipe uses stochastic depth instead of uniformly
suppressing encoder updates. For Tiny, Meta's published command fixes maximum
DropPath at `0.2`, distributed linearly across its 18 blocks.
The implementation and fixed rate follow Meta's official
[ConvNeXt-V2 model](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py)
and [Tiny fine-tuning recipe](https://github.com/facebookresearch/ConvNeXt-V2/blob/main/TRAINING.md).

Round74 independently implements that mechanism in the local UNeXt2 encoder.
During training, each sample randomly drops residual branches and rescales kept
branches by their survival probability. The rate grows linearly from zero to
0.2 with depth. Decoder blocks remain at zero. DropPath has no parameters or
buffers, preserves every state-dict key, and is exact identity during eval, so
inference cost and checkpoint compatibility are unchanged.

## Fair Screen

Fold0 and fold2 reuse the same official-FCMAE800 encoders as Round52, with
SHA-256 values `50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9`
and `c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7`.
The candidate is trained fresh for the unchanged 14+14 supervised epochs at
full learning rate. Seed, data order, effective batch, loss, halo64 supervision,
thumbnail FiLM, global-MSE selection, D4, and q95 match Round52 exactly.

## Gates

The direct model must gain at least `+0.05` q95 over the matched no-DropPath
control, remain positive in float and both folds, improve 7/10 ROIs, and keep
the worst ROI above `-0.12`. Replacement requires that mechanism plus `+0.06`
over `t70_o50`, both folds, 8/10 ROIs, and a `-0.08` floor. An independent
half-split diversity route requires `+0.04` under the same distribution checks.

A passing route must expand unchanged to all five folds before test inference.
Failure closes the official fixed 0.2 Tiny recipe; no DropPath-rate grid,
decoder DropPath, epoch rescue, or leaderboard probe follows.

## Status

Implementation is complete. All 149 focused local UNeXt2, ConvNeXt, training-
safety, and CV tests pass, together with Python compilation, launcher syntax,
and diff checks. Tests verify linear 18-block rates, decoder exclusion, exact
eval identity, unchanged state keys, and stochastic training behavior. The
launcher includes hash-bound encoder reuse, a resource-gated 64/32 real-data
smoke, and adaptive two-GPU fold scheduling. Linux validation, preflight,
smoke, and launch remain pending.

On Linux, `git fsck` confirms the repaired remote object database is healthy,
and all 296 tests pass. The hash-bound preflight validates both reused encoder
hard links and confirms that no formal output exists. Round74 launched from
commit `5d9488f` in detached Screen `round74` at 2026-08-26 11:57:39 +08:00.

The scheduler found GPU0 externally occupied at 54% and GPU1 idle, so the real-
data smoke ran only on GPU1. All runtime assertions passed and the verified
smoke directory was removed at 11:59:03. Fold0 then launched on GPU1. A fresh
GPU0 sample remained at 56%, so fold2 stays queued instead of competing with
the unrelated process. Formal completion, OOF, and gate status remain pending.

## Completion Audit

The scheduler kept GPU0 excluded throughout its external load and ran fold0,
then fold2, serially on GPU1. The pipeline exits `0`; the smoke is removed, all
four supervised stages contain 14 metric rows, and OOF covers 2,706 images and
10 fixed ROIs in 1,787.04 seconds. All 10 checkpoint hashes match, while the
historical official-FCMAE and `t70_o50` controls exactly replay Round62. The
OOF SHA-256 is
`64c7701b0ff2ed22269aff337bd679a2d9d41647cf6789974c1a52377a230f6b`;
the gate SHA-256 is
`8577a92245b3e8908b67b51b20834e831ba0034833f78d14ec3bff4017c95378`.
Independent local gate evaluation exactly reproduces the saved result.

## Result

The DropPath model is effectively tied in aggregate: `74.847894` versus
`74.847851`, only `+0.000044` q95, despite a positive `+0.004614` float delta.
The apparent tie hides strong redistribution. Fold0 loses `-0.079279`, fold2
gains `+0.074370`, only 3/10 ROIs improve, and the worst ROI loses `-0.346084`.
The model gains a little SSIM (`0.839265` versus `0.839017`) but loses PSNR
(`26.832243` versus `26.861077`).

Full replacement loses `-0.034732` versus `t70_o50`. The half-split diversity
candidate gains only `+0.002637`, is positive on one fold and 4/10 ROIs, and
misses every aggregate/fold/ROI threshold except its ROI floor. Every gate
fails and `promotion.allowed=false`.

Round74 closes the official fixed Tiny maximum DropPath rate of 0.2. No rate
grid, decoder stochastic depth, five-fold expansion, test inference, or
leaderboard probe is authorized. The result suggests stochastic depth trades
SSIM against PSNR and shifts ROI behavior rather than improving cross-domain
accuracy.

# Round 73: FCMAE Encoder Differential Learning Rate

## Motivation

Round59 establishes a steep labelled-data curve, while Round71 shows that
larger Base capacity transfers only a fraction of its OOF gain. Round72 closes
high-frequency correction even under generous target-aware oracles. The next
portable data-efficiency question is whether supervised fine-tuning destroys
too much of the in-domain FCMAE representation.

Current UNeXt2 training applies the same learning rate to the pretrained encoder
and randomly initialized decoder/FiLM/head. Round73 fixes the encoder at one-
tenth of the downstream rate: Stage1 uses encoder/decoder rates `3e-5/3e-4`,
and Stage2 uses `1e-5/1e-4`. Both cosine schedules reach zero together. No
parameter is frozen and the architecture, update count, seed, data order, loss,
context, D4 inference, JPEG path, and checkpoint selection remain unchanged.

## Fair Screen

Fold0 and fold2 reuse the exact Round52 official-FCMAE800 encoders by hard link:

- fold0 SHA-256:
  `50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9`;
- fold2 SHA-256:
  `c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7`.

The existing Round52 full-rate runs are the exact matched controls. Fresh
Round73 stage1/stage2 trajectories use seed 2044, 14+14 epochs, effective batch
16, halo64 full-target supervision, thumbnail FiLM, Uniform-7 hybrid loss,
global-MSE selection, D4, and q95. No 800-epoch pretraining is repeated.

## Gates

The direct LLRD model must gain at least `+0.05` q95 over full-rate official
FCMAE, remain positive in float and both folds, improve 7/10 ROIs, and keep the
worst ROI above `-0.12`. Replacement in `t70_o50` requires that mechanism plus
`+0.06`, both folds, 8/10 ROIs, and a `-0.08` floor. An independent fixed
half-split diversity route requires `+0.04` under the same distribution checks.

A passing route must expand unchanged to all five folds before test inference.
Failure closes the fixed 0.1 encoder multiplier; no multiplier grid, freezing
schedule, epoch rescue, or leaderboard probe follows.

## Status

Implementation is complete. The optimizer extension and fixed protocol pass 103
focused local UNeXt2, ConvNeXt, and CV tests, together with Python compilation,
launcher syntax, and diff checks. Parameter-group tests prove that every UNeXt2
parameter belongs to exactly one encoder or decoder/head group. Linux
validation, hard-link preflight, real-data smoke, and launch remain pending.

Because both A100s were externally busy at validation time, the formal launcher
contains a gated smoke phase rather than competing for a card. It waits for a
GPU below 30% utilization with at least 27 GB free, trains 64/32 real images for
two epochs, verifies exact encoder provenance, complete disjoint parameter
groups, the scheduled 10:1 rate ratio, and a selected checkpoint, then removes
only the verified smoke directory before scheduling fold0/fold2. Formal
training cannot start if any smoke invariant fails.

Round73 launched from commit `ef8b521` in detached Screen `round73` at
2026-08-25 21:11:43 +08:00. The first smoke scheduler samples found both GPUs
at 100% utilization, with only 15,159 MB free on GPU0 and 645 MB on GPU1, so no
project GPU process was started. A later point sample found GPU1 at 0%
utilization but still 39,804 MB resident, which also fails the 27 GB free-memory
requirement. The Screen remains in its five-minute resource loop; smoke,
training, OOF, and gate status are all pending.

## Completion Audit

The scheduler waited until GPU1 met both resource thresholds, completed and
removed the real-data smoke, then trained fold0 on GPU1. It launched fold2 on
GPU0 only after that card's measured utilization fell to zero. Both folds exit
successfully and all four supervised stages contain 14 metric rows. Every
config records the exact fold-local epoch-800 encoder hash, disjoint optimizer
groups covering all parameters, and Stage1/Stage2 encoder rates `3e-5/1e-5`
against decoder rates `3e-4/1e-4`. The scheduled ratio remains 10:1 until both
groups reach zero.

OOF covers all 2,706 images and 10 fixed ROIs in 1,775.88 seconds. All 10
referenced checkpoint hashes match the server files, and the historical
official-FCMAE and `t70_o50` controls exactly replay Round62. The OOF SHA-256
is `9972576c4c95440b11af7d6d2093a26c5adafa8d3ffbe76d604603fc1dec505e`;
the gate SHA-256 is
`8622e2f78635fcd7c2d0c6dcc35fc3acdf4dc3994f40dc67b09e8eb9ff6f621b`.
Independent local gate evaluation exactly reproduces the saved result.

## Result

The differential-rate model scores `74.017860` versus `74.847851` for the
matched full-rate model, a q95 loss of `-0.829990` and float loss of
`-0.831315`. Both folds regress (`-0.949167`, `-0.718321`), only ROI000 is
slightly positive, and the worst ROI loses `-1.491240`. Full replacement loses
`-0.340463` versus `t70_o50`; even the half-split diversity candidate loses
`-0.122997`, with both folds negative and only ROI000 positive.

Every gate fails and `promotion.allowed=false`. The result is too large and
uniform to attribute to threshold noise: the pretrained encoder must adapt
substantially to the staining objective under this architecture. Round73 closes
the fixed 0.1 multiplier and does not authorize a multiplier grid, freezing,
five-fold expansion, test inference, or submission.

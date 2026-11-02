# Round209: matched CD68-primary domain-pretraining screen

Question: does fold-local domain FCMAE improve CD68 when it is the primary
target, rather than a5% auxiliary output? Test fold0/fold3 using both public
and corresponding Round207 domain encoders. Hold architecture, seed2026,
microbatch4/accumulation4, losses, validation, 14+14 cosine periods and
stage1 epoch14/stage2 epoch12 endpoints fixed. Only encoder initialization
differs within each public/domain pair. CD68 loss weight0.85, others0.05.
Validate Gaussian11 q95; report both CD68 and CD45RO. No stage clock search
or submission is included. CD68 weights will not replace CD45RO without
matched evidence and an appropriate final-model confirmation.

First complete the missing Round207 seven-ROI D4 audit at stage2 epoch12,
using the identical Round204 baseline predictions and all four channels.
Do not imply that it was already completed when launching Round208.

Reuse the two existing epoch800 encoders, checking excluded validation ROIs
and recording their hashes. No new FCMAE training. Four14+14 supervised runs
cost approximately4.7h at the recent Round206 throughput, plus inference;
shared GPU load can change this. GPU1 only, memory floor16GiB, no waiting
for idle compute. Round208 is now confirmed at78.9741. Initial test confirms paired command
equality except encoder and run paths; the training path was exercised by
Round204/207. This is a mechanistic comparison, not a complete restoration
of relational guidance/distillation or a new cell-level architecture.

## Completed

Exit0; all eight14-epoch trajectories verified with finite losses/scores,
seed2026, CD68 primary target, correct held-out ROIs and saved optimizer
updates. Summed supervised epoch times:4.90h. Fixed epoch12 validation:

| Fold | Public CD68 | Domain CD68 | Change |
| --- | ---: | ---: | ---: |
| fold0 | 68.12646 | 69.95411 | +1.82765 |
| fold3 | 68.72566 | 69.20045 | +0.47479 |

Domain pretraining benefits CD68, but this recipe remains well behind
CD45RO on the same folds. No CD68 package is justified, and no claim is
made about other participants' implementations. These CD68 scores lack D4.

The completed Round207 D4 audit gives CD45RO fold scores77.45609/76.80009,
gains+1.54149/+1.41701 over Round204. On Round207's no-D4 validation curves,
stage2 epoch14 exceeds epoch12 by just0.049535 in the weighted two-fold
mean; epoch11 exceeds it by about0.060085. These exploratory comparisons
suggest modest checkpoint headroom, not another one-point gain. Prioritize
one matched relational-guidance test instead of a platform epoch sweep.

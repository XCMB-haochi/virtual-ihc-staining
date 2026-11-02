# Round 79: Base-Capacity Auxiliary Supervision

## Motivation

Round78 rejects target-intensity L1 weighting. The strongest remaining
supervised signal is the small but repeatable auxiliary-marker effect: Round55
showed that CD45RO, HLA-DR, and Vimentin gradients improve a matched Tiny
four-output control. Separately, Base FCMAE is the only capacity change that
passed complete five-fold diversity gates and transferred positively to the
hidden set. Their interaction has not been tested.

Round79 asks whether Base capacity can use aligned auxiliary labels more
effectively than Tiny. It does not repeat the 800-epoch Base pretraining. Both
folds reuse the exact audited Round62 fold-local Base encoders.

## Matched Screen

`configs/round79_cv2_base_auxiliary.json` trains two fresh four-output Base
models per fold. Both share the same architecture, encoder, seed, data order,
`14+14` epochs, micro-batch 8 with two-step accumulation, halo64 supervision,
thumbnail FiLM, Uniform-7 hybrid loss, and CD68-only checkpoint selection:

- `baseauxcontrol`: marker weights `1/0/0/0`;
- `baseauxiliary15`: marker weights `0.85/0.05/0.05/0.05`.

The matched control isolates auxiliary gradients from the four-output head and
changed random-number path. Auxiliary predictions never enter the competition
score.

Direct auxiliary effect requires `+0.05` q95 over matched control, positive
float, both folds, 7/10 ROIs, and a `-0.12` floor. A fixed half-split of the
current Base share must first gain `+0.02` over its matched-control split, then
gain `+0.04` over `t70_base_split`, with both folds, 8/10 ROIs, and a `-0.08`
deployment floor. Full Base-member replacement requires the direct mechanism
plus the same deployment checks.

No auxiliary weight, marker subset, capacity, epoch, checkpoint, ensemble
weight, fold, or ROI route will be searched after the result. A passing route
must expand unchanged to all five folds before test inference; this screen
cannot generate a submission.

## Status

Implementation and prelaunch validation are complete. The focused local suite
passes 147 tests. On Linux, all 306 tests and 80 subtests pass, together with
Python compilation, shell syntax, exact eight-command dry run, diff checks,
the two Base encoder hard-link hashes, and the hash-bound no-side-effect
preflight.

Round79 launched from commit `205e08b` in detached Screen `round79` at
2026-08-28 07:12:20 +08:00. A one-epoch four-output Base smoke on GPU1 verified
the marker order, weights `0.85/0.05/0.05/0.05`, effective batch 16, exact
fold0 Base encoder provenance, complete metric row, and selected checkpoint,
then removed only its verified smoke directory. The first post-smoke sample
still caught residual utilization, so the scheduler waited one interval.
At 07:19:44 GPU1 was idle with 40,427 MB free and launched formal
`fold0:baseauxcontrol`. A later sample measured 98% utilization, 27,293 MB
resident, and about 218 W. GPU0 remains excluded under unrelated 94% load.
The other three tasks, OOF, and gates remain pending.

All four formal fold/recipe tasks later exited zero. Each of their eight
supervised stages contains exactly 14 contiguous metric rows and a selected
checkpoint. The OOF evaluator covered all 2,706 images and ten ROIs in
4,370.32 seconds. All 14 recorded checkpoint hashes independently match their
files. The OOF SHA-256 is
`faaf75fd3aafeb2a2e227abfc58e0f6b6bb147c19bb2a23fb085cdbba0ef2fe2`;
the gate SHA-256 is
`2b00c283c12337f07d5691192c1ea6a3011f57f3c6d06a3f4c374be4e9189094`.
Independent local gate evaluation reproduces every metric and decision.

## Result

Base auxiliary15 scores `74.793586` versus `74.608049` for its matched
four-output control, a `+0.185536` q95 effect. Both folds improve, but only
6/10 ROIs are positive and ROI022 loses `-0.243590`; the direct mechanism gate
therefore fails its ROI count and floor despite passing aggregate and fold
checks. Relative to historical single-output Base, auxiliary15 is only
`+0.038452`.

The fixed Base-half auxiliary split gains only `+0.004770` over its matched
control split, with fold0 negative and 5/10 positive ROIs. Versus the current
Base endpoint, replacement gains `+0.013531` and the half-split gains
`+0.014945`; both remain negative on fold0 and improve only 6/10 ROIs. Every
deployment gate fails and `promotion.allowed=false`.

Round79 closes Base auxiliary composition at the fixed 15% marker weights. No
auxiliary-weight, marker-subset, capacity, epoch, ensemble-weight, five-fold,
test-inference, or submission branch follows.

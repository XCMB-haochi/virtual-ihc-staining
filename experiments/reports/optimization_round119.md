# Round119: Corrected Primary-Gradient Projection

Round118 tested asymmetric projection of auxiliary gradients away from the
CD68 gradient, but its AMP-scaled implementation recorded 4--5 overflow skips
per stage. Its near-neutral OOF (`-0.006165` q95) is therefore contaminated and
is not used to close the hypothesis.

Round119 keeps exactly the same fold0/fold2 Tiny UNeXt2, Round52 FCMAE800
initialization, Round55 auxiliary15 weights, seed, losses, effective batch16,
14+14 schedule, context, D4, q95 and offset. The corrected implementation
accumulates unscaled primary gradients, unscales total optimizer gradients
before projection, and applies projection once per completed effective batch.
The primary gradient is preserved and only an opposing auxiliary component is
removed. Non-finite gradients remain visible to GradScaler.

The correction is the only code change relative to Round118. It is registered
as an independent two-fold screen with the unchanged `+0.05`, positive-float,
both-fold, 7/10-ROI and `-0.12` floor gate. A failed corrected screen closes
this exact projection method; no coefficient or layer search follows.

Only physical GPU1 is authorized. The queue waits for existing compute
processes to exit before running a real two-stage smoke; GPU0 remains reserved
for the user's other project. No training has started until the smoke passes.

## Smoke And Formal Launch

The real smoke passed both stages. Stage1 had 3 finite projection updates and
one ordinary AMP non-finite skip; stage2 had 4/4 finite projected updates and
4 conflicts. Peak memory was 15.21 GiB. This skip is recorded explicitly and
is distinct from Round118's projected-gradient overflow; the formal trainer
retains GradScaler's normal skip behavior.

The formal queue then launched fold0 on GPU1 from source commit
`852c38035968a6552e3161b4d980fa7bc376eabd`, with no GPU0 process. The queue
receipt is `round119_queue.json` and launch receipt is `round119_launch.json`.
Round119 is currently training; OOF and gate are pending.

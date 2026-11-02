# Round118: Project Conflicting Auxiliary Gradients

The corrected complete-ROI CPU diagnostic measured conflicts in 6/16
training-endpoint batches. The maximum opposing component was 0.409780 of the
weighted CD68 gradient norm squared. This supports testing interference, not a
claim that projection improves generalization. Round60/61 tested static
encoder/decoder separation; they did not retain aligned gradients and remove
only an opposing component. History search found no such trained screen.

Let `gp = grad(0.85 L_CD68)` and `ga = grad(0.05 sum L_aux)` over all trainable
parameters after accumulating effective batch16. Apply
`g = gp + ga - min(dot(ga,gp),0)/dot(gp,gp) * gp`, treating zero primary norm
as no projection. This asymmetric, primary-preserving variant is inspired by
[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html);
it is not the original symmetric multi-task algorithm. AdamW's preconditioning
means the projection is not a guarantee of descent after the optimizer step.

Fold0/fold2 use fresh supervised Tiny UNeXt2, existing fold-local Round52
FCMAE800 encoders, Round55 auxiliary15, seed2044, 14+14 epochs, `8x2` batch,
context64 training/context128 inference, D4/q95 and `[0,-1]`. The global-MSE
selector and model architecture stay fixed. The primary comparison is the
matched Round55 auxiliary checkpoint. No coefficient, layer, marker-weight or
ensemble search is registered. Gate: q95 +0.05, positive float and both folds,
7/10 positive ROIs and worst ROI >= -0.12. Failed evidence ends this recipe.

The extra primary backward is accumulated under the same AMP scale as total
gradients. Projection occurs once per effective batch, including the shorter
last batch. Non-finite components are exposed to GradScaler so it skips the
update. Epoch records include activation and overflow counts.

## Cost And Queue

Recent Tiny runs measured 90-92.5 minutes per two-stage fold. One extra backward
is expected to make this screen approximately 1.5-2x as expensive; this estimate
is provisional until GPU smoke measures the implementation. Two folds avoid
three additional folds of a rejected objective. Unlike large/huge, no new
domain pretraining or large checkpoint preparation is needed. Larger models
remain allowed if later cost/benefit evidence supports them.

Only GPU1 is authorized for this queue; GPU0 runs the user's other project.
The first run was contaminated by AMP-scaled gradient overflow; Round119 is the
corrected registration. `--queue --gpus 1 --wait-for-empty-gpu` requires existing compute processes to
exit and resource samples to pass, then runs both real-data smoke stages. Any
smoke failure stops before formal training. The existing memory-margin check
also prevents launch if measured peak plus 4 GiB exceeds the 24,000 MiB budget.
Training then runs both folds sequentially and writes complete OOF/gate. Source
is pinned in an isolated deployment, with datasets/checkpoints shared by links.
Queue registration is not evidence that training has started or smoke passed.
The first run later proved implementation-contaminated by AMP-scaled projection
overflow; its OOF is retained only for that audit. Round119 is the corrected
independent registration.

Round118's formal result is therefore classified as implementation-contaminated:
its four stages recorded 4--5 projection overflow skips per stage. Its OOF and
gate remain immutable audit artifacts, but are not evidence for or against the
corrected mechanism.

## Validation And Registration

147 tests passed locally and in the isolated Linux deployment, including actual
Tiny two-backward integration, weighted accumulation, projection geometry,
overflow visibility, CLI/protocol compatibility and empty-GPU scheduling. The
two baseline checkpoint hashes and preflight passed. Screen
`round118_primary_projection` is registered with `gpu_candidates=[1]` and
`wait_for_empty_gpu=true`; initial state is `waiting_for_smoke`.
Receipt: `round118_queue_receipt.json`. The live state is `round118_queue.json`;
GPU smoke, formal training and OOF are pending, not claimed successful.

# Round121: Paired CD68 Relations in the DAPI Encoder

Question: does training the DAPI encoder to preserve target-image relationships
improve held-out CD68 reconstruction? Round109 found predictable pooled CD68
features across held-out ROIs (standardized R2 0.654895). That supports a
learnable target, not an expected image-score gain. Round110 constrained the
generated image through a frozen perceptual encoder and failed; Round96 used
independent reconstruction of unpaired marker images. Neither directly guides
the DAPI encoder using paired target relationships. Historical reports/configs
were searched for relational distillation, paired feature alignment and
privileged information; no matching experiment was found.

The distance-wise objective follows [Relational Knowledge Distillation,
CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html).
Its published benchmarks do not establish efficacy for DAPI-to-CD68 translation.

Fixed candidate: Round55 auxiliary15 architecture and fold-local Round52
FCMAE800 initialization, fresh supervised fold0/fold2, seed2044, batch16,
14+14 epochs. No gradient projection. A frozen, fold-matched Round96 encoder
reads only paired training CD68. For each batch, spatially pool stage4 target
features from the center 256 crop; pool the corresponding center 16x16 raw
DAPI bottleneck before FiLM. Match all unordered pairwise Euclidean distances,
each modality divided by its own mean pair distance, with smooth-L1 loss and
fixed weight 0.05. Pooling and distance normalization avoid forcing two modality
encoders into identical channel coordinates or matching their large common
feature mean. They do not eliminate modality-exclusive information. The DAPI
features retain their normal input halo; teacher features use the diagnostic's
central crop, so receptive fields are not identical. This is image-level
relational guidance, not a claim of exact spatial feature correspondence.

The same training forward supplies image predictions and encoder features.
There is no projection head, teacher backward, new inference input or new
checkpoint architecture. Degenerate batches below three images contribute zero
relational loss. Training retains full batch16 (no microbatch approximation of
batch relations). Teacher hashes and validation exclusions are enforced before
each stage. The teacher is absent from inference/model state.

Comparison: fixed Round55 auxiliary15, context128+D4/q95, global-MSE selector,
output shift [0,-1]. One direct candidate only. Automatic five-fold expansion
requires q95 >= +0.05, positive float and both folds, >=7/10 positive ROIs,
and worst ROI >= -0.12. Report borderline outcomes separately; a missed gate
does not authorize a weight/layer/batch sweep or a new leaderboard probe.

Cost: recent ordinary Tiny training measured roughly 90-93 minutes/fold;
Round120's extra-backward recipe cost 2.30 hours on its uncontended fold2.
This recipe adds only a no-grad teacher forward, so use two folds first to
avoid roughly 4.5-7 hours for three additional folds if negative. Measure
real two-stage smoke time/memory before launch. GPU1 only; GPU0 remains reserved.
Pipeline completion includes the fixed 2,706-image OOF and gate, and then stops.


## Validation and launch (2026-09-10)

132 targeted tests pass on both Windows and the isolated Linux deployment.
The Git archive and every extracted file were verified before preflight.
Two real smoke stages each completed four full-batch optimizer updates with
finite selected weights; feature losses were 0.035181 and 0.065409, and peaks
were 26.583 / 26.701 GiB. Stage1 loads the intended fold-local FCMAE encoder;
stage2 restores strictly. Teacher hashes and fold exclusions pass both stages.

The formal two-fold job is detached in Screen `round121_encoder_relational`,
using GPU1 only and waiting for an empty device at phase boundaries. Source
commit is `fb31dd42cac09799d0e001d3476d77500b140f64`.
Runtime and reports are under `transfers/round121_runtime`; the run directory
links to the canonical project runs. Launch receipt and exit marker will be
`reports/round121_launch.json` and `reports/round121_pipeline.exit` there.
The 36,000-MiB launch threshold exceeds measured smoke peak by over 4 GiB.
Estimated two-fold training plus OOF is 4-6 hours without new contention, based
on prior full-fold costs; the short smoke is not a steady-throughput benchmark.
No formal result exists at launch.


## Completion (2026-09-10)

Complete-fail under the preregistered gate: exit `0`; both folds and both
stages have complete trajectories and finite checkpoints; OOF covers 2,706
images / 10 ROIs; checkpoint hashes and local gate replay match.

| comparison with Round55 auxiliary15 | q95 delta |
| --- | ---: |
| overall | +0.093161 |
| fold0 | +0.025023 |
| fold2 | +0.157007 |
| worst ROI (ROI007) | -0.335121 |

Candidate q95 is 75.106434 versus 75.013272, and float delta is +0.093126.
It improves 7/10 ROIs and both folds, but fails the fixed worst-ROI floor by a
large margin. This is a real borderline-positive screen, not a clear negative;
no weight, batch, teacher or ROI-specific search is authorized. The stable
Round88 submission remains the deployment artifact.

The result is materially better than Round110 perceptual guidance and suggests
that training-only target relations can help the DAPI representation. The
fold-level asymmetry and ROI007 loss mean it is not safe to expand automatically
to five folds or replace the retained package yet. The formal result should be
used as evidence for the next design decision, not as a submission candidate.

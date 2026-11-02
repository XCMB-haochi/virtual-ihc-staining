# Round 76: Direct Cytoland Virtual-Staining Transfer

## Motivation

Round75 rejects a Cytoland encoder followed by 800 epochs of fold-local DAPI
FCMAE: it loses `-0.250145` q95 to the official ImageNet-FCMAE800 model, and
even a fixed half-split loses `-0.013767` on both folds. That experiment uses
only the source encoder. Its reconstruction phase discards Cytoland's trained
virtual-staining decoder and may overwrite task-specific translation features.

Round76 tests the distinct direct-transfer hypothesis. It starts supervised
CD68 training directly from the public Cytoland checkpoint, without DAPI
FCMAE. An encoder-only recipe is the matched control. The candidate also loads
the two decoder stages whose dimensions exactly match the local one-output
UNeXt2. The incompatible final two-output source stage, local thumbnail FiLM,
and CD68 output bias remain at the same deterministic local initialization.

## Audited Mapping

The public checkpoint and SHA-256 remain fixed at
`1abdbc1c727aba33dad0d174dba57d128a12019b712db946ce06012ecd64c1fe`.
Both recipes load 178 encoder tensors and 27,854,112 encoder parameters. The
full-transfer candidate additionally loads 48 decoder tensors and 3,302,784
parameters from decoder stages 0 and 1. Eight 1x1 convolution tensors are
reshaped losslessly to the local channel-last linear representation.

No source layer subset is selected after observing validation. The source
decoder stage 2 is excluded before training solely because its two-output
channel dimensions are incompatible with the one-output CD68 model. Tests bind
the exact source key set, shapes, dtypes, hash, loaded counts, and preservation
of stage 2, conditioning, and output bias.

## Fixed Screen

`configs/round76_cv2_unext2_cytoland_direct.json` preserves Round52 fold0/fold2,
seed 2044, `14+14` supervised epochs, effective batch 16, halo64 supervision,
spatial-thumbnail FiLM, Uniform-7 hybrid loss, global-MSE checkpoint selection,
D4, and JPEG q95. It trains two fresh recipes:

- `cytolandencoder`: public encoder, local decoder;
- `cytolandtransfer`: the same encoder plus compatible decoder stages 0 and 1.

Historical scratch UNeXt2, official-FCMAE800, and the current Base-diversity
endpoint are reused with immutable checkpoint provenance. The decoder effect
requires `+0.05` q95 over encoder-only, both folds, 7/10 ROIs, positive float,
and a `-0.12` ROI floor. The full transfer must then gain `+0.05` over official
FCMAE800 under the same distribution checks.

Two deployment routes are fixed: replace the current `0.175` official-Tiny
share with direct transfer, or split that share equally. Each requires `+0.04`
over the Base endpoint, both folds, 8/10 ROIs, positive float, and a `-0.08`
floor. Replacement depends on the decoder and prior gates; the fixed diversity
split remains independent. No epoch, source scope, checkpoint, loss, blend
weight, fold, or ROI route will be searched after the result.

A passing route must expand unchanged to all five folds and pass complete OOF
before test inference. Round76 cannot generate a submission from two folds.

## Status

Implementation and prelaunch validation are complete. The focused local suite
passes 144 tests. On Linux, all 301 tests and 77 subtests pass, together with
Python compilation, shell syntax, exact eight-command dry run, diff checks,
and the hash-bound no-side-effect preflight. Loading the real checkpoint on
CPU independently reproduces the fixed 178 encoder tensors, 48 compatible
decoder tensors, 3,302,784 decoder parameters, and eight lossless 1x1 weight
reshapes.

Round76 launched from commit `143debb` in detached Screen `round76` at
2026-08-27 14:57:31 +08:00. The first resource sample found GPU0 at 50%
external utilization despite 35,054 MB free, while GPU1 was fully occupied
with only 12,713 MB free. Neither card met the fixed training gate of at most
30% utilization and at least 27,000 MB free, so no smoke or project GPU process
started. The scheduler remains in its five-minute resource loop. Once eligible,
it will run both removable real-data smokes, then dynamically schedule the four
fold/recipe jobs and final OOF without changing the protocol.

The original waiting Screen later disappeared without a nonempty exit record
and before creating any smoke or formal run. At 16:27, ten two-second samples
showed GPU1 at 0% compute and memory utilization with only 3,176 MB resident
and 37,265 MB free; five per-process samples also showed no SM or memory
activity for its three unrelated resident processes. GPU0 remained actively
contended near 100%. Round76 therefore restarted at commit `3876aa1` in Screen
`round76` at 16:37:00 and selected only GPU1.

Both one-epoch real-data smokes completed there. The first post-smoke audit
failed because the launcher read `config.initialization` instead of the actual
`config.init_checkpoint_report`; both smoke directories nevertheless contained
complete metrics and loadable selected checkpoints. The corrected audit
independently verified the exact hash, scopes, 178 encoder tensors, 0/48
decoder tensors, and 0/3,302,784 decoder parameters before those temporary
directories were removed. The fixed launcher reran and passed both smokes,
then launched formal `fold0:cytolandencoder` on GPU1 at 16:39:19. A later
sample measured 97% utilization, 30,714 MB resident, and about 237 W. GPU0
remained excluded while its unrelated tasks were active.

The scheduler completed both encoder-only folds on GPU1. It attempted fold0
full transfer on GPU0 after a three-sample window reached 0% utilization and
27,258 MB free, but unrelated processes resumed after launch. The candidate
peaked near 25.4 GiB by itself and failed before epoch one when the device had
only 394 MB free. The failed directory was moved under
`/workspace/tmp-meimy/round76-failed-gpu0`; no checkpoint or metric row
from it entered evaluation. A recovery worker waited for fold2 encoder stage 2
to have exactly 14 rows and no live trainer, then ran both full-transfer folds
serially on GPU1. Its exit code is zero and all eight formal stages contain
exactly 14 contiguous metric rows with selected checkpoints.

The unchanged OOF evaluator then ran on idle GPU1, exited successfully, covered
all 2,706 images and ten fixed ROIs, and took 3,621.33 seconds. All 16 referenced
checkpoint hashes independently match the files. The OOF SHA-256 is
`1f7223566a654f730212fa23ac1436e20888cb9cf8f4fc9720c449af8e580312`;
the gate SHA-256 is
`d34690c569d4e6b5a350ea228f37c9034aebbb1373a864deacba27d3ed56ac13`.
Independent local gate evaluation reproduces every metric and decision.

## Result

Direct Cytoland encoder initialization is a real representation gain over the
matched scratch UNeXt2: `74.708110` versus `73.455490`, or `+1.252620` q95.
Both folds and all 10 ROIs improve; the worst ROI still gains `+0.606600`.
This establishes a strong microscopy-domain prior without DAPI FCMAE, although
it remains `-0.139740` below official ImageNet-FCMAE800.

Loading compatible Cytoland decoder stages is harmful. Full transfer scores
`74.621115`, a `-0.086995` q95 change versus encoder-only. Fold0 gains
`+0.101697`, but fold2 loses `-0.263800`; only 5/10 ROIs improve and ROI008
loses `-0.701247`. The decoder-effect gate fails decisively.

Consequently, full transfer is `-0.226735` below official FCMAE800. Replacing
the official-Tiny share in the Base endpoint loses `-0.038930`. Its fixed
half-split loses `-0.008080`, is negative on both folds, and improves only
2/10 ROIs. Every full-transfer deployment gate fails and
`promotion.allowed=false`. No decoder-layer subset, epoch, blend-weight,
five-fold expansion, test inference, or submission follows.

The positive encoder-only mechanism was not included in Round76's deployment
candidates because the round was preregistered around the decoder-effect
hypothesis. One separately named fixed 50/50 split of the official-Tiny share
with this independently passing encoder endpoint is the only justified bounded
composition check; it is not permission for a weight grid.

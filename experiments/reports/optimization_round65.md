# Round 65: Shared All-Train-DAPI FCMAE

## Motivation

Round64 closes further calibration of the existing auxiliary endpoints: the
best fixed route gains only about `+0.04` complete-OOF q95 and fails its
distribution gate. Round63 Base pretraining has stronger diversity evidence
but requires roughly two more days and changes capacity. Round65 instead tests
a data-efficiency change inside the successful Tiny FCMAE mechanism.

Existing fold-local FCMAE encoders see only the 20 supervised training ROIs of
their fold, even though masked DAPI reconstruction uses no IHC label. Round65
pretrains one shared encoder on DAPI from all 25 official training ROIs. The
supervised fold models still exclude every held-out ROI's IHC targets exactly
as before. No test DAPI, hidden label, leaderboard feedback, or ROI-specific
route is used.

## Update-Matched Protocol

The current fold0/fold2 official FCMAE800 controls process 4,987 and 4,899 DAPI
images per epoch. At batch 32 they make 124,800 and 123,200 optimizer updates
over 800 epochs. The new shared encoder processes all 6,296 DAPI images, or 197
updates per epoch, for 630 epochs: 124,110 updates. This is within 0.75% of
both controls. Eight warmup epochs give 1,576 warmup updates, within 2.4% of
the controls' 1,560 and 1,540.

Everything else is fixed to Round52: audited official ConvNeXt-V2 Tiny FCMAE
initialization, seed 2044, effective batch 32, mask ratio 0.5, AdamW, cosine
decay, and the same reconstruction objective. Both supervised folds initialize
from the exact same final encoder, then independently train the unchanged
14+14 UNeXt2 recipe with label-isolated folds, halo64 thumbnail FiLM,
Uniform-7 hybrid loss, global-MSE checkpoint selection, D4, and JPEG q95.

## Gates

The direct representation must gain at least `+0.05` q95 over matched
fold-local official FCMAE800, remain positive in float and both folds, improve
7/10 ROIs, and keep the worst ROI above `-0.12`. Replacement in `t70_o50`
requires that mechanism plus `+0.04`, both folds, 8/10 ROIs, and a `-0.08`
floor. An independent half-split diversity route has the same deployment
thresholds without depending on direct superiority.

No epoch, DAPI subset, checkpoint, ensemble weight, fold, or ROI route will be
changed after seeing the screen. A passing recipe must expand unchanged to all
five supervised folds and pass complete OOF before any test inference.

## Interpretation Limit

The shared self-supervised encoder sees DAPI inputs from the held-out supervised
ROIs, but never their IHC targets. This makes the screen semi-transductive at the
representation stage: it tests whether a fixed encoder trained on all available
official training DAPI is useful, but its OOF gain may overestimate performance
on unseen test ROIs. The result therefore cannot be promoted from aggregate q95
alone. It must satisfy the preregistered fold/ROI distribution gates, expand
unchanged to complete OOF, and remain subject to a reproducible platform check.

## Status

The 71 focused local unit tests, dry runs, Python compilation, and launcher
shell syntax checks pass. Full Windows discovery remains blocked by the known
duplicate OpenMP runtime error; the unsafe duplicate-runtime override was not
used. On Linux, all 282 tests pass. A one-epoch, 32-image smoke test selected
all 25 training ROIs with no excluded validation ROI, loaded 99.998% of the
official encoder parameters, completed one optimizer step in 14.75 seconds,
and wrote both resumable and final checkpoints. Its peak allocated GPU memory
was 22.04 GB. A multiprocessing temporary-directory cleanup warning occurred
after completion on the shared filesystem, but the process returned success
and all required artifacts were verified before the smoke directory was
removed.

Round65 launched at 2026-08-21 20:49:28 +08:00 in detached Screen `round65`
from commit `2f251ca`. Five launch-time samples found physical GPU0 at 0%
utilization but with 19.8 GB resident memory and GPU1 at 0% with 17 MB resident,
so shared pretraining was assigned to GPU1. Initial verification reached epoch
2/630 at 70.51 seconds per epoch, 98% GPU1 utilization, and 22.28 GB peak
allocated memory. GPU0 had meanwhile risen to 97% utilization under another
task and was not used. The recurring multiprocessing temporary-directory
cleanup warning is non-fatal: metrics continue and the same warning appeared
after the successful smoke run.

Shared pretraining completed at epoch 630 with 630 metric rows and final loss
`0.00129252`. Its config records `data_scope=all-train`, 6,296 images, all 25
ROIs, no excluded validation ROI, and the audited official initialization hash.
The final shared encoder SHA-256 is
`f4a8c93c616310ce4a981f9b1fe6830ba82bf78ff91bdbe52d471cf1c0164554`.
At the supervised boundary both GPUs were eligible, so fold0 and fold2 ran in
parallel. All four supervised stages contain 14 metric rows and both stage-1
runs record the same epoch-630 encoder hash.

The pipeline exited `0`, Screen closed, and the OOF evaluated all 2,706 images
from the fixed 10 ROIs in 1,785.49 seconds. All 10 referenced checkpoint hashes
match the files on the server. The historical official-FCMAE model and its four
source checkpoint records exactly replay Round52; `t70_o50` exactly replays the
matched Round62 baseline. The OOF SHA-256 is
`56a91690be0438569499f903ff0797ad233131c1af28b5c78c3e063477453316`,
and an independent local gate evaluation exactly reproduces the saved result.

## Result

At JPEG q95, the direct all-DAPI model scores `74.888495` versus `74.847851`
for fold-local official FCMAE, a gain of `+0.040644`. It fails the direct gate:
fold0 is `-0.002139`, only 6/10 ROIs improve, and the worst ROI is `-0.252095`.
Full replacement scores `75.219172` versus `75.180556` (`+0.038616`) and also
fails its threshold and distribution requirements.

The preregistered diversity candidate `t70_all_dapi_split` scores `75.223522`,
an OOF gain of `+0.042966` over `t70_o50`. It is positive in float, both folds
(`+0.021224`, `+0.063338`), and 9/10 ROIs; the worst ROI is only `-0.011572`.
Every diversity-gate check passes, so `promotion.allowed=true` for unchanged
five-fold expansion. This supports complementarity between fold-local and
all-train-DAPI encoders, not direct superiority of the all-DAPI model. No test
inference or submission is authorized until the complete 25-ROI OOF passes.

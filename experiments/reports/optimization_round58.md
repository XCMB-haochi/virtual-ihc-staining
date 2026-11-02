# Round 58: Fresh 1200-Epoch Official FCMAE

## Motivation

Round56 and Round57 reject both asymmetric schedules for auxiliary-marker
supervision. Earlier architecture, context, augmentation, and postprocessing
branches have also failed their controlled gates. The only mechanism that has
produced a large, stable five-fold gain is fold-local DAPI FCMAE: Round46 gained
`+0.509991` q95 by extending scratch pretraining from 100 to 800 epochs, and
Round52 gained another `+0.205827` in the fixed ensemble by adding an
official-initialized FCMAE800 member.

The late FCMAE reconstruction loss is not flat. Across all five official-init
folds, the mean loss in epochs 751--800 is 9.1--9.4% below epochs 351--400,
and the minimum occurs between epochs 745 and 797. Reconstruction loss is not
the downstream metric, but this is enough evidence for one bounded duration
test before changing architecture.

Round58 retains ConvNeXt-V2 Tiny. The current VisCy 2D Cytoland recipe uses the
same `[3, 3, 9, 3]` depths and `[96, 192, 384, 768]` dimensions, while a Base
encoder would change capacity, memory, and optimization together. Previous
capacity increases in this dataset were negative. A larger backbone therefore
has weaker controlled evidence than extending the successful Tiny pretraining
mechanism.

## Fixed Screen

`configs/round58_cv2_unext2_official_fcmae1200.json` changes the official-init
fold-local DAPI FCMAE horizon from 800 to 1200 epochs. It starts afresh from the
same audited official ConvNeXt-V2 Tiny ImageNet-1K FCMAE checkpoint rather than
resuming the Round52 endpoint. This gives the 1200-epoch run its own complete
warmup/cosine schedule and avoids inheriting an 800-epoch terminal learning
rate trajectory.

Everything downstream remains fixed: fold0/fold2, seed 2044, batch 32 and four
workers for FCMAE, mask ratio 0.5, the same 14+14 supervised UNeXt2 training,
halo64 spatial-thumbnail FiLM, Uniform-7 hybrid loss, global-MSE checkpoint
selection, D4 inference, and JPEG q95 scoring. Historical members are reused
by their recorded run sources.

The preregistered candidates are:

- direct `officialfcmae1200` versus `officialfcmae800`;
- replacement of the 0.35 official member in retained `t70_o50`;
- a fixed diversity split that assigns 0.175 each to official FCMAE800 and
  official FCMAE1200 while leaving every other weight unchanged.

Direct improvement must reach `+0.05` q95, remain positive in float score and
both folds, improve at least 7/10 ROIs, and keep the worst ROI above `-0.12`.
Replacement requires that direct gate plus `+0.04` ensemble q95, both folds,
8/10 ROIs, and a `-0.08` floor. The diversity route is independent of the
direct gate, as correlated models can improve a fixed ensemble without one
being uniformly superior; it must still gain `+0.04`, improve both folds and
8/10 ROIs, and satisfy the same `-0.08` floor.

No epoch endpoint, checkpoint, ensemble weight, fold, or ROI route will be
selected after observing this screen. A passing candidate must be expanded
unchanged to all five folds and pass a complete-OOF gate before test inference.

## Status

Local validation passes: the JSON protocol parses, 59 focused CV protocol
tests pass, relevant Python files compile, the shell runner passes `bash -n`,
and exact dry-runs expand to two fresh 1200-epoch pretraining jobs followed by
only the four expected 14-epoch supervised stages. Neither pretraining command
contains `--resume`, and both point to the audited official initialization.

Full local test discovery remains blocked by duplicate Intel OpenMP runtimes in
the Windows Anaconda environment. The unsafe `KMP_DUPLICATE_LIB_OK` override is
not used.

The clean server at commit `6d87f56` passes all 256 tests, shell syntax, Python
compilation, JSON loading, and the exact two-pretraining/four-supervised-job
dry-run. The public initialization artifact again matches its registered
SHA-256, `b25eadb89ad634b5c925ea04cdacebafe3e462486baf9482dc2d832bb9af57ef`.

A physical-GPU1 real-data smoke run loaded that artifact at `99.998277%`
parameter coverage and completed two optimizer steps with 32 fold0 training
images, batch 32, four workers, and AMP. Its losses were `0.021163680` and
`0.019794930`; peak allocated memory was `22,280,386,560` bytes
(`20.750 GiB`). The final encoder SHA-256 was
`e698ed271e5ce663db4fa680aab2bfd953a6efe8479ca3e1c467177533448965`.
After the completion record, metric count, checkpoints, and hash were verified,
the temporary smoke directory was removed.

Immediately before formal launch, five resource samples found physical GPU1 at
0% utilization with 40.4 GiB free. GPU0 had one pre-existing 4.1-GiB compute
process and bursty utilization of `47, 52, 0, 0, 0%`, with 36.3 GiB free. The
measured 20.75-GiB peak leaves a sufficient memory margin on either card. To
reduce the projected straggler, the historically slower fold0 is assigned to
idle GPU1 and the slightly faster fold2 to intermittently loaded GPU0; final
OOF evaluation is assigned to GPU1. Based on the Round52 fold runtimes, the
expected wall time is roughly 21--26 hours if GPU0 contention remains
intermittent.

## Execution and Recovery Audit

The formal run launched at audited commit `6d87f56`, initially assigning fold0
to physical GPU1 and fold2 to physical GPU0. Both jobs were later stopped with
`SIGTERM` when the GPUs were temporarily needed elsewhere. Fold0 had written
335 metric rows but its latest durable checkpoint was epoch 325; fold2 had
written 610 rows with a durable epoch-600 checkpoint. The original scheduler
exited and released both devices before the resume was launched.

The resume used the same fixed protocol and restored model, optimizer, scaler,
and RNG state from those durable checkpoints. The runner discarded the stale
metric tails and replayed them. The completed metric files now contain exactly
epochs 1--1200, in order and without gaps or duplicates, for both folds. The
completion records are true at epoch 1200, and all four downstream supervised
stages contain exactly 14 metric rows. This establishes a completed resume,
not merely an exited Screen session.

Both folds retained the preregistered ImageNet-1K official FCMAE initialization
with checkpoint SHA-256
`b25eadb89ad634b5c925ea04cdacebafe3e462486baf9482dc2d832bb9af57ef`
and `99.998277%` encoder-parameter coverage. The final fold-local encoder hashes
are:

- fold0: `4b61e4125074daf7ea10a5d247e1afad883a8bfc6666f30921eb6c389e19fe84`;
- fold2: `012bd082ff837948326d94463e224023b7ed363d96be376a2726f6b567e156c3`.

The selected downstream `best_global_mse.pt` checkpoints are fold0 epoch 10,
SHA-256 `b975e91ff1b84184c8fbcf736db5eff871eaee0b365c18a5d202c12d30a4030c`,
and fold2 epoch 5, SHA-256
`b2a842b89cf332c7b1b98d5917c26c57cf45270bfc9c53d1e68164da5d24b885`.
Independent server-side hashing matched every one of the ten historical and
new checkpoints referenced by the OOF provenance.

## Formal Results

The complete two-fold OOF evaluation covered 2,706 validation images from the
ten fixed ROIs and took 1,787.06 seconds. Scores below use the official
per-image metric and JPEG q95 path.

| Candidate | q95 score | Float score | q95 delta |
| --- | ---: | ---: | ---: |
| `officialfcmae800` | 74.847851 | 74.921061 | baseline |
| `officialfcmae1200` | 74.770093 | 74.840240 | -0.077758 |
| `t70_o50` | 75.180556 | 75.255122 | baseline |
| `t70_official1200_replace` | 75.203745 | 75.276897 | +0.023189 |
| `t70_official1200_split` | 75.214865 | 75.288431 | +0.034309 |

All three preregistered gates fail:

- Direct 1200 versus 800 loses `0.077758` q95 overall. Its fold deltas are
  `+0.016822` and `-0.166379`, only 5/10 ROIs improve, and the worst ROI delta
  is `-0.778242`.
- Replacing the 800 member gains only `0.023189`; its fold deltas are
  `+0.048034` and `-0.000092`, only 6/10 ROIs improve, and the worst ROI delta
  is `-0.215810`. It also depends on the failed direct gate.
- Splitting the official-member weight between 800 and 1200 is positive in
  both folds (`+0.047239`, `+0.022193`) and gains `0.034309` overall, but misses
  the fixed `+0.04` threshold, improves only 7/10 ROIs, and has a worst ROI
  delta of `-0.086131` against the `-0.08` floor.

The gate artifact therefore records `promotion.allowed=false`. No five-fold
expansion, test inference, or submission archive is authorized from Round58.

The reconstruction objective itself continued to improve: mean loss in epochs
1151--1200 was `0.001179447` for fold0 and `0.001177528` for fold2, about 7%
below epochs 751--800. The minima occurred at epochs 1176 and 1184. The lower
reconstruction loss did not transfer to robust virtual-staining performance,
so the result rejects additional Tiny official-FCMAE duration as the next
scaling direction. The retained deployed recipe remains Round53 `t70_o50`.

## Completion Artifacts

The detached pipeline exited with code 0. The copied formal artifacts match
their server hashes:

- OOF report: `096cbb89280d334d35fa6f91775c99167df6fe3ef91240e9959e0fc723654289`;
- gate report: `d381f9f7ab9b5d97b9d8bc7fbede82686118c353aa18b276a4ab609a7e66ab1e`.

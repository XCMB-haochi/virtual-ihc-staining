# Round 52: Official FCMAE Initialization

## Motivation

Round51 establishes a small cross-marker benefit but does not pass a deployment
gate. The strongest portable mechanism in the retained system remains
fold-local DAPI FCMAE: extending pretraining from 100 to 800 epochs produced a
large, stable gain in Round46. Round52 tests whether the same microscopy
pretraining benefits from a stronger initialization before seeing any
competition image.

This is not a repeat of Round15. That experiment used an ImageNet-1K
classification ConvNeXt-Tiny inside a different generic encoder-decoder.
Round52 uses the official self-supervised ConvNeXt-V2 Tiny FCMAE checkpoint,
whose block depths and channel dimensions exactly match the current
`FCMAEEncoder`, and then runs the unchanged fold-local DAPI FCMAE800 and
UNeXt2 recipe.

## Audited Public Artifact

The fixed source is Meta's official ImageNet-1K FCMAE checkpoint:

- URL: `https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt`;
- license reported by the official repository: `CC-BY-NC-4.0`;
- SHA-256: `b25eadb89ad634b5c925ea04cdacebafe3e462486baf9482dc2d832bb9af57ef`;
- file size: `111,520,513` bytes.

The official RGB stride-4 input convolution cannot be copied into the local
single-channel stride-2 stem without a heuristic spatial conversion. Its
weight and bias therefore remain at the existing seeded random initialization.
All other tensors map structurally: the three downsampling layers and all 18
ConvNeXt-V2 blocks. GRN gamma/beta tensors only drop the official leading
singleton dimension. The audited transfer covers 194/196 target tensors and
`27,860,256 / 27,860,736` encoder parameters (`99.998277%`). Loading is strict
on source keys, shapes, dtypes, mode, and checkpoint hash.

The rules explicitly permit public pretrained weights. Round52 does not use
official test inputs, hidden labels, leaderboard feedback, or any external
microscopy data.

## Fixed Protocol

The machine-readable protocol is
`configs/round52_cv2_unext2_official_fcmae.json`. The candidate changes only
the initial encoder state. It retains fold0/fold2, seed 2044, 800 DAPI FCMAE
epochs, batch 32, four workers, the same warmup/cosine schedule and masking,
then the same 14+14 supervised epochs, halo64 spatial-thumbnail FiLM,
Uniform-7 hybrid loss, global-MSE checkpoint selection, D4 inference, and JPEG
q95 scoring.

The scratch FCMAE800 and Round35 members reuse their existing checkpoints.
Three comparisons are fixed before training:

- direct official-initialized FCMAE800 versus scratch FCMAE800;
- a 60% replacement in the retained Round47 ensemble;
- a 30% scratch / 30% official-initialized diversity split.

The direct mechanism requires `+0.05` q95, positive float gain, 2/2 folds,
7/10 positive ROIs, and a worst ROI above `-0.12`. Deployment requires either
the 60% replacement to gain `+0.08` or the fixed diversity split to gain
`+0.06`; both require positive float gain, 2/2 folds, 8/10 positive ROIs, and
a worst ROI above `-0.08`.

No stem conversion, public checkpoint choice, downstream schedule, seed,
checkpoint, ensemble weight, or ROI-specific route will be searched after the
two-fold result. A passing deployment candidate must be expanded unchanged to
all five folds and pass a complete-OOF gate before test inference.

## Execution Status

The implementation passes Python compilation, JSON parsing, Git Bash pipeline
syntax, `git diff --check`, the exact six-command protocol dry-run, and 63
focused protocol/FCMAE tests locally. The Windows Anaconda environment cannot
run the entire test discovery process because two installed libraries load
duplicate Intel OpenMP runtimes; no unsafe duplicate-runtime override is used.
The clean server environment passes the full 246-test suite.

The official artifact was transferred to the server and revalidated at
`111,520,513` bytes with the exact preregistered SHA-256. A physical-GPU1
smoke run used 32 real fold0 training images, batch 32, four workers, AMP, and
one optimizer step. It took `12.433 s`, peaked at `22,041,952,256` allocated
bytes (`20.528 GiB`), exited zero, and wrote a loadable 196-tensor final
encoder plus the full 194-tensor initialization report. A same-configuration
resume initially exposed an existing PyTorch 2.8 incompatibility: checkpoint
loading had moved saved CUDA RNG states onto CUDA although
`set_rng_state_all` requires CPU byte tensors. Commit `0c91a97` adds strict
validation and CPU restoration; the identical resume then exited zero and
rewrote a valid final artifact. The verified smoke directory was removed.

The two formal pretraining runs completed all 800 epochs and exited zero.
Fold0 took `46,016.921 s` and reached a last-50 mean loss of `0.001262100`,
versus `0.001367265` for the matched scratch run. Fold2 took `44,439.942 s`
and reached `0.001256987`, versus `0.001362739`. The final encoder SHA-256
values are respectively
`50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9`
and
`c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7`.
Both downstream stages contain 14 consecutive metric rows and 14 epoch
checkpoints. The selected stage-2 global-MSE scores improve over scratch by
`+0.186558` on fold0 and `+0.106101` on fold2.

## Two-Fold Result

The complete fixed OOF evaluation covers all 2,706 images from the 10 held-out
ROIs and took `1,579.743 s`. It uses halo64, D4, and JPEG q95 exactly as
registered. All eight referenced checkpoint files were independently hashed
after evaluation and match the provenance embedded in the report. The OOF
SHA-256 is
`a10a20a521a2eab8a4462d79ec346e620187531b9e5eee33bfbb677cbf7645e9`.

The official-initialized model alone gains only `+0.025058` q95 over scratch.
Fold0 improves `+0.157293`, but fold2 loses `-0.098847`; its worst ROI is
ROI000 at `-0.470987`. It therefore fails the direct-model gate. Replacing the
entire 60% FCMAE member gains `+0.114470` overall and is positive on both
folds and 8/10 ROIs, but ROI000 loses `-0.253656`, so the fixed ROI floor
correctly rejects that deployment route.

The preregistered diversity split is stable. Keeping 30% scratch FCMAE800 and
adding 30% official-initialized FCMAE800 raises q95 from `74.960051` to
`75.138136`, a gain of `+0.178084`; float gain is `+0.176595`. Fold gains are
`+0.232699` and `+0.126910`. Nine of 10 ROIs improve, and the only negative
ROI is ROI000 at `-0.007323`, comfortably above the `-0.08` floor. Every
fixed diversity check passes, so `promotion.allowed=true` uniquely selects
`round47_official_split30`. No alternative weight, checkpoint, calibration,
or ROI route was searched.

## Five-Fold Promotion

`configs/round52_cv5_unext2_official_fcmae_promotion.json` preserves the same
protocol namespace so fold0/fold2 are reused and trains only fold1/fold3/fold4.
The candidate weights remain base32 `0.28`, base48 `0.12`, scratch FCMAE800
`0.30`, and official-initialized FCMAE800 `0.30`. The complete-OOF gate keeps
the `+0.06` q95 and `-0.08` ROI-floor thresholds, scales the distribution
requirements to 4/5 positive folds and 20/25 positive ROIs, and requires a
positive float gain. Test inference remains forbidden until this unchanged
five-fold gate passes.

All three remaining fold-local runs completed successfully. Fold1 and fold3
selected stage-2 epoch 11. Fold4 selected stage-1 epoch 13 and stage-2 epoch
6; its official-initialized stage-2 selection score is `74.887172`, versus
`74.408667` for the matched scratch FCMAE800 checkpoint (`+0.478505`). Fold4
pretraining contains exactly 800 metric rows and records the audited public
checkpoint hash, `99.998277%` parameter coverage, runtime commit `fe726ee`,
and final-encoder SHA-256
`8c71c765cb217496bcc8a693030c5910106a39b898f4f28573f10738c03d1d59`.

The final OOF evaluation covers all 6,296 images from all 25 ROIs. It took
`3,324.087 s` and uses the fixed halo64, D4, JPEG-q95, per-image official
metric. All 20 distinct checkpoint files were independently rehashed and
match the nested provenance in the report. The OOF report SHA-256 is
`2c7ca6f5054845f2b9ddb2cde2c43229daabf5769a8716a75ce4c64fd14734fb`;
the gate report SHA-256 is
`ef8ae0eba874932d25c7b0b843d52c11fd13621536978ab4b6dbf4ae8524390e`,
and the gate binds the exact OOF digest.

The retained Round47 ensemble scores `75.324015` at JPEG q95. The fixed
scratch/official split scores `75.529842`, a gain of `+0.205827`; float gain
is `+0.205135`. Every fold improves: fold0 `+0.232699`, fold1 `+0.266629`,
fold2 `+0.126910`, fold3 `+0.200667`, and fold4 `+0.215348`. Twenty-three of
25 ROIs improve. The only negative ROIs are ROI000 at `-0.007323` and ROI023
at `-0.064396`, so the worst ROI remains above the fixed `-0.08` floor. Every
distribution check passes and promotion selects
`round47_official_split30`.

This is a robust incremental improvement rather than a new performance
regime. Before test inference replaces the retained submission, Round53 fixes
a bounded 3-by-3 OOF surface: total FCMAE weight in `50/60/70%` crossed with
official-initialized share in `1/3, 1/2, 2/3`. It reuses the exact 20 approved
checkpoints and performs no training. The current `60%` total, equal-split
candidate is the baseline; promotion requires at least `+0.02` q95, positive
float gain, 4/5 positive folds, 20/25 positive ROIs, and a worst ROI delta no
lower than `-0.04`.

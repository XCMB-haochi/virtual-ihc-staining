# Round 75: Cytoland Microscopy-Prior FCMAE

## Motivation

Round74 rejects the official ConvNeXt-V2 Tiny DropPath recipe. Round71 also
shows that only `23.9%` of the Base-diversity complete-OOF gain transfers to
the preliminary hidden set. Another ImageNet-scale regularizer or ensemble
weight probe is therefore less compelling than a representation prior that
matches the image domain.

Biohub publishes the BSD-3-Clause
[VSCyto2D checkpoint](https://huggingface.co/biohub/VSCyto2D) used by
[Cytoland](https://github.com/mehta-lab/VisCy). It is a single-channel UNeXt2
Tiny trained for microscopy virtual staining from label-free images to nucleus
and membrane fluorescence. Its encoder has the same `2x2` single-channel stem,
depths `[3,3,9,3]`, dimensions `[96,192,384,768]`, and ConvNeXt-V2 blocks as
the local Tiny encoder. This is materially closer to DAPI-to-IHC than the
current natural-image prior while remaining public, fixed, and unrelated to
the competition test set.

## Audited Initialization

The source file is `epoch=399-step=23200.ckpt`, SHA-256
`1abdbc1c727aba33dad0d174dba57d128a12019b712db946ce06012ecd64c1fe`.
The strict loader accepts only that hash and the exact published encoder
schema. It maps 178 tensors and 27,854,112 of 27,860,736 local encoder
parameters (`99.9762%`). Unlike ImageNet FCMAE, the single-channel input stem
is transferred exactly.

The source also contains a 3D stem, which is not part of the 2D model and is
explicitly skipped. Biohub's depthwise convolutions omit bias; the 18
corresponding local biases remain at their deterministic zero initialization.
The source decoder and nucleus/membrane output head are not loaded. Round75
then performs a fresh fold-local 800-epoch DAPI FCMAE run, so downstream
supervision never inherits the unrelated source output semantics.

## Fixed Two-Fold Screen

`configs/round75_cv2_unext2_cytoland_fcmae.json` uses the same fold0/fold2
ROIs, seed, FCMAE800 schedule, `14+14` supervised schedule, halo64 target,
spatial-thumbnail FiLM, Uniform-7 hybrid loss, global-MSE checkpoint selector,
D4 inference, and JPEG-q95 metric as Round52. The sole training difference is
the public encoder initialization before fold-local DAPI FCMAE.

The direct candidate is compared with Round52 official ImageNet-FCMAE800. Two
fixed deployment tests start from the current two-fold Base-diversity endpoint:

- replace its `0.175` official-Tiny share with Cytoland-FCMAE800;
- split that share equally between official Tiny and Cytoland while preserving
  the `0.175` Base member and every other weight.

Direct gain requires `+0.05` q95, both folds, 7/10 ROIs, positive float score,
and a `-0.12` ROI floor. Replacement and diversity each require `+0.04` over
the current Base endpoint, both folds, 8/10 ROIs, positive float score, and a
`-0.08` floor. Replacement depends on the direct mechanism; the fixed split
may pass independently as a diversity route. No source layer subset, FCMAE
epoch, supervised epoch, checkpoint, ensemble weight, or ROI route will be
selected after the result.

A passing route must be expanded unchanged to all five folds and pass complete
OOF before test inference. This two-fold screen cannot create a submission.

## Status

Implementation and prelaunch validation are complete. The focused local suite
passes 99 tests. On Linux, all 299 tests and 77 subtests pass, together with
Python compilation, launcher syntax, exact pretraining/training dry runs, and
diff checks. Loading the real public checkpoint independently reproduces 252
source tensors, 180 encoder tensors, 178 loaded tensors, 27,854,112 loaded
parameters, and the registered SHA-256.

Round75 launched from commit `5509c9a` in detached Screen `round75` at
2026-08-26 18:43:06 +08:00. A two-epoch, 32-image real-data smoke ran on GPU0,
verified the exact initialization mode, hash, tensor count, parameter count,
metric rows, and final encoder, then removed only its verified smoke directory.
The scheduler sampled GPU0/GPU1 at 0--1% utilization with 35,056/40,424 MB
free, then launched fold0 on GPU0 and fold2 on GPU1 at 18:43:44/18:43:46.
Subsequent sampling measured 100%/99% utilization at about 237/230 W. Formal
training completed successfully. Both pretraining runs contain exactly 800
contiguous epochs, `complete=true`, the exact public initialization hash, and
final encoder SHA-256 values
`bf65d68494adcd364b04106ee17cfe0e181e6a6521bc2ac42fd534e2be67ff34`
and
`d2e019f8f14e29297aea8c51c0381c878eae275e057d3dfc882b1e6b5ebd7561`.
All four supervised stages contain 14 contiguous metric rows. Their selected
stage-2 checkpoints are fold0 epoch 13, SHA-256
`a8427771c845ec9278aab56daa98cef5818e10702aa181bb30791700ece4b228`,
and fold2 epoch 11, SHA-256
`d295b3eebd82cd21e69819c1fac6da8ab771b91c2968372a4ccbdee7cf7daf08`.

The original scheduler then waited because its 27 GB training-memory threshold
was also applied to OOF evaluation. A full six-model, one-image D4 probe on
idle GPU1 measured only 1.20 GiB peak allocation and 1.71 GiB peak reservation.
The sleeping scheduler was stopped and the unchanged evaluator ran on GPU1,
which had 20.8 GB free. OOF exited zero, covered all 2,706 images and ten ROIs,
and took 4,350.35 seconds. All 12 recorded checkpoint hashes independently
match their files. The OOF SHA-256 is
`0a53d7184f46952737cf275d15f313a37876dd307c8b7985c89efcc03791f4ec`;
the gate SHA-256 is
`19523bf65cbc59bedab2ac03cb11c22c2ce8f13a7bc80904a3d668fc89c555dc`.
Independent local gate evaluation reproduces every metric and decision.

## Result

The Cytoland-initialized FCMAE model scores `74.597706` q95 versus
`74.847851` for official ImageNet-FCMAE800, a direct loss of `-0.250145`.
Both folds regress (`-0.421191/-0.089873`), only ROI008 improves, and ROI001
loses `-1.110855`. SSIM falls from `0.839017` to `0.836633`, while mean-image
PSNR falls from `26.861077` to `26.722323` dB.

Replacing the official-Tiny share in the current two-fold Base endpoint loses
`-0.049339`. The fixed half-split diversity route also loses `-0.013767`, is
negative on both folds, and improves only 3/10 ROIs. Every gate fails and
`promotion.allowed=false`; no five-fold expansion, test inference, submission,
FCMAE-duration search, or ensemble-weight search follows.

Round75 rejects the hypothesis that a Cytoland encoder remains beneficial after
a full 800-epoch fold-local DAPI-FCMAE trajectory. It does not test Cytoland's
compatible virtual-staining decoder, which FCMAE discards entirely; that is a
separate direct-transfer mechanism rather than an epoch rescue.

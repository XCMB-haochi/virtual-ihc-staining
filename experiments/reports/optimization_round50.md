# Round 50: Paired Spatial Augmentation Screen

## Motivation

Round49 rejected a wider high-resolution decoder head. The direct model lost
`-0.311588` q95 to FCMAE800, and even a fixed 30%/30% diversity split remained
`-0.016307` below the Round47 two-fold analogue. Together with prior negative
base48, Restormer, Swin, NAFNet, and whole-ROI results, this closes another
capacity-oriented route.

The remaining untested data-efficiency mechanism is continuous paired spatial
augmentation. Training currently uses only exact D4 transforms. Official VisCy
configurations apply the same random affine to source and target and then
center-crop an oversized field. Those configurations use probability `0.8`,
arbitrary in-plane rotation, and broad scale jitter. This round adapts that
mechanism without changing labels, architecture, inference, or supervision
weights.

## Fixed Augmentation

The candidate loads DAPI and CD68 with outer halo 256, producing a 768x768
paired field. With probability `0.8`, both channels receive one bilinear affine
with residual rotation in `[-45, 45]` degrees and isotropic scale in
`[0.72, 1.30]`, followed by an exact center crop back to the existing 384x384
halo64 training tensor. Existing random D4 remains enabled.

Random D4 already chooses one of four right-angle rotations uniformly. Adding
a residual angle from `[-45, 45]` therefore spans the same complete orientation
range as an arbitrary in-plane rotation, without redundant parameterization.
The minimum scale is raised from VisCy's `0.70` to `0.72` because the fixed
768-to-384 geometry requires at least `384*sqrt(2)/768 = 0.707107` coverage at
45 degrees. This prevents zero-filled pixels from entering the supervised
crop. No translation, shear, elastic deformation, source-only photometric
change, or thumbnail transformation is added. The affine is centered, so the
D4-invariant neighborhood thumbnail continues to describe the same patch and
ROI rather than a shifted location.

## Fair Two-Fold Screen

The protocol is `configs/round50_cv2_unext2_paired_affine.json`. It reuses the
same SHA-identical fold-local FCMAE800 encoders as Round46, but trains the
supervised candidate fresh. Fold0/fold2, seed 2044, architecture, thumbnail
FiLM, 14+14 epochs, batch 16, workers 4, learning rates, loss, full halo64
supervision, global-MSE checkpoint selection, D4 inference, and JPEG q95 are
unchanged. The additional random numbers use each DataLoader worker's PyTorch
RNG, while the existing D4 path keeps its independent Python RNG stream.

The fixed candidates are direct FCMAE800 versus paired-affine FCMAE800, the
Round47 two-fold 28%/12%/60% analogue with direct replacement, and one
28%/12%/30%/30% diversity split. No affine parameter, epoch, checkpoint,
ensemble weight, calibration, or ROI route is searched.

## Preregistered Gates

The direct mechanism requires at least `+0.05` q95, positive float gain, both
folds and at least 7/10 ROIs positive, and no ROI below `-0.12`. Replacement
requires the mechanism gate plus at least `+0.08` q95, both folds, 8/10 ROIs,
and a worst ROI above `-0.08`. The independent diversity route requires at
least `+0.06` q95 with the same distribution constraints.

This pipeline cannot create a test package. A passing route must first be
expanded unchanged to all five folds and pass complete-OOF gates. Longer
supervised training is deliberately deferred: it becomes a separate follow-up
only if augmentation itself passes and selected checkpoints remain late.

## Execution Status

The local implementation and protocol are complete. Focused dataset tests
verify shared source/target resampling, exact 384x384 output shapes, and no
zero-filled pixels at the fixed 45-degree/0.72-scale extreme. Protocol tests
verify complete parameter metadata, reject an outer field that cannot cover
the crop, and confirm all four fold-stage commands. Python compilation, JSON
parsing, the exact protocol dry-run, and `git diff --check` pass. The broader
Windows test process still reaches the repository's known NumPy/MKL duplicate
OpenMP native termination, so complete regression was run on the clean Linux
server.

The first Linux pass caught a default-path regression when source and target
halos differed while affine augmentation was disabled. The implementation was
corrected so only the opt-in affine branch concatenates channels; the default
dataset path again performs the original independent conversions. After that
fix, the complete server suite passes `241 tests` plus `71 subtests`.
Compilation, pipeline shell syntax, the exact four-command dry-run, and
`git diff --check` also pass at runtime commit `05b7812`.

A physical-GPU1 smoke run used the real fold0 FCMAE800 encoder (SHA-256
`cfd93eaf4fb67b41b44b3bc829b5edd6a5d6ce253a6903599b0b2feae2a4aa04`),
64 training images, 32 validation images, batch 16, and workers 4. One epoch
took `18.130 s` and peaked at `25.270 GiB` of PyTorch GPU allocation. This is
not slower than the comparable Round49 smoke (`19.406 s`), so oversized-field
loading and CPU affine resampling do not create a material throughput
bottleneck at the fixed worker count. The checkpoint records all five affine
parameters exactly and loaded 196 encoder tensors. The process exited zero;
trailing `pymp-*` NFS cleanup messages were the known worker-finalizer noise.
The validated smoke directory was removed.

The final five-second GPU sample found physical GPU0 at `31--40%` utilization
with `32.380 GiB` free and physical GPU1 at `15--17%` with `32.251 GiB` free.
Both exceed the measured peak by about 7 GiB. Round49 also showed fold2 taking
longer than fold0, so the parallel scheduler assigns shorter fold0 to the more
contended GPU0 and longer fold2 to the less-contended GPU1; OOF evaluation is
fixed to GPU1 after both queues complete. This is expected to beat a serial
GPU1 schedule despite GPU0 contention and avoids assigning the intrinsic
straggler to the slower queue.

The formal pipeline launched from runtime commit `adf4c7a` at 2026-07-27
13:40 CST in detached Screen `round50_paired_affine`. Fold logs, the pipeline
log, and the machine-readable exit marker remain under `reports/`. The pipeline
exited zero. Each of the four supervised stages produced 14 metric rows and 14
epoch checkpoints, and all selected checkpoints load successfully. Stage-1
best epochs were 9/14 for fold0/fold2 and stage-2 best epochs were 13/14 and
14/14. Peak PyTorch allocation was `25.270 GiB` in stage 1 and `25.389 GiB` in
stage 2. Trailing `pymp-*` NFS cleanup tracebacks occurred only after the
explicit completion markers and are non-fatal worker-finalizer noise.

## Formal Outcome

The OOF report contains all 2,706 held-out images across the fixed 10 ROIs and
uses the organizer-confirmed per-image metric. Its SHA-256 is
`f4f6bac31311acc2d1c4c847bb7939d24ac55952bbc1a80148929fbd3d4e208e`; the
gate report SHA-256 is
`2c30674c8a18cbdde44f09c06276c70ce039a12b884081283b12cf1692173fd0`.

Direct paired affine scores `74.774993` q95 versus `74.822793` for FCMAE800, a
`-0.047799` change. Both folds are negative (`-0.069317/-0.027637`), only 5/10
ROIs improve, and ROI024 falls `-0.455767`. The fixed 60% replacement is also
negative by `-0.045882` q95, with both folds negative and a worst-ROI change of
`-0.296962`.

The independent 30%/30% diversity split is mildly positive: `+0.015077` q95
and `+0.015762` float, with both folds positive. It nevertheless improves only
7/10 ROIs, misses the preregistered `+0.06` magnitude threshold, and violates
the `-0.08` ROI floor at ROI024 (`-0.103836`). All three gates therefore fail.
No five-fold expansion, longer-training rescue, test inference, package, or
adjacent augmentation/ensemble search follows. Strong continuous affine
augmentation is closed for this recipe; its small diversity gain is retained
as evidence, not promoted as a candidate.

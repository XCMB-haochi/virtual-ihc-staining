# Round 96: Fold-Local Multi-Marker FCMAE

## Motivation

Round95 closes the texture-thumbnail branch after complete five-fold OOF. Its
`+0.019837` q95 gain is supported by only three folds and 12/25 ROIs, with a
`-0.935751` worst-ROI regression. Another local context feature or blend probe
does not have enough evidence.

The remaining unused official information is the image distribution of the
three auxiliary IHC channels and CD68 itself before supervised translation.
Round51/55 showed that aligned auxiliary labels can change the useful error
structure, while Round65 showed that DAPI FCMAE endpoints can add diversity.
Round96 tests a different representation prior: one shared masked-reconstruction
encoder learns the morphology of all five official channels, then transfers
only its encoder to the unchanged DAPI-to-CD68 model.

Public VirTues weights are not used. Their pretraining spans many spatial-
proteomics datasets, and the current source audit cannot exclude hidden-test
overlap. The public repository for this exact competition is also not used as
evidence because it publishes no comparable grouped OOF, checkpoint, or
leaderboard artifact.

## Exposure-Matched Pretraining

For each outer fold, validation ROI filenames are excluded from DAPI, CD68,
CD45RO, HLA-DR, and Vimentin before pretraining. The five remaining fold-train
directories must have exactly matching filenames. Each image is independently
reconstructed by the same one-channel FCMAE; channel identity and paired
cross-channel targets are not provided. This isolates a generic multi-stain
morphology prior rather than adding another supervised translation loss.

The historical control uses DAPI-only FCMAE for `800` epochs. Round96 presents
all five channels once per epoch for `160` epochs. With the same batch 32,
this yields exactly the same number of image exposures. Fold0 also has exactly
the same `124800` optimizer steps. Because each epoch flushes its final short
batch, fold2 has `122560` steps versus the control's `123200` (`-0.52%`). The
alternative would repeat selected images solely to manufacture extra updates,
so the fixed protocol keeps exact exposure matching and records this bounded
step difference. Warmup is reduced from 10 to 2 epochs and checkpoint interval
from 25 to 5, preserving their exposure fractions. Learning rate, cosine
floor, weight decay, mask ratio, seed, official ImageNet-1K FCMAE
initialization, and model size remain unchanged.

## Fixed Downstream Screen

Only fold0 and fold2 are trained initially. The downstream candidate keeps
UNeXt2-Tiny, halo64 training, spatial-thumbnail FiLM, Uniform-7 hybrid loss,
global-MSE checkpoint selection, `14+14` epochs, effective batch 16, seed 2044,
context128 D4 inference, JPEG q95, and the fixed output offset `[0,-1]`.

The direct model is compared with the historical fold-local DAPI-only official
FCMAE800 model. Two fixed Base-endpoint comparisons are also registered:

- replace the `0.175` official-Tiny share with the multi-marker endpoint;
- split that share equally (`0.0875/0.0875`) between the old and new Tiny
  endpoints.

Direct representation gain requires `+0.05` q95, positive float gain, both
folds, 7/10 positive ROIs, and a `-0.12` ROI floor. Replacement and diversity
each require `+0.04`, positive float gain, both folds, 8/10 positive ROIs, and
a `-0.08` floor; replacement also depends on the direct gate. No marker ratio,
duration, seed, checkpoint, ensemble weight, ROI route, or leaderboard result
will be searched after the screen. A passing route must expand unchanged to
all five folds before test inference.

## Completion Audit

The detached pipeline exited with `reports/round96_pipeline.exit=0`; its Screen
session and all producer processes are gone. The server ran the formal jobs at
runtime commit `0e6a73bc2a6352b2be00aece4b65cb2286f8fa05` after all 337 Linux
tests passed. The pretraining and downstream real-data smokes also completed
before the formal launch.

Both fold-local pretraining runs contain 160 consecutive metric rows and a
`complete.json` record with `complete=true`, epoch 160. Each of the four
downstream stages contains exactly 14 consecutive rows from epoch 1 through
14, and both fold status files record the two supervised stages as complete.
The final encoder SHA-256 values are:

- fold0: `fa2d96bfa3196b913f954a15862987c94e45d7cc2e0088edde19be2029c9449c`;
- fold2: `ce54525418ccd5ef3dc08085e2fa3614d76aa2b51bffdff57f48e00d52129c72`.

The selected stage-2 checkpoint is epoch 10 in both folds. Independent hashes
match the OOF provenance:

- fold0: `478228adba18c4cd3d76859f446c82e6817e28b11c1b13fa89d2bfc751bed691`;
- fold2: `1b46d08bced76fcc91f8af0869300b7c6a9416d85442f43e2b3be99de6ba2f69`.

The OOF report covers all 2,706 images and all ten registered validation ROIs
and took `4440.639 s`. Its SHA-256 is
`e5c1b182ddadd1d9c5ee604f68f71621f08008a6c69f569d52b6e038d1e04a06`;
the gate SHA-256 is
`04cb3a3fbac72a554b96d099866d4b9aae6123c07f4de71e78280b46e7e060ae`.
An independent local gate replay reproduced every gate field, the OOF hash,
and `promotion.allowed=false`.

## Result

At JPEG q95, the direct multi-marker representation scores `74.947841` versus
`74.987515` for the exposure-matched DAPI-only official-FCMAE800 control, a
`-0.039674` change. Float changes by `-0.035566`. Fold0 loses `-0.137545`,
fold2 gains `+0.052033`, only 4/10 ROIs improve, and ROI012 loses `-0.551434`.
The direct representation therefore fails every preregistered distribution
check except its dependency check.

Replacing the fixed Base endpoint's official-Tiny share with the new endpoint
gains only `+0.010265` q95. Fold0 is negative, only 5/10 ROIs improve, and
ROI012 loses `-0.095268`, below the fixed `-0.08` floor. The half-split diversity
candidate is slightly safer but still only gains `+0.012357`: fold0 changes by
`-0.001194`, 7/10 ROIs improve, and the worst ROI change is `-0.041462`. It
misses the fixed `+0.04`, 2/2-fold, and 8/10-ROI requirements. No gate passes.

## Boundary

Round96 closes exposure-matched independent reconstruction of all five marker
channels as the next FCMAE prior. The fold disagreement and direct loss do not
support changing marker ratios, duration, seed, or ensemble weights, nor an
unchanged five-fold expansion or leaderboard submission. The small split gain
is retained as covariance evidence only; it is not large or stable enough to
justify a composition search.

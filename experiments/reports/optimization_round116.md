# Round116: Shared All-Train-DAPI Encoder With Auxiliary15 Supervision

Round65 found that an all-train DAPI FCMAE encoder is useful as a diversity
endpoint, while Round55 found a small auxiliary15 supervision effect. Round67
showed that these endpoint families are complementary in a fixed ensemble. This
screen tests their composition inside one supervised model: initialize UNeXt2
from the already audited shared Round65 FCMAE630 encoder, then use the unchanged
Round55 auxiliary15 14+14 supervised recipe.

The screen is fold0/fold2 only and uses one physical GPU by request. It keeps
seed2044, effective batch16 (`8x2`), context64 training, context128+D4/q95
inference, global-MSE selection and offset `[0,-1]`. The shared encoder saw all
official DAPI inputs but no IHC labels; this is the same semi-transductive
boundary documented for Round65. It is not a continuation of a supervised
checkpoint and does not search auxiliary weights or ensemble weights.

The fixed gate requires q95 `+0.05`, positive float, both folds, 7/10 positive
ROIs and worst ROI `>= -0.12`. Failure closes this unified route; a pass would
authorize only one unchanged five-fold expansion.

The all-train encoder source is fixed at SHA-256
`f4a8c93c616310ce4a981f9b1fe6830ba82bf78ff91bdbe52d471cf1c0164554`.

## Validation And Launch

The shared-source override passed local protocol validation, 95 focused tests,
remote preflight and a real two-stage smoke. The smoke loaded all 196 encoder
tensors, strictly restored stage2 and peaked at 13.21 GiB. Formal Screen
`round116_all_dapi_aux` is now detached with `gpu_candidates=[1]`; physical
GPU0 is excluded for this entire invocation. The launch receipt records source
commit `84b87da0e42c791629179edbb3c1d779f32dd878`. Formal fold0 has started with
4987 training and 1309 validation images. No OOF or gate result exists yet.

## Result

The single-GPU pipeline exited `0`. All four stages contain 14 contiguous
epochs, finite losses, valid checkpoints and the registered shared encoder
SHA-256; the OOF covers 2,706 images and 10 ROIs, and independent gate replay
passed. Selected epochs were fold0 `14/1` and fold2 `13/7`. Training took about
90-92.5 minutes per two-stage fold and OOF took 25.8 minutes.

| Metric | Round55 Baseline | Round116 | Change |
| --- | ---: | ---: | ---: |
| q95 | 75.013272 | 75.023209 | +0.009937 |
| float | 75.083722 | 75.095129 | +0.011407 |

Fold0/fold2 q95 deltas are `-0.139333/+0.149804`. Only ROI000, ROI001 and
ROI008 improve; the worst ROI is ROI020 at `-0.400201`. The unified all-train
DAPI plus auxiliary15 recipe is closed without five-fold expansion or parameter
search. Its positive aggregate is fold2-driven and does not support combining
the two endpoint families inside one model.

Artifacts: `round116_cv2_all_dapi_auxiliary_oof.json`,
`round116_cv2_all_dapi_auxiliary_gate.json`, `round116_pipeline.exit`.

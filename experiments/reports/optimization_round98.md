# Round 98: Cross-Marker Logit Feedback

## Completion audit

The detached Round98 pipeline exited with status `0`. Its Screen session and
producer processes are gone. Fold0 and fold2 each contain complete stage1 and
stage2 runs with 14 metric rows per stage, valid `best_global_mse.pt` files,
and the expected four-marker metadata. The OOF report covers 2,706 images over
the ten fixed ROIs. The candidate loaded the fold-local Round52 DAPI-FCMAE
encoder with the recorded SHA-256 values and used the new strict model
specification. Peak allocation was about 26.2 GiB on the A100.

The local and remote protocol/syntax/test checks passed before launch. The
known multiprocessing temporary-directory cleanup warning appeared after
successful training and did not change the zero exit status or completeness
checks.

## Mechanism

`unext2-fcmae-tiny-cross-marker-feedback` adds a small residual head from the
three auxiliary marker logits to the CD68 logit. Its final projection is zero
initialized, so the fresh model starts with the same forward output as the
ordinary shared decoder. All four outputs still depend only on DAPI at
inference. The candidate used the fixed `[0.85, 0.05, 0.05, 0.05]` marker loss
weights, context128 inference, D4 TTA, and the existing 14+14 supervised
clock.

## Result

| candidate | float official | JPEG-q95 official |
| --- | ---: | ---: |
| Round55 auxiliary15 reference | 75.083722 | 75.013272 |
| Round98 cross-marker feedback | 74.875576 | 74.807699 |

The JPEG-q95 delta is `-0.205574`; the float delta is `-0.208146`. Fold deltas
are `-0.227418` and `-0.185106`. Only one of ten ROIs improved and the minimum
ROI delta was `-0.455293`, so every distribution gate check failed and
`promotion.allowed=false`.

## Interpretation and boundary

An additional provenance check resolves the apparent path difference. The
Round55 `officialauxiliary15` stage-1 reports use the same fold-local encoder
SHA-256 values as Round98 (`50231215826b6d533975594030e48c7dca1316f0f2593f426bf02732ff33b0c9`
and `c3f71f31ed8652618d98092ae07d56c1fc1ed1967854bfe00c902af359efc1b7`), and
the downstream recipes share the same marker order, loss weights, condition,
seed, context, and 14+14 clock. The different directory names therefore do
not represent different encoder states. Round98 is a valid matched comparison
against the Round55 auxiliary model, and the feedback-head route is closed by
the negative q95/float deltas and unstable ROI changes.

The in-progress zero-feedback inference is diagnostic only; it cannot rescue
the architecture or authorize a submission, weight search, epoch search, or
ROI-specific post-processing.

## Zero-feedback ablation

The learned feedback projection was then zeroed at inference while keeping the
same two selected Round98 checkpoints, D4 path, context, offset, and OOF image
set. The ablated output scores only `58.925485` q95 (`58.968291` float), versus
`74.807699` q95 with the learned projection. This is not an evaluation or
checkpoint-loading failure: a direct batch check shows the ablation removes a
large primary-channel residual, while auxiliary channels and checkpoint
metadata remain valid. The result indicates that training let the new residual
head carry a substantial part of the CD68 prediction, but that decomposition
still underperforms the ordinary auxiliary model. It does not justify a gate,
residual-strength search, or another submission.

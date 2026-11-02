# Round218: fixed DAPI-to-marker offset diagnostic

Question: does a fixed one-pixel output shift improve new-stage OOF before
any training change? Evaluate complete five-fold domain endpoint predictions
at offsets row0/col -1,0,+1 under Gaussian11. This is a training-set
diagnostic only; no offset is applied to Round216 output and no test data or
leaderboard result is used. A stable direction across folds/ROIs would justify
a matched two-fold training target-offset screen; a tiny or mixed change
closes the offset hypothesis.

No model, checkpoint, loss, package or submission changes. GPU1 only; GPU0
remains reserved by standing project policy even when idle.

## Completed

Exit0. CD45RO Gaussian11 SSIM across the complete five-fold domain OOF:

| Offset | SSIM | Relative to `[0,0]` |
| --- | ---: | ---: |
| `[0,-1]` | 0.887214 | -0.001073 |
| `[0,0]` | **0.888287** | 0 |
| `[0,+1]` | 0.887229 | -0.001057 |

The zero offset wins every fold, but the difference is small and this
diagnostic reports SSIM only for shift comparison. Under the semifinal rule
banning prediction post-processing, applying a fixed output shift to a
submission would be risky automatic correction. No offset is added to the
Round216 package, and no target-offset training is opened from this result.

# Semifinal Round200: Grouped Five-Fold Convergence Baseline

The semifinal dataset audit found a new 17-ROI training set (`2380` paired
patches) and four-ROI test DAPI set (`560` patches). ROI IDs were reassigned,
but every ROI retains a complete `10x14` grid and all five training directories
are exactly paired. No Colon checkpoint or Colon data is used.

The fixed baseline uses seed `2026`, grouped ROI folds with validation sets:

- fold0: ROI005, ROI052, ROI087
- fold1: ROI010, ROI058, ROI093
- fold2: ROI019, ROI061, ROI100
- fold3: ROI044, ROI067, ROI074, ROI078
- fold4: ROI045, ROI072, ROI076, ROI081

The model is UNeXt2-FCMAE-Tiny with spatial-thumbnail conditioning, the fixed
four-marker auxiliary loss, D4 inference, JPEG quality 95, and output offset
`[0,0]` for the stricter semifinal output rule. The encoder starts from a
public ConvNeXt-V2 FCMAE Tiny artifact; no external or Colon data is used.
This is intentionally a convergence baseline, not a literal copy of Colon
Round136: it has no Colon relational teacher, no Colon checkpoint, and no
distillation. Those mechanisms can only be reintroduced after a new-stage
fold-local teacher is trained from the semifinal data.

The runner records all 14+14 supervised epochs for each fold. The complete OOF
trajectory will determine one common stage1/stage2 checkpoint clock from
Train/Val only. A later full-data student and leakage-safe fold-out teacher
cache will be considered only after this baseline convergence evidence.

## Launch (2026-09-21)

- protocol: `configs/semifinal_cv5_baseline.json`
- source commit: `b5c24e7`
- detached Screen: `round200_semifinal_cv5`
- physical device: GPU1 only
- status: fold0 stage1 training active; an interrupted foreground attempt was
  preserved under `transfers/round200_interrupted/` and excluded from formal
  artifacts.

## Completion (2026-09-21)

- pipeline exit: `0`; all five folds and both 14-epoch stages completed;
  complete OOF covers `2380` images and all 17 training ROIs
- overall q95 score: `71.8536`; float score: `71.9231`
- fold q95 scores: `67.7642`, `71.2195`, `78.3737`, `67.5443`, `74.8154`
- fold-best stage1 epochs: approximately `12,12,11,14,10`
- fold-best stage2 epochs: approximately `2,1,14,13,12`

The two protocol candidates are the same model and therefore have identical
OOF scores; the second entry exists only to satisfy the current evaluator
schema. This result is a convergence baseline, not a final package. The wide
fold spread means a single fold-best epoch is not a safe final clock. The next
registered step is a full-data seed-2026 student with a fixed grouped-OOF
teacher and a common checkpoint clock near the fold medians (stage1 12,
stage2 12).

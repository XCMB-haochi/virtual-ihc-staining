# Round217: complete domain-teacher five-fold OOF

Now all five fixed domain teachers exist from Round207/216. Evaluate their
epoch12 endpoints on their held-out ROIs under D4/halo128/offset0/q95, all
four output channels, using the checked evaluator and manifest hashes.
Compare to Round204's five-fold public-initialization baseline in the
Round205 OOF report. No new weights, checkpoint selection, output tuning,
training or test access. This completes the previously two-fold domain
evidence and can expose weaknesses outside folds0/3 for subsequent work.
It is teacher OOF, not an estimate of the full-data distilled student's
score. Round216 platform submission can proceed independently.

GPU1 only; recent full evaluation took about18min, but load may change it.
Detached session `round217_domain_cv5_oof`; no submission package generated.

## Completed

Exit0; all five fixed domain teachers were evaluated at epoch12 over2,380
images and17 ROIs. Relative to the Round205 public-initialization baseline,
CD45RO Gaussian11 score improves **+1.03110 overall**; fold gains are
**+1.54149,+1.02683,+0.80196,+1.41701,+0.43745**, with all17 ROIs positive.
CD68 gains+2.89725 overall, HLA-DR+1.90747, and Vimentin+1.54626, with
only Vimentin ROI072 negative. This completes the full-fold evidence that
domain FCMAE is the main recovered mechanism behind Round208/216. It is
teacher OOF evidence, not an independent validation of the full-data
distilled student, and no submission package is produced.

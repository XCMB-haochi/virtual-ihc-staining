# Round228: complete base-model update-budget validation

Round227's paired budget test found the base benefits while the signal
auxiliary model loses its former advantage. Complete the base-only five-fold
evidence, not another signal-model parameter search. Reuse fold0/fold3
Round227 checkpoints and train only folds1/2/4 from their exact existing
Round216 domain800 encoders. Keep the same formula-derived clocks:
fold1/2 use17+17 with endpoint15; fold4 uses18+18 with endpoint16.
No new encoder pretraining, loss, condition, batch, marker, seed or output
change. Fixed D4/halo128/offset0/q95 evaluation over2380 images/17 ROIs.

This is a validation calibration study. Existing Round208 already has the
full-data14-epoch update budget which these folds approximate. A positive
OOF result does not prescribe increasing full-data training again. It may
yield a better-supported teacher set or better evidence for the base model;
any subsequent training requires a separate interpretation. No full-data
student, cache replacement, checkpoint search or submission is automated.

Three new supervised folds cost about4.8h serial based on Round227's1.57/
1.65h per fold. GPU0 has25.3GiB free and idle compute. GPU1 has enough memory
but another workload uses compute; final allocation will use live checks,
giving it at most one fold while GPU0 handles the longer queue if parallel
execution remains worthwhile. The recipe's peak near7GiB permits coexistence;
no other process is touched. Tests verify exact expansion command equivalence.

## Completed

Exit0; ten stage audit entries with actual optimizer updates; full OOF
covers2380 images. CD45RO score78.621778, gain+0.069143 versus the original
budget. Fold changes:+0.064250,-0.007434,+0.110326,+0.284219,-0.115716.
Thirteen of17ROIs improve, but ROI072 loses0.953131. The two-fold gain
does not establish a broadly robust longer-budget prescription. Do not
extend the full-data model, replace the distillation cache, or submit this
validation-only configuration. Close the budget expansion; preserve
Round21679.0198. The independent teacher-bank checkpoints remain available.

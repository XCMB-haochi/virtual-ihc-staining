# Round316: bounded late Stomach weighting test

Question: does a late objective0.1/0.1/0.8 improve on Round313's0.1/0.2/0.7?
Keep the exact Round307 stage1 and RNG, Round313 stage2 recipe (teacher0.2,
consistency0.1), all image exposures,14-epoch horizon and original grouped Val.
Only stage2 coefficients change to N*w/N_organ; mean coefficient stays1.
Colon stays0.1, Liver decreases0.2 to0.1, Stomach increases0.7 to0.8. No organ
input or inference change. This does not compound Round315 teacher0.4.

Related evidence: Round307 changed natural per-image proportions to0.1/0.2/0.7
through both stages; Stomach improved0.16894 but total only0.01270 with losses
in other organs. This supports a limited follow-up, not a guaranteed gain or
a broad weight search. Historical0.1/0.2/0.7 replacement sampling also differs.
A late modest increase preserves the shared first-stage representation, but
may still overfit3 held-out Stomach ROIs or harm Liver enough to lose overall.

Run epochs15-28 and report weighted Val, organ/ROI metrics and endpoint curves.
No automatic full fit or Test-based tuning. Expected2-3h after315; overall queue
approximately6-9h including corrected314 packaging.315 launches316 directly
in its entry point because the already-running parent queue has fixed code in
memory; merely editing that parent's source would not append a task.
Deployment: archive hash verified remotely; real-data two-update smoke, finite optimizer audit and config comparison passed. Only organ weighting fields differ from313. Existing314 continues;315 loads the successor entry point at runtime. No formal316 result exists yet.

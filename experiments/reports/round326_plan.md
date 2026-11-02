# Round326: final bounded consistency comparison

Train/Val rationale only: consistency0/0.1/0.2 produced weighted best
76.2247099/76.2500301/76.2672365. Test whether one further doubling to0.4
helps. No prior0.4 consistency trial found; Round315 changed teacher fraction,
not consistency. No platform-feedback-based choice or inference change.

Only change from318 is stage2 consistency0.2->0.4; same307 stage1/RNG,
800 encoder, teacher0.2, organ weights, seed2026, order, LR and14-epoch horizon.
Validate15-28 with identical Gaussian11/JPEG95 metrics; report organ/ROI and
matched trajectory. No automatic refit; positive best alone is insufficient.
If persuasive, fresh full supervised fit from fixed encoder, not Val weights.
No pretraining rerun, EMA, TTA, checkpoint averaging or output processing.

November1 09:20: approximately2.5h Val plus4-5h fresh full training and delivery
can fit before20:00. Reassess measured timing before refit and preserve at least
one hour for delivery/submission. Otherwise keep existing artifacts.

Launch: revision8e6b1c6 deployed with three source SHA256 matches. Remote real
GPU smoke passed finite optimizer audit, baseline config comparison and report
serialization.5090 idle with29934MiB free before launch. Hidden AIC-Round326
Scheduled Task running, actual pythonw training process confirmed. No automatic
full refit; user reports completion for result review.

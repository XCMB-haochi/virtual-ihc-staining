# Round301: matched fourfold supervised budget

Round300 finished48 cycles successfully. Best grouped validation74.3855805 at
cycle48: Colon/CD68=70.2320482, Liver/CD45RO=76.3087289 (rounded),
Stomach/CD68=74.4294713 (rounded). See round300_validation_048.json for exact
metrics and all ROIs. This is local validation, not a platform result.

Late validation36/40/44/48:74.3125/74.3599/74.3309/74.3856. The small late
gain with near-zero LR does not establish sufficient training. With98304
sampled tasks and organ probabilities0.1/0.2/0.7, expected per-image per-marker
exposures are0.491/2.701/13.653 for Colon/Liver/Stomach Train respectively.
Insufficient task coverage is a concrete concern, particularly for Colon.

Change only training budget and its associated cosine duration:192 x256 updates
x8 samples, from identical public initialization and seed2026. Same architecture,
organ/marker sampling, losses, fixed split, JPEG95 validation, no Test use.
Validate every16 cycles, retaining12 scheduled validation endpoints as in the
baseline. Do not resume or reset the LR of the mature Round300 checkpoint.

Decision: compare weighted validation and all organ/marker/ROI outcomes. Do not
declare generalization from only3 Stomach ROIs. Use this result to decide whether
to add domain FCMAE or change sampling; neither mechanism is rejected by Round300.
No submission or automatic long follow-up chain. Keep Round300 as fallback.

Cost: measured Round300 ~74.7s per2048-task cycle, ~81s per complete validation.
Projected Round301 compute ~4.25h plus loading/checkpoint overhead; budget4-5h.
RTX5090 observed idle with31104MiB free at result review. Launch using a hidden
one-off user Scheduled Task, logs and resumable checkpoints. User reports finish.

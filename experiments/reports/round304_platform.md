# Round304 platform result (October29)

User reported Colon73.6982, Liver78.7480, Stomach72.3313; fixed weights
0.1/0.2/0.7 give73.75133, displayed73.7513. Package/checkpoint identifiers
are in round304_submission.json. One October29 submission has been used.

Compared with submitted Round302: organ gains+1.5832,+0.3747,+0.5221;
weighted contributions+0.15832,+0.07494,+0.36547, total+0.59873.
All three improve. This combines recipe changes and all-Train supervision;
Round303 has no reported platform score, so it is not a clean measure of
full-data training or any one mechanism. No independent Val score for304.

Stomach priority was already motivated by published70% score weight and the
Train inventory (1680 images/12 ROIs; outer fit1260/9 and Val420/3).
Round307 already tests organ loss weighting in joint training, and Round308
is queued with a fixed Train/Val selection protocol. Keep those plans unchanged.
One model/checkpoint does not require equal training loss weights or sampling
rates by organ; organ identity is never supplied to model.forward. Isolated
Stomach experiments can diagnose mechanisms but cannot establish improvement
of the jointly trained three-organ submission. Prioritize joint matched trials
and per-ROI Stomach analysis over a separate single-organ search now.

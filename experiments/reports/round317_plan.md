# Round317: lower second-stage learning rate

Train/Val rationale: Round313 endpoint scores vary75.85-76.25, including drops
at18,21,25. This can reflect update/order sensitivity; it does not prove the
learning rate is excessive.315 teacher0.4 and316 Stomach0.8 did not improve
weighted Val. Search of semifinal plans found no matched half-LR test.

Compare against313, changing only stage2 base LR1e-4 to5e-5 across all parameter
groups. Same encoder depth ratios, AdamW reset,14-epoch cosine horizon, zero
minimum, clipping, image order, geometry RNG,307 stage1, teacher0.2, consistency0.1,
organ objective0.1/0.2/0.7 and grouped split. Evaluate epochs15-28. No new encoder
pretraining or Test-based choice. Could underfit within14 epochs; no guaranteed
benefit or automatic extension. No automatic full refit.

Inspect weighted score, every organ/ROI and matched endpoint trajectory. A small
single-split difference is uncertain. If selected, full refit can reuse verified
all-Train309 stage1 with the same LR intervention. Estimated2-3h on5090.
Smoke must execute actual report serialization/reload, optimizer update audit,
and config equality except learning-rate fields; prior315 smoke missed reporting.
Launch October30: verified source archive; real GPU smoke passed optimizer/config audits and actual report save/reload. Observed all stage2 group rates halved. AIC-Round317 started with project pythonw.exe, process and formal config confirmed, no stderr. Source revision recorded in round317_launch_commit.txt.

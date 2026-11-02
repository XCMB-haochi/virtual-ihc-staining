# Round316 result

Completed14 stage2 epochs,7084 optimizer updates; finite weights and final/last
checkpoint equality audit passed, no stderr. Best epoch23 weighted Val76.2433370
versus Round313 best epoch26 76.2500301: -0.0066931 (effectively tied).
At the selected endpoints: Colon -0.1400501, Liver -0.0985013,
Stomach +0.0385888 (2/3 Stomach ROIs positive). Weighted contributions are
approximately -0.0140050,-0.0197003,+0.0270122.

Matched-epoch comparison:316 is lower at12/14 endpoints; only16 and23 improve.
At epoch26,31676.2402081 versus31376.2500301 (-0.0098220).
Do not claim statistically demonstrated harm from this single split; there is
no affirmative total-score evidence to justify a full refit or a weight sweep.
Keep313-selected full refit314 as the submission candidate.314 ZIP already
local, checks passed. Recommend submitting314 now rather than waiting for
unplanned experiments. No316 full refit or additional training launched.

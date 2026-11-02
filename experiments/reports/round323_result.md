# Round323 result: small paired crops

Completed14 stage2 epochs/7084 updates, finite/final checkpoint and replayed
crop-count audits passed, no stderr. Best epoch23 Val76.2781952 versus318's
best epoch26 76.2672365: +0.0109587. Selected-endpoint organ changes:
Colon-0.1471678, Liver-0.0751023, Stomach+0.0581370. Those endpoint comparisons
include stopping-time differences; do not attribute every organ change purely
to cropping. Stomach ROI000/002/006: -0.1580582/+0.1945469/+0.1379224.
Only3/12 ROI changes are positive at the selected endpoints.

Matched epochs:7/14 positive,7/14 negative, mean-0.0356848; last3 mean+0.0152939.
Epoch26 also reaches76.2763836 versus31876.2672365, so the small gain is not
confined to the selected epoch23. Evidence is weak and heterogeneous, not a
reliable generalized improvement. Single-split selection and3 Stomach Val ROIs
remain material limitations. No new augmentation strengths or crop-size sweep.

Decision: build one full-data candidate324 while time permits, using fixed
epoch23 and the registered crop protocol. This is a bounded candidate based on
weighted Val and supportive late endpoints, not a claim of statistical benefit.
Preserve319, do not automatically submit. Full supervised refit begins again
from the fixed800 encoder and random task initialization; no supervised Val
checkpoint is expanded to allTrain. See round324_plan.md.

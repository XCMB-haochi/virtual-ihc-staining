# Round302 platform result

User reported October28: Colon72.1150, Liver78.3733, Stomach71.8092.
Weights0.1/0.2/0.7 give73.1526 exactly. Reported rank42/64; leader74.1960.
Package: submissions/semifinal_round302_mixed_q95.zip (see submission manifest).

Grouped Val was75.4728314: Colon73.0953529, Liver76.6485249,
Stomach75.4765588. Platform-minus-Val differences are-0.9803529,+1.7247751,
-3.6673588; weighted difference-2.3202314. The previous rough75.8 platform
estimate based on historical Liver offsets was not reliable for this stage.
These are different image populations, not paired model comparisons; they do
not identify a bug or establish a causal explanation for the gap.

Round303 protocol was fixed and launched before this score; keep it unchanged.
Use Train-derived grouped Val for subsequent selection, not platform scores.
This submitted checkpoint used held-out Train ROIs, not all-Train supervision.
No assumed gain from full-data retraining; only3 Stomach Val ROIs limit inference.

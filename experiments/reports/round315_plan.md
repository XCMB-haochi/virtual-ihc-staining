# Round315: stronger teacher supervision, one matched test

Compare teacher fraction0.4 with Round313's0.2, preserving consistency0.1,
organ objective0.1/0.2/0.7, original grouped split, Round307 stage1+RNG,
teacher cache, encoder, optimizer, augmentation, seed2026 and14-epoch horizon.
Fit epochs15-28; report every endpoint and organ/ROI. No automatic full refit.

Rationale: matched distillation308 improved307 by0.1211988, all3 organ means
and9/12 ROIs positive. This is stronger evidence than307's organ-weighting gain
of0.012697 with Colon/Liver losses. Teacher smoothing may reduce limited-data
variance; raising its weight may also amplify bias and blur targets.0.4 keeps
GT dominant (0.6), and is a single bounded probe rather than a grid search.

Alternatives reviewed: Gaussian11 objective312 was -0.006555; DAPI gain311
was +0.00619 against matched A100 control; all-CD68305 lost0.80134. Raising
Stomach to0.8 remains untested in this exact setting, but is lower priority than
this test given existing cross-organ tradeoffs. These results reject only the
tested configurations, not entire method families. No Test feedback is used.

Estimate2-3h after corrected314 packaging, based on313 measured throughput.
Review weighted Val, Stomach and other organ/ROI changes, plus endpoint curves;
no arbitrary one-ROI veto and no claim of statistical certainty from3 ROIs.

# Round322 result: marker-specific output heads

Completed14 stage2 epochs/7084 shared optimizer updates. Audit passed finite
weights, final/last state equality and per-head optimizer update counts; stderr
empty at completion review. Best epoch26 matches the reference endpoint.

| Grouped Val | Round318 | Round322 | Difference |
|---|---:|---:|---:|
| Weighted |76.2672365|76.2612881|-0.0059484|
| Colon |74.4382610|74.4498625|+0.0116016|
| Liver |77.5651494|77.5935295|+0.0283802|
| Stomach |76.1576865|76.1394228|-0.0182637|

Stomach ROI000/002/006 changes: -0.0090514/-0.0083497/-0.0373901.
Matched endpoints:6/14 higher,8/14 lower; mean change-0.0027962, last3 mean
-0.0033278. Differences are small, with no persuasive overall benefit. The
decision is not an arbitrary single-ROI veto or evidence of universal harm.

No full-data refit: the tested stage2 head split does not improve the weighted
selection metric, and neither late trajectory nor Stomach ROI distribution
provides a positive supporting signal. This does not test separate heads from
the start of stage1 or larger marker-specific decoders. No parameter search or
further training is authorized by this result alone; no new job launched at
this review. Keep all322 checkpoints and metrics.319 remains the highest
reported submission,74.0243. Test scores were not used to evaluate322.

Reviewed October31 about19:24 Beijing; submission deadline November1 20:00.

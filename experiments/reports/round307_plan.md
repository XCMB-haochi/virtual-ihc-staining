# Round307: fixed organ-weighted loss, same per-epoch coverage

Question: under Round303's improved two-stage recipe, does weighting organ mean
losses by the published0.1/0.2/0.7 objective improve grouped Val? Current equal
per-image training implicitly weights organs5004/8084,1820/8084,1260/8084
(61.9%,22.5%,15.6%). This is a mismatch to the scoring objective, not proof that
the scoring weights are the optimal training weights. Unequal scales/gradient
variance and overfitting are risks. No gain is assumed.

Round300/301/302 already sampled organs0.1/0.2/0.7 with replacement. This round
tests the same intended organ objective using complete Train traversals under
the Round303 recipe, without extra Stomach repeats. It is not a claim that
organ weighting is new. Search/review found no previous matched comparison of
this configuration. The rationale uses Train counts and published weights,
not the Round302 Test score. Round304 package remains available now.

Only change: multiply each training loss by N*w_org/N_org, computed from Train
counts; mean coefficient across Train equals1. Each image appears once/epoch.
No per-batch renormalization by the observed organ mix. Unweighted losses remain
in the diagnostic log. Gradient clipping remains1, so weighting changes which
updates clip as part of this intervention. Model receives DAPI and marker only;
the training loop's organ loss coefficient is not an inference input.

Keep mixed markers, grouped split, completed800 encoder, seed2026, same sample
order/geometry, architecture, optimizer,14+14 schedule, BF16, accumulation16,
uniform7 loss and Gaussian11/JPEG95 Val. Best checkpoint over28 endpoints;
compare to Round30376.0908141 and show every ROI, with only3 Stomach ROIs.
If weighted Val is strictly higher, package this one best checkpoint and download
it; no other ROI threshold. Otherwise retain Round304. No automatic full refit
or additional training chain; review next after completion. Estimate2.5-3h.

Tests cover weighted mean/gradient equivalence, Train-only coefficient derivation,
unchanged sample coverage/schedules and existing queued pipeline branches.
Real two-stage GPU smoke must pass model/optimizer/config/split audit before
launch. Remote task AIC-Round307 invokes training, audit and optional package
in one Python process; local AIC-Round307-Download uses the tested download code.

Launch October29: local11 relevant tests passed; remote weighting tests and
real-data two-stage smoke passed, including identical baseline config/split
except loss-weight fields and finite optimizer updates. Verified archive
SHA3ab8c13ce9d38d7a6747b2a7b494f25c3e46a2cb0a47efd2d1a3cb2edb0755c0;
source commit recorded in round307_launch_commit.txt. GPU idle at launch with
ample memory. Remote training and local delivery scheduled tasks started.

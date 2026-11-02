# Round 92: D4 Consistency on the Auxiliary UNeXt2 Endpoint

## Motivation

Round90 showed that the auxiliary/all-train-DAPI endpoint benefits materially
from a context128 inference view in aggregate, but its fixed global ROI floor
failed. The next structural test is a different question: can an explicit
equivariance loss make the current UNeXt2 endpoint less sensitive to the D4
views already used at inference? The old U-Net D4-consistency experiment was
near zero; it does not settle this question for the FCMAE-UNeXt2 model.

## Fixed protocol

`configs/round92_cv2_unext2_d4_consistency_context128.json` trains only a new
four-output `d4auxiliary15` recipe on fold0 and fold2. It starts from the exact
fold-local official FCMAE-Tiny encoders already used by Round55, keeps the
auxiliary loss weights `0.85/0.05/0.05/0.05`, and adds one fixed
`d4_consistency_weight=0.1`. The effective batch remains 16 via micro-batch
8 and two-step accumulation. Both the original and one deterministic non-
identity D4 forward contribute gradients; all model, data, optimizer, epoch,
context, output offset `[0,-1]`, D4 inference, and JPEG-q95 settings are fixed.

The OOF also evaluates a fixed replacement and half-split in the completed
Round90 context128 endpoint. No consistency-weight grid, D4 subset, checkpoint,
epoch, blend, ROI route, test-label statistic, or leaderboard feedback is used.
The gate requires the direct mechanism to pass before either deployment split
can be promoted; a passing route would still require a complete five-fold
expansion.

## Status

The protocol, runner support, focused tests, and shell syntax checks pass
locally. The remote preflight and real-data smoke must verify the two-forward
loss path, the 4-marker metadata, the effective batch, and the reused encoder
hashes before formal training starts. The detached pipeline will schedule
folds from live GPU telemetry and keep all artifacts under `/workspace`.

Remote compilation, the full test suite, shell syntax, and the hash-bound
preflight passed. The first detached attempt launched Screen
`round92_unext2_d4` at `2026-08-31 00:02:06` (Asia/Shanghai). It selected GPU0
after seeing low utilization and about 26 GB free, but the real-data smoke
consumed the available memory and exited with `torch.OutOfMemoryError`; no
formal fold training or OOF result was produced. The original evidence is
preserved as `reports/round92_attempt1_pipeline.exit`,
`reports/round92_attempt1_scheduler.log`, and
`reports/round92_attempt1_smoke.log`.

The retry keeps the protocol and all model settings unchanged. Its scheduler
uses a fixed minimum of 36,000 MB free for smoke/training (rather than treating
26 GB as sufficient for this four-output UNeXt2 run) and enables PyTorch's
expandable CUDA segments. This is a resource-safety correction, not a model or
data change. The retry Screen `round92_unext2_d4_retry` launched at
`2026-08-31 03:55:49` and entered the real-data smoke on physical GPU1 with
`40,427` MB free.

## Completion audit

The retry exited with status `0`. Both fold logs contain the expected
`COMPLETE ...: 2 stages` marker. Each of the four stage directories has exactly
14 metric rows, `best_global_mse.pt`, `best_mean_image.pt`, `best.pt`, and
`last.pt`. The OOF covers `2706` validation images across the ten protocol ROIs;
its checkpoint selector is `best_global_mse.pt`, and the two referenced stage2
checkpoint hashes match the files on the server:

- fold0: `6aad6fd7b227604df5569461c25520c7062edf5b6e71aa0a5ecda843258a8f4c`
- fold2: `79baf11b382fa7b78553d398ff42566e8de62b2d62fc20cfd28ad51b504aff66`

The OOF SHA-256 is
`68e70ca9344d3e83dcb2b52798474a3d84baf01131418960a3fe35e7cc3d75a3`. The
remote gate report has the same OOF hash, and an independent local execution of
`evaluate_cv_gate.py` produced identical gate objects and promotion state.
The training logs contain repeated Python multiprocessing temporary-directory
cleanup warnings (`Errno 16`, resource busy) after completed stages; they did
not change the stage markers, exit status, checkpoints, or OOF/gate hashes.

## Result and decision

The direct D4-consistency candidate versus the official auxiliary baseline had
q95 deltas `[-0.017848, +0.018556]` overall and fold deltas
`fold0 [-0.066587, -0.034915]`, `fold2 [+0.027821, +0.068638]`; it failed the
fixed `+0.02` overall and two-fold requirements. Replacing the Round90
auxiliary member improved q95 by only `+0.015894` to `+0.023076`, below its
fixed `+0.03` gate. The half-split improved by `+0.011989` to `+0.015511`,
below its `+0.02` gate. All promotion states are therefore `allowed=false`.

This closes the D4-consistency branch. No five-fold expansion, test submission,
weight search, or leaderboard-guided rescue is justified. The next experiment
must test a materially different structural/data-scope hypothesis.

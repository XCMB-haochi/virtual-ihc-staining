# Round 90: Auxiliary/All-DAPI Endpoint with Context128 Inference

## Motivation

Round67 produced the only recent complete-OOF endpoint that combines two
different positive mechanisms: auxiliary-marker supervision and an all-train
DAPI FCMAE encoder. Its fixed 30-member candidate
tri_auxhalf_residual_half scored 75.610623 q95 and passed all five-fold
distribution checks, but it was never evaluated with the context128 inference
view.

Round85/88 independently evaluated context128 inference on the Base-diversity
endpoint. The OOF gain was +0.026427, and the fixed preliminary package
returned a positive hidden-set gain of +0.0214. Round90 asks one bounded
question: does that inference-only mechanism remain useful when applied to the
already-promoted auxiliary/all-DAPI composition?

## Fixed protocol

configs/round90_cv5_auxiliary_context128.json reuses exactly the 30 Round67
checkpoint files, fold assignments, candidate weights, D4 TTA, JPEG quality 95,
and output source offset [0,-1]. Every recipe is evaluated with source
context halo 128; no checkpoint is retrained, reselected, or reweighted. The
baseline is the immutable Round67 context64 OOF report. Test inputs and
test-derived statistics are disabled.

The fixed gate requires q95 gain at least +0.02, positive float gain, at least
4/5 positive folds, at least 18/25 positive ROIs, and a worst ROI change no
lower than -0.05. A passing result would authorize one fixed test inference
using the same 30-member recipe. A failed result closes this composition;
there will be no adjacent halo, weight, offset, or leaderboard search.

## Validation and launch

The new evaluator and gate perform immutable source OOF/gate hash checks,
re-hash all 30 checkpoint files, run a real-data six-recipe context128 smoke,
and compare the same fold/ROI universe. Local Bash/JSON/Python checks pass.
The remote preflight will run in the storage-scoped environment, and the
detached scheduler will choose a physical GPU from sustained utilization and
free memory without interrupting other users.

The source preflight passed on the storage-scoped server at runtime commit
1875237. A fresh resource sample selected physical GPU1 (GPU0 was carrying
another computation), and the detached Screen
round90_auxiliary_context128 launched at 2026-08-30 21:11:03 (Asia/Shanghai).
The real-data six-recipe smoke passed its resource gate; the full OOF is now
running on GPU1. The final runtime, OOF/gate hashes, and completion status will
be appended after the detached job finishes. This is a fixed
mechanism-composition screen, not a submission probe.

## Completion audit and result

The detached Screen exited with status `0` and no Round90 producer process
remained. The OOF covers all `6,296` training images and all `25` expected
ROIs. Each of the six source recipes has five complete stage-2 directories
with `14` metric rows and a loadable `best_global_mse.pt`; all `30` checkpoint
files referenced by the OOF report independently match their recorded
SHA-256 values. The final OOF and gate files are:

- OOF SHA-256: `44fe7a90ce712821ad80a1274443d74e38eaf3ba538fb3c8593cbd8ba4035d73`;
- gate SHA-256: `a8fb185946784df4ae017b6d48ee8d23a1a8b1fe77ca27d054ef3c77a2d486da`;
- OOF elapsed time: `7548.57 s`.

At JPEG q95, the fixed context128 candidate scores `75.748793`, versus
`75.610623` for the identical Round67 context64 composition. The q95 gain is
`+0.138170`; the float gain is `+0.138618`. All five folds improve:
`+0.135488`, `+0.103403`, `+0.155568`, `+0.122852`, and `+0.168722`.
Twenty-two of 25 ROIs improve. The two failing ROIs are `ROI005` (`-0.235068`)
and `ROI006` (`-0.080082`); the other smallest changes are `ROI010=-0.023724`
and positive `ROI020=+0.006478`. The aggregate, float, fold-count, and
positive-ROI checks pass, but the fixed worst-ROI floor `-0.05` fails, so
`promotion.allowed=false`.

This is strong evidence that the context128 inference view composes with the
auxiliary/all-train-DAPI endpoint in aggregate, but it is not a stable global
replacement under the preregistered distribution rule. No test prediction,
weight rescue, ROI routing, alternate halo, offset, or leaderboard probe is
authorized from this result. The next structural route remains a fresh
current-UNeXt2 data-efficiency screen, followed by the previously untested
UNeXt2 D4-consistency screen.

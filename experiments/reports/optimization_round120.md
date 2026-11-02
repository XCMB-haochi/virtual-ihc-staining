# Round120: Clean Primary-Gradient Projection Rerun

Round119 used unscaled primary gradients and unscaled total gradients, but its
non-finite projection branch wrote `inf` into one gradient before GradScaler.
Round120 removes that write; a non-finite projection statistic leaves the
original total gradient untouched and lets GradScaler apply its normal skip.

The protocol is otherwise byte-for-byte the Round119 recipe: Tiny UNeXt2,
fold-local Round52 FCMAE800, auxiliary15, fold0/fold2, seed2044, effective
batch16, 14+14 epochs, context64 training, context128+D4/q95, global-MSE and
offset `[0,-1]`. Gate and candidates are unchanged. This is an implementation
correction, not a new projection coefficient or parameter search.

Only GPU1 is allowed. The queue waits for existing GPU1 computation to exit,
runs a real two-stage smoke and then trains the two folds sequentially. The
Round119 OOF remains an audit artifact and is not used as the corrected result.

## Queue Registration

Local and isolated Linux tests pass 147 cases, and the protocol preflight passes
with the corrected source. Screen `round120_projection_clean` is detached with
`gpu_candidates=[1]` and `wait_for_empty_gpu=true`; its state is
`waiting_for_smoke`. It will not touch GPU0 or start formal training until GPU1
has no existing compute process and the real two-stage smoke passes.

## Completion (2026-09-10)

Complete-fail: exit 0; fold0/fold2 each contain both 14-epoch stages; all
selected weights are finite. Deployed source hashes match the launch receipt,
and all four OOF checkpoint hashes match their files. Complete OOF covers
2,706 images / 10 ROIs; local gate replay reproduces every decision.

| Comparison with Round55 auxiliary15 | q95 delta |
| --- | ---: |
| Overall | -0.172130 |
| fold0 | -0.278694 |
| fold2 | -0.072278 |
| Worst ROI (ROI022) | -0.686034 |

Candidate q95 is 74.841143 versus 75.013272; float delta is -0.169785.
Only 2/10 ROIs improve. This is a clear negative, not a borderline gate miss.
Close the registered projection method; no five-fold expansion follows.

All four selected model state dictionaries are tensor-identical to Round119,
although they have distinct run/config artifacts and checkpoint file hashes.
Round120 did run with the corrected source and fresh 14-row trajectories;
removing the forced-inf write did not alter these selected models. Therefore
this rerun resolves the implementation concern but is not an independent
seed confirmation. Projection skips are 4/4/5/4 across fold0 stage1/stage2 and
fold2 stage1/stage2, confined to epoch1; these counts do not by themselves
indicate contaminated weights. The clean branch preserves original gradients
and leaves actual overflow handling to GradScaler.

Measured training time totals 5.72 hours (one stage suffered contention);
OOF took 26.42 minutes. Operational logs remain on the server. Durable artifacts:
`round120_launch.json`, `round120_pipeline.exit`,
`round120_cv2_primary_projection_clean_oof.json` and its gate JSON.

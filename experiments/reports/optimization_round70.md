# Round 70: Base and Composition Interpolation

## Motivation

Round67 and Round69 independently pass complete-OOF gates using different
mechanisms. Round67 combines auxiliary supervision with all-train-DAPI
pretraining for `+0.050199`; Round69 adds ConvNeXt-V2 Base diversity for
`+0.083681`. Round70 tests whether their endpoint errors compose, without new
training, leaderboard feedback, or a continuous weight search.

## Fixed Candidates

Both endpoints share the fixed core: base32 `0.21`, base48 `0.09`, and scratch
FCMAE800 `0.35`. The remaining `0.35` pool is interpolated at three
preregistered Base-endpoint shares:

- `blend_base33`: one-third Base endpoint and two-thirds Round67 composition;
- `blend_base50`: the exact midpoint;
- `blend_base67`: two-thirds Base endpoint and one-third Round67 composition.

The midpoint has first priority because it is the unique symmetric combination;
the Base-favoring and composition-favoring alternatives follow. This bounded
three-point comparison still reuses the same complete OOF for selection, so it
is treated as a small multiple-comparison screen rather than independent proof.

## Gates

Any blended candidate must gain at least `+0.005` q95 over the already promoted
`t70_base_split`, remain positive in float, improve at least 4/5 folds and
16/25 ROIs relative to Base, and keep its worst relative ROI above `-0.04`.
It must also gain at least `+0.085` over `t70_o50`, improve all five folds and
20/25 ROIs, and keep its worst baseline-relative ROI above `-0.08`. If no
interpolation passes both checks, the independently validated Base endpoint is
retained without changing its weights.

The launcher binds Round67 OOF/gate SHA-256 values
`8551eac7ca9f57bd4de716ba9dc98cacc161ca3a249d1d3aee614a1a3a6301c2` and
`cd918b3333b9e33267e81449cc6a8c399bd383ae06ac39812a72c460b1f6dedc`,
plus Round69 OOF/gate SHA-256 values
`cf3b9af9ae5b2da3757107bcec5b8d1aa6b013d0e2181369df6cc9ee9e2b9c53` and
`fb5994f347e440ddf69d4c16f821ef09286264bb2d92e72c4c0e68d5112f9d77`.

## Status

Implementation is complete. All 71 focused local protocol tests, Python
compilation, launcher syntax, and diff checks pass. Linux validation, the
no-side-effect source preflight, and formal OOF launch remain pending.

On Linux, all 285 tests pass. The no-side-effect preflight validates all four
bound source hashes, both exact promotion objects, complete 6,296-image/25-ROI
coverage, and matching retained baselines. All 35 unique checkpoint files in
the union of the two promoted endpoints match their recorded SHA-256 values,
and no Round70 OOF, gate, or exit artifact exists before launch.

Round70 launched from commit `82e721e` in detached Screen `round70` at
2026-08-25 14:07:49 +08:00. Five launch-time samples found both GPUs at 0%
utilization, but GPU0 had 13,219 MB resident while GPU1 had only 17 MB, so the
Base-containing OOF was assigned to GPU1. Initial verification showed the
Round70 evaluator alive with the fixed protocol and GPU1 at 99--100%
utilization. An unrelated GPU0 task then rose to about 67% utilization and
25,409 MB resident; Round70 does not use or interfere with it. Completion and
gate status remain unverified.

## Result

The pipeline exited `0` and evaluated all 6,296 images from all 25 ROIs in
9,072.54 seconds. All 35 checkpoint hashes match the server files. The retained
baseline and both promoted source endpoints exactly replay Round67/Round69 at
overall, fold, ROI, and provenance levels. Independent local gate evaluation
exactly reproduces the saved report. The OOF SHA-256 is
`e3e854ecf193e95fe0b1d9b721a87361e98c3ac3b0c11684f302d22f18eec372`;
the gate SHA-256 is
`7cd5de3e89be461134056072279efe848a85bde7b98c781945883ab44ffc6bad`.

No interpolation improves the Base endpoint. `blend_base33` scores
`75.632775` (`-0.011330` versus Base), the midpoint scores `75.639852`
(`-0.004253`), and `blend_base67` scores `75.643899` (`-0.000206`). Every
advance gate fails, including float, fold, and ROI requirements. The exact Base
endpoint remains `75.644105`, passes its retained complete-OOF gate, and is
selected as `promotion.candidate=t70_base_split`.

This closes weight interpolation. No additional blend, calibration, or
leaderboard-guided candidate will be created. The unique fixed next action is
test inference and packaging of `t70_base_split` using its 25 members.

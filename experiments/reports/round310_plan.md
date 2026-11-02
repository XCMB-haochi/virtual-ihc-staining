# Round310/311: independent A100 control and DAPI-gain experiment

User restored campus-direct A100 access and authorized independent useful work;
do not interrupt or split the currently running5090 Round308 queue. No teacher
offload was launched. Two A100s can instead run a matched short stage2 control.

Question: does mild training-only input gain improve held-out-ROI generalization
of the selected Round307 recipe? Small independent ROI counts and lack of any
current input photometric augmentation motivate this bounded test. Gain can also
remove useful intensity cues, so no improvement is presumed. Historical reports
contain intensity correction/postprocessing and texture experiments, but no
matched instance of this exact training-only gain configuration was identified.
No Test statistics or scores set the gain range. R03 permits DAPI-only brightness
augmentation; paired target labels are kept unchanged. No organ input is added.

Both arms start from the exact Round307 stage1 epoch14 model and RNG, fresh
stage2 AdamW as before,14-epoch horizon lr1e-4, stage decay0.8, BF16, accumulation16,
clip1, uniform7, same fixed grouped split, mixed markers and organ-loss weights.
Control Round310 is unchanged; candidate Round311 multiplies training DAPI by
Uniform[0.9,1.1] and clips to[0,1]. A separate Random(2026+100000*epoch) stream
preserves sample order and paired geometry RNG. All organs use the same gain
distribution. Validation/inference use unmodified DAPI, no gain, TTA or correction.

Train once per image/epoch, no epoch/strength/seed grid. Evaluate all14 stage2
endpoints; compare best weighted Val and each organ/ROI. A candidate must exceed
both its matched A100 control and retained Round307 to warrant later review;
no one-ROI veto, no automatic refit or submission. Controls on the same GPU family
help distinguish the intervention from the move off5090; bitwise equivalence
across hardware is not promised. Completed800 encoder remains unchanged.

Dedicated compute workspace under
/workspace/virtual_ihc_staining_generation/transfers/round310_a100.
Transfer only the20712 Train/Val DAPI+selected-marker images, source snapshot,
encoder and stage1 checkpoint. Verify every input byte hash and decoded pixels
for96 fixed samples; preserve original Windows split strings in checkpoints and
translate separators only for actual I/O. Train filenames/Val exclusion unchanged.
GPU1 receives candidate; GPU0 control may share compute with a process observed
after initial availability. Recheck free memory/utilization before launch and
estimate time from smoke, not nominal A100 speed. Other processes remain untouched.
Use two detached Screen sessions and durable logs/checkpoints/exit files.

First run local/remote gradient/RNG-input tests, checkpoint/split/image checks and
one32-sample stage2 execution smoke per arm. Full estimate after measured timing.
No new pretraining, no dependency of5090 training on these jobs or local SSH.

October29 setup: local and A100 unit tests passed; shell launch syntax passed
after enforcing LF. Final source archive SHA
7f742d1ba728d057a293b90d92e5aef1221e6214976c23e6844aa2e508d951bd verified;
commit in round310_launch_commit.txt. Input archive646502400 bytes, SHA
4acc45b53e5fc98d271599384a9a08f0a1abb926df04f36f7cb69e4f41d11d75.
Transfer still in progress at handoff; don't claim data/GPU smoke passed yet.
Local Scheduled Task AIC-Round310-Transfer tracks the existing SCP, retries a
failed copy and publishes readiness only after remote checksum. Detached Screen
round310_a100 executes prepare_round310_a100.py with durable status/logs; it
extracts/verifies inputs, runs both GPU smokes, then rechecks both GPUs three
times before scheduling dual or GPU1 sequential. This is independent of the
5090 queue and survives loss of the interactive SSH connection. Software checks
cannot prevent shared-host shutdown/session termination; per-epoch resume exists.

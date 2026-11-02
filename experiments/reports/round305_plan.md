# Round305: queued uniform-CD68 control under the Round303 recipe

October29 user prefers waiting until morning before consuming a submission slot
and authorizes useful overnight experiments within23.5h. Do not submit Round303
automatically. Round304 continues unchanged and its package downloads first.

One bounded control: all organs/CD68 with the exact Round303 grouped split,
800-epoch encoder, architecture, seed2026,14+14 traversals, loss, LR groups,
precision, accumulation and validation. Only Liver supervised target/marker ID
changes. The previous0.943 mixed-marker advantage was under a different recipe;
interaction with the new recipe is unresolved. No new pretraining or Test tuning.
Train/Val filename order and augmentation RNG are unchanged by the fixed marker
choice. Compare best28 scheduled Val endpoints to Round303 best76.0908141.
Report per-organ/ROI, including declines; only3 Stomach ROIs remain a limitation.

If Round305 weighted Val is strictly higher, run one full-data refit (Round306)
from the same encoder, retaining the original14-epoch cosine horizons and
stopping at Round305's selected epoch. No additional ROI/fold gates or threshold
grid. Exact ties retain mixed Round304. If no improvement, end the queue.
This selection does not guarantee a platform gain. Preserve both independent
models as artifacts; each output package always uses only one checkpoint.

Remote AIC-Round305 waits for Round304's packaged manifest, then runs the control,
audits fixed settings and steps, decides from Val, optionally trains/packages
Round306. No dependency on a local SSH connection. Local AIC-Round306-Download
waits for a ready package or the not-selected decision; no model heartbeat.
Round306 delivery is submissions/semifinal_round306_cd68_q95.zip.

Estimate from measured throughput: remaining Round304 about3h, control2.5h,
optional refit up to3.3h, plus load/audit/transfers; budget9-10h from00:30 Beijing.
Queue dependency timeout12h; task time limit20h. External failures may still
require intervention; do not represent projected delivery as guaranteed.

Launch: six local/remote tests passed. Both grouped and full-CD68 two-stage
smoke paths passed real-data finite-loss/weight/optimizer-count audits.
Archive SHA3fea6e69975f0eec3db1d276a4fbad21eaeeceb9f7782038ab13db9654d4f45c
verified before deployment; commit captured in round305_launch_commit.txt.
Remote AIC-Round305 and local AIC-Round306-Download scheduled tasks started.
Round304 remains running with its already loaded training implementation.

Handoff verification: dependency/win/loss/tie control-flow integration tests
passed locally and on5090 with isolated fixtures (not new model results).
Both actual GPU smoke paths were already successful. A real existing Round303
package was downloaded through the delivery code into transfers/round305_delivery_check
and its hash/2186-member ZIP passed. Status queries return WAIT successfully.
Completion manifests and decisions now publish atomically; download network
failures retry and record connection state. Waiting Round305 was restarted to
load this control-only revision, with no production Round305 config present;
Round304 training was not interrupted. Local download tasks reloaded too.
Exact control revision is in round305_handoff_commit.txt. Hardware/session
shutdown remains outside this software check; do not promise zero failure risk.

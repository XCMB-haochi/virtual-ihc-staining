# Round304: full-Train refit selected by Round303 Val

Same model, fixed markers CD68/CD45RO/CD68, seed2026, loss, precision,
accumulation16, clip1, augmentation, optimizer and stage decay as Round303.
Use all official Train: Colon6296, Liver2380, Stomach1680 (10356 total).
Preserve original selection split separately. No Test images enter training.
Reuse unchanged Round302 encoder trained on grouped Train8084; newly included
Val images enter supervised fitting only. This is not full-data pretraining.

Fresh supervised initialization; do not continue from the fitted Round303 model.
Stage1:14 full traversals, cosine horizon14, lr3e-4. Reset AdamW as in Round303.
Stage2:13 full traversals of the original14-epoch cosine horizon, lr1e-4.
Fixed global endpoint27 from Round303 best grouped Val. Do not compress the
second-stage cosine to13 or choose checkpoint from training loss/platform.
This transfers epochs (per-image exposure), not optimizer-step count, to larger
Train. Expected279612 exposures and17496 updates; stage2 optimizer steps8424.
No validation during full fitting. Preserve resumable state every epoch.

Expected3h at measured27 images/s plus load/save. GPU availability checked at
launch. Remote Scheduled Task directly executes round304.py; that same process
audits training then calls package_round302.main(304). One checkpoint,2186
JPEG95 RGB results. Local Scheduled Task download_round304.py waits for the
package with lightweight file checks/reconnects and verifies SHA256/ZIP before
moving the completed file into local submissions. No model heartbeat, no
dependence on a long-lived local SSH session, no automatic platform submission.

No expectation of a guaranteed gain; single grouped validation remains limited,
especially3 Stomach ROIs. Review this full-data result before another experiment.

October29 launch: four schedule/data-coverage tests passed locally and remotely.
Two-stage GPU smoke passed real-label loss, finite weights, optimizer counts and
fixed endpoint checks. Git archive SHA332e2f88139f28c045e92f6368ca524ae9f2ab970e001606c4a274faca10aa53
verified before extraction. Source commit in round304_launch_commit.txt. Both
remote training and local download Scheduled Tasks started; no changes to old
pretraining weights. Round303 package delivered and checked before launch.

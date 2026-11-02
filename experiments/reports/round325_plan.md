# Round325: marker heads from stage1

Round322 split the CD68/CD45RO output heads only at the stage2 boundary and
scored76.2612881 versus31876.2672365 (-0.0059484). That does not test whether
the shared stage1 output head already created target interference. Round325
keeps the shared FCMAE encoder/UNeXt decoder but creates identical CD68/CD45RO
heads before stage1. Colon and Stomach share CD68. Each task executes one head
selected only by target marker; R05 explicitly permits target output heads.

Train from the fixed Round302800 encoder and seed2026 with no supervised
checkpoint initial. Stage1 hard-label14 epochs; stage2 teacher0.2 and
training-only D4 consistency0.2, LR/organ weights/order/teacher cache matching
Round318. Validate complete images on the same grouped split. No organ/file
input, multiple outputs, ensemble, TTA or postprocessing. Initial two heads are
bitwise copies of the original head, so the architecture change is isolated.

This is a bounded validation test, not an automatic full refit. A queue waits
for Round324 completion, runs smoke, then the full grouped Val test. If its
weighted score does not exceed318, retain319/324 and stop. If it clearly wins,
register a separate fresh all-Train refit from the fixed encoder; do not extend
the grouped checkpoint to allTrain. Test/leaderboard feedback is not used.

Expected grouped run about5h on5090; it occupies the overnight window after
324 and should finish well before November1 20:00. No EMA is scheduled because
its historical gain was negligible and its semifinal R09 boundary remains
unclear.
October31 23:55 Beijing: remote archive verified and current smoke rerun passed after correcting stage-reset Adam audit; prior traceback in append-only err log was stale from the first audit version. AIC-Round325-Queue is running and waiting for runs/round324/complete.json, then runs smoke and grouped Val. No full refit is automatic.

November1 03:00 review: Round324 finished packaging at02:07. Round325 had
already completed five stage1 epochs with unintended consistency0.2. Stopped
only the Round325 queue/processes and preserved runs/round325 and its smoke
under *_superseded_consistency. Revision6741842 restricts from-start-head
consistency and checkpoint selection to epoch15 onward, asserts the actual
per-epoch loss schedule, and waits for324 submission manifest before launch.
Three deployed source SHA256 values matched local files. Restarted the existing
hidden queue with a fresh two-stage smoke before formal training. No pretraining
rerun and no automatic full refit. 08:00 assistant review remains scheduled.

Corrected smoke completed all14 short stage1 epochs and one stage2 epoch;
actual loss-schedule, stage2 optimizer and decision-report assertions passed.
Formal runs/round325/config.json confirmed smoke=false, initial=null,
consistency_start_epoch=15 and selection_start_epoch=15. Fresh formal training
started after smoke. Append-only round325.err still contains the historical
pre-correction audit traceback; it is not a current-run error indicator.

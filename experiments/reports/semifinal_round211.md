# Round211: restore full batch16 relational comparison

The user clarified that the GPU1 memory reservation had expired and
authorized a rerun without compromising the comparison. Round210 batch8
is a valid local ablation but cannot decide whether the original full
batch16 relation mechanism transfers. Round211 repeats both no-guidance
and relational students on fold0/fold3 with full batch16, accumulation1,
and all16 samples in the relation loss (120 pairs).

Reuse the unchanged completed fold-local Round210 five-modality160 teachers
and Round207 DAPI800 encoders, verifying hashes and validation exclusions.
These are fixed prerequisites, not batch8 student weights. Each student
stage1 starts fresh from the same domain encoder; stage2 loads its own
Round211 stage1 epoch14. No Round210 student or optimizer state is reused.
Keep all other settings and final stage2 epoch12 fixed. This restores the
batch-dependent relation calculation, not every detail of the Colon recipe.

Compare both recipes by D4/halo128/offset0/q95 on the same980 held-out images,
using the same paired reporting as Round210. Candidate must be considered
against both its matched control and the retained Round207 baseline.
No automatic submission or full-data promotion; Round20878.9741 is preserved.

Real batch16 two-stage smoke passed with6/4 actual optimizer updates,
positive finite guidance, and peak allocations26.60/26.73GiB. A focused
command test verifies full batch16, unchanged teacher paths, and fresh
student initialization. No teacher pretraining is repeated.

Detached `round211_batch16` waits only for the currently running fold3
teacher to finish. The Round210 Python scheduler is suspended so it cannot
launch further batch8 students; its teacher child continues normally. After
teacher completion and child exit, terminate the old scheduler and start
four fresh batch16 students. The partial batch8 experiment is superseded,
not reported as complete. Teacher metadata/exclusions are checked and hashes
recorded directly even if Round210 never wrote its final source receipt.
GPU1 only,31,000MiB free-memory requirement for this workload;
there is no reservation for another task. Allow roughly4-6h after the
existing pipeline finishes; this is an estimate, not a fixed deadline.

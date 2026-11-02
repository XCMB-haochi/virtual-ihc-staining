# Round215: nested teacher preparation for valid CD45RO distillation

Round214 showed no material matched improvement. Pixel distillation remains
unverified, because Round212 failed through target coverage, wrong-channel
routing and an invalid proposal to use global OOF teachers inside outer-fold
student validation. This round builds the correctly isolated prerequisites;
teacher completion alone is not a distillation result.

Outer validation: original fold0 and fold3, unchanged. For each outer fold,
materialize a dataset containing only the other14 or13 full training ROIs,
using hardlinks to original unchanged training images. The test/DAPI directory
is empty and exists only to satisfy the data-root contract. No outer-validation
image or actual test input exists in this teacher dataset, including during
condition construction or validation. Never edit the linked source images.

Partition that outer-training ROI list lexically into five groups using
round-robin assignment. Each inner teacher pretrains on DAPI excluding its
inner holdout, then performs14+14 supervised training on the same allowed
ROIs. Inner holdout is the only teacher validation set. Keep stage1 epoch14
and stage2 epoch12 fixed; no inner-best checkpoint selection. Every outer
training ROI is held out by exactly one teacher, so the later target cache
can cover all1,960/1,820 student training images once with both exclusions.
The nested teacher training subsets are smaller than the student set, as
required for cross-fitting; a future screen result is conditional on these
teachers, not every possible distillation design.

Ten teachers total. DAPI FCMAE800, immutable public epoch0 initialization,
seed2026, full batch32/accumulation1, same effective pretraining batch and
schedule. Tiny domain-initialized CD45RO-primary supervision, batch16,
uniform7 loss, Gaussian11 validation. Periodic FCMAE resume checkpoints and
supervised per-epoch weights are retained. Do not reuse the outer pretrained
encoder for inner teachers: it has seen their held-out DAPI images.

The student's later comparison is fixed now: reuse each Round211 control
stage1 epoch14, reset stage2 optimizer, unchanged14-epoch cosine schedule,
batch16, marker order and supervised loss; add0.2 pixel distillation on
CD45RO only, evaluate epoch12 with D4 versus the completed matched control.
Do not train this student until all targets, provenance and disjointness have
been verified. No submission, ensemble or final-model promotion in this job.

Trainer repair adds explicit `--distillation-marker` and derives its output
index, preserving historical first-channel default. A gradient-routing test
proves a CD45RO teacher supervises channel1 only, not CD68. Missing target
coverage now fails before model training. Nested partition tests verify exact
coverage and both exclusion boundaries; historical distillation tests pass.

Cost is materially higher than the discarded Round212 estimate. A full
batch32 FCMAE smoke measured3.139s per256 images after startup,15 actual
updates, and20.76GiB peak allocation. With the10 nested subsets' combined
15,120 images per epoch,800 epochs extrapolate to about41h pretraining;
supervision adds roughly9h. Allow approximately45-55h depending on load and
I/O. This is a one-time fixed teacher bank, not a parameter search, and leaves
time before the October5 deadline. GPU1 exclusive; GPU0 untouched.

The corrected student also completed a128-image execution smoke with4
actual Adam updates and saved `distillation_marker_index=1`. Those smoke
targets were copies of permitted training CD45RO labels solely to exercise
the interface; they are never used for formal distillation or efficacy claims.

## Superseded by user-authorized direct full-data route

The user approved stopping the expensive nested expansion in favor of
completing three global teachers and performing one fixed full-data student
confirmation. The first nested teacher was stopped with SIGTERM after its
parent scheduler was suspended; partial checkpoints/logs are retained.
No nested teacher is repurposed as a global fold teacher. This run is
incomplete, not a negative distillation result. Round216 supersedes it.

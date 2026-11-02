# Round212: CD45RO pixel-distillation screen

Round211 closed full-batch relational guidance. Round212 tests the remaining
unverified component from the Colon route: fold-excluded CD45RO pixel targets
from the completed Round207 domain teachers. Targets use D4, halo128 and no
output offset, are stored as lossless uint8 PNG, and cover only the held-out
ROIs for each teacher exactly once.

The intended student reused the Round211 batch16 no-guidance stage1
checkpoint for the same fold and added stage2 distillation at weight0.2.
The attempted implementation does not establish that comparison; see the
design and channel errors below. Do not relaunch this runner as written.

## Attempt outcome

Pipeline exit1 before the first student completed. The target builder wrote
980 held-out images from fold0/fold3, but the student training set contains
the other training ROIs as well; it therefore failed on a missing target
(`ROI076_05_08.jpg`). Fold0 student training needs targets for its1,960
training images; fold3 needs1,820, with valid provenance for each. No
checkpoint or submission was promoted.

A subsequent code audit found another error: teacher targets are CD45RO,
but `training_batch_losses` applies pixel distillation to `prediction[:,:1]`.
The configured first output is CD68. Filling the missing files alone would
silently train the wrong channel. Correct marker routing requires explicit
validation and a gradient-routing test before any rerun.

Global five-fold OOF targets can train a final full-data student, but do not
automatically support unbiased outer-fold student validation: teachers for
other folds may have trained on the student's held-out validation ROIs.
A valid outer-fold experiment must exclude those ROIs throughout teacher
pretraining, teacher supervised training and target generation. If inner
OOF teachers are used, they also exclude the target image's own ROI.
Simply completing the remaining three global teachers is therefore not a
sufficient repair of this two-fold screen. The prior cost estimate was not
based on a correct validation design and is withdrawn.

Status: **implementation failed; semifinal efficacy untested; method not
rejected**. Historical matched single-model Colon platform comparison was
Round13474.1432 versus Round13674.1985: a positive gain of0.0553. This does
not bound the possible gain on new data and is insufficient grounds to
reject semifinal distillation. The earlier decision to close the method is
withdrawn. A corrected experiment must specify coverage, marker routing,
teacher/student exclusion boundaries and measured cost before launch.

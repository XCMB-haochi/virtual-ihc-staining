# Round 95: Five-Fold Texture Thumbnail Expansion

## Decision

Round94 passed its fixed fold0/fold2 texture-thumbnail screen. Round95 therefore
performed the one authorized unchanged expansion: reuse the audited fold0 and
fold2 candidate checkpoints, train only folds1, 3, and 4, then evaluate one
complete 25-ROI OOF comparison against the fixed Round55 auxiliary baseline.
No texture channel, loss weight, epoch, checkpoint rule, output offset, blend,
ROI route, or leaderboard result was changed after the screen.

## Completion Audit

The detached pipeline exited with `reports/round95_pipeline.exit=0`; its Screen
session and producer processes are gone. Folds1, 3, and 4 each completed two
14-epoch supervised stages. Every new stage contains consecutive metric rows
from epoch 1 through 14 and all four required checkpoints. Fold0 and fold2 are
symbolic links to the exact audited Round94 runs. The selected stage-2
`best_global_mse.pt` SHA-256 values are:

- fold0: `0d1ce37609ca19225a75065bc7c245e78d373b87247ef0294429989b516ea6c3`;
- fold1: `65b949debe68ab05491cb80ca6ea8bcce7199b9586cc9450cbc4c87ae47579f2`;
- fold2: `c126199f46795a98e888e8238a4238c7e34b7cdf4aab89ba02c7f481b606c1bc`;
- fold3: `911befac83f018eb2b166cd7ad351ebc5be58121bacb6c2577495b53fe5210fc`;
- fold4: `9bc83ae4b084ce4b44e0c77ff4fe84825f17f2bc9c3533d8ae555abc458cf1ad`.

All five fold-local FCMAE initialization files are hard links to their
registered Round52 encoders and reproduce the registered hashes. The complete
OOF covers all 6,296 images and all 25 ROIs in `3617.59 s`. Its SHA-256 is
`7a2e2d304632683bb48387787bcf8a3afed42bf38662baf99434af9b334f4dae`;
the gate SHA-256 is
`68aaa2b6b81358efdeb40778ac5a0b38c19e463c30b4440af844e9857cca7506`.

The repeated `pymp-*` `EBUSY` tracebacks occur during successful DataLoader
temporary-directory cleanup. They do not change the zero pipeline exit,
complete metric rows, checkpoints, OOF coverage, or report hashes.

## Result

At JPEG q95, `textureauxiliary15_context128` scores `75.431648` versus
`75.411811` for `officialauxiliary15_context128`, a gain of only `+0.019837`.
The float gain is `+0.021885`. Fold deltas are:

- fold0 `+0.002875`;
- fold1 `-0.053668`;
- fold2 `+0.142090`;
- fold3 `+0.015371`;
- fold4 `-0.028090`.

Only three of five folds and 12 of 25 ROIs improve. The worst ROI is ROI015 at
`-0.935751`; ROI001, ROI005, ROI006, ROI009, ROI011, and ROI014 also have
material negative changes. The candidate fails the fixed overall-q95,
positive-fold, positive-ROI, and ROI-floor checks, so
`promotion.allowed=false`.

## Boundary

The texture-thumbnail branch is closed. The result does not authorize a lower
gate, texture-channel variant, post-hoc blend, ROI routing, test inference, or
leaderboard probe. The two-fold pass was dominated by fold2 and did not
generalize to the remaining folds; this is exactly the failure that the
five-fold expansion was designed to detect.

The next structural investigation must use a materially different prior. A
newly released spatial-proteomics foundation model may be relevant only if its
official weights and reconstruction interface can condition on DAPI alone and
produce or initialize CD68 reconstruction without test-label information.
Feasibility must be established before registering a GPU experiment.

# Round126: Independent Seed for Median Relational Guidance

Round124's fixed five-fold relation package scored `74.1630` on the hidden
platform, `+0.1315` over the verified Round88 package, while remaining below
the recorded external leaderboard high `75.4966`. Its complete OOF gain was `+0.094465`, with
four positive folds and one large negative fold. This supports a small hidden
benefit but leaves room for trajectory variance.

Round126 changes exactly one training variable: the supervised student seed
from `2044` to the pre-existing independent seed `2027`. The median-normalized
stage-4 CD68 relational guidance, weight `0.05`, fold-local Round96 teachers,
Round52 DAPI FCMAE800 initialization, four-marker auxiliary weights, model,
context, epochs, optimizer, checkpoint selector, TTA, JPEG quality and output
offset remain fixed. No teacher is retrained, and no test information is used.

The screen uses fold0/fold2 and the unchanged auxiliary15 controls. It requires
the same direct gate as Round124: q95 gain at least `+0.05`, positive float,
both folds positive, at least 7/10 positive ROIs, and worst ROI no lower than
`-0.12`. A failed screen closes this seed branch. A pass permits one fixed
five-fold expansion; it does not authorize seed, weight, epoch or ROI grids.

This is a diversity screen, not a response to the hidden score. Seed 2027 has
prior independent-seed evidence in Round84, but that earlier route produced
only small OOF gains and does not predict this relation recipe. Expected cost
is about 3.2 hours of two-fold student training plus 25--30 minutes OOF on an
uncontended GPU1. The then-current personal-best Round124 package remains
untouched.

The protocol, source archive and preflight were verified on the storage-scoped
Linux runtime. A real two-stage seed-2027 smoke completed four optimizer steps
per stage with finite weights and the expected 26.58/26.70 GiB peaks. An initial
formal `--queue` invocation correctly refused to recreate the existing smoke
directories and exited before training. The formal run was then launched using
the already verified smoke receipt, with source commit
`28c0a9881934ec776eb454167a248cb061a8f73e`, on GPU1 only. Its output is in
`transfers/round126_runtime/reports/round126_screen.log` and will produce the
two-fold OOF and fixed gate before any expansion decision.

## Completion (2026-09-14)

Round126 completed with exit `0`; all four supervised stages have 14 epochs,
finite checkpoints, and matching provenance hashes. The OOF covers 2,706
images / 10 ROIs and the gate replay passes every condition:

| metric | result |
| --- | ---: |
| q95 gain vs auxiliary15 | `+0.293073` |
| float gain | `+0.297946` |
| fold0 / fold2 | `+0.278796 / +0.306451` |
| positive ROIs | `9/10` |
| worst ROI | `-0.058121` |

The independent seed substantially improves the two-fold screen over the
seed-2044 relation result. This authorizes exactly one unchanged five-fold
expansion. Existing fold-local teachers are reusable; fold1/fold3/fold4
students will be trained with seed2027, followed by complete 6,296-image OOF
and the fixed gate. No seed or weight search is opened.

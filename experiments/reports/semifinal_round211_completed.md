# Round211 completion: full batch16 relational control

Round211 was rerun after the temporary GPU memory reservation expired. The
previously trained fold-local multimarker160 teachers and Round207 DAPI800
encoders were hash-checked and reused; both batch16 students were freshly
initialized. This isolates relation-batch behavior without inheriting the
superseded batch8 student states.

All four supervised trajectories completed 14+14 epochs with finite metrics,
actual optimizer updates, batch16, accumulation1, and peak allocation about
26.73 GiB. D4/halo128/offset0/q95 evaluation covered 980 images and seven
held-out ROIs.

For CD45RO, the batch16 no-guidance control gains **+0.15298** over the
Round207 domain baseline, while the relational candidate is **-1.53250**
relative to that baseline and **-1.68547** relative to its matched control.
Both folds decline. The relational candidate also declines for CD68,
HLA-DR and Vimentin. The tested checkpoint is not promoted; this alone does
not establish that the method family is ineffective. See the implementation
and optimization audit below. No submission follows this failed candidate.

The control change is a two-fold development estimate with mixed fold
changes; its absolute OOF score is not directly comparable to Round208's
platform score. Round208 remains the submission reference. Do not tune
against the platform to rescue this candidate. A concrete implementation
or optimization diagnosis can justify a separately registered correction.

## Re-audit after the Round212 implementation failure

Compared actual saved configs across all eight stage runs: apart from
guidance options and artifact paths, both arms match. Full batch16 with
accumulation1; stage1 uses the same immutable DAPI encoder, with no student
init checkpoint; stage2 loads each arm's own Round211 epoch14. No batch8
student/optimizer is reused. Teacher hashes match recorded sources,
epoch160, five modalities and fold exclusions; each teacher saw exactly
five times its fold-training image count. The shared filesystem versions
of the trainer, model, loss and data modules match local code after line
ending normalization. Compared with the initial-stage source commit, the
relation formula is unchanged; only frozen-teacher RNG isolation was added.

The saved optimizer states confirm about1,589-1,718 actual Adam updates
per stage, with4-7 AMP-skipped updates rather than a failed training loop.
Four real batch16 checks (two deterministic batches per fold) hook the
actual teacher input: it equals the central CD45RO target crop, index1.
An independent explicit120-pair distance and lower-median implementation
matches the loss within7.5e-9 and the feature-vector gradient within4.8e-7.
These checks directly address the wrong-channel bug found in Round212.

However, fold3's relational checkpoint shows an optimization warning:
pooled student-feature pair-distance medians about0.124-0.127, compared to
fold0's23.8-26.7. The no-guidance fold3 control on those same batches has
medians29.5-32.7, over200 times larger. On the sampled fold3 batches, weighted relation gradients
at the raw bottleneck dominate the reconstruction gradients by millions.
The diagnostic uses scaled backward to avoid unscaled FP16-gradient
underflow; teacher medians are nonzero, with no zero-distance pairs in the
samples. This is evidence of a possibly degenerate representation or poor
objective balance at the trained endpoint, not proof of its origin.
Four sampled batches do not reconstruct the entire training history and
cannot establish whether gradient imbalance caused or followed degradation.

Conclusion: no batch-handoff, channel-routing or pair-formula error found
in the audited paths; the -1.68547 checkpoint outcome stands. Withdraw the
broader assertion that this proves relational guidance cannot transfer.
Optimization stability remains a specific open question. This audit performs
no optimizer step and changes no existing training weights or submission.
Evidence: `round211_implementation_audit.json` and its reproducible audit script.

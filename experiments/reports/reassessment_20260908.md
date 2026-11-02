# Plateau Reassessment — 2026-09-08

Scope: inspect the active pipeline and existing evidence within the user-approved
one-to-two-day maximum. Preserve the best submission. No large/huge model code,
experiments or automatic training chain. CPU diagnostics run in an isolated
checkout under `transfers/reassessment_20260908`.

## Confirmed Defects

1. **Sparse diagnostic conditions differ from training conditions.** Spatial
   statistics and thumbnail builders use the supplied filenames as their whole
   context universe. The Round113 knockout selected 8 images/ROI, so 80/80
   query vectors differed from normal complete-ROI conditions (mean absolute
   difference 0.380418 across the vectors). With the same existing Tiny
   checkpoint and two fixed images, changing only those conditions produced
   prediction RMS 6.950509/255 and q95 71.005753 versus 75.559717. This is a
   two-image input-contract probe, not a model improvement estimate.
   Evaluation, prediction and gradient-diagnostic entry points now compute
   conditions from complete DAPI ROIs, then select query images. Low-level
   feature builders retain explicit-universe semantics for deliberate ablations.

2. **Training validation clips PSNR after averaging.** Official scoring and
   full OOF clip each image's normalized PSNR before averaging. The old training
   validator gave 65.013994 on a synthetic perfect-plus-poor pair versus the
   correct 51.820174; the repaired validator matches the latter. The global-MSE
   selector is unchanged. In saved Round55, 85, 113 and 117 OOF reports, the
   clipping-order difference is below 3e-13, so this defect is not demonstrated
   to explain the plateau or to have changed those models' selected epochs.

## Formal-Pipeline Checks

- All 2,706 full validation condition vectors from the registered ten ROIs are
  identical to those extracted from the all-training-ROI condition table.
  The sparse-input bug does not invalidate the complete-ROI OOF comparisons.
- For matched Round55 auxiliary checkpoints and context128, rounds
  98/99/100/103/105/107/110/111/113/114/116 have one identical checkpoint hash
  per fold and exactly zero baseline score range on fold0 and fold2.
- Tests match training-dataset and inference tensor construction at halos
  0/64/128, including missing neighbors, and match NumPy/Torch shifts and
  JPEG evaluation/output encoding. These establish the tested contracts,
  not a proof against every possible pipeline bug.
- The training shuffle uses a dedicated seeded generator since Round11. Its
  presence is not a recent change that invalidates the Round52 comparison.
- Training at halo64 and inference at halo128/D4 is a real, intentional
  difference. Round85/88 tested it; Round86/87 and Round83 tested larger-view
  training and fixed target alignment. Finding those differences again does
  not establish a new correction or justify rerunning the same recipes.

## Interpretation Corrections

- Withdraw the normal-input interpretation of the Round113 80-image knockout
  and the Round115 sparse-ROI gradient probes. Keep their original JSON files
  for provenance. Formal Round113/114/115 OOF and their gate outcomes stand.
- Round94 actually passed its screen; its later instability does not measure
  how many valid methods a stricter gate would reject. Most failed methods
  were never expanded, so that false-negative rate is unknown. Fold signs,
  ROI signs and mean gain are correlated checks, not independent evidence.
- Repeated reuse of the same validation ROIs supports development comparisons,
  not an untouched generalization estimate. Patch count is not the count of
  independent ROI/patient samples. No new threshold is selected in this audit.
- Base versus Tiny does not establish that scaling can never help; DINO's
  failure does not establish that every backbone must use FCMAE. No larger
  model work is authorized here.
- Round96 reconstructed independent marker images without paired-channel
  input; Round110 imposed a fixed feature loss. These do not exhaust paired
  cross-modal representation learning, but are insufficient positive evidence
  to fund another such model now.
- Correct Round116 cost: 45 minutes was one stage, not a whole fold. Its two
  stages cost 90-92.5 minutes/fold. Repeated baseline inference is cacheable,
  but only after binding checkpoint, complete ROI inputs, conditioning,
  evaluator, TTA, encoding and offset; no cache rewrite is bundled here.

## Literature Cross-Check

[MAS (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11378090/) combines
autofluorescence and DAPI and separately pretrains their masked encoders. Its
results are not a DAPI-only CD68 benchmark. The retrieved
[marker-conditioned diffusion study](https://ojs.aaai.org/index.php/AAAI/article/view/37764)
uses H&E inputs and pretrained latent diffusion. Neither result proves a
feasible drop-in gain for this dataset. They also do not justify declaring
generative or cross-modal methods universally ineffective.

## Evidence And Decision

`reassessment_20260908_measurements.json` records the CPU experiment (114.9 s),
checkpoint/source hashes, fixed image names, full-ROI comparisons and the
synthetic metric counterexample. `tests/test_reassessment_contracts.py` adds
regression coverage. The 155 affected tests pass locally; after extending the
entry-point fix, 93 relevant tests pass on Linux. Local affine tests encountered
the known duplicate-OpenMP runtime abort and were verified on Linux instead.
The retained Round88 submission ZIP still matches
its recorded SHA-256 and size (11,740,002 bytes).

The complete-ROI CPU float32 gradient recheck completed in 1,975.6 seconds,
retaining the original 64 training images/fold and four batch16 groups at both
saved endpoints. Six of sixteen full-model and encoder batches have opposing
gradients; median cosines are `+0.501234/+0.565307`. Full-model opposing primary
fractions have mean `0.051866` and maximum `0.409780`; the zero median must not
hide this minority. Ten decoder batches have negative dots, but median cosine
is only `-0.008403` and median opposing primary fraction is `0.001773`.
This is a valid normal-input endpoint probe, not a training-dynamics study or
evidence that gradient surgery improves OOF. Both input context and arithmetic
precision differ from the old CUDA probe, so its numerical change cannot be
attributed solely to conditioning. Conflict cannot be ruled out, but its
presence alone does not establish that it is harmful to held-out accuracy.
`reassessment_20260908_gradient_summary.json` binds the raw 16-record report by
SHA-256; the detached collector exited `0`. Its one-use helper stays in the
remote operational checkout. CPU runtime was underestimated; future diagnostics
should measure throughput before choosing the batch size. No training is queued.

The audit found repairable defects in diagnostic reliability, but no measured
improvement to the current best full pipeline. Repairing these defects must not
be reported as a leaderboard gain. Do not restart the failed training chain or
claim all possible routes are exhausted. Any next training proposal needs an
explicit rationale beyond the fact that GPUs are available.

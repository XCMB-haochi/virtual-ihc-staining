# Round236: TRAIN cross-marker correspondence audit

Question: do same-name DAPI/marker pairs correspond spatially, beyond the
within-directory continuity established earlier? Audit original TRAIN images
only; no labels, names, transforms, training parameters or submissions changed.
Existing Round234/235 jobs are unaffected; this diagnostic used CPU only.

## Method and limitations

All2380 filenames must match across DAPI and four markers. Use fixed Gaussian
sigma3 smoothing followed by stride4 subsampling and16 intensity bins for
symmetric entropy-normalized mutual information (NMI). Within each ROI compare
same-name patches against cyclic tile offsets7/31/73 and the same grid position
in each of the16 other ROIs. Only matched pairs with both coarse-image standard
deviations>=4 enter the statistic; controls use the same DAPI index subset.
Very low-entropy comparisons score0, not fabricated high agreement. Fixed
identity/independent/constant-image analytic tests pass.

Control difficulty differs by tissue texture/entropy. NMI is not calibrated
pairing probability, and the best-other-ROI criterion is a heuristic, not
proof of biological correspondence. Shared background can contribute. No
search chooses a replacement ROI or modifies the ground-truth mapping.

## Coarse results

| Marker | Same ROI ranks first among17 | Mean same-pair NMI | Mean wrong-tile NMI |
| --- | ---: | ---: | ---: |
| CD45RO | 17/17 | 0.054825 | 0.015655 |
| CD68 | 16/17 | 0.046267 | 0.014633 |
| HLA-DR | 14/17 | 0.027537 | 0.015450 |
| Vimentin | 17/17 | 0.040888 | 0.015998 |

For all four markers, every ROI's same-name aggregate exceeds each of the
three within-ROI wrong-tile controls. Weak exceptions for cross-ROI ranking:
CD68 ROI010; HLA-DR ROI005/010/081. Their strongest other-ROI advantage is
only roughly0.00019-0.00056 NMI; they are not diagnosed as mismatched labels.

## Local offset evidence and visual review

For each ROI/marker choose up to6 deterministic informative patches. Compare
Gaussian sigma2 features on a common central support at shifts -8..8 in2px
steps. This is coarse translation evidence, not a subpixel/rotation/nonrigid
registration algorithm. Predicted optima vary widely across patches and
markers; many low-information optima hit the search boundary. For CD45RO,
6/17 ROI-mean optima are zero, with median improvement over zero only0.000249
NMI. There is no coherent universal offset warranting correction. This does
not prove every image is aligned to subpixel precision.

Reviewed12 fixed representative overlays and12 additional examples for the
four weak-ranking pairs (median/upper-quartile/max target standard deviation).
Several marker contours surround DAPI nuclear structures, consistent with
different stained compartments; no obvious gross wrong-tissue mapping was
identified in these examples. Visual selection favors informative images,
not an exhaustive blinded landmark assessment. Precision registration would
need stronger shared anatomical landmarks or acquisition metadata.

Decision: no evidence justifies ROI reassignment, target shifting or any test
output correction. CD45RO's global correspondence is supported, but isolated
patch errors, small local deformation and exact acquisition provenance remain
unverified. No TEST marker labels exist for direct checking. Keep the original
data and continue registered learning experiments. This narrows a suspected
data problem; it does not identify the score bottleneck or promise a gain.

Artifacts: `round236_pairing_audit.json`, local diagnostic mosaics
`round236_pairing_examples.jpg` and `round236_weak_pair_examples.jpg`.

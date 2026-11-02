# Round127: Fixed Five-Fold Seed-2027 Relational Expansion

Round126 passed its pre-registered two-fold gate with median-normalized
relational guidance and supervised seed2027: q95 gain `+0.293073`, fold gains
`+0.278796/+0.306451`, 9/10 positive ROIs and worst ROI `-0.058121`. The OOF
covers 2,706 images, exit is `0`, and all checkpoint hashes match. This result
authorizes one unchanged five-fold expansion.

Round127 changes no method. It keeps seed2027, the same fold-local Round96
teachers, Round52 DAPI FCMAE800 initialization, auxiliary15 weights, median
pair-distance normalization, relation weight `0.05`, 14+14 epochs, context64
training, context128+D4/q95 inference and offset `[0,-1]`. Existing fold0 and
fold2 student checkpoints are reused after hash verification; fold1/fold3/fold4
students are trained with the same fixed recipe. No weights, seed, ROI, epoch or
checkpoint search is permitted.

The five-fold gate remains q95 gain `+0.05`, positive float, at least 4/5
positive folds, at least 18/25 positive ROIs, and worst ROI no lower than
`-0.12`. A failure closes this route for Colon; a pass permits one fixed test
package confirmation. The then-current personal-best Round124 package remains
unchanged.

Only GPU1 is eligible. Existing fold-local teachers are already complete, so
the remaining work is three supervised folds plus complete OOF/gate. Expected
cost is about 4.8 hours of student training and 45-60 minutes of OOF on an
uncontended GPU1.

The fixed config passed local protocol loading, the seed-contract tests, the CV
protocol suite and Python compilation. The Round126 smoke receipt was reused
after independent audit, so no redundant smoke was launched. A detached Screen
`round127_seed2027_expansion` is now running on GPU1 with source commit
`8ebf8ed`; fold1 is the active missing fold. GPU0 remains reserved for the
other project. The script will run fold1/fold3/fold4 sequentially, then complete
OOF and the unchanged distribution gate.

## Completion (2026-09-14)

Round127 completed with exit `0`. The three new seed2027 fold trajectories each
contain 14 epochs; the two reused fold checkpoints and all three new checkpoints
are finite and independently match their OOF provenance hashes. The complete
OOF covers 6,296 images across 25 ROIs and was generated after all producers
exited.

| fixed seed2027 relational candidate vs auxiliary15 | q95 |
| --- | ---: |
| overall gain | `+0.214328` |
| fold0 / fold1 / fold2 | `+0.278796 / +0.207337 / +0.306451` |
| fold3 / fold4 | `+0.170603 / +0.090857` |
| positive ROIs | `19/25` |
| worst ROI (ROI013) | `-0.252999` |

The candidate scores `75.626139` q95 versus `75.411811` for the matched
auxiliary15 ensemble; float gain is `+0.215556`. All five folds and the
aggregate checks pass, while the fixed worst-ROI floor fails. This is a useful
near-pass, but not a formally promotable route. No fold4, ROI, weight or
checkpoint rescue is opened. One fixed complete-candidate platform package is
allowed because the preliminary stage is still being used for confirmation and
the Round124 personal-best package remains unchanged.

The fixed package script is `tools/run_round127_submission.sh`; it binds the
protocol SHA-256 `1d917dbc2a1fa521d49ca9d7e0249dcef215f2017b2c53395b65e9ae759ba53f`,
OOF SHA-256 `c9c446c35b31776c7cf6b2559bc404fabd8a228aa9374db4608c383ef237ca39`,
and gate SHA-256 `f9c5f6ca6478a5840f224b30a841580ba09cd22722ebaa583588328e4cc3e0fe`.
It will produce a separate ZIP on GPU1 only and will not alter existing packages.

The package script passed remote preflight with the semantically identical
deployed protocol (normalized-CRLF byte hash
`5566436efb625f40e4988c7ea91ae34554199f71c2602000f996e0e5d4f43037`) and was
detached as Screen `round127_submission` on 2026-09-14. It is waiting for GPU1
to have no compute process before inference; GPU0 remains untouched. The output
ZIP, prediction report and status marker will be audited before any platform
submission decision.

## Submission package completion (2026-09-14)

The fixed package job exited `0` on GPU1. It generated 1,346 test predictions
and passed the remote package validator. A second local audit found exactly
1,346 ZIP entries under `results/test/CD68/`, all 256x256 RGB JPEGs with equal
channels; `zipfile.testzip()` returned no error. The remote and local archive
bytes match:

- file: `submissions/cd68_round127_cv5_relational_seed2027_q95_plain.zip`;
- size: `11,964,493` bytes;
- SHA-256: `2d477e60c7d0d357e3eac12322ade72539334ef7f5d7b2869348fc95d0e94a23`;
- prediction report: `reports/round127_cv5_relational_seed2027_test_prediction.json`;
- exit marker: `reports/round127_submission.exit` = `0`.

This is a fixed confirmation package, not a promotion override. Submit it once
if a platform confirmation is desired, while retaining Round124 as fallback.
Do not tune from its score.

The user reported platform score `74.2222` (`SSIM=0.833`, `PSNR=26.483`). This
is `+0.0592` versus the Round124 package (`74.1630`) and `+0.1907` versus the
verified Round88 package (`74.0315`). It remains `-1.2744` below the recorded
external leaderboard high `75.4966`. Round127 became the project's personal
best at that point. The result supports retaining seed2027 as a
candidate, but it is not evidence to select a fold-specific rescue or a new
blend ratio. A prospective 50/50 seed-blend OOF check is registered separately
in Round128.

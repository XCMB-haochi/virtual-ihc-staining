# Round 61: Decoder-Only Auxiliary Gradient Routing

## Motivation

Round55's aligned auxiliary-marker objective improves its shared-decoder model
by `+0.078951` q95 over the exact matched primary-only control. Round60 then
separates the marker decoders: the isolated-decoder control remains competitive,
but auxiliary supervision loses `-0.124665` q95, regresses both folds, and
improves only 2/10 ROIs. The two results reject both removing decoder sharing
and leaving auxiliary gradients only in the shared encoder.

Round61 tests the remaining localized mechanism. CD68 uses the ordinary shared
encoder, thumbnail FiLM, and shared decoder path. The three auxiliary outputs
reuse that same decoder but receive detached encoder features. Auxiliary losses
therefore update the shared decoder and their output channels without updating
the encoder or FiLM. The architecture, predictions, initialized parameters,
loss weights, and inference graph remain identical to Round55; only the
backward route changes.

## Fixed Screen

`configs/round61_cv2_unext2_decoder_only_auxiliary.json` retains the Round55
fold0/fold2 ROIs, seed 2044, fold-local official-FCMAE800 initialization,
14+14 supervised epochs, effective batch 16, halo64 thumbnail FiLM, Uniform-7
hybrid loss, global-MSE checkpoint selection, D4 inference, and JPEG q95. The
only new recipe is `decoderonlyauxiliary15`, with fixed marker order
`CD68,CD45RO,HLA-DR,Vimentin`, weights `0.85/0.05/0.05/0.05`, and
`auxiliary_gradient_scope=decoder-only` in both stages.

The immutable Round55 `officialauxcontrol` and `officialauxiliary15` runs are
the matched controls. Re-training the former would duplicate an identical
forward, loss, initialization, data order, and optimizer trajectory. Before
formal launch, unit and real-GPU smokes must instead prove exact forward
identity, primary-gradient preservation, auxiliary-gradient blocking at the
decoder input, and a valid batch-16 optimizer step. If batch 16 cannot fit, the
only permitted pre-launch adjustment is micro-batch 8 with two-step gradient
accumulation for effective batch 16; the protocol and this report must be
updated before the formal run.

## Preregistered Gates

The mechanism must gain `+0.08` q95 over `officialauxcontrol`, improve both
folds and at least 7/10 ROIs, and keep the worst ROI above `-0.12`. Its fixed
0.175 split must gain `+0.02` over the corresponding control split. The direct
model must then beat Round55 `officialauxiliary15` by `+0.05`, and the new split
must beat the accepted Round55 split by `+0.01`, with the registered fold, ROI,
and floor requirements.

Deployment requires the new split to gain `+0.06` over `t70_o50`, with both
folds, at least 8/10 positive ROIs, and worst ROI above `-0.08`. The replacement
route requires `+0.08` under the same distribution requirements. The gates are
dependency chained so a deployment cannot pass on ensemble covariance alone
when its isolated mechanism fails. No auxiliary weight, epoch, checkpoint,
ensemble weight, or ROI-specific selection will be changed after seeing the
OOF result. A passing route must be expanded unchanged to all five folds before
test inference; failure closes this gradient-routing mechanism and authorizes
no platform submission.

## Status

The implementation passes 16 UNeXt2/FCMAE tests, 43 training-safety tests, and
64 CV protocol tests locally. The server passes all 271 tests, Python
compilation, JSON parsing, shell syntax, and the exact four-job dry run. The
local monolithic suite remains unavailable because its pre-existing affine
calibration test loads two incompatible Intel OpenMP runtimes; no unsafe
duplicate-runtime override was used.

A CUDA float16 identity check reports an exact maximum output difference of
zero between the ordinary and routed forwards. An auxiliary-only loss produces
zero total gradient in both the encoder and FiLM and nonzero gradient in the
shared decoder; a primary-only loss still produces nonzero encoder gradient.

The real-data batch-16 smoke at commit `ddbe250` trains 64 images, validates 32,
writes four optimizer steps and a valid selected checkpoint, and peaks at
31.031 GiB. Both A100s had about 40.4 GiB free, so the fixed batch 16 fits with
adequate headroom and no accumulation fallback is needed.

## Formal Outcome

Both folds complete both 14-epoch stages at training commit `d37188d`. Every
stage has 14 consecutive metric rows, nonempty selected and final checkpoints,
seed 2044, effective batch 16, the registered marker weights, and
`auxiliary_gradient_scope=decoder-only`. Fold0 executes 4,368 optimizer updates
per stage and fold2 executes 4,298; peak allocation is 31.150 GiB. Stage 1 uses
the registered fold-local FCMAE800 encoder hashes `50231215...` and
`c3f71f31...`; Stage 2 strict-loads its matching Stage-1 selected checkpoint.

The first pipeline invocation finishes all training but exits before OOF
because the launcher omitted the fold2 process-ID assignment. Commit `8f3b575`
adds the missing assignment and a regression test. The resumed pipeline
recognizes and skips all four complete training stages, then exits zero after
OOF and gate evaluation; no weights are retrained or overwritten.

OOF evaluation covers all 2,706 images from the fixed ten validation ROIs and
takes 2,773.370 seconds. The selected Stage-2 epochs are 10 for fold0 and 3 for
fold2. Their SHA-256 values are `37800d9e...` and `b458793a...`, and independent
recomputation matches the OOF provenance. All 14 member checkpoint hashes also
match.

| Candidate | q95 | Float | q95 delta vs `t70_o50` |
| --- | ---: | ---: | ---: |
| `officialauxcontrol` | 74.794670 | 74.866941 | n/a |
| `officialauxiliary15` | 74.873621 | 74.943107 | n/a |
| `decoderonlyauxiliary15` | 74.788275 | 74.856690 | n/a |
| `t70_o50` | 75.180556 | 75.255122 | baseline |
| `t70_official_auxcontrol_split` | 75.195485 | 75.269552 | +0.014929 |
| `t70_official_auxiliary_split` | 75.221576 | 75.295013 | +0.041020 |
| `t70_decoderonly_auxiliary_split` | 75.203918 | 75.277486 | +0.023362 |
| `t70_decoderonly_auxiliary_replace` | 75.185833 | 75.258513 | +0.005277 |

The routed model is effectively tied with the primary-only control: q95 loses
`-0.006395`, float loses `-0.010251`, fold0 loses `-0.022012`, and fold2 gains
`+0.008239`. It improves 6/10 ROIs but reaches a worst-ROI delta of
`-0.179049`. It also loses `-0.085346` q95 to the original Round55 full-gradient
auxiliary model, with both folds negative, only 2/10 positive ROIs, and worst
ROI `-0.521832`.

The fixed split has a small, stable diversity gain of `+0.008433` over the
primary-only control split: both folds and 8/10 ROIs improve, with worst ROI
`-0.026101`. This misses the preregistered `+0.02` threshold and depends on a
failed isolated mechanism. More importantly, it loses `-0.017658` to the
accepted Round55 auxiliary split on both folds and improves only 3/10 ROIs.
Against `t70_o50`, it gains `+0.023362`, below the `+0.06` deployment threshold
and with only 6/10 positive ROIs. The replacement gains only `+0.005277`.

Every gate fails and `promotion.allowed=false`; no five-fold expansion or test
inference is authorized. Round61 shows that auxiliary gradients sent only to
the shared decoder recover a primary-control-like model but not Round55's
benefit. Together with Round60, the evidence indicates that the useful
auxiliary effect requires the shared encoder and shared decoder to adapt
jointly; neither encoder-only nor decoder-only routing is sufficient. This
closes selective auxiliary-gradient routing at the fixed 15% objective. The
accepted Round55 split and current submission remain unchanged.

The OOF SHA-256 is
`47d167db08c26fc5b82524ace60dfdc4d7520c079702bdbb4944bb8961916b65`;
the gate SHA-256 is
`2a3a68b001ff68d3d33157cd9f6e36d342a65e48a036050155a94005f7568439`.

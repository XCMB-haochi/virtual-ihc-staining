# Round123: Additional-Fold Confirmation of Round122

Round122 gains +0.200209 q95 on both development folds with 8/10 positive ROIs,
but ROI007 loses -0.178934 and the original -0.12 floor fails. The new decision
is whether this fixed recipe generalizes to the predefined fold1. This is a
separate confirmation experiment, not a relaxed replay of Round122's gate.

Freeze all student settings from Round122: median distance scale, weight0.05,
pooled stage4, all batch16 relations, fresh Round52 fold-local DAPI FCMAE800
initialization, auxiliary15, seed2044, 14+14 epochs, halo64 training,
halo128+D4/q95 inference, global-MSE selection and [0,-1] output shift.
Use fold1 ROI002/009/018/015/005 and the existing matched Round55 fold1 control.
No alternative candidate or retrospective fold choice is evaluated.

The new frozen teacher follows Round96 exactly: official Tiny FCMAE weights,
160 epochs of five-marker reconstruction, batch32, warmup2, seed2044; every
fold1 validation ROI is excluded from every modality. Student initialization
still uses its separate, existing DAPI-only fold1 encoder. The committed
teacher recipe determines its output; the runner replaces only the template's
placeholder teacher path/hash after verifying completion, all epochs, finite
weights and fold exclusions. The resolved protocol is saved before student
training. The runner checks that no other Round122 settings changed.

Confirmation requires q95 gain >=+0.05, positive float, at least 4/5 positive
ROIs, and worst ROI >=-0.12 (same single-fold policy used in Round115).
`confirmation_only=true` prevents any pass from authorizing submission.
If confirmed, consider completing folds3/4 unchanged, followed by one fixed
deployment decision. If mean gain is nonpositive, close this recipe and move
the capacity/initialization comparison ahead. A positive but incomplete pass
is reported as inconclusive, without automatic further training or tuning.
The pipeline stops after fold1 OOF and this confirmation gate in all cases.

Teacher training costs 12.3-12.5 hours per historical fold, student training
about 1.6 hours/fold, plus OOF. Budget 14-16 hours without contention, versus
roughly 42-48 hours for all three missing teachers/students. Only GPU1 is
eligible; resource checks repeat at teacher/student/OOF boundaries. Teacher
resume state is saved every five epochs; supervised resume state every epoch.

Larger models remain a separate option, but the current measured positive
route has higher priority. The earlier Huge cost estimate of twice Base lacked
measurement. The [official ConvNeXt-V2 repository](https://github.com/facebookresearch/ConvNeXt-V2)
provides Huge FCMAE weights and lists 660M parameters / 115G FLOPs at 224,
versus Base 89M / 15.4G and Large 198M / 34.4G. These classification FLOPs are
not runtime estimates for this stride16 dense implementation. No ten-day
completion promise or rejection of public-weight initialization follows from
them. A future capacity screen needs measured local throughput first.


## Validation and launch

108 targeted tests pass on Windows and Linux, including frozen-recipe equality,
unchanged median loss/gradients, teacher binding and confirmation-only gates.
The first teacher smoke was rejected before training because its short epoch
count equaled warmup2; the corrected three-epoch smoke preserves that warmup.
It completed with a finite, fold-excluded encoder. Both student smoke stages
then completed four batch16 optimizer updates, strict stage2 restoration and
finite checkpoints. Peaks: 26.583/26.701 GiB. Smoke loss is not a quality estimate.

Source commit `a8a6c28bc1b1804a730526eb10ea80c91bf78c24` was deployed
as a verified Git archive. Formal Screen `round123_relational_confirmation`
is launched on GPU1 only. Runtime: `transfers/round123_runtime`; the sequence
is fresh teacher160, teacher audit/binding, actual-teacher supervised smoke,
fold1 14+14 training, OOF and confirmation gate, then exit. The orchestration
receipt records source hashes; the resolved protocol and downstream launch
receipt are generated only after the formal teacher passes audit. No new
manual launch is needed between these phases.


## Completion (2026-09-12)

Round123 is a successful additional-fold confirmation. Pipeline exit is `0`;
the fold1 teacher has 160 epochs, finite weights, and excludes all five fold1
validation ROIs. Both student stages have 14 epochs and finite checkpoints;
all checkpoint hashes and the OOF hash match. OOF covers 1,145 images / 5 ROIs.

| fold1 confirmation | result |
| --- | ---: |
| q95 gain | +0.136267 |
| float gain | +0.129609 |
| positive ROIs | 4/5 |
| worst ROI | -0.085853 |

The fixed confirmation gate passes every check. `confirmation_only=true` still
keeps `promotion.allowed=false`; this result is evidence for expanding the
unchanged recipe, not permission to submit a one-fold model. Together with
Round122, the recipe is positive on fold0, fold1 and fold2, with gains
+0.227592, +0.136267 and +0.174551 under their respective matched controls.

Decision: proceed to a fixed five-fold expansion of the median-normalized
relation recipe. No weight, ROI, teacher, seed or architecture search is opened.
Fold3/fold4 require fresh five-marker FCMAE teachers because the existing
Round96 teachers excluded different validation ROIs. Existing fold0/fold1/fold2
student checkpoints will be reused after independent hash checks. Only GPU1 is
eligible; GPU0 remains reserved.


## Five-fold expansion launch

The fixed recipe passed the predefined fold1 confirmation. Fold0/fold1/fold2
student gains are +0.227592/+0.136267/+0.174551. The next step is not a
new method: it is an unchanged five-fold expansion. Existing fold0/fold1/fold2
students are retained after hash audit; fresh fold3/fold4 five-marker teachers
are required because teacher validation exclusions must match their folds.

Round124 teacher expansion is detached as Screen `round124_relational_teachers`,
GPU1 only. It trains fold3 then fold4, each 160 epochs, with checkpoint every
five epochs and fold-local exclusions. GPU0 was observed at 39% utilization and
about 3.9 GiB free, so it was excluded; GPU1 had 40.4 GiB free. After both
teachers pass audit, the unchanged student stages for fold3/fold4 and complete
OOF will be scheduled.


## Round124 completion (2026-09-13)

The five-fold expansion completed with exit `0`; fold3/fold4 students used the
new fold-local teachers, and the OOF covers 6,296 images across all 25 ROIs.
The OOF/gate artifacts were generated after all producers exited; the remote
checkpoint hashes and launch provenance were checked.

| fixed median-relational candidate vs auxiliary15 | q95 |
| --- | ---: |
| overall gain | +0.094465 |
| fold0 | +0.227592 |
| fold1 | +0.136267 |
| fold2 | +0.174551 |
| fold3 | +0.127388 |
| fold4 | -0.209331 |
| positive ROIs | 17/25 |
| worst ROI (ROI016) | -0.586862 |

The candidate scores 75.506276 q95 versus 75.411811 for the matched auxiliary
ensemble, with float gain +0.093495. The fixed gate fails its 18/25 ROI and
-0.12 worst-ROI requirements. This is a useful aggregate gain but not a stable
five-fold improvement: four folds are positive and fold4 is substantially
negative. The fold4 failure is too large to explain as a small gate margin.

Do not make this the primary submission package and do not tune fold4, weights,
ROI subsets or normalization after seeing this result. Preserve the current
best submission. The relation recipe is closed as a general five-fold
promotion route, but retained as a representation lead for the new-data stage.
With the preliminary deadline extended to 20 September and the later data
being new, the next valuable work is a capacity feasibility smoke or a
new-organ transfer check; another Colon-only local patch is lower value.

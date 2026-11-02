# Round232: isolate auxiliary-marker supervision for CD45RO

Current four-marker loss weights came from the successful Colon CD68 route.
Rounds51/55 found auxiliary-label benefits there, but that does not establish
the same benefit for new-stage CD45RO. Round209 varied which marker receives
85% weight, not whether auxiliary supervision is beneficial. The fitting probe
and small gains from local refinements leave this task-allocation question open.

Change only weights CD68/CD45RO/HLA-DR/Vimentin from0.05/0.85/0.05/0.05
to0/1/0/0. Keep the same four-output network, initialization, seed2026, domain
FCMAE800 encoders, data order,4x4 effective16,14+14 cosine stages, stage1
epoch14 and final stage2 epoch12. Do not switch to a one-channel head, which
would change capacity and initialization. Unsupervised output channels are
not meaningful competing marker models; only CD45RO decides the experiment.

No distillation, relational guidance, DropPath, signal-occupancy head, texture
condition or target rewrite. Normal image losses and D4/halo128/offset0/q95
remain fixed. Test original five folds directly with existing encoders, then
compare all2380 OOF images/17ROIs to Round217. No automatic final student,
package or submission. Positive evidence motivates further confirmation,
not a weight grid; negative results concern this primary-only configuration.

Paired command test verifies only target weights and paths change. A gradient
test verifies other marker labels do not affect loss and their prediction
channels receive zero loss gradient. Two real smoke stages completed7/8
Adam updates, exact0/1/0/0 weights, correct CD45RO channel and6.99GiB peak.
These are execution checks, not efficacy evidence.

Both GPUs are currently idle with25.3/39.5GiB free. GPU0 handles folds0/3;
GPU1 handles folds1/2/4 and final OOF. Expect approximately4-5h including
scoring; no new FCMAE pretraining. Allocation is launch-specific, not a
permanent assumption. Best platform package remains Round21679.0198.

## Completed

Exit0. Full2380-image D4 mean change+0.000479, essentially a tie. Fold
deltas:-0.117859,-0.112134,+0.068239,+0.290525,-0.167174. Seven of17 ROIs
positive; worst ROI072-0.564707. Removing all auxiliary supervision does
not show a robust advantage. Do not train a full-data candidate or search
marker-weight mixtures; keep the retained four-marker objective unchanged.

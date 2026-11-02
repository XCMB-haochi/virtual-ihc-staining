# Round225: full-data local-signal auxiliary confirmation

Round224 supplies complete five-fold positive mean evidence (+0.16117),
four positive folds and12/17 positive ROIs. Preserve the fixed auxiliary
recipe and train one full-data model using the existing all-data DAPI800
encoder, seed2026,4x4 effective16,14+14 cosine periods, original spatial
thumbnail and image loss. Stage2 starts from its own stage1 epoch14 and
the final checkpoint is stage2 epoch12. No new pretraining.

No pixel distillation or relational guidance: this matches the grouped
screen and isolates the mechanism against Round20878.9741. The best actual
submission remains Round21679.0198, which includes distillation. Candidate
must exceed that best score to replace it; a gain against Round208 alone
does not suffice. Do not assume effects of distillation and auxiliary
supervision add independently.

One model, DAPI-only inference, D4/halo128/offset0/q95,560 outputs. The
training-only occupancy head is serialized but not used to modify inference
predictions. Verify ZIP and hashes. One manual platform confirmation after
completion; no checkpoint/weight search. Latest user released both cards,
both idle at check; choose GPU1 for the single model because it has more
free memory. Additional data parallelism would alter the matched recipe.
Expected1.5-2h including inference, based on recent all-data runs.

Matched-command test passed. Real two-stage all-data-path smoke completed
with7/8 actual optimizer updates, nonzero finite auxiliary losses and peak
allocation6.99GiB. Strict stage2 restore passed. Smoke validates execution
only; no smoke score is used for selection.

## Completed

Pipeline exit0. Both stages completed14 epochs with finite image/auxiliary
losses,2,380 training images, no validation images, seed2026, auxiliary
weight0.05 and no pixel distillation. Actual saved Adam updates2081/2083;
summed epoch runtime1.31h. Final stage2 epoch12 tensors and both checkpoint
and package SHA256 match the server receipt. The package contains560 CD45RO
RGB256x256 images and uses unchanged D4/halo128/offset0/q95.
After a slow initial transfer, a fresh download over the restored connection
passed the recorded SHA256, ZIP CRC and decoding of all560 images locally.

## Platform confirmation

User reported **78.8423**, SSIM **0.896**, PSNR **26.870**. Change versus
the matched no-distillation Round20878.9741 is **-0.1318**; change versus
best Round21679.0198 is **-0.1775**. Both displayed SSIM and PSNR are lower
than Round208's0.898/26.894. The loss cannot be attributed solely to absent
pixel distillation. The five-fold mean gain+0.16117 did not transfer to this
full-data model's platform confirmation.

Do not promote this fixed auxiliary configuration. Preserve Round216 and
its exact package. Do not automatically add distillation, search auxiliary
weights/thresholds or try neighboring checkpoints to rescue the score.
The combination with distillation remains untested, not proven negative.
This result also does not establish that every cell/local-structure approach
fails. Both recent texture and auxiliary candidates lost on the platform
despite positive mean OOF, so reassessing validation-to-full-data transfer
is warranted before another speculative module.

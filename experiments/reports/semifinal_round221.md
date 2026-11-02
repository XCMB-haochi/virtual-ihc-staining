# Round221: bounded full-data texture confirmation

Round220 provides complete but mixed five-fold evidence: mean+0.05517,
three folds positive, two negative; benefit concentrated in fold3. This is
not a stable promotion. With the all-data encoder and validated global OOF
cache already available, one full-data candidate costs roughly80-100 minutes
plus inference, without new pretraining or a parameter sweep.

Relative to Round216: preserve the full-data DAPI800 encoder, seed2026,
four-output order/weights, microbatch4/accumulation4,14+14 cosine periods,
uniform7 loss, no relational guidance, fixed epoch12 endpoint. Train stage1
fresh with spatial-thumbnail-texture-film, then initialize stage2 from its
own stage1 epoch14. Retain exactly the Round216 CD45RO OOF target cache
and0.2 distillation weight; do not introduce texture teachers simultaneously.
This tests whether the condition change remains useful in the retained
distilled recipe. No mature baseline student is silently loaded into a
different condition structure.

Final inference is one checkpoint,D4,halo128,offset0,q95. Validate560 RGB
256x256 outputs, ZIP CRC and hashes. No automatic submission; recommend
one fixed platform comparison if artifacts pass. Round21679.0198 remains
best until an actual higher platform score. GPU0 only, GPU1 occupied.
Command-equivalence test fixes all non-condition settings and target routing;
two-stage smoke checks real updates and strict stage2 restoration.

Both real smoke stages passed:7/8 actual Adam updates, finite losses,
correct CD45RO distillation index and peak allocation6.99GiB. These checks
validate execution only, not performance. Full training retains14-epoch
cosine periods and the fixed epoch12 endpoint.

## Completed

Exit0. Both full-data stages completed14 epochs on2,380 training images,
with zero validation images, seed2026, finite losses and the intended
texture condition. Stage2 used CD45RO channel1 distillation at0.2 with the
retained target cache; final epoch12 tensors are finite and hash-verified.
Training times summed to1.28h. Package:3,200,390 bytes,560 unique CD45RO
RGB256x256 JPEG images. Local SHA256 matches the server, ZIP CRC and every
image passed. Hashes and concise audit are in `round221_hashes.json` and
`round221_completion_audit.json`.

## Platform confirmation

User reported **78.967**, SSIM **0.898**, PSNR **26.810**. Compared with
Round21679.0198 (0.899,26.833), the total decreases **0.0528**, with both
displayed components lower. The small/mixed five-fold mean gain did not
translate into an improvement for this fixed full-data distilled candidate.
Do not promote Round221 or search texture channels, weights or checkpoints
against this platform result. Preserve Round216 as the best submission.
This rejects the tested configuration, not every local-structure approach.

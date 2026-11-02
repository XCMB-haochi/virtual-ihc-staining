# Round 100: Zero-Gated Shallow DAPI Skip

## Motivation

The Round99 high-pass logit residual lowered the official score and is closed.
The current UNeXt2 decoder reaches the full-resolution output only after the
DAPI has passed a stride-2 stem and four encoder stages, so fine local signal
must be reconstructed from downsampled features. Round100 tests a different
placement of the same information: a learned shallow DAPI feature path is
fused before the decoder's final PixelShuffle.

## Fixed mechanism

`unext2-fcmae-tiny-shallow-dapi-skip` keeps the UNeXt2-Tiny encoder, decoder,
thumbnail FiLM, four-marker auxiliary objective, and all training settings.
The raw DAPI input passes through a stride-2 5x5 convolution and a 3x3 feature
block. A zero-initialized 1x1 projection adds four channels to the primary
CD68 pre-shuffle logits; auxiliary pre-shuffle channels are unchanged. The
projection starts at exact zero, making the initial output identical to the
ordinary shared decoder while allowing the new path to learn during training.

The comparison uses the same fold-local Round52 FCMAE encoders and the same
Round55 `officialauxiliary15` recipe: marker order
`CD68,CD45RO,HLA-DR,Vimentin`, weights `0.85/0.05/0.05/0.05`, seed 2044,
context64 training, context128 inference, D4, fixed `[0,-1]` output offset,
and 14+14 supervised epochs. Only fold0 and fold2 are screened first.

## Gate

The candidate must gain at least `+0.05` JPEG-q95 official score, have a
positive float gain, improve both folds and at least 7/10 ROIs, and keep the
worst ROI above `-0.12`. No shallow width, projection scale, epoch, checkpoint,
blend, ROI route, or leaderboard search is authorized after the screen.

## Completion audit and result

The detached pipeline exited with status `0`; its Screen and producer processes
are gone. Fold0 and fold2 each completed both 14-epoch stages with nonempty
selected checkpoints and matching fold-local FCMAE provenance. The OOF covers
all 2,706 registered images across ten ROIs. The known `pymp-*` cleanup warnings
occurred after successful stages and did not affect the completion contract.

At JPEG q95 the matched auxiliary15 baseline scores `75.013272`, while the
shallow-skip candidate scores `74.953469`, a delta of `-0.059803`. Float changes
by `-0.056667`. Fold deltas are `-0.014890` and `-0.101888`; only 4/10 ROIs
improve, and ROI007 is the worst at `-0.380856`. Every registered distribution
check fails and `promotion.allowed=false`.

The new projection learned nonzero weights in both folds, but its output has
lower SSIM and lower PSNR than the matched baseline. Round100 therefore closes
the shallow-DAPI skip at this fusion point and does not authorize width,
projection, checkpoint, blend, five-fold, or leaderboard follow-ups.

The OOF SHA-256 is
`14be4fb0d7c350be21e55a468fb2c07ee68bb35b6fe32ebdf4afec47a66b79b8`;
the gate SHA-256 is
`f06970d549e110929464dabafd20853b11e5bd090d106bbcbd4f9ca928a38c10`.

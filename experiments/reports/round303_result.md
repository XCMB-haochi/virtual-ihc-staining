# Round303 completed

All28 epochs complete; best global epoch27 (stage2 epoch13):76.0908141.
Round302 mixed75.4728314; difference+0.6179827 on the same grouped split.
Colon74.8907161 (+1.7953632), Liver77.5821999 (+0.9336750),
Stomach75.8361465 (+0.3595877). ROI improvements:5/5 Colon,3/4 Liver,
2/3 Stomach; Stomach gains+0.7518825,+0.5555743,-0.2286937.
Only3 Stomach Val ROIs and different checkpoint-search counts limit certainty.
Stage1 best76.0233959 at13; stage2 best adds0.0674182. This recipe bundle
improves local validation; it does not isolate SSIM, sampling or learning rates.

Package uses the one best checkpoint, fixed mixed markers, no TTA/context,
JPEG95. Downloaded and verified in local submissions after the transient local
delivery process disappeared without a failure record. Training did complete;
automatic packaging had not run. Do not claim its trigger worked.

Decision before further platform feedback: submit Round303 and refit this recipe
on all official Train (Round304). Uniform-CD68 under this recipe is untested;
the prior0.943 gap supports lower priority than using all Train now. No guarantee
of an online gain. The best encoder remains the completed800-epoch Round302
encoder; do not rerun it. Old Liver score and new three-organ weighted score
are different tasks, so their numerical gap is not a paired regression.

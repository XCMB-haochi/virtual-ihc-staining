# Round326 result and Round327 full refit

326 completed14 stage2 epochs/7084 updates, finite-weight audit passed.
Val76.2806863090 at26 vs31876.2672365203: +0.0134497887.
Organ deltas Colon+0.0046891,Liver+0.0211359,Stomach+0.0125053.
Stomach ROI deltas +0.0170478,+0.0244472,-0.0039792.
11/14 matched epochs improve; last9 all improve. Small single-split benefit,
not guaranteed Test gain, but sustained trajectory supports one final refit.

327 uses fresh full10356-image supervised stage1 for14 epochs from fixed800
encoder and seed2026 task initialization, then own stage1 checkpoint/RNG,
reset AdamW,12 stage2 epochs ending26 (cosine horizon14). Teacher0.2,
consistency0.4, unchanged organ weights/LR/geometry/architecture. No crop,
EMA/TTA/averaging, Val-weight expansion or new pretraining. Same full teacher
cache as309. No full-data validation or checkpoint selection.

Run tools/round327.py --log. Two-stage smoke exercises14 short stage1 epochs
and one stage2 epoch, provenance and optimizer checks. After training automatically
verify provenance and package327; hidden local downloader writes submissions.
At12:12, remaining7h48. Expected4-5h plus delivery, preserve submission buffer.

Launch verification: two-stage real-data smoke passed14 short stage1 epochs,
one stage2 epoch, optimizer checks and final-package provenance assertions.
Deployed327/packager SHA256 matched local revision16661e7. Hidden remote
AIC-Round327 and local AIC-Round327-Download registered and started. Downloader
uses existing CREATE_NO_WINDOW child handling and SHA256/CRC/2186-entry checks.

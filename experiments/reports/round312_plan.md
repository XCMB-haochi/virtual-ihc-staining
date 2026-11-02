# Round312: matched Gaussian-valid late loss under current distillation

One second-stage comparison only, reusing exact Round307 stage1 epoch14 weights
and RNG plus Round308's8084 label-excluded teacher targets. No new teacher or
domain pretraining, no Test inspection. Same mixed markers, split, seed2026,
organ weighting, optimizer,14-epoch stage2 cosine, BF16, accumulation16 and clip1.
Only change uniform7 training SSIM to existing gaussian11_valid, for both hard
and0.2 teacher losses. Gaussian11 validation, JPEG95 and architecture unchanged.

Historical Round206 Gaussian-reflect was negative. Round241 late Gaussian-valid
Liver OOF was effectively tied(-0.0030955); this is not a new mechanism or a
correction of a known bug. Current multi-organ single-image distilled setting
differs; direct alignment to the assumed validation objective and low stage2
cost justify one bounded comparison, not a grid. Sigma1.5/population/valid
remain local assumptions, not fully confirmed organizer implementation.

Compare best14 Val endpoints to Round30876.2247099 and all ROI differences.
One split with3 Stomach ROIs is uncertain; no extra ROI gate or predicted online
gain. No automatic full refit, package, or additional experiment chain.
Expected1-1.5h on5090; current A100GPU0 heavily occupied, GPU1 not needed.
Tests verify loss/gradient agreement and real-data stage2/config equivalence.
Scheduled Task AIC-Round312 directly executes Python with per-epoch resume.

October29 launch: numerical loss/gradient tests passed locally and remotely;
real-data stage2 smoke passed finite-state/update audit and config comparison
against Round308. Verified source archive SHA
875dcb42093e91f645fd22337ccd3cd0be167de2c38ec0a225cbcf90035bb64f;
source commit in round312_launch_commit.txt. AIC-Round312 started on5090.

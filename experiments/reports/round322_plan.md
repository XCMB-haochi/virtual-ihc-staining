# Round322: target-marker-specific lightweight output heads

Compare grouped-Val31876.2672365, not platform ordering. Current model conditions
encoder features by marker but shares decoder and output head. Hypothesis:
separating the final CD68/CD45RO mappings can reduce cross-target interference;
no measured gradient-conflict claim and no guaranteed benefit. Shared features
may already suffice, and smaller per-head datasets may hurt generalization.

At fixed stage boundary14, copy the shared PixelShuffle/conv/GELU/conv head
into two identical heads. No random initialization or RNG consumption; initial
outputs must match bitwise. One shared encoder/decoder; select one head solely
by marker0/1. Colon/Stomach both use CD68 head0. No organ input, other-marker
outputs, averaging or independent full models. R05 explicitly permits target
output heads; R06/R07 single image/task and one output path preserved.

Use same307 grouped stage1+RNG, encoder800, teacher cache, original sample order,
LR1e-4/cosine14, teacher0.2, consistency0.2, organ0.1/0.2/0.7. Train15-28.
Audit Adam steps per active head (inactive head correctly receives no update),
shared parameter steps, strict reload, full coverage and report serialization.
Val comparison permits only model field difference. Expected2-3h on5090.
No automatic package/full refit. If selected, full candidate must rerun complete
supervised training from fixed pretrained initialization using allTrain, shared
head stage1 then declared split-head stage2; never expand a Val-selected model
onto allTrain. Pretraining itself remains reused. Reserve delivery time before
November1 20:00 Beijing. Current best reported319 preserved.
Launch October31: verified archive; local and remote head invariance/routing tests passed. Real GPU smoke/config/report/per-head optimizer audit passed. GPU idle29907MiB free before launch. AIC-Round322 process and grouped initialization hash confirmed; no stderr. Source commit in round322_launch_commit.txt.

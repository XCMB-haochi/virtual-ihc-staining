# Round327 semifinal delivery

Folder: deliverables/AIC-2026-51865112-通信速通生物-半决赛-团队成员.
Same outer layout as Liver delivery; PDF deferred. Only final-recipe code,
public initialization, final checkpoint, result images and concise execution
instructions included. SHA256 omitted from delivery markdown at user request;
internal verification retained here and in submission manifest.

Final checkpoint matched4b6a42149167282de0eda7140d4631408576ea60c37be77cbec6b704ac683825.
All2186 image payloads in results.zip match submitted327 ZIP byte-for-byte.
Extracted public initialization tensors match original implementation.
Six actual GPU inference outputs (two per organ) and JPEG bytes match submitted
results exactly. Loss scalars match; nondeterministic CUDA backward differences
are about4e-9, also observed on repeated original backward calls. Under
deterministic kernels, BF16 distillation+consistency loss and AdamW update
produce bitwise-identical model tensors. Both shipped training stages passed
finite-weight, optimizer-count and cross-stage smoke audits. Full training was
not repeated; no claim of bitwise reproducibility for a fresh nondeterministic
multi-epoch training run. Final-weight inference parity verified as above.

No server tools, experimental models, check scripts, logs, cache files or PDFs
in delivered folder. Reproduction uses five17-epoch teachers, shared800-epoch
outer-Train encoder, full10356-image fresh student14+12 epochs. Fixed markers,
seed2026, teacher0.2, consistency0.4, single FP32 JPEG95 inference retained.

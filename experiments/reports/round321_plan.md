# Round320 result / Round321 full refit

320 completed14 stage2 epochs/7084 updates, finite checkpoint audit passed,
no stderr. Best epoch26 Val76.2927807 versus31876.2672365 (+0.0255441).
Colon +0.0157210, Liver +0.0099273, Stomach +0.0314094. Stomach ROI000
-0.0198077, ROI002 +0.0526848, ROI006 +0.0613511.8/14 matched endpoints improve,
including26-28; early trajectory not uniformly better. Small one-split evidence,
not demonstrated generalization or proof that gradient variance decreased.

321 uses verified309 full-Train stage1 epoch14 and exact309 teacher cache,
trains full10356 images once/epoch for epochs15-26, same LR/teacher0.2/
consistency0.2/organ0.1-0.2-0.7. Only sample interleaving differs from319;
per-image geometry and consistency views mapped to original full-data order.
No new pretraining/teachers. Package requires full-stage1 provenance and update
order config. Single-image FP32 inference, JPEG95,2186 images; no TTA/context.

Expected~2h training plus inference/transfer. AIC-Round321 directly trains and
packages; local downloader retrieves submissions/semifinal_round321_mixed_q95.zip.
Keep319 current best reported. No follow-up chain or platform-based choice.
Launch October31: archive verified; real-data smoke completed and wrote audit/provenance successfully. GPU idle30021MiB free before launch. Formal AIC-Round321 process/config confirmed; local download task waiting for package. Source revision in round321_launch_commit.txt.

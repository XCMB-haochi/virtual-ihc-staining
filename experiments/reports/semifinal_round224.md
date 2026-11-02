# Round224: unchanged local-signal five-fold expansion on two GPUs

Round223's fixed two-fold gain+0.246920 is positive in both folds and six
of seven ROIs. Reuse those exact student endpoints and train only folds1/2/4
using their existing Round216 domain-FCMAE800 encoders. Keep the model,
signal threshold/pooling/loss weights, CD45RO channel, seed2026,4x4 effective16,
14+14 cosine periods, stage1 epoch14 and stage2 epoch12 unchanged.

No extra FCMAE, distillation, texture condition, checkpoint search or output
processing. Evaluate all2,380 images with D4/halo128/offset0/q95 against the
complete Round217 baseline. Record fold/ROI and all-marker deltas. No automatic
full-data run or package; the complete evidence determines that next decision.

User explicitly authorized both GPUs for this expansion. Live free memory
was25.3GiB on GPU0 and39.5GiB on GPU1, both idle; the recipe peaks near7GiB.
GPU0 trains fold1; GPU1 trains fold2 then the smaller fold4. The coordinator
sets CUDA_VISIBLE_DEVICES separately for each child and pins the storage-scoped
Python, avoiding the Round219 system-Python launch failure. OOF runs on GPU1
after all three new folds complete. One detached coordinator records failure
if any child fails. No other user's process is stopped or modified.

Tests verify command equivalence apart from paths and explicit physical-device
mapping. The identical model/training path has already completed Round223.
Expect roughly3-4h for the longest two-fold queue plus full OOF, depending on
load. This use of both cards is not a permanent assumption about availability.

## Completed

Exit0; all three added folds completed14+14 epochs with finite total/auxiliary
losses and actual saved optimizer updates. Fixed epoch12 full OOF covers
2,380 images/17ROIs. CD45RO mean gain+0.161170; fold changes
+0.101765,-0.142670,+0.025552,+0.355786,+0.340703. Twelve of17ROIs improve;
worst ROI058-0.248049. Four folds positive is stronger than the texture
screen, but fold1 is consistently negative and platform transfer is unknown.
Proceed to one fixed full-data auxiliary-only candidate, not an untested
auxiliary-plus-distillation combination. Keep Round21679.0198 until confirmed.

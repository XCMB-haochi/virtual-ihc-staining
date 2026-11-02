import argparse,hashlib,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['original','delivery']);p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
sys.path[:0]=[str(a.root/'src'),str(a.root/'tools') if a.mode=='original' else str(a.root)]
from ihc_unext2 import UNeXt2VirtualStainer,UNeXt2LocalConditionVirtualStainer
from ihc_torch import CompetitionLoss
if a.mode=='original':
 from train_unet import build_optimizer,training_batch_losses
else:
 from train_stage import build_optimizer,batch_loss

def hash_state(model):
 h=hashlib.sha256()
 for n,v in model.state_dict().items():h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()

torch.set_num_threads(2);report={}
for role in ('student','teacher'):
 torch.manual_seed(2026)
 cls=UNeXt2VirtualStainer if role=='student' else UNeXt2LocalConditionVirtualStainer
 model=cls(4,'spatial-thumbnail-film')
 state=hash_state(model)
 x=torch.rand(2,1,64,64);target=torch.rand(2,4,64,64);condition=torch.rand(2,12901);soft=torch.rand(2,1,32,32) if role=='student' else None
 lossfn=CompetitionLoss(ssim_weight=.7,l1_weight=.2,mse_weight=.1,ssim_mode='uniform7',channel_weights=(.05,.85,.05,.05))
 dist=CompetitionLoss(ssim_weight=.7,l1_weight=.2,mse_weight=.1,ssim_mode='uniform7') if soft is not None else None
 if a.mode=='original':
  opt,_=build_optimizer(model,'unext2-fcmae-tiny' if role=='student' else 'unext2-fcmae-tiny-local-condition',.0001,None,.0001,.8 if role=='student' else 1)
  loss=training_batch_losses(model,x,target,condition,lossfn,0.,distillation_target=soft,distillation_criterion=dist,distillation_weight=.2 if soft is not None else 0,distillation_marker_index=1)[0]
 else:
  opt=build_optimizer(model,.0001,.8 if role=='student' else 1)
  loss=batch_loss(model,x,target,condition,lossfn,soft,dist)
 loss.backward();opt.step()
 report[role]={'initial':state,'loss':float(loss.detach()),'after_update':hash_state(model)}
 del opt,model
print(json.dumps(report));a.output.write_text(json.dumps(report,indent=2))

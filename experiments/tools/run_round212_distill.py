"""Two-fold CD45RO pixel-distillation screen using fold-local domain teachers."""
from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
from run_round204_cd45_cv import DATASET

FOLDS=[{'name':'fold0','val_rois':['ROI005','ROI052','ROI087']},{'name':'fold3','val_rois':['ROI044','ROI067','ROI074','ROI078']}]

def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def make_targets():
 import torch
 from torch.utils.data import DataLoader
 sys.path.insert(0,'src');sys.path.insert(0,'tools')
 from ihc_data import image_names,parse_patch_name,resolve_data_root
 from ihc_torch import PairedPatchDataset,build_model_from_checkpoint,batch_condition_tensor,center_crop_like,predict_with_tta,checkpoint_conditioning
 from ihc_spatial import apply_complete_roi_conditioning
 root=resolve_data_root(Path(DATASET)); out=Path('runs/round212_distill_targets');
 if out.exists():raise FileExistsError(out)
 (out/'targets').mkdir(parents=True)
 all_names=image_names(root/'train'/'DAPI'); rec=[]
 for fold in FOLDS:
  ck=Path(f"runs/round207_{fold['name']}_stage2/epochs/epoch_012.pt")
  payload=torch.load(ck,map_location='cuda:0',weights_only=False); model=build_model_from_checkpoint(payload,torch.device('cuda:0')).eval()
  names=[n for n in all_names if parse_patch_name(n).roi in fold['val_rois']]
  cond=apply_complete_roi_conditioning(root/'train'/'DAPI',names,checkpoint_conditioning(payload))
  loader=DataLoader(PairedPatchDataset(root,names,markers=tuple(payload['markers']),augment=False,context_halo=128),batch_size=2,shuffle=False,num_workers=0)
  with torch.inference_mode():
   for source,target,bn in loader:
    c=batch_condition_tensor(bn,cond,torch.device('cuda:0')); pred=center_crop_like(predict_with_tta(model,source.cuda(),'d4',c),target)[:,1]
    for name,value in zip(bn,pred.cpu().numpy()*255):
     from PIL import Image
     Image.fromarray(value.clip(0,255).round().astype('uint8')).save(out/'targets'/name,format='PNG')
  rec.append({'fold':fold,'checkpoint':str(ck.resolve()),'sha256':digest(ck)})
  del model;torch.cuda.empty_cache()
 (out/'manifest.json').write_text(json.dumps({'method':'fold-excluded-cd45ro-domain-d4-teacher','images':980,'folds':rec},indent=2)+'\n')

def stage2_command(fold):
 out=f"runs/round212_{fold['name']}_distill_stage2"
 return [sys.executable,'tools/train_unet.py',DATASET,'--output-dir',out,'--val-rois',','.join(fold['val_rois']),'--markers','CD68,CD45RO,HLA-DR,Vimentin','--marker-loss-weights','0.05,0.85,0.05,0.05','--epochs','14','--batch-size','16','--model-arch','unext2-fcmae-tiny','--base-channels','32','--block-type','residual','--condition-mode','spatial-thumbnail-film','--context-halo','64','--train-target-context-halo','64','--learning-rate','0.0001','--minimum-learning-rate','0','--weight-decay','0.0001','--ssim-weight','0.7','--l1-weight','0.2','--mse-weight','0.1','--loss-profile','hybrid','--ssim-mode','uniform7','--selection-marker','CD45RO','--selection-ssim-mode','gaussian11','--selection-psnr-aggregation','mean-image','--selection-jpeg-quality','95','--workers','0','--seed','2026','--save-every-epoch','--init-checkpoint',f"runs/round211_{fold['name']}_control_stage1/epochs/epoch_014.pt",'--distillation-target-dir','runs/round212_distill_targets/targets','--distillation-weight','0.2','--device','cuda:0'],out

def main():
 make_targets()
 for fold in FOLDS:
  cmd,out=stage2_command(fold)
  with Path(f'reports/round212_{fold["name"]}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  rows=[json.loads(x) for x in (Path(out)/'metrics.jsonl').read_text().splitlines()];assert len(rows)==14
  print(fold['name'],rows[11]['validation']['markers']['CD45RO']['competition_score_mean_image_psnr'],flush=True)
 print('Training screen complete; evaluate fixed epoch12 with D4 before promotion.',flush=True)
if __name__=='__main__':main()

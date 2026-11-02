"""Generate one ROI-excluded CD45RO teacher target per TRAIN tile; never used at test."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from ihc_data import image_names,load_context_gray,parse_patch_name,resolve_data_root
from ihc_spatial import apply_complete_roi_conditioning
from ihc_torch import build_model_from_checkpoint,checkpoint_conditioning,batch_condition_tensor,predict_with_tta
from common import load_checkpoint,sha256,json_write

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2026)
    data=resolve_data_root(args.dataset);names=image_names(data/'train/DAPI')
    folds=json.loads((ROOT/'configs/recipe.json').read_text(encoding='utf-8'))['folds']
    out=args.work/'targets';out.mkdir(parents=True,exist_ok=False);covered=[];sources=[];device=torch.device(args.device)
    for fold in folds:
        path=args.work/fold['name']/'stage2/epochs/epoch_012.pt';ck=load_checkpoint(path)
        assert ck['epoch']==12 and not ck['args']['train_all']
        assert set(ck['args']['val_rois'].split(','))==set(fold['val_rois'])
        assert not set(ck['args']['train_rois_resolved'])&set(fold['val_rois'])
        model=build_model_from_checkpoint(ck,device).eval()
        queried=[n for n in names if parse_patch_name(n).roi in fold['val_rois']]
        conditions=apply_complete_roi_conditioning(data/'train/DAPI',queried,checkpoint_conditioning(ck))
        for i in range(0,len(queried),2):
            batch=queried[i:i+2];x=torch.from_numpy(np.stack([load_context_gray(data/'train/DAPI',n,128) for n in batch])[:,None]/255.).float().to(device)
            y=predict_with_tta(model,x,'d4',batch_condition_tensor(batch,conditions,device))[:,1,128:384,128:384].cpu().numpy()*255
            if not np.isfinite(y).all():raise RuntimeError('Nonfinite target')
            for n,v in zip(batch,y):Image.fromarray(np.rint(v).clip(0,255).astype(np.uint8)).save(out/n,format='PNG')
        covered+=queried;sources.append({**fold,'checkpoint_sha256':sha256(path),'images':len(queried)})
        del model,ck
        if device.type=='cuda':torch.cuda.empty_cache()
    assert len(covered)==len(set(covered))==len(names)==2380 and set(covered)==set(names)==set(image_names(out))
    json_write(args.work/'teacher_manifest.json',{'images':len(names),'marker':'CD45RO','folds':sources,'scope':'TRAIN only; full-data student loss only'})

if __name__=='__main__':main()

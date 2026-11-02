"""Fixed-shift OOF diagnostic; no training or submission postprocessing."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from skimage.metrics import structural_similarity
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'tools')]
from ihc_data import image_names,parse_patch_name,resolve_data_root
from ihc_torch import shift_array_edge
from audit_semifinal_markers import model_predictions

DATASET='dataset/复赛数据集(包括训练集和测试集输入)'
MARKERS=('CD68','CD45RO','HLA-DR','Vimentin')

def main():
 import torch
 root=resolve_data_root(Path(DATASET)); cache=Path('runs/round216_oof_targets')
 manifest=json.loads((cache/'manifest.json').read_text())
 for fold in manifest['folds']:
  fold['checkpoint_sha256']=fold.get('checkpoint_sha256',fold.get('sha256'))
 names=image_names(root/'train'/'DAPI')
 assignment={}
 for f in manifest['folds']:
  for roi in f['val_rois']:assignment[roi]=f['name']
 # Use the complete five same-architecture domain endpoint checkpoints.
 items=list(model_predictions(root,manifest,names,'cuda:0'))
 grouped={}
 for name,fold,marker,pred,truth in items:
  for offset in (-1,0,1):
   shifted=shift_array_edge(pred,(0,offset))
   s=float(structural_similarity(truth.astype(np.float64),shifted.astype(np.float64),data_range=255,
       gaussian_weights=True,sigma=1.5,win_size=11,use_sample_covariance=False))
   grouped.setdefault((offset,marker,fold),[]).append(s)
 result={}
 for offset in (-1,0,1):
  result[str(offset)]={}
  for marker in MARKERS:
   fold_scores={f:float(np.mean(grouped[(offset,marker,f)])) for f in sorted(set(assignment.values()))}
   vals=[v for (o,m,f),rows in grouped.items() if o==offset and m==marker for v in rows]
   result[str(offset)][marker]={'overall_ssim':float(np.mean(vals)),'score':70*float(np.mean(vals)),'fold_ssim':fold_scores}
 Path('reports/round218_offset_oof.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result['0'],indent=2),flush=True)

if __name__=='__main__':main()

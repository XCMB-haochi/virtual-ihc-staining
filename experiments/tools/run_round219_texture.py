import argparse,json,subprocess,sys
from pathlib import Path
from run_round204_cd45_cv import training_command

FOLDS=[{'name':'fold0','val_rois':['ROI005','ROI052','ROI087']},{'name':'fold3','val_rois':['ROI044','ROI067','ROI074','ROI078']}]

def command(fold,stage):
 c,_=training_command(fold,stage,14)
 c[c.index('--condition-mode')+1]='spatial-thumbnail-texture-film'
 c[c.index('--output-dir')+1]=f"runs/round219_{fold['name']}_stage{stage}"
 if stage==1:c[c.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt"
 else:c[c.index('--init-checkpoint')+1]=f"runs/round219_{fold['name']}_stage1/epochs/epoch_014.pt"
 return c

def main():
 p=argparse.ArgumentParser();p.add_argument('--fold',required=True);a=p.parse_args()
 fold=next(x for x in FOLDS if x['name']==a.fold)
 for stage in (1,2):
  c=command(fold,stage);out=Path(c[c.index('--output-dir')+1])
  if out.exists():raise FileExistsError(out)
  with Path(f'reports/round219_{a.fold}_stage{stage}.log').open('w') as log:
   subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Round219 fold complete',a.fold,flush=True)

if __name__=='__main__':main()

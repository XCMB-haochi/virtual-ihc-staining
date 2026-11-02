"""Fixed endpoint D4 evaluation, one fold at a time with a matched baseline."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from run_round219_texture import FOLDS
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',required=True,choices=['fold0','fold3'])
    args=parser.parse_args()
    fold=next(f for f in FOLDS if f['name']==args.fold)
    checkpoint=Path(f"runs/round219_{args.fold}_stage2/epochs/epoch_012.pt")
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['epoch']==12 and payload['args']['seed']==2026
    assert payload['args']['condition_mode']=='spatial-thumbnail-texture-film'
    assert set(payload['args']['val_rois'].split(','))==set(fold['val_rois'])
    directory=Path(f'runs/round219_{args.fold}_oof_sources')
    directory.mkdir(exist_ok=False)
    manifest={'method':'fold-excluded-single-recipe-d4-teacher','output_source_offset_applied':False,
              'images':len(fold['val_rois'])*140,'folds':[{**fold,'images':len(fold['val_rois'])*140,
              'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':digest(checkpoint)}]}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    output=Path(f'reports/round219_{args.fold}_d4_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(directory),str(output),
                    '--mode','models','--subset-rois',','.join(fold['val_rois']),'--device','cuda:0'],check=True)
    current=json.loads(output.read_text())['markers']
    baseline=json.loads(Path('reports/round217_domain_cv5_oof.json').read_text())['markers']
    gains={m:{scope:row['gaussian11_score']-baseline[m][scope]['gaussian11_score']
              for scope,row in scopes.items() if scope!='overall'} for m,scopes in current.items()}
    Path(f'reports/round219_{args.fold}_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('Fixed endpoint CD45RO delta',gains['CD45RO'][args.fold],flush=True)


if __name__=='__main__':main()

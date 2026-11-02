"""Stage2-only intensity-weighted L1 on the retained domain CD45RO baseline."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from run_round211_batch16 import command_for as batch16_command
from run_round210_relational import digest,run_logged
from run_round204_cd45_cv import DATASET


def command_for(fold,smoke=False):
    command=batch16_command(fold,2,'control',None,None)
    root='runs/round214_smoke' if smoke else f"runs/round214_{fold['name']}_stage2"
    command[command.index('--output-dir')+1]=root
    # Same checkpoint/channel order/optimizer as the completed control stage2.
    command+=['--target-l1-weight-strength','2.0']
    if smoke:
        command[command.index('--epochs')+1]='1'
        command+=['--max-train-images','128','--max-val-images','8']
    return command,Path(root)


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    folds=[f for f in json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds'] if f['name'] in ('fold0','fold3')]
    receipt={}
    for fold in (folds[:1] if args.smoke else folds):
        command,root=command_for(fold,args.smoke)
        if root.exists():raise FileExistsError(root)
        initial=Path(command[command.index('--init-checkpoint')+1])
        baseline=json.loads(Path(f"runs/round211_{fold['name']}_control_stage2/config.json").read_text())
        assert initial.resolve()==Path(baseline['init_checkpoint']).resolve()
        assert baseline['target_l1_weight_strength']==0 and baseline['batch_size']==16
        ck=torch.load(initial,map_location='cpu',weights_only=False)
        assert ck['epoch']==14 and ck['args']['seed']==2026
        assert ck['args']['feature_guidance_weight']==0 and ck['args']['distillation_weight']==0
        assert set(ck['args']['val_rois'].split(','))==set(fold['val_rois'])
        receipt[fold['name']]={'stage1_sha256':digest(initial)}
        print('Training weighted L1 stage2',fold['name'],flush=True)
        run_logged(command,f"reports/round214_{'smoke_train' if args.smoke else fold['name']}.log")
        rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in rows]==list(range(1,(1 if args.smoke else 14)+1))
        assert all(math.isfinite(r['train_loss']) for r in rows)
        final=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        steps=[float(s['step']) for s in final['optimizer']['state'].values() if 'step' in s]
        assert steps and min(steps)>0
        assert final['args']['target_l1_weight_strength']==2
        print('Verified updates',min(steps),'peak GiB',rows[-1]['peak_gpu_gib'],flush=True)
    if args.smoke:return
    Path('reports/round214_sources.json').write_text(json.dumps(receipt,indent=2)+'\n')
    root=Path('runs/round214_oof_sources');root.mkdir(exist_ok=False)
    sources=[]
    for fold in folds:
        path=Path(f"runs/round214_{fold['name']}_stage2/epochs/epoch_012.pt")
        sources.append({**fold,'checkpoint':str(path.resolve()),'checkpoint_sha256':digest(path),'images':len(fold['val_rois'])*140})
    (root/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher',
             'output_source_offset_applied':False,'images':980,'folds':sources},indent=2)+'\n')
    output=Path('reports/round214_d4_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(root),str(output),
                    '--mode','models','--subset-rois',','.join(r for f in folds for r in f['val_rois']),
                    '--device','cuda:0'],check=True)
    current=json.loads(output.read_text())['markers']
    control=json.loads(Path('reports/round211_control_d4_oof.json').read_text())['markers']
    prior=json.loads(Path('reports/round207_d4_oof.json').read_text())['markers']
    result={m:{s:{'candidate_minus_control':v['gaussian11_score']-control[m][s]['gaussian11_score'],
                  'candidate_minus_prior':v['gaussian11_score']-prior[m][s]['gaussian11_score']}
               for s,v in scopes.items()} for m,scopes in current.items()}
    Path('reports/round214_paired_gains.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Fixed epoch12 results',json.dumps(result['CD45RO']),flush=True)


if __name__=='__main__':main()

"""Matched CD45RO local-signal auxiliary task; no label or output rewriting."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from run_round204_cd45_cv import DATASET, training_command
from run_round210_relational import digest, run_logged


def command(fold,stage,smoke=False):
    cmd,_=training_command(fold,stage,14,smoke)
    prefix='runs/round223_smoke' if smoke else f"runs/round223_{fold['name']}"
    cmd[cmd.index('--model-arch')+1]='unext2-fcmae-tiny-signal-aux'
    cmd[cmd.index('--output-dir')+1]=f'{prefix}_stage{stage}'
    cmd+=['--signal-auxiliary-weight','0.05']
    if stage==1:
        cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt"
    else:
        cmd[cmd.index('--init-checkpoint')+1]=f"{prefix}_stage1/epochs/epoch_{1 if smoke else 14:03d}.pt"
    return cmd


def main():
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    folds=[f for f in json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds'] if f['name'] in ('fold0','fold3')]
    sources=[]
    for fold in (folds[:1] if args.smoke else folds):
        encoder=Path(f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt")
        expected=json.loads(Path('reports/round209_encoder_hashes.json').read_text())[str(encoder)]
        assert digest(encoder)==expected
        for stage in (1,2):
            cmd=command(fold,stage,args.smoke);out=Path(cmd[cmd.index('--output-dir')+1])
            if out.exists():raise FileExistsError(out)
            print('Training local-signal auxiliary task',fold['name'],stage,flush=True)
            run_logged(cmd,f"reports/round223_{'smoke' if args.smoke else fold['name']}_stage{stage}.log")
            rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in rows]==list(range(1,(1 if args.smoke else 14)+1))
            assert all(math.isfinite(r['train_loss']) and math.isfinite(r['train_signal_auxiliary_loss']) for r in rows)
            assert any(r['train_signal_auxiliary_loss']>0 for r in rows)
            p=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            steps=[float(v['step']) for v in p['optimizer']['state'].values() if 'step' in v]
            assert steps and min(steps)>0
            assert p['args']['selection_marker']=='CD45RO' and p['args']['signal_auxiliary_weight']==.05
            assert float(p['model']['signal_head.2.weight'][1].abs().sum())>0
            print('Verified',stage,'updates',min(steps),'aux loss',rows[-1]['train_signal_auxiliary_loss'],'peak',rows[-1]['peak_gpu_gib'],flush=True)
        if not args.smoke:
            path=out/'epochs'/'epoch_012.pt'
            sources.append({**fold,'images':140*len(fold['val_rois']),'checkpoint':str(path.resolve()),'checkpoint_sha256':digest(path)})
    if args.smoke:return
    root=Path('runs/round223_oof_sources');root.mkdir(exist_ok=False)
    (root/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher',
        'images':980,'output_source_offset_applied':False,'folds':sources},indent=2)+'\n')
    output=Path('reports/round223_d4_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(root),str(output),'--mode','models',
        '--subset-rois',','.join(r for f in folds for r in f['val_rois']),'--device','cuda:0'],check=True)
    current=json.loads(output.read_text())['markers'];baseline=json.loads(Path('reports/round207_d4_oof.json').read_text())['markers']
    gains={m:{scope:row['gaussian11_score']-baseline[m][scope]['gaussian11_score'] for scope,row in scopes.items()} for m,scopes in current.items()}
    Path('reports/round223_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('CD45RO fixed endpoint gains',json.dumps(gains['CD45RO']),flush=True)


if __name__=='__main__':main()

"""Complete fixed-budget base-model evidence without new full-data training."""
import argparse
import concurrent.futures
import json
import math
import subprocess
import sys
from pathlib import Path
from run_round227_update_control import clock,command as screen_command
from run_round224_cv5 import device_environment
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest


def command(fold,stage):
    cmd=screen_command(fold,stage,'base')
    cmd=[v.replace('round227_','round228_').replace('round207_','round216_') for v in cmd]
    return cmd


def worker(folds,gpu):
    for fold in folds:
        for stage in (1,2):
            print('Training fixed budget',fold['name'],stage,'GPU',gpu,flush=True)
            with Path(f"reports/round228_{fold['name']}_stage{stage}.log").open('w') as log:
                subprocess.run(command(fold,stage),stdout=log,stderr=subprocess.STDOUT,
                               env=device_environment(gpu),check=True)


def main():
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',default='0',choices=('0','0,1'))
    args=parser.parse_args()
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    remaining=[f for f in folds if f['name'] not in ('fold0','fold3')]
    for fold in remaining:
        encoder=Path(f"runs/round216_{fold['name']}_fcmae800/final_encoder.pt")
        baseline=torch.load(f"runs/round216_{fold['name']}_stage1/epochs/epoch_014.pt",map_location='cpu',weights_only=False)
        assert digest(encoder)==baseline['args']['init_checkpoint_report']['checkpoint_sha256']
        enc=torch.load(encoder,map_location='cpu',weights_only=False)
        assert enc['epoch']==800 and set(enc['val_rois'])==set(fold['val_rois'])
        for stage in (1,2):
            cmd=command(fold,stage)
            if Path(cmd[cmd.index('--output-dir')+1]).exists():raise FileExistsError('Existing run')
    jobs=[(remaining,0)] if args.gpus=='0' else [(remaining[:1],1),(remaining[1:],0)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures=[pool.submit(worker,items,gpu) for items,gpu in jobs]
        for future in concurrent.futures.as_completed(futures):future.result()
    sources=[];audit=[]
    for fold in folds:
        prefix='round227' if fold['name'] in ('fold0','fold3') else 'round228'
        plan=clock(2380-140*len(fold['val_rois']))
        for stage in (1,2):
            root=Path(f"runs/{prefix}_{fold['name']}_base_stage{stage}")
            rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in rows]==list(range(1,plan['epochs']+1))
            assert all(math.isfinite(r['train_loss']) and math.isfinite(r['selection_score']) for r in rows)
            ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
            assert ck['args']['model_arch']=='unext2-fcmae-tiny' and ck['args']['seed']==2026
            steps=[float(v['step']) for v in ck['optimizer']['state'].values() if 'step' in v];assert min(steps)>0
            audit.append({'fold':fold['name'],'stage':stage,'epochs':plan['epochs'],
                          'actual_updates':min(steps),'attempted_updates':sum(r['optimizer_steps'] for r in rows)})
        path=root/'epochs'/f"epoch_{plan['endpoint']:03d}.pt"
        sources.append({**fold,'images':len(fold['val_rois'])*140,'checkpoint':str(path.resolve()),'checkpoint_sha256':digest(path)})
    directory=Path('runs/round228_oof_sources');directory.mkdir(exist_ok=False)
    (directory/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher',
          'output_source_offset_applied':False,'images':2380,'folds':sources},indent=2)+'\n')
    Path('reports/round228_completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    output=Path('reports/round228_budget_cv5_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(directory),str(output),
                    '--mode','models','--device','cuda:0'],env=device_environment(0),check=True)
    current=json.loads(output.read_text())['markers']
    reference=json.loads(Path('reports/round217_domain_cv5_oof.json').read_text())['markers']
    gains={m:{s:v['gaussian11_score']-reference[m][s]['gaussian11_score'] for s,v in scopes.items()} for m,scopes in current.items()}
    Path('reports/round228_budget_cv5_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('Budget-adjusted validation changes',json.dumps(gains['CD45RO']),flush=True)


if __name__=='__main__':main()

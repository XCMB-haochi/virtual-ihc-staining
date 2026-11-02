"""Two-GPU unchanged expansion of the local signal auxiliary screen."""
import concurrent.futures
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from run_round223_signal_aux import command as screen_command
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest


def command(fold,stage):
    cmd=screen_command(fold,stage)
    cmd[cmd.index('--output-dir')+1]=f"runs/round224_{fold['name']}_stage{stage}"
    if stage==1:
        cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/round216_{fold['name']}_fcmae800/final_encoder.pt"
    else:
        cmd[cmd.index('--init-checkpoint')+1]=f"runs/round224_{fold['name']}_stage1/epochs/epoch_014.pt"
    return cmd


def device_environment(gpu):
    if gpu not in (0,1):raise ValueError(gpu)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
    env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='1'
    return env


def worker(folds,gpu):
    for fold in folds:
        for stage in (1,2):
            cmd=command(fold,stage)
            print('Training',fold['name'],stage,'physical GPU',gpu,flush=True)
            with Path(f"reports/round224_{fold['name']}_stage{stage}.log").open('w') as log:
                subprocess.run(cmd,env=device_environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    import torch
    torch.set_num_threads(4)
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    lookup={f['name']:f for f in folds}
    for name in ('fold1','fold2','fold4'):
        f=lookup[name]
        encoder=Path(f'runs/round216_{name}_fcmae800/final_encoder.pt')
        reference=torch.load(f'runs/round216_{name}_stage1/epochs/epoch_014.pt',map_location='cpu',weights_only=False)
        assert digest(encoder)==reference['args']['init_checkpoint_report']['checkpoint_sha256']
        enc=torch.load(encoder,map_location='cpu',weights_only=False)
        assert enc['epoch']==800 and set(enc['val_rois'])==set(f['val_rois'])
        for stage in (1,2):
            cmd=command(f,stage)
            if Path(cmd[cmd.index('--output-dir')+1]).exists():raise FileExistsError('Existing expansion output')
    # One longer queue on the card with most memory; shorter fold4 follows fold2.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,[lookup['fold1']],0),
              pool.submit(worker,[lookup['fold2'],lookup['fold4']],1)]
        for job in concurrent.futures.as_completed(jobs):job.result()
    sources=[]
    for fold in folds:
        prefix='round223' if fold['name'] in ('fold0','fold3') else 'round224'
        for stage in (1,2):
            root=Path(f"runs/{prefix}_{fold['name']}_stage{stage}")
            rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in rows]==list(range(1,15))
            assert all(math.isfinite(r['train_loss']) and math.isfinite(r['train_signal_auxiliary_loss']) for r in rows)
            ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
            assert ck['args']['model_arch']=='unext2-fcmae-tiny-signal-aux' and ck['args']['signal_auxiliary_weight']==.05
            assert ck['args']['selection_marker']=='CD45RO' and ck['args']['seed']==2026
            steps=[float(v['step']) for v in ck['optimizer']['state'].values() if 'step' in v];assert steps and min(steps)>0
        path=root/'epochs'/'epoch_012.pt'
        sources.append({**fold,'images':len(fold['val_rois'])*140,
                        'checkpoint':str(path.resolve()),'checkpoint_sha256':digest(path)})
    out=Path('runs/round224_oof_sources');out.mkdir(exist_ok=False)
    (out/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher',
        'output_source_offset_applied':False,'images':2380,'folds':sources},indent=2)+'\n')
    output=Path('reports/round224_cv5_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(out),str(output),
                    '--mode','models','--device','cuda:0'],env=device_environment(1),check=True)
    current=json.loads(output.read_text())['markers'];base=json.loads(Path('reports/round217_domain_cv5_oof.json').read_text())['markers']
    gains={m:{scope:row['gaussian11_score']-base[m][scope]['gaussian11_score'] for scope,row in scopes.items()} for m,scopes in current.items()}
    Path('reports/round224_cv5_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('Full five-fold CD45RO changes',json.dumps(gains['CD45RO']),flush=True)


if __name__=='__main__':main()

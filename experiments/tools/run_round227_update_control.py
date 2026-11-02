"""Paired validation test of full-data-sized update budgets; no platform tuning."""
import concurrent.futures
import json
import math
import subprocess
import sys
from pathlib import Path
from run_round223_signal_aux import command as signal_command
from run_round204_cd45_cv import DATASET,training_command
from run_round224_cv5 import device_environment
from run_round210_relational import digest


def clock(train_images, full_images=2380, effective_batch=16):
    steps=math.ceil(train_images/effective_batch)
    full_steps=math.ceil(full_images/effective_batch)
    epochs=int(14*full_steps/steps+.5)
    endpoint=int(12*full_steps/steps+.5)
    return {'steps_per_epoch':steps,'full_steps_per_epoch':full_steps,
            'epochs':epochs,'endpoint':endpoint,
            'cycle_update_ratio':epochs*steps/(14*full_steps),
            'endpoint_update_ratio':endpoint*steps/(12*full_steps)}


def command(fold,stage,recipe):
    c=clock(2380-140*len(fold['val_rois']))
    if recipe=='signal':cmd=signal_command(fold,stage)
    elif recipe=='base':
        cmd,_=training_command(fold,stage,14)
        if stage==1:
            cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt"
    else:raise ValueError(recipe)
    prefix=f"runs/round227_{fold['name']}_{recipe}"
    cmd[cmd.index('--output-dir')+1]=f'{prefix}_stage{stage}'
    cmd[cmd.index('--epochs')+1]=str(c['epochs'])
    if stage==2:
        cmd[cmd.index('--init-checkpoint')+1]=f"{prefix}_stage1/epochs/epoch_{c['epochs']:03d}.pt"
    return cmd


def worker(fold,gpu):
    for recipe in ('base','signal'):
        for stage in (1,2):
            print('Training update-budget control',fold['name'],recipe,stage,'GPU',gpu,flush=True)
            with Path(f"reports/round227_{fold['name']}_{recipe}_stage{stage}.log").open('w') as log:
                subprocess.run(command(fold,stage,recipe),stdout=log,stderr=subprocess.STDOUT,
                               env=device_environment(gpu),check=True)


def main():
    import torch
    torch.set_num_threads(4)
    folds=[f for f in json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds'] if f['name'] in ('fold0','fold3')]
    receipts=json.loads(Path('reports/round209_encoder_hashes.json').read_text())
    for fold in folds:
        path=Path(f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt")
        assert digest(path)==receipts[str(path)]
        for recipe in ('base','signal'):
            for stage in (1,2):
                cmd=command(fold,stage,recipe)
                if Path(cmd[cmd.index('--output-dir')+1]).exists():raise FileExistsError('Existing run')
    plans={f['name']:clock(2380-140*len(f['val_rois'])) for f in folds}
    Path('reports/round227_clocks.json').write_text(json.dumps(plans,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,fold,gpu) for fold,gpu in zip(folds,(0,1))]
        for job in concurrent.futures.as_completed(jobs):job.result()
    for recipe in ('base','signal'):
        sources=[]
        for fold in folds:
            plan=plans[fold['name']]
            for stage in (1,2):
                root=Path(f"runs/round227_{fold['name']}_{recipe}_stage{stage}")
                rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
                assert [r['epoch'] for r in rows]==list(range(1,plan['epochs']+1))
                assert all(math.isfinite(r['train_loss']) for r in rows)
                last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
                steps=[float(v['step']) for v in last['optimizer']['state'].values() if 'step'in v];assert min(steps)>0
            p=root/'epochs'/f"epoch_{plan['endpoint']:03d}.pt"
            sources.append({**fold,'images':140*len(fold['val_rois']),'checkpoint':str(p.resolve()),'checkpoint_sha256':digest(p)})
        directory=Path(f'runs/round227_{recipe}_sources');directory.mkdir(exist_ok=False)
        (directory/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher',
            'output_source_offset_applied':False,'images':980,'folds':sources},indent=2)+'\n')
        subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(directory),
            f'reports/round227_{recipe}_d4.json','--mode','models','--subset-rois',
            ','.join(r for f in folds for r in f['val_rois']),'--device','cuda:0'],env=device_environment(1),check=True)
    expanded={r:json.loads(Path(f'reports/round227_{r}_d4.json').read_text())['markers']['CD45RO'] for r in ('base','signal')}
    old={r:json.loads(Path('reports',name).read_text())['markers']['CD45RO'] for r,name in
         [('base','round207_d4_oof.json'),('signal','round223_d4_oof.json')]}
    comparison={scope:{'signal_minus_base_long':v['gaussian11_score']-expanded['base'][scope]['gaussian11_score'],
        'signal_minus_base_original':old['signal'][scope]['gaussian11_score']-old['base'][scope]['gaussian11_score'],
        'signal_long_minus_original':v['gaussian11_score']-old['signal'][scope]['gaussian11_score'],
        'base_long_minus_original':expanded['base'][scope]['gaussian11_score']-old['base'][scope]['gaussian11_score']}
        for scope,v in expanded['signal'].items()}
    Path('reports/round227_budget_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print('Update-budget comparison',json.dumps(comparison),flush=True)


if __name__=='__main__':main()

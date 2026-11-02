"""Complete three global teachers, then one fixed full-data CD45RO student."""
import json
import subprocess
import sys
from pathlib import Path
from run_round204_cd45_cv import DATASET, training_command
from run_round207_domain import pretrain_command
from run_round208_full_domain import supervised_command
from run_round210_relational import digest, run_logged


def teacher_commands(fold):
    pretrain, _ = pretrain_command(fold)
    pretrain[pretrain.index('--output-dir')+1]=f"runs/round216_{fold['name']}_fcmae800"
    commands=[pretrain]
    for stage in (1,2):
        command,_=training_command(fold,stage,14)
        command[command.index('--output-dir')+1]=f"runs/round216_{fold['name']}_stage{stage}"
        if stage==1:
            command[command.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/round216_{fold['name']}_fcmae800/final_encoder.pt"
        else:
            command[command.index('--init-checkpoint')+1]=f"runs/round216_{fold['name']}_stage1/epochs/epoch_014.pt"
        commands.append(command)
    return commands


def student_command():
    command=supervised_command(2)
    command[command.index('--output-dir')+1]='runs/round216_full_stage2'
    command+=['--distillation-target-dir','runs/round216_oof_targets/targets',
              '--distillation-marker','CD45RO','--distillation-weight','0.2']
    return command


def main():
    import torch
    import numpy as np
    from PIL import Image
    from torch.utils.data import DataLoader
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from ihc_data import image_names,parse_patch_name,resolve_data_root
    from ihc_torch import PairedPatchDataset
    from evaluate_cv_ensemble import predict_fold
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    root=resolve_data_root(Path(DATASET)); names=image_names(root/'train'/'DAPI')
    assert len(names)==2380
    for fold in folds:
        if fold['name'] in ('fold0','fold3'):continue
        for phase,command in zip(('fcmae800','stage1','stage2'),teacher_commands(fold)):
            output=Path(command[command.index('--output-dir')+1])
            if output.exists():raise FileExistsError(output)
            print('Training missing teacher',fold['name'],phase,flush=True)
            run_logged(command,f"reports/round216_{fold['name']}_{phase}.log")
    cache=Path('runs/round216_oof_targets'); cache.mkdir(exist_ok=False)
    targets=cache/'targets';targets.mkdir()
    sources=[]; covered=[]
    prior=json.loads(Path('runs/round209_domain_audit_sources/manifest.json').read_text())['folds']
    for fold in folds:
        prefix='round207' if fold['name'] in ('fold0','fold3') else 'round216'
        checkpoint=Path(f"runs/{prefix}_{fold['name']}_stage2/epochs/epoch_012.pt")
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ck['epoch']==12 and ck['args']['seed']==2026
        assert not ck['args']['train_all'] and ck['args']['distillation_weight']==0
        assert ck['args']['batch_size']==4 and ck['args']['gradient_accumulation_steps']==4
        assert ck['args']['selection_marker']=='CD45RO' and ck['args']['feature_guidance_weight']==0
        if prefix=='round207':
            source=next(p for p in prior if p['name']==fold['name'])
            assert digest(checkpoint)==source['checkpoint_sha256']
        encoder=torch.load(f"runs/{prefix}_{fold['name']}_fcmae800/final_encoder.pt",map_location='cpu',weights_only=False)
        assert encoder['epoch']==800 and set(encoder['val_rois'])==set(fold['val_rois'])
        assert not set(encoder['config']['train_rois']) & set(fold['val_rois'])
        assert set(encoder['config']['train_rois'])=={parse_patch_name(n).roi for n in names}-set(fold['val_rois'])
        val_names=[n for n in names if parse_patch_name(n).roi in fold['val_rois']]
        loader=DataLoader(PairedPatchDataset(root,val_names,markers=('CD45RO',),augment=False,context_halo=128),batch_size=2,num_workers=0)
        pred,_,actual,_=predict_fold({'teacher':checkpoint},loader,val_names,set(fold['val_rois']),root,'CD45RO',device,'d4',128,{'teacher':128})
        assert actual==val_names
        assert pred['teacher'].shape==(len(actual),256,256) and np.isfinite(pred['teacher']).all()
        for name,value in zip(actual,pred['teacher']):
            Image.fromarray(np.rint(value).clip(0,255).astype(np.uint8)).save(targets/name,format='PNG')
        covered.extend(actual)
        sources.append({**fold,'checkpoint':str(checkpoint.resolve()),'sha256':digest(checkpoint),'images':len(actual)})
        print('Cached CD45RO',fold['name'],len(actual),flush=True)
    assert len(covered)==len(set(covered))==2380 and set(covered)==set(names)
    assert set(image_names(targets))==set(names)
    (cache/'manifest.json').write_text(json.dumps({'marker':'CD45RO','images':2380,'folds':sources,
        'purpose':'full-data student only; not an outer-fold student validation cache'},indent=2)+'\n')
    command=student_command()
    config=json.loads(Path('runs/round208_full_stage2/config.json').read_text())
    assert Path(config['init_checkpoint']).resolve()==Path(command[command.index('--init-checkpoint')+1]).resolve()
    assert config['batch_size']==4 and config['gradient_accumulation_steps']==4 and config['epochs']==14
    assert config['target_l1_weight_strength']==0 and config['distillation_weight']==0
    Path('reports/round216_student_initialization.json').write_text(json.dumps({
        'checkpoint':config['init_checkpoint'],'sha256':digest(config['init_checkpoint'])},indent=2)+'\n')
    if Path('runs/round216_full_stage2').exists():raise FileExistsError('Student exists')
    print('Training full-data stage2 CD45RO distillation',flush=True)
    run_logged(command,'reports/round216_full_stage2.log')
    rows=[json.loads(line) for line in Path('runs/round216_full_stage2/metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,15))
    assert all(np.isfinite(r['train_loss']) for r in rows)
    last=torch.load('runs/round216_full_stage2/last.pt',map_location='cpu',weights_only=False)
    steps=[float(s['step']) for s in last['optimizer']['state'].values() if 'step' in s]
    assert steps and min(steps)>0
    del last
    checkpoint=Path('runs/round216_full_stage2/epochs/epoch_012.pt')
    final=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert final['args']['distillation_marker_index']==1 and final['epoch']==12
    predictions='predictions/round216_cd45ro_q95';package='submissions/semifinal_round216_cd45ro_q95.zip'
    if Path(predictions).exists() or Path(package).exists():raise FileExistsError('Package output exists')
    subprocess.run([sys.executable,'tools/predict_unet.py',DATASET,str(checkpoint),predictions,'--marker','CD45RO',
        '--jpeg-quality','95','--tta','d4','--context-halo','128','--output-source-offset','0,0','--batch-size','2','--device','cuda:0'],check=True)
    subprocess.run([sys.executable,'tools/package_submission.py',predictions,str(root/'test'/'DAPI'),package,'--marker','CD45RO'],check=True)
    subprocess.run([sys.executable,'-m','zipfile','-t',package],check=True)
    Path('reports/round216_hashes.json').write_text(json.dumps({str(checkpoint):digest(checkpoint),package:digest(package)},indent=2)+'\n')


if __name__=='__main__':main()

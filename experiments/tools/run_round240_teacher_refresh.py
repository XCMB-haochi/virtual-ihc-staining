"""Use validated Round239 fold teachers for one fixed full-data student."""
import argparse,json,math,subprocess,sys,zipfile
from pathlib import Path
from run_round239_stage_decay import command as reference_command
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest,run_logged

CACHE=Path('runs/round240_oof_targets')

def command(smoke=False):
    cmd=reference_command(None,2)
    cmd[cmd.index('--output-dir')+1]='runs/round240_smoke_stage2' if smoke else 'runs/round240_full_stage2'
    cmd[cmd.index('--distillation-target-dir')+1]=(CACHE/'targets').as_posix()
    if smoke:
        cmd[cmd.index('--epochs')+1]='1'
        cmd+=['--max-train-images','128']
    return cmd


def main():
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from ihc_data import image_names,parse_patch_name,resolve_data_root
    from ihc_torch import PairedPatchDataset
    from evaluate_cv_ensemble import predict_fold
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    root=resolve_data_root(Path(DATASET));names=image_names(root/'train'/'DAPI')
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    reference=json.loads(Path('reports/round239_oof.json').read_text())
    assert reference['markers']['CD45RO']['overall']['images']==len(names)==2380
    sources=[]
    for fold in folds:
        path=Path(f"runs/round239_{fold['name']}_stage2/epochs/epoch_012.pt")
        ck=torch.load(path,map_location='cpu',weights_only=False);a=ck['args']
        assert ck['epoch']==12 and not a['train_all'] and a['distillation_weight']==0
        assert a['seed']==2026 and a['encoder_stage_decay']==.8
        assert set(a['val_rois'].split(','))==set(fold['val_rois'])
        assert not set(a['train_rois_resolved'])&set(fold['val_rois'])
        source=next(x for x in reference['fold_sources'] if x['name']==fold['name'])
        sha=digest(path);assert sha==source['checkpoint_sha256']
        sources.append({**fold,'checkpoint':str(path.resolve()),'sha256':sha,'images':140*len(fold['val_rois'])})
    init=Path('runs/round239_full_stage1/epochs/epoch_014.pt')
    initsha=digest(init)
    old=torch.load('runs/round239_full_stage2/epochs/epoch_012.pt',map_location='cpu',weights_only=False)
    assert Path(old['args']['init_checkpoint']).resolve()==init.resolve()
    original=torch.load(init,map_location='cpu',weights_only=False)
    assert original['epoch']==14 and original['args']['train_all'] and original['args']['encoder_stage_decay']==.8
    assert all(torch.isfinite(v).all() for v in original['model'].values())
    del old,original,ck
    if args.preflight:
        print('All five evaluated teacher hashes, fold exclusion and fixed full-data stage1 verified',flush=True)
        return
    if CACHE.exists() or Path('runs/round240_full_stage2').exists():raise FileExistsError('Existing trajectory/cache')
    targets=CACHE/'targets';targets.mkdir(parents=True)
    covered=[]
    for source in sources:
        val=[n for n in names if parse_patch_name(n).roi in source['val_rois']]
        loader=DataLoader(PairedPatchDataset(root,val,markers=('CD45RO',),augment=False,context_halo=128),batch_size=2,num_workers=0)
        pred,_,actual,_=predict_fold({'teacher':Path(source['checkpoint'])},loader,val,set(source['val_rois']),root,'CD45RO',torch.device('cuda:0'),'d4',128,{'teacher':128})
        assert actual==val and pred['teacher'].shape==(len(val),256,256) and np.isfinite(pred['teacher']).all()
        for name,value in zip(actual,pred['teacher']):
            Image.fromarray(np.rint(value).clip(0,255).astype(np.uint8)).save(targets/name,format='PNG')
        covered+=val
        print('Cached fold-excluded targets',source['name'],len(val),flush=True)
    assert len(covered)==len(set(covered))==2380 and set(covered)==set(names)==set(image_names(targets))
    (CACHE/'manifest.json').write_text(json.dumps({'marker':'CD45RO','images':2380,'folds':sources,'purpose':'full-data student only; not an outer-fold student validation cache'},indent=2)+'\n')
    cmd=command();run_logged(cmd,'reports/round240_full_stage2.log')
    out=Path('runs/round240_full_stage2');rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,15)) and all(math.isfinite(r['train_loss']) for r in rows)
    config=json.loads((out/'config.json').read_text());assert config['train_images']==2380 and config['val_images']==0
    last=torch.load(out/'last.pt',map_location='cpu',weights_only=False);a=last['args']
    assert a['train_all'] and a['distillation_weight']==.2 and a['distillation_marker_index']==1 and a['encoder_stage_decay']==.8 and a['seed']==2026
    assert Path(a['init_checkpoint']).resolve()==init.resolve() and digest(init)==initsha
    assert Path(a['distillation_target_dir']).resolve()==targets.resolve()
    steps=[float(s['step']) for s in last['optimizer']['state'].values() if 'step' in s];assert steps and min(steps)>0
    model=out/'epochs'/'epoch_012.pt';ck=torch.load(model,map_location='cpu',weights_only=False)
    assert ck['epoch']==12 and all(torch.isfinite(v).all() for v in ck['model'].values())
    pred='predictions/round240_cd45ro_q95';package='submissions/semifinal_round240_cd45ro_q95.zip'
    if Path(pred).exists() or Path(package).exists():raise FileExistsError('Package output exists')
    subprocess.run([sys.executable,'tools/predict_unet.py',DATASET,str(model),pred,'--marker','CD45RO','--jpeg-quality','95','--tta','d4','--context-halo','128','--output-source-offset','0,0','--batch-size','2','--device','cuda:0'],check=True)
    subprocess.run([sys.executable,'tools/package_submission.py',pred,str(root/'test'/'DAPI'),package,'--marker','CD45RO'],check=True)
    with zipfile.ZipFile(package) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==560
    Path('reports/round240_artifacts.json').write_text(json.dumps({'stage1_sha256':initsha,'checkpoint_sha256':digest(model),'package':package,'package_sha256':digest(Path(package)),'actual_updates':min(steps),'teachers':sources},indent=2)+'\n')

if __name__=='__main__':main()

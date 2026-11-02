"""Fixed late-stage Gaussian-valid objective on the current single-model baseline."""
import argparse,concurrent.futures,json,math,subprocess,sys,zipfile
from pathlib import Path
from run_round239_stage_decay import command as fold_command
from run_round240_teacher_refresh import command as full_command
from run_round204_cd45_cv import DATASET
from run_round224_cv5 import device_environment
from run_round210_relational import digest


def command(fold,smoke=False):
    cmd=full_command() if fold is None else fold_command(fold,2)
    name='full' if fold is None else fold['name']
    cmd[cmd.index('--output-dir')+1]=f"runs/round241_{'smoke_' if smoke else ''}{name}_stage2"
    cmd[cmd.index('--ssim-mode')+1]='gaussian11_valid'
    if smoke:
        cmd[cmd.index('--epochs')+1]='1';cmd+=['--max-train-images','128','--max-val-images','8']
    return cmd


def preflight(fold):
    import torch
    cmd=command(fold);init=Path(cmd[cmd.index('--init-checkpoint')+1])
    name='full' if fold is None else fold['name']
    base=Path('runs/round240_full_stage2') if fold is None else Path(f'runs/round239_{name}_stage2')
    cfg=json.loads((base/'config.json').read_text())
    assert Path(cfg['init_checkpoint']).resolve()==init.resolve()
    ck=torch.load(init,map_location='cpu',weights_only=False);a=ck['args']
    assert ck['epoch']==14 and a['seed']==2026 and a['encoder_stage_decay']==.8
    assert a['train_all']==(fold is None)
    if fold is not None:
        assert set(a['val_rois'].split(','))==set(fold['val_rois'])
        assert not set(a['train_rois_resolved'])&set(fold['val_rois'])
    return digest(init)


def worker(items,gpu,smoke=False):
    import torch
    records=[]
    for fold in items:
        initsha=preflight(fold);cmd=command(fold,smoke)
        out=Path(cmd[cmd.index('--output-dir')+1]);name='full' if fold is None else fold['name']
        print('Training late Gaussian',name,'GPU',gpu,flush=True)
        with Path(f"reports/round241_{'smoke_' if smoke else ''}{name}.log").open('w') as log:
            subprocess.run(cmd,env=device_environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in rows]==list(range(1,(1 if smoke else 14)+1))
        assert all(math.isfinite(r['train_loss']) for r in rows)
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);a=ck['args']
        cfg=json.loads((out/'config.json').read_text())
        assert a['ssim_mode']=='gaussian11_valid' and a['encoder_stage_decay']==.8 and a['seed']==2026
        assert digest(Path(a['init_checkpoint']))==initsha
        assert a['train_all']==(fold is None)
        if fold is None:
            assert a['distillation_weight']==.2 and a['distillation_marker_index']==1
            assert Path(a['distillation_target_dir']).resolve()==Path('runs/round240_oof_targets/targets').resolve()
            if not smoke:assert cfg['train_images']==2380 and cfg['val_images']==0
        else:
            assert a['distillation_weight']==0 and set(a['val_rois'].split(','))==set(fold['val_rois'])
            assert not set(a['train_rois_resolved'])&set(fold['val_rois'])
            if not smoke:assert cfg['val_images']==140*len(fold['val_rois']) and cfg['train_images']==2380-cfg['val_images']
        steps=[float(s['step']) for s in ck['optimizer']['state'].values() if 'step' in s];assert steps and min(steps)>0
        assert all(torch.isfinite(v).all() for v in ck['model'].values())
        rec={'name':name,'init_sha256':initsha,'actual_steps':min(steps),'peak_gpu_gib':max(r['peak_gpu_gib'] for r in rows)}
        records.append(rec);print(json.dumps(rec),flush=True)
    return records


def main():
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    for f in ([None,folds[0]] if args.smoke else [None,*folds]):
        cmd=command(f,args.smoke)
        if Path(cmd[cmd.index('--output-dir')+1]).exists():raise FileExistsError('Existing stage')
        preflight(f)
    cache=Path('runs/round240_oof_targets');manifest=json.loads((cache/'manifest.json').read_text())
    assert manifest['marker']=='CD45RO' and manifest['images']==2380
    assert {p.name for p in (cache/'targets').glob('*.jpg')}=={p.name for p in (Path(DATASET)/'train'/'DAPI').glob('*.jpg')}
    # GPU1 is contended: shorter queue prevents it dominating total time.
    queues=[([None,folds[0]],0)] if args.smoke else [([None,folds[0],folds[2],folds[3]],0),([folds[1],folds[4]],1)]
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queues)) as pool:
        futures=[pool.submit(worker,items,gpu,args.smoke) for items,gpu in queues]
        for future in concurrent.futures.as_completed(futures):records+=future.result()
    Path(f"reports/round241_{'smoke_' if args.smoke else ''}audit.json").write_text(json.dumps(records,indent=2)+'\n')
    if args.smoke:return
    sources=[]
    for f in folds:
        p=Path(f"runs/round241_{f['name']}_stage2/epochs/epoch_012.pt")
        sources.append({**f,'images':140*len(f['val_rois']),'checkpoint':str(p.resolve()),'checkpoint_sha256':digest(p)})
    root=Path('runs/round241_oof_sources');root.mkdir(exist_ok=False)
    (root/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher','output_source_offset_applied':False,'images':2380,'folds':sources},indent=2)+'\n')
    output=Path('reports/round241_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(root),str(output),'--mode','models','--device','cuda:0'],env=device_environment(0),check=True)
    current=json.loads(output.read_text())['markers']['CD45RO'];base=json.loads(Path('reports/round239_oof.json').read_text())['markers']['CD45RO']
    assert current['overall']['images']==2380
    gains={s:r['gaussian11_score']-base[s]['gaussian11_score'] for s,r in current.items()}
    decision={'gains':gains,'package':gains['overall']>=.03 and sum(gains[f['name']]>0 for f in folds)>=3}
    Path('reports/round241_decision.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision),flush=True)
    if not decision['package']:return
    model='runs/round241_full_stage2/epochs/epoch_012.pt';pred='predictions/round241_cd45ro_q95';package='submissions/semifinal_round241_cd45ro_q95.zip'
    if Path(pred).exists() or Path(package).exists():raise FileExistsError('Existing output')
    subprocess.run([sys.executable,'tools/predict_unet.py',DATASET,model,pred,'--marker','CD45RO','--jpeg-quality','95','--tta','d4','--context-halo','128','--output-source-offset','0,0','--batch-size','2','--device','cuda:0'],env=device_environment(0),check=True)
    subprocess.run([sys.executable,'tools/package_submission.py',pred,DATASET+'/test/DAPI',package,'--marker','CD45RO'],check=True)
    with zipfile.ZipFile(package) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==560
    Path('reports/round241_hashes.json').write_text(json.dumps({model:digest(Path(model)),package:digest(Path(package))},indent=2)+'\n')

if __name__=='__main__':main()

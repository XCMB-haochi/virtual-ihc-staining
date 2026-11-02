"""Stage-wise encoder adaptation: parallel full candidate and complete grouped validation."""
import argparse,concurrent.futures,json,math,subprocess,sys,zipfile
from pathlib import Path
from run_round204_cd45_cv import DATASET,training_command
from run_round208_full_domain import supervised_command
from run_round216_distill import student_command
from run_round224_cv5 import device_environment
from run_round210_relational import digest


def command(fold,stage,smoke=False):
    full=fold is None
    if full:cmd=supervised_command(1) if stage==1 else student_command()
    else:cmd,_=training_command(fold,stage,14)
    name='full' if full else fold['name']
    prefix=f"runs/round239_{'smoke_' if smoke else ''}{name}"
    cmd[cmd.index('--output-dir')+1]=f'{prefix}_stage{stage}'
    if stage==1 and not full:
        old='round207' if name in ('fold0','fold3') else 'round216'
        cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1]=f'runs/{old}_{name}_fcmae800/final_encoder.pt'
    if stage==2:cmd[cmd.index('--init-checkpoint')+1]=f"{prefix}_stage1/epochs/epoch_{1 if smoke else 14:03d}.pt"
    cmd+=['--encoder-stage-decay','0.8']
    if smoke:
        cmd[cmd.index('--epochs')+1]='1'
        cmd+=['--max-train-images','128','--max-val-images','8']
    assert ('--train-all' in cmd)==full
    assert ('--distillation-target-dir' in cmd)==(full and stage==2)
    return cmd


def audit(fold,stage,smoke=False):
    import torch
    cmd=command(fold,stage,smoke);out=Path(cmd[cmd.index('--output-dir')+1])
    ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    c=json.loads((out/'config.json').read_text());a=ck['args']
    rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
    epochs=1 if smoke else 14
    assert ck['epoch']==epochs and [r['epoch'] for r in rows]==list(range(1,epochs+1))
    assert all(math.isfinite(r['train_loss']) for r in rows)
    assert a['seed']==2026 and a['encoder_stage_decay']==.8
    assert a['context_halo']==a['train_target_context_halo']==64
    assert a['batch_size']==4 and a['gradient_accumulation_steps']==4
    assert a['train_all']==(fold is None)
    if fold is not None:
        assert set(a['val_rois'].split(','))==set(fold['val_rois'])
        assert not set(a['train_rois_resolved'])&set(fold['val_rois'])
        assert a['distillation_weight']==0
        if not smoke:assert c['val_images']==140*len(fold['val_rois']) and c['train_images']==2380-c['val_images']
    else:
        if not smoke:assert c['train_images']==2380 and c['val_images']==0
        if stage==2:assert a['distillation_weight']==.2 and a['distillation_marker_index']==1
    if stage==2:assert str(a['init_checkpoint']).endswith(cmd[cmd.index('--init-checkpoint')+1])
    groups=a['optimizer_parameter_groups'];rates=[g['initial_learning_rate'] for g in groups]
    lr=3e-4 if stage==1 else 1e-4
    assert len(rates)==6 and all(math.isclose(x,y,rel_tol=1e-10) for x,y in zip(rates,[lr,*[lr*.8**e for e in (4,3,2,1,0)]]))
    steps=[float(v['step']) for v in ck['optimizer']['state'].values() if 'step' in v]
    assert steps and min(steps)>0
    assert all(torch.isfinite(v).all() for v in ck['model'].values())
    return {'name':'full' if fold is None else fold['name'],'stage':stage,'actual_steps':min(steps),
            'seconds':sum(r['seconds'] for r in rows),'peak_gpu_gib':max(r['peak_gpu_gib'] for r in rows),'group_rates':rates}


def preflight(fold):
    import torch
    cmd=command(fold,1);enc=Path(cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1])
    if fold is None:baseline='runs/round208_full_stage1/epochs/epoch_014.pt'
    else:
        old='round207' if fold['name'] in ('fold0','fold3') else 'round216'
        baseline=f"runs/{old}_{fold['name']}_stage1/epochs/epoch_014.pt"
    ref=torch.load(baseline,map_location='cpu',weights_only=False)
    assert digest(enc)==ref['args']['init_checkpoint_report']['checkpoint_sha256']
    state=torch.load(enc,map_location='cpu',weights_only=False)
    assert state['epoch']==800
    if fold is not None:
        assert set(state['val_rois'])==set(fold['val_rois'])
        assert not set(state['config']['train_rois'])&set(fold['val_rois'])
    else:assert state['val_rois']==[]


def worker(items,gpu,smoke=False):
    records=[]
    for fold in items:
        for stage in (1,2):
            cmd=command(fold,stage,smoke);name='full' if fold is None else fold['name']
            print('Starting',name,stage,'physical GPU',gpu,flush=True)
            with Path(f"reports/round239_{'smoke_' if smoke else ''}{name}_stage{stage}.log").open('w') as log:
                subprocess.run(cmd,env=device_environment(gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
            record=audit(fold,stage,smoke);records.append(record);print(json.dumps(record),flush=True)
    return records


def main():
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    items=[None,folds[0]] if args.smoke else [None,*folds]
    for fold in items:
        preflight(fold)
        for stage in (1,2):
            cmd=command(fold,stage,args.smoke)
            if Path(cmd[cmd.index('--output-dir')+1]).exists():raise FileExistsError('Existing training directory')
    cache=Path('runs/round216_oof_targets');m=json.loads((cache/'manifest.json').read_text())
    assert m['marker']=='CD45RO' and m['images']==2380
    assert {p.name for p in (cache/'targets').glob('*.jpg')}=={p.name for p in (Path(DATASET)/'train'/'DAPI').glob('*.jpg')}
    queues=[([None],0),([folds[0]],1)] if args.smoke else [([None,folds[0],folds[3]],0),([folds[1],folds[2],folds[4]],1)]
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,jobs,gpu,args.smoke) for jobs,gpu in queues]
        for result in concurrent.futures.as_completed(jobs):records+=result.result()
    Path(f"reports/round239_{'smoke_' if args.smoke else ''}training_audit.json").write_text(json.dumps(records,indent=2)+'\n')
    if args.smoke:return
    sources=[]
    for fold in folds:
        p=Path(f"runs/round239_{fold['name']}_stage2/epochs/epoch_012.pt")
        sources.append({**fold,'images':len(fold['val_rois'])*140,'checkpoint':str(p.resolve()),'checkpoint_sha256':digest(p)})
    root=Path('runs/round239_oof_sources');root.mkdir(exist_ok=False)
    (root/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher','output_source_offset_applied':False,'images':2380,'folds':sources},indent=2)+'\n')
    output=Path('reports/round239_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(root),str(output),'--mode','models','--device','cuda:0'],env=device_environment(0),check=True)
    current=json.loads(output.read_text())['markers']['CD45RO'];base=json.loads(Path('reports/round217_domain_cv5_oof.json').read_text())['markers']['CD45RO']
    assert current['overall']['images']==2380
    gains={s:r['gaussian11_score']-base[s]['gaussian11_score'] for s,r in current.items()}
    decision={'gains':gains,'package':gains['overall']>=.03 and sum(gains[f['name']]>0 for f in folds)>=3}
    Path('reports/round239_decision.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision),flush=True)
    if not decision['package']:return
    model='runs/round239_full_stage2/epochs/epoch_012.pt';pred='predictions/round239_cd45ro_q95';package='submissions/semifinal_round239_cd45ro_q95.zip'
    if Path(pred).exists() or Path(package).exists():raise FileExistsError('Existing package output')
    subprocess.run([sys.executable,'tools/predict_unet.py',DATASET,model,pred,'--marker','CD45RO','--jpeg-quality','95','--tta','d4','--context-halo','128','--output-source-offset','0,0','--batch-size','2','--device','cuda:0'],env=device_environment(0),check=True)
    subprocess.run([sys.executable,'tools/package_submission.py',pred,DATASET+'/test/DAPI',package,'--marker','CD45RO'],check=True)
    with zipfile.ZipFile(package) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==560
    Path('reports/round239_hashes.json').write_text(json.dumps({model:digest(Path(model)),package:digest(Path(package))},indent=2)+'\n')

if __name__=='__main__':main()

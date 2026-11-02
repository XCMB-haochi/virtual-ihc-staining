"""Complete seed2026 Colon workflow with resource-aware detached scheduling."""
import argparse,concurrent.futures,json,math,os,subprocess,sys,time,zipfile
from collections import defaultdict
from pathlib import Path
from run_round210_relational import digest
from run_round224_cv5 import device_environment

CONFIG=Path('configs/round137_colon_seed2026.json')
C=json.loads(CONFIG.read_text(encoding='utf-8'))
ROOT=Path('runs/round137_colon')
REPORTS=Path('reports/round137_colon')
SCOPES={'full':[],**{f['name']:f['val_rois'] for f in C['folds']}}


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def pretrain_command(scope,kind):
    full=scope=='full';teacher=kind=='teacher'
    epochs=C['feature_teacher_epochs'] if teacher else C['dapi_epochs_full' if full else 'dapi_epochs_fold']
    return [sys.executable,'tools/train_dapi_fcmae.py',C['dataset'],'--output-dir',str(ROOT/scope/kind),
      '--val-rois',','.join(SCOPES[scope]),'--data-scope','all-train' if full else 'fold-train',
      '--pretraining-markers','DAPI,CD68,CD45RO,HLA-DR,Vimentin' if teacher else 'DAPI',
      '--epochs',str(epochs),'--batch-size','32','--gradient-accumulation-steps','1','--workers','4',
      '--learning-rate','0.0002','--minimum-learning-rate','0.000001','--warmup-epochs','2' if teacher else ('8' if full else '10'),
      '--weight-decay','0.05','--mask-ratio','0.5','--checkpoint-interval','5','--seed','2026','--model-size','tiny',
      '--encoder-init-checkpoint',C['public_checkpoint'],'--encoder-init-mode','official-convnextv2-fcmae-imagenet1k-tiny',
      '--encoder-init-sha256',C['public_sha256'],'--device','cuda:0']


def student_command(scope,arm,stage,teacher_sha):
    full=scope=='full';base=ROOT/scope;out=base/arm/f'stage{stage}'
    cmd=[sys.executable,'tools/train_unet.py',C['dataset'],'--output-dir',str(out),'--val-rois',','.join(SCOPES[scope]),
      '--markers','CD68,CD45RO,HLA-DR,Vimentin','--marker-loss-weights','0.85,0.05,0.05,0.05','--epochs','14',
      '--batch-size','16','--gradient-accumulation-steps','1','--model-arch','unext2-fcmae-tiny','--base-channels','32',
      '--block-type','residual','--condition-mode','spatial-thumbnail-film','--context-halo','64','--train-target-context-halo','64',
      '--learning-rate','0.0003' if stage==1 else '0.0001','--minimum-learning-rate','0','--gradient-clip-norm','0',
      '--weight-decay','0.0001','--ssim-weight','0.7','--l1-weight','0.2','--mse-weight','0.1','--loss-profile','hybrid',
      '--ssim-mode','uniform7','--selection-marker','CD68','--selection-ssim-mode','uniform7',
      '--selection-psnr-aggregation','global-mse','--selection-jpeg-quality','95','--workers','4','--seed','2026',
      '--device','cuda:0','--save-every-epoch','--encoder-stage-decay',str(C['arms'][arm]),
      '--feature-guidance-checkpoint',str(base/'teacher'/'final_encoder.pt'),'--feature-guidance-checkpoint-sha256',teacher_sha,
      '--feature-guidance-marker','CD68','--feature-guidance-weight','0.05','--feature-guidance-mode','encoder-relational-robust',
      '--feature-guidance-stage','4','--feature-guidance-crop-size','256','--feature-guidance-batch-samples','16']
    if full:cmd+=['--train-all']
    if stage==1:cmd+=['--fcmae-encoder-init-checkpoint',str(base/'dapi'/'final_encoder.pt')]
    else:cmd+=['--init-checkpoint',str(base/arm/'stage1'/'epochs'/'epoch_014.pt')]
    if full and stage==2:cmd+=['--distillation-target-dir',str(ROOT/'oof_targets'/'targets'),'--distillation-marker','CD68','--distillation-weight','0.2']
    return cmd


def plan():
    jobs=[]
    for scope in SCOPES:
        for kind in ('dapi','teacher'):
            jobs.append({'id':f'{scope}.{kind}','deps':[],'memory':24500,'kind':kind,'scope':scope})
        for arm in C['arms']:
            for stage in (1,2):
                deps=[f'{scope}.dapi',f'{scope}.teacher'] if stage==1 else [f'{scope}.{arm}.s1']
                if scope=='full' and stage==2:deps+=['cache']
                jobs.append({'id':f'{scope}.{arm}.s{stage}','deps':deps,'memory':32000,'kind':'student','scope':scope,'arm':arm,'stage':stage})
            jobs.append({'id':f'{scope}.{arm}.average','deps':[f'{scope}.{arm}.s2'],'memory':0,'kind':'average','scope':scope,'arm':arm})
        if scope!='full':
            jobs.append({'id':f'{scope}.evaluate','deps':[f'{scope}.{a}.average' for a in C['arms']],'memory':8500,'kind':'evaluate','scope':scope})
    jobs.append({'id':'cache','deps':[f'{f["name"]}.evaluate' for f in C['folds']],'memory':0,'kind':'cache'})
    for arm in C['arms']:
        jobs.append({'id':f'full.{arm}.package','deps':[f'full.{arm}.average'],'memory':8500,'kind':'package','scope':'full','arm':arm})
    return jobs


def launch(cmd):
    print('COMMAND',json.dumps(cmd,ensure_ascii=False),flush=True)
    subprocess.run(cmd,check=True)


def audit_training(path,scope,epochs,kind,arm=None,stage=None):
    import torch
    config=json.loads((path/'config.json').read_text())
    assert config['seed']==2026
    assert Path(config['dataset']).resolve()==Path(C['dataset']).resolve()
    rows=[json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,epochs+1))
    if kind in ('dapi','teacher'):
        ck=torch.load(path/'final_encoder.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==epochs and ck['config']['seed']==2026
        assert set(ck['val_rois'])==set(SCOPES[scope])
        assert not set(ck['config']['train_rois'])&set(SCOPES[scope])
        assert all(torch.isfinite(t).all() for t in ck['encoder'].values())
        last=torch.load(path/'latest.pt',map_location='cpu',weights_only=False)
        for r in rows:
            assert all(not isinstance(v,float) or math.isfinite(v) for k,v in r.items() if 'loss' in k)
    else:
        last=torch.load(path/'last.pt',map_location='cpu',weights_only=False);a=last['args']
        assert last['epoch']==14 and a['seed']==2026 and a['batch_size']==16 and a['gradient_accumulation_steps']==1
        assert a['encoder_stage_decay']==C['arms'][arm] and a['feature_guidance_batch_samples']==16
        assert a['feature_guidance_weight']==.05 and a['feature_guidance_mode']=='encoder-relational-robust'
        assert a['train_all']==(scope=='full') and set(a['val_rois'].split(','))-{''}==set(SCOPES[scope])
        assert not set(a['train_rois_resolved'])&set(SCOPES[scope])
        assert config['train_images']+config['val_images']==C['train_images']
        assert all(math.isfinite(r['train_loss']) for r in rows)
        assert all(torch.isfinite(t).all() for t in last['model'].values())
        if scope=='full' and stage==2:assert a['distillation_marker_index']==0 and a['distillation_weight']==.2
        else:assert a['distillation_weight']==0
        if stage==2:assert Path(a['init_checkpoint']).resolve()==(ROOT/scope/arm/'stage1/epochs/epoch_014.pt').resolve()
    steps=[float(s['step']) for s in last['optimizer']['state'].values() if 'step' in s];assert steps and min(steps)>0
    return {'epochs':epochs,'actual_updates':min(steps),'seed':2026}


def evaluate(scope):
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from evaluate_cv_ensemble import predict_fold
    from audit_semifinal_markers import image_metrics,summarize
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from ihc_data import image_names,parse_patch_name,resolve_data_root
    from ihc_torch import PairedPatchDataset,shift_array_edge
    root=resolve_data_root(Path(C['dataset']));names=[n for n in image_names(root/'train/DAPI') if parse_patch_name(n).roi in SCOPES[scope]]
    report={'scope':scope,'images':len(names),'arms':{}}
    for arm in C['arms']:
        ck=ROOT/scope/arm/'model.pt'
        loader=DataLoader(PairedPatchDataset(root,names,markers=('CD68',),augment=False,context_halo=128),batch_size=2,num_workers=0)
        pred,target,actual,_=predict_fold({'model':ck},loader,names,set(SCOPES[scope]),root,'CD68',torch.device('cuda:0'),'d4',128,{'model':128})
        assert actual==names and np.isfinite(pred['model']).all()
        if arm=='control':
            folder=ROOT/'oof_targets'/'targets';folder.mkdir(parents=True,exist_ok=True)
            for name,value in zip(names,pred['model']):
                p=folder/name
                if p.exists():raise FileExistsError(p)
                Image.fromarray(np.rint(value).clip(0,255).astype(np.uint8)).save(p,format='PNG')
        results={}
        for label,offset in [('zero',(0,0)),('shift',(0,-1))]:
            values=shift_array_edge(pred['model'],offset)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                metrics=list(pool.map(lambda pair:image_metrics(*pair),zip(values,target)))
            byroi=defaultdict(list)
            for name,row in zip(names,metrics):byroi[parse_patch_name(name).roi].append(row)
            results[label]={'overall':summarize(metrics),'rois':{r:summarize(rows) for r,rows in byroi.items()}}
        report['arms'][arm]={'checkpoint_sha256':digest(ck),'metrics':results}
        del pred,target
    output=REPORTS/f'{scope}_paired_offsets.json';write_json(output,report)
    return [output]


def finalize_cache():
    from ihc_data import image_names
    names=image_names(Path(C['dataset'])/'train/DAPI');folder=ROOT/'oof_targets'
    assert len(names)==6296 and set(names)==set(image_names(folder/'targets'))
    reports=[json.loads((REPORTS/f'{f["name"]}_paired_offsets.json').read_text()) for f in C['folds']]
    assert sum(r['images'] for r in reports)==6296
    summary={}
    for arm in C['arms']:
        summary[arm]={}
        for offset in ('zero','shift'):
            groups=[r['arms'][arm]['metrics'][offset]['overall'] for r in reports]
            summary[arm][offset]={k:sum(g[k]*g['images'] for g in groups)/6296 for k in groups[0] if k not in ('images','psnr_global_mse')}
            summary[arm][offset]['images']=6296
    sources=[{'name':f['name'],'val_rois':f['val_rois'],'checkpoint':str((ROOT/f['name']/'control/model.pt').resolve()),'sha256':digest(ROOT/f['name']/'control/model.pt')} for f in C['folds']]
    write_json(folder/'manifest.json',{'seed':2026,'images':6296,'marker':'CD68','purpose':'full-data-student-only; baseline teacher cache shared by both arms','folds':sources})
    write_json(REPORTS/'cv5_summary.json',{'metrics':summary,'fold_reports':[str(REPORTS/f'{f["name"]}_paired_offsets.json') for f in C['folds']],
      'limitations':'Repeated development ROIs; provisional Gaussian11 profile. Final full-data distilled students have no independent OOF score. Shift diagnosis is not a submission.'})
    return [folder/'manifest.json',REPORTS/'cv5_summary.json']


def execute_task(task):
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    scope=task.get('scope');kind=task['kind'];audit=None
    if kind in ('dapi','teacher'):
        cmd=pretrain_command(scope,kind);out=ROOT/scope/kind
        if (out/'final_encoder.pt').exists():raise FileExistsError('Unrecorded completed pretraining needs audit')
        if out.exists():
            if not (out/'latest.pt').is_file():raise RuntimeError('Incomplete pretraining without resumable state')
            cmd+=['--resume']
        launch(cmd);audit=audit_training(out,scope,int(cmd[cmd.index('--epochs')+1]),kind)
        files=[out/'final_encoder.pt']
    elif kind=='student':
        arm=task['arm'];stage=task['stage'];teacher=ROOT/scope/'teacher/final_encoder.pt'
        cmd=student_command(scope,arm,stage,digest(teacher));out=ROOT/scope/arm/f'stage{stage}'
        if out.exists():raise FileExistsError('Do not silently restart or reuse partial supervised training')
        launch(cmd);audit=audit_training(out,scope,14,kind,arm,stage)
        files=[out/'epochs/epoch_014.pt'] if stage==1 else [out/'epochs'/f'epoch_{e:03d}.pt' for e in C['final_epochs']]
    elif kind=='average':
        from average_checkpoints import average_checkpoints
        base=ROOT/scope/task['arm'];out=base/'model.pt'
        if out.exists():raise FileExistsError(out)
        audit=average_checkpoints([base/'stage2/epochs'/f'epoch_{e:03d}.pt' for e in C['final_epochs']],C['final_weights'],out,buffer_policy='latest')
        files=[out]
    elif kind=='evaluate':files=evaluate(scope)
    elif kind=='cache':files=finalize_cache()
    elif kind=='package':
        arm=task['arm'];model=ROOT/'full'/arm/'model.pt';pred=Path(f'predictions/round137_colon_{arm}_zero');package=Path(f'submissions/colon_round137_{arm}_seed2026_zero_q95.zip')
        if pred.exists() or package.exists():raise FileExistsError('Existing package output')
        launch([sys.executable,'tools/predict_unet.py',C['dataset'],str(model),str(pred),'--marker','CD68','--jpeg-quality','95','--tta','d4','--context-halo','128','--output-source-offset','0,0','--batch-size','2','--device','cuda:0'])
        launch([sys.executable,'tools/package_submission.py',str(pred),C['dataset']+'/test/DAPI',str(package),'--marker','CD68'])
        with zipfile.ZipFile(package) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==C['test_images']
        files=[model,package]
    else:raise ValueError(kind)
    write_json(REPORTS/'done'/f"{task['id']}.json",{'task':task,'audit':audit,'files':{str(p):digest(p) for p in files}})


def preflight():
    from ihc_data import image_names,parse_patch_name,resolve_data_root
    root=resolve_data_root(Path(C['dataset']));names=image_names(root/'train/DAPI')
    assert len(names)==C['train_images'] and len(image_names(root/'test/DAPI'))==C['test_images']
    rois=[r for f in C['folds'] for r in f['val_rois']]
    assert len(rois)==len(set(rois))==25 and set(rois)=={parse_patch_name(n).roi for n in names}
    for marker in ('CD68','CD45RO','HLA-DR','Vimentin'):assert set(image_names(root/'train'/marker))==set(names)
    assert digest(Path(C['public_checkpoint']))==C['public_sha256']
    assert C['seed']==2026 and C['submission_offset']==[0,0]
    print('Colon inventory, fold partition and public-only initialization verified',flush=True)


def gpu_resources():
    out=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True)
    return [tuple(int(x.strip()) for x in line.split(',')) for line in out.strip().splitlines()]


def schedule():
    REPORTS.mkdir(parents=True,exist_ok=True)
    tasks=plan();write_json(REPORTS/'plan.json',tasks)
    if C['wait_for_round241']:
        marker=Path('reports/round241_pipeline.exit')
        print('Waiting for existing Round241 phase to end; no interference',flush=True)
        while not marker.exists():time.sleep(60)
        print('Round241 phase ended with exit',marker.read_text().strip(),flush=True)
    completed=set()
    for task in tasks:
        p=REPORTS/'done'/f"{task['id']}.json"
        if p.exists():
            record=json.loads(p.read_text());assert record['task']==task
            assert all(Path(f).is_file() and digest(Path(f))==sha for f,sha in record['files'].items())
            completed.add(task['id'])
    running={};failure=None
    while len(completed)<len(tasks):
        for name,(proc,gpu,log) in list(running.items()):
            code=proc.poll()
            if code is None:continue
            log.close();del running[name]
            if code!=0:failure=f'{name} exited{code}; remaining active tasks are allowed to finish'
            else:
                assert (REPORTS/'done'/f'{name}.json').is_file();completed.add(name)
                print('Completed',name,flush=True)
        if failure:
            write_json(REPORTS/'status.json',{'state':'failed-draining','reason':failure,'running':list(running),'completed':sorted(completed)})
            if not running:raise RuntimeError(failure)
            time.sleep(15);continue
        busy={gpu for proc,gpu,log in running.values() if gpu is not None}
        resources=gpu_resources()
        for task in tasks:
            name=task['id']
            if name in completed or name in running or not set(task['deps'])<=completed:continue
            if len(running)>=2:break
            gpu=None
            if task['memory']:
                possible=[(i,u,m) for i,u,m in resources if i not in busy and m>=task['memory'] and u<=40]
                if not possible:continue
                gpu=min(possible,key=lambda x:(x[1],-x[2]))[0];busy.add(gpu)
            log=(REPORTS/f'{name}.log').open('a')
            env=device_environment(gpu) if gpu is not None else os.environ.copy()
            env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='1'
            proc=subprocess.Popen([sys.executable,__file__,'--task',name],env=env,stdout=log,stderr=subprocess.STDOUT)
            running[name]=(proc,gpu,log);print('Launched',name,'physical GPU',gpu,flush=True)
        write_json(REPORTS/'status.json',{'state':'running' if running else 'waiting-resources','running':{k:v[1] for k,v in running.items()},'completed':sorted(completed),'total_tasks':len(tasks)})
        time.sleep(15 if running else 60)
    print('COMPLETE: both seed2026 single-model packages and paired CV5 offset reports',flush=True)


def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true');parser.add_argument('--task');args=parser.parse_args()
    if args.task:execute_task(next(t for t in plan() if t['id']==args.task));return
    preflight()
    if not args.preflight:schedule()

if __name__=='__main__':main()

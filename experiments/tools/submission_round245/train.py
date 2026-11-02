"""完整训练入口：6个域内编码器、5个教师、折外目标、全量学生。"""
import argparse,json,subprocess,sys
from pathlib import Path
from common import sha256,json_write,load_checkpoint
ROOT=Path(__file__).resolve().parent

def commands(dataset,work,device):
    cfg=json.loads((ROOT/'configs/recipe.json').read_text(encoding='utf-8'));jobs=[]
    for f in [*cfg['folds'],{'name':'full','val_rois':[]}]:
        name=f['name'];scope=work/name;val=','.join(f['val_rois'])
        jobs.append((name+'.pretrain',[sys.executable,str(ROOT/'pretrain.py'),'--dataset',str(dataset),'--output',str(scope/'fcmae'),
                     '--val-rois',val,'--device',device],scope/'fcmae/final_encoder.pt'))
        if name!='full':
            for stage in (1,2):
                init=scope/'fcmae/final_encoder.pt' if stage==1 else scope/'stage1/epochs/epoch_014.pt'
                jobs.append((f'{name}.teacher{stage}',[sys.executable,str(ROOT/'train_stage.py'),'--dataset',str(dataset),'--output',str(scope/f'stage{stage}'),
                    '--role','teacher','--stage',str(stage),'--init',str(init),'--val-rois',val,'--device',device],scope/f'stage{stage}/epochs/epoch_014.pt'))
    jobs.append(('oof_targets',[sys.executable,str(ROOT/'build_targets.py'),'--dataset',str(dataset),'--work',str(work),'--device',device],work/'teacher_manifest.json'))
    for stage in (1,2):
        scope=work/'student';init=work/'full/fcmae/final_encoder.pt' if stage==1 else scope/'stage1/epochs/epoch_014.pt'
        cmd=[sys.executable,str(ROOT/'train_stage.py'),'--dataset',str(dataset),'--output',str(scope/f'stage{stage}'),
             '--role','student','--stage',str(stage),'--init',str(init),'--device',device]
        if stage==2:cmd+=['--targets',str(work/'targets')]
        jobs.append((f'student.stage{stage}',cmd,scope/f'stage{stage}/epochs/epoch_014.pt'))
    return jobs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--work',required=True,type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--plan',action='store_true',help='仅打印命令')
    p.add_argument('--resume-completed',action='store_true',help='跳过哈希核验通过的已完成阶段；中断的预训练使用 pretrain.py --resume')
    args=p.parse_args();args.dataset=args.dataset.resolve();args.work=args.work.resolve()
    jobs=commands(args.dataset,args.work,args.device)
    if args.plan:
        print(json.dumps([{'stage':n,'command':cmd,'output':str(out)} for n,cmd,out in jobs],ensure_ascii=False,indent=2));return
    sys.path.insert(0,str(ROOT/'src'))
    from ihc_data import image_names,parse_patch_name
    cfg=json.loads((ROOT/'configs/recipe.json').read_text(encoding='utf-8'))
    names=image_names(args.dataset/'train/DAPI')
    rois=[r for f in cfg['folds'] for r in f['val_rois']]
    if len(names)!=2380 or len(rois)!=len(set(rois)) or set(rois)!={parse_patch_name(n).roi for n in names}:
        raise ValueError('Official Liver training inventory / fold partition mismatch')
    for marker in cfg['markers']:
        if set(image_names(args.dataset/'train'/marker))!=set(names):raise ValueError('Unpaired marker names: '+marker)
    args.work.mkdir(parents=True,exist_ok=True);done=args.work/'completed';done.mkdir(exist_ok=True)
    logs=args.work/'logs';logs.mkdir(exist_ok=True)
    for name,cmd,artifact in jobs:
        record=done/(name+'.json')
        if record.exists():
            if not args.resume_completed:raise FileExistsError(record)
            previous=json.loads(record.read_text(encoding='utf-8'))
            if previous['command']!=cmd or sha256(artifact)!=previous['sha256']:raise ValueError('Completed stage changed')
            print('Reuse',name,flush=True);continue
        if artifact.exists():
            if not (args.resume_completed and name.endswith('.pretrain')):raise FileExistsError(artifact)
            ck=load_checkpoint(artifact)
            expected=set(filter(None,cmd[cmd.index('--val-rois')+1].split(',')))
            assert ck['epoch']==800 and ck['config']['seed']==2026 and set(ck['val_rois'])==expected
            assert Path(ck['config']['dataset']).resolve()==args.dataset
            del ck
        else:
            execute=list(cmd)
            if name.endswith('.pretrain') and artifact.parent.exists():
                if not args.resume_completed:raise FileExistsError(artifact.parent)
                execute+=['--resume']
            print('Run',name,flush=True)
            with (logs/(name+'.log')).open('a',encoding='utf-8') as f:subprocess.run(execute,stdout=f,stderr=subprocess.STDOUT,check=True)
        if not artifact.is_file():raise FileNotFoundError(artifact)
        json_write(record,{'command':cmd,'artifact':str(artifact),'sha256':sha256(artifact)})
    print('Final checkpoint:',args.work/'student/stage2/epochs/epoch_012.pt',flush=True)

if __name__=='__main__':main()

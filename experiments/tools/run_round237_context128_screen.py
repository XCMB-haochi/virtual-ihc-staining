"""Leakage-safe source-halo screen; matched non-distilled fold baselines."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from run_round204_cd45_cv import DATASET, training_command
from run_round210_relational import digest, run_logged


def selected_folds():
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    return [f for f in folds if f['name'] in ('fold0','fold3')]


def command(fold, stage, smoke=False):
    cmd,_=training_command(fold,stage,14,smoke)
    prefix='runs/round237_smoke' if smoke else f"runs/round237_{fold['name']}"
    cmd[cmd.index('--output-dir')+1]=f'{prefix}_stage{stage}'
    cmd[cmd.index('--context-halo')+1]='128'
    if stage==1:
        cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt"
    else:
        cmd[cmd.index('--init-checkpoint')+1]=f"{prefix}_stage1/epochs/epoch_{1 if smoke else 14:03d}.pt"
    validate_command(cmd,fold)
    return cmd


def validate_command(cmd,fold):
    if '--train-all' in cmd or '--distillation-target-dir' in cmd or '--distillation-weight' in cmd:
        raise ValueError('Fold screen cannot use full-data training or global teacher targets')
    if set(cmd[cmd.index('--val-rois')+1].split(','))!=set(fold['val_rois']):
        raise ValueError('Validation exclusion mismatch')


def audit_checkpoint(ck,fold,smoke=False,config=None):
    a=ck['args'];excluded=set(fold['val_rois'])
    assert not a['train_all'] and a['distillation_weight']==0
    assert set(a['val_rois'].split(','))==excluded
    assert not set(a['train_rois_resolved']) & excluded
    assert a['context_halo']==128 and a['train_target_context_halo']==64
    assert a['seed']==2026 and a['batch_size']==4 and a['gradient_accumulation_steps']==4
    if not smoke:
        assert config is not None
        assert config['val_images']==140*len(excluded)
        assert config['train_images']==2380-config['val_images']
    steps=[float(s['step']) for s in ck['optimizer']['state'].values() if 'step' in s]
    assert steps and min(steps)>0
    return min(steps)


def compare(current,baseline,folds):
    scopes=[f['name'] for f in folds]+[r for f in folds for r in f['val_rois']]
    result={}
    for marker,values in current.items():
        changes={s:values[s]['gaussian11_score']-baseline[marker][s]['gaussian11_score'] for s in scopes}
        count=sum(values[f['name']]['images'] for f in folds)
        matched=sum(baseline[marker][f['name']]['gaussian11_score']*values[f['name']]['images'] for f in folds)/count
        assert values['overall']['images']==count==980
        changes['overall']=values['overall']['gaussian11_score']-matched
        result[marker]=changes
    return result


def audit_completed(out,fold,stage,smoke=False):
    import torch
    cmd=command(fold,stage,smoke)
    ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    config=json.loads((out/'config.json').read_text())
    expected_epochs=1 if smoke else 14
    assert ck['epoch']==expected_epochs
    # Compare every explicitly requested training option before reusing a stage.
    i=3
    while i<len(cmd):
        token=cmd[i]
        if not token.startswith('--'):
            i+=1;continue
        key=token[2:].replace('-','_')
        if i+1<len(cmd) and not cmd[i+1].startswith('--'):
            expected=cmd[i+1];actual=ck['args'][key]
            if key in ('output_dir','init_checkpoint','fcmae_encoder_init_checkpoint'):
                assert str(actual).replace('\\','/').endswith(expected),(key,actual,expected)
            elif key=='markers':assert (actual.split(',') if isinstance(actual,str) else list(actual))==expected.split(',')
            elif isinstance(actual,(float,int)) and not isinstance(actual,bool):assert float(actual)==float(expected)
            else:assert str(actual)==expected,(key,actual,expected)
            i+=2
        else:
            assert ck['args'][key] is True
            i+=1
    rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows]==list(range(1,expected_epochs+1))
    assert all(math.isfinite(r['train_loss']) for r in rows)
    assert all(r['validation']['count']==config['val_images'] for r in rows)
    steps=audit_checkpoint(ck,fold,smoke,config)
    endpoint=out/'epochs'/f'epoch_{expected_epochs:03d}.pt'
    ep=torch.load(endpoint,map_location='cpu',weights_only=False)
    assert ep['epoch']==expected_epochs
    assert all(torch.isfinite(v).all() and torch.equal(v,ep['model'][k]) for k,v in ck['model'].items())
    print('Audited completed stage',out,'actual updates',steps,'endpoint SHA256',digest(endpoint),flush=True)


def main():
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume-completed',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args();folds=selected_folds()
    if args.smoke:folds=folds[:1]
    for fold in folds:
        encpath=Path(f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt")
        ref=torch.load(f"runs/round207_{fold['name']}_stage1/epochs/epoch_014.pt",map_location='cpu',weights_only=False)
        assert digest(encpath)==ref['args']['init_checkpoint_report']['checkpoint_sha256']
        enc=torch.load(encpath,map_location='cpu',weights_only=False)
        assert enc['epoch']==800 and set(enc['val_rois'])==set(fold['val_rois'])
        assert not set(enc['config']['train_rois']) & set(fold['val_rois'])
        del enc,ref
        for stage in (1,2):
            cmd=command(fold,stage,args.smoke)
            out=Path(cmd[cmd.index('--output-dir')+1])
            if out.exists():
                if not args.resume_completed:raise FileExistsError(out)
                audit_completed(out,fold,stage,args.smoke)
    if args.audit_only:return
    sources=[]
    for fold in folds:
        for stage in (1,2):
            cmd=command(fold,stage,args.smoke);out=Path(cmd[cmd.index('--output-dir')+1])
            if out.exists():
                print('Reusing audited completed stage',out,flush=True)
            else:
                print('Training',fold['name'],stage,flush=True)
                run_logged(cmd,f"reports/round237_{'smoke' if args.smoke else fold['name']}_stage{stage}.log")
            rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in rows]==list(range(1,(1 if args.smoke else 14)+1))
            assert all(math.isfinite(r['train_loss']) for r in rows)
            ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
            steps=audit_checkpoint(ck,fold,args.smoke,json.loads((out/'config.json').read_text()))
            if stage==2:
                assert str(ck['args']['init_checkpoint']).endswith(cmd[cmd.index('--init-checkpoint')+1])
            print('Verified actual updates',steps,'peak GiB',rows[-1]['peak_gpu_gib'],flush=True)
            del ck
        if not args.smoke:
            path=out/'epochs'/'epoch_012.pt'
            sources.append({**fold,'images':140*len(fold['val_rois']),'checkpoint':str(path.resolve()),'checkpoint_sha256':digest(path)})
    if args.smoke:return
    root=Path('runs/round237_oof_sources');root.mkdir(exist_ok=args.resume_completed)
    (root/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher','images':sum(s['images'] for s in sources),'output_source_offset_applied':False,'folds':sources},indent=2)+'\n')
    out=Path('reports/round237_d4_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(root),str(out),'--mode','models','--subset-rois',','.join(r for f in folds for r in f['val_rois']),'--device','cuda:0'],check=True)
    current=json.loads(out.read_text())['markers'];base=json.loads(Path('reports/round217_domain_cv5_oof.json').read_text())['markers']
    gains=compare(current,base,folds)
    Path('reports/round237_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('CD45RO gains',json.dumps(gains['CD45RO']),flush=True)


if __name__=='__main__':main()

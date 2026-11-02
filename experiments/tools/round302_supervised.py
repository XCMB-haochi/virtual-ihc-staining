"""Bounded fixed-marker comparison after the completed Round302 pretraining."""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from round300 import ORGANS, inventory, marker_policy, write_json

ROOT=Path(__file__).resolve().parents[1]


def project_path(value):
    path=(ROOT/value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('A continuation path is outside the project')
    return path


def load_plan(path):
    plan=json.loads(path.read_text(encoding='utf-8'))
    if plan['kind']!='round302-fixed-marker-comparison' or plan['seed']!=2026:
        raise ValueError('Unrecognized comparison or seed')
    if (plan['domain_epochs'],plan['epochs'],plan['validation_every'])!=(800,192,16):
        raise ValueError('The registered800/192-cycle schedule must remain fixed')
    if len(plan['jobs'])!=2 or [job['name'] for job in plan['jobs']]!=['mixed','cd68']:
        raise ValueError('Exactly the two registered candidates are permitted')
    for key in ('domain_encoder','public','data','output_root'):
        project_path(plan[key])
    for job in plan['jobs']:
        if not re.fullmatch('[a-z][a-z0-9_]*',job['name']) or set(job['organ_markers'])!=set(ORGANS):
            raise ValueError('Invalid job name or organ mapping')
        marker_policy([job['organ_markers'][organ] for organ in ORGANS])
    expected=[{'colon':'CD68','liver':'CD45RO','stomach':'CD68'},dict.fromkeys(ORGANS,'CD68')]
    if [job['organ_markers'] for job in plan['jobs']]!=expected:
        raise ValueError('The registered marker mappings must remain fixed')
    return plan


def command_for(plan,job,smoke=False,resume=False):
    name=('smoke_' if smoke else '')+job['name']
    command=[sys.executable,'-u',str(ROOT/'tools/round300.py'),
             '--data',plan['data'],
             '--output',str(project_path(plan['output_root'])/name),
             '--public',str(project_path(plan['public'])),
             '--domain-encoder',str(project_path(plan['domain_encoder'])),
             '--epochs',str(plan['epochs']),'--validation-every',str(plan['validation_every']),
             '--organ-markers',*[job['organ_markers'][organ] for organ in ORGANS]]
    if smoke: command.append('--smoke')
    if resume: command.append('--resume')
    return command


def check_inherited(plan,args):
    # These are the arguments already parsed into the running PowerShell script.
    for key,actual in (('output_root',args.output),('domain_encoder',args.domain_encoder),
                       ('public',args.public),('data',args.data)):
        if actual is None or project_path(plan[key])!=actual.resolve():
            raise ValueError(f'Queued {key} differs from the registered comparison')
    if args.epochs!=plan['epochs'] or args.validation_every!=plan['validation_every']:
        raise ValueError('Queued supervised schedule differs from the plan')


def audit_candidate(plan,job,domain_hash,smoke=False):
    import torch
    output=project_path(plan['output_root'])/(('smoke_' if smoke else '')+job['name'])
    config=json.loads((output/'config.json').read_text())
    expected={organ:[job['organ_markers'][organ]] for organ in ORGANS}
    assert config['supervised_markers']==expected and config['seed']==2026
    assert config['domain_encoder_sha256']==domain_hash and config['smoke']==smoke
    assert config['epochs']==192 and config['validation_every']==16
    assert config['organ_probabilities']==[.1,.2,.7] and config['batch']==1
    rows=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    count=1 if smoke else plan['epochs']
    assert [row['epoch'] for row in rows]==list(range(1,count+1))
    for row in rows:
        assert math.isfinite(row['train_loss'])
        assert set(row['sampled_targets'])==set(ORGANS)
        assert all(set(row['sampled_targets'][organ])==set(expected[organ]) for organ in ORGANS)
        assert sum(sum(counts.values()) for counts in row['sampled_targets'].values())==(32 if smoke else 2048)
    checkpoint=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    assert checkpoint['config']==config and checkpoint['markers']==job['organ_markers']
    assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())
    validation=json.loads((output/f"validation_{checkpoint['epoch']:03d}.json").read_text())
    assert validation['selected_markers']==job['organ_markers']
    assert all(set(validation['organs'][organ])==set(expected[organ]) for organ in ORGANS)
    score=validation['weighted_score']
    assert math.isfinite(score) and abs(score-max(row['validation_score'] for row in rows if 'validation_score' in row))<1e-10
    complete=json.loads((output/'complete.json').read_text())
    assert complete['epochs']==count and complete['best_validation_score']==score
    last=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    steps={int(state['step']) for state in last['optimizer']['state'].values() if 'step' in state}
    expected_steps=4 if smoke else plan['epochs']*256
    assert last['epoch']==count and steps=={expected_steps}
    del last
    return {'name':job['name'],'organ_markers':job['organ_markers'],
            'checkpoint':str((output/'best.pt').relative_to(ROOT)),
            'best_epoch':checkpoint['epoch'],'weighted_score':score,
            'organs':validation['organs'],'domain_encoder_sha256':domain_hash,
            'sampled_targets_verified':True,'finite_weights':True,'optimizer_steps':expected_steps}


def launch(command,label):
    # Check resources at each new independent job; never alter other processes.
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'],text=True)
    devices=[line.split(',') for line in hardware.strip().splitlines()]
    if len(devices)!=1 or int(devices[0][1].strip())<4096:
        raise RuntimeError('Expected one GPU with at least4096MiB free; preserve candidates and stop')
    print(json.dumps({'stage':label,'gpu':hardware.strip(),'command':command}),flush=True)
    with (ROOT/f'reports/round302_{label}.log').open('w',encoding='utf-8') as stdout, \
         (ROOT/f'reports/round302_{label}.err').open('w',encoding='utf-8') as stderr:
        subprocess.run(command,cwd=ROOT,stdout=stdout,stderr=stderr,check=True)


def run_plan(path,check_only=False,inherited=None,resume=False):
    plan=load_plan(path)
    if inherited is not None: check_inherited(plan,inherited)
    if Path.cwd().resolve()!=ROOT:
        raise ValueError('Run the continuation from the project root')
    pretrain_split=project_path(plan['domain_encoder']).parent/'split.json'
    # Keep the data path representation identical to the live pretraining's
    # manifest, as well as comparing every actual Train/Val filename and ROI.
    assert inventory(Path(plan['data']))==json.loads(pretrain_split.read_text(encoding='utf-8')), 'Supervised split must exactly match pretraining'
    commands=[command_for(plan,job) for job in plan['jobs']]
    if check_only:
        print(json.dumps({'check_only':True,'split_matches_pretraining':True,'plan':plan,'commands':commands}),flush=True)
        return
    import torch
    torch.set_num_threads(4)
    root=project_path(plan['output_root']); root.mkdir(parents=True,exist_ok=True)
    status=ROOT/'reports/round302_supervised_status.json'
    record=root/'pipeline_config.json'
    if record.exists():
        if not resume or json.loads(record.read_text())!=plan:
            raise FileExistsError('Use the same plan with explicit --resume for an existing comparison')
    else: write_json(record,plan)
    try:
        audit=json.loads((ROOT/'reports/round302_pretrain_audit.json').read_text())
        domain_hash=hashlib.sha256(project_path(plan['domain_encoder']).read_bytes()).hexdigest()
        assert audit['epochs']==800 and audit['optimizer_steps']==202400 and audit['val_excluded']
        assert audit['encoder_sha256']==domain_hash
        # Test both new label paths before committing time to either full run.
        for job in plan['jobs']:
            smoke_output=root/('smoke_'+job['name'])
            write_json(status,{'phase':'smoke','candidate':job['name']})
            if not (resume and (smoke_output/'complete.json').exists()):
                launch(command_for(plan,job,smoke=True,resume=resume and (smoke_output/'last.pt').exists()),
                       'smoke_'+job['name'])
            audit_candidate(plan,job,domain_hash,smoke=True)
        candidates=[]
        for job in plan['jobs']:
            output=root/job['name']; write_json(status,{'phase':'training','candidate':job['name']})
            if not (resume and (output/'complete.json').exists()):
                launch(command_for(plan,job,resume=resume and (output/'last.pt').exists()),job['name'])
            candidates.append(audit_candidate(plan,job,domain_hash))
            write_json(root/(job['name']+'_audit.json'),candidates[-1])
        assert json.loads((root/'mixed/split.json').read_text())==json.loads((root/'cd68/split.json').read_text())
        configs=[json.loads((root/job['name']/'config.json').read_text()) for job in plan['jobs']]
        for key in configs[0]:
            if key not in ('supervised_markers','marker_probabilities_by_organ'):
                assert configs[0][key]==configs[1][key],key
        winner=max(candidates,key=lambda candidate:candidate['weighted_score'])
        result={'candidates':candidates,'selected_candidate':winner['name'],
                'selection_basis':'Train-derived grouped Val only; no formal Test information',
                'mixed_minus_cd68':candidates[0]['weighted_score']-candidates[1]['weighted_score'],
                'uncertainty':'One grouped split; only3 Stomach Val ROIs; no guarantee of Test gain',
                'test_inference':False,'pretraining_restarted':False}
        write_json(root/'comparison.json',result)
        write_json(status,{'phase':'complete','selected_candidate':winner['name'],
                           'mixed_minus_cd68':result['mixed_minus_cd68']})
        print(json.dumps(result),flush=True)
    except Exception as error:
        write_json(status,{'phase':'failed','error':repr(error)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path('configs/round302_supervised.json'))
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--resume',action='store_true')
    arguments=parser.parse_args()
    run_plan(arguments.plan,check_only=arguments.check,resume=arguments.resume)

"""Strict outer/inner ROI isolation for OOF pixel-distillation teachers."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from run_round207_domain import pretrain_command
from run_round204_cd45_cv import DATASET, training_command


def partitions(rois, excluded):
    train=sorted(set(rois)-set(excluded))
    return [{'inner':i,'holdout':train[i::5],
             'train':sorted(set(train)-set(train[i::5]))} for i in range(5)]


def pretraining_command(data, fold, inner, smoke=False):
    command,_=pretrain_command({'name':fold,'val_rois':inner['holdout']},smoke)
    command[2]=str(data)
    output=f"runs/round215_{fold}_inner{inner['inner']}_fcmae800" if not smoke else 'runs/round215_smoke_fcmae'
    command[command.index('--output-dir')+1]=output
    command[command.index('--batch-size')+1]='32'
    command[command.index('--gradient-accumulation-steps')+1]='1'
    return command,Path(output)


def main():
    import torch
    torch.set_num_threads(4)
    source=Path(DATASET).resolve()
    protocol=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())
    rois=protocol['expected_rois']; receipts=[]
    for outer in [f for f in protocol['folds'] if f['name'] in ('fold0','fold3')]:
        fold=outer['name']; groups=partitions(rois,outer['val_rois'])
        data=Path(f'runs/round215_{fold}_data')
        if data.exists():raise FileExistsError(data)
        allowed=set(rois)-set(outer['val_rois'])
        (data/'test'/'DAPI').mkdir(parents=True)
        for marker in ('DAPI','CD68','CD45RO','HLA-DR','Vimentin'):
            folder=data/'train'/marker;folder.mkdir(parents=True)
            for p in (source/'train'/marker).glob('*.jpg'):
                if p.name.split('_')[0] in allowed:os.link(p,folder/p.name)
            assert len(list(folder.glob('*.jpg')))==140*len(allowed)
        plan={'outer':outer,'outer_train_rois':sorted(allowed),'inner_folds':groups,
              'dataset':str(data.resolve()),'test_inputs':'empty directory; no real test data'}
        (data/'split.json').write_text(json.dumps(plan,indent=2)+'\n')
        for inner in groups:
            command,output=pretraining_command(data,fold,inner)
            if output.exists():raise FileExistsError(output)
            print('Nested domain pretraining',fold,inner['inner'],flush=True)
            with Path(f"reports/round215_{fold}_inner{inner['inner']}_fcmae.log").open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            encoder=output/'final_encoder.pt'
            payload=torch.load(encoder,map_location='cpu',weights_only=False)
            assert payload['epoch']==800
            assert set(payload['config']['train_rois'])==set(inner['train'])
            assert not set(payload['config']['train_rois']) & (set(inner['holdout'])|set(outer['val_rois']))
            for stage in (1,2):
                name=f"round215_{fold}_inner{inner['inner']}"
                cmd,_=training_command({'name':fold,'val_rois':inner['holdout']},stage,14)
                cmd[2]=str(data)
                cmd[cmd.index('--output-dir')+1]=f'runs/{name}_stage{stage}'
                cmd[cmd.index('--batch-size')+1]='16'
                cmd[cmd.index('--gradient-accumulation-steps')+1]='1'
                if stage==1:cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1]=str(encoder)
                else:cmd[cmd.index('--init-checkpoint')+1]=f'runs/{name}_stage1/epochs/epoch_014.pt'
                if Path(cmd[cmd.index('--output-dir')+1]).exists():raise FileExistsError('Student output exists')
                print('Nested supervised teacher',fold,inner['inner'],stage,flush=True)
                with Path(f'reports/{name}_stage{stage}.log').open('w') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            checkpoint=Path(f'runs/{name}_stage2/epochs/epoch_012.pt')
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert ck['epoch']==12 and ck['args']['selection_marker']=='CD45RO'
            with checkpoint.open('rb') as handle:sha=hashlib.file_digest(handle,'sha256').hexdigest()
            receipts.append({'outer':fold,**inner,'checkpoint':str(checkpoint.resolve()),'sha256':sha,'outer_excluded':outer['val_rois']})
            Path('reports/round215_teacher_sources.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print('Ten nested teachers completed. Next: verify all targets, then paired stage2 distillation.',flush=True)


if __name__=='__main__':main()

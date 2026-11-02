"""固定最终方案：域内预训练、五个教师、训练目标、全量学生。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import torch
from data import inventory,write_json,ORGANS
from training_utils import all_train_manifest
from training import fit,audit_fit,fixed_folds,teacher_split,assign_heldout_rois,predict_targets,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--stage',choices=('all','pretrain','teachers','targets','student'),default='all')
    p.add_argument('--plan',action='store_true')
    args=p.parse_args()
    if args.plan:
        print('1. outer-Train DAPI FCMAE: 800 epochs\n2. five ROI-excluded teachers: 17 epochs each\n3. one teacher target per official Train image\n4. fresh full-Train student: 14 + 12 epochs; distillation0.2, consistency0.4')
        return
    torch.set_num_threads(4)
    args.dataset=args.dataset.resolve();work=args.work.resolve();work.mkdir(parents=True,exist_ok=True)
    split=inventory(args.dataset);full=all_train_manifest(split)
    assert sum(len(s['train']) for s in split.values())==8084
    assert sum(len(s['train']) for s in full.values())==10356
    write_json(work/'split.json',split)
    domain=work/'pretrain/final_encoder.pt'
    if args.stage in ('all','pretrain'):
        if not domain.exists():
            cmd=[sys.executable,str(ROOT/'pretrain.py'),'--dataset',str(args.dataset),'--public',str(ROOT/'assets/convnextv2_tiny_1k_224_fcmae.pt'),'--output',str(domain.parent)]
            if (domain.parent/'last.pt').exists():cmd+=['--resume']
            subprocess.run(cmd,check=True)
        if args.stage=='pretrain':return
    ck=torch.load(domain,map_location='cpu',weights_only=False)
    assert ck['epoch']==800 and ck['split']==split
    encoder=ck['encoder']
    cfg=json.loads((ROOT/'configs/recipe.json').read_text(encoding='utf-8'))['base_config']
    cfg['domain_encoder_sha256']=digest(domain)
    folds=fixed_folds(split);write_json(work/'folds.json',folds)
    if args.stage in ('all','teachers'):
        for i,excluded in enumerate(folds):
            subset=teacher_split(split,excluded)
            folder=work/f'teachers/teacher{i}'
            fit(folder,subset,cfg,17,encoder)
            audit_fit(folder,subset,17)
        if args.stage=='teachers':return
    targets=work/'targets';targets.mkdir(exist_ok=True)
    if args.stage in ('all','targets'):
        items={}
        # Training ROIs excluded from each teacher's supervised fitting.
        for i,names in enumerate(folds):
            predict_targets(work/f'teachers/teacher{i}/final.pt',split,names,targets,i,items)
        # All teachers exclude outer Val labels; assign each entire ROI to one.
        for i,names in enumerate(assign_heldout_rois(split)):
            predict_targets(work/f'teachers/teacher{i}/final.pt',split,names,targets,i,items)
        assert set(items)=={(o,n) for o in ORGANS for n in full[o]['train']}
        write_json(targets/'manifest.json',{'items':list(items.values()),'scope':'all official Train after fixed Val selection'})
        if args.stage=='targets':return
    first=work/'student_stage1';second=work/'student'
    # Full supervised refit starts with random task layers and the fixed encoder.
    fit(first,full,cfg,14,encoder)
    audit_fit(first,full,14)
    fit(second,full,cfg,26,encoder,initial=first/'stage1_final.pt',targets=targets,consistency_weight=.4)
    audit_fit(second,full,26,initial=True)
    print('Final checkpoint:',second/'final.pt',flush=True)


if __name__=='__main__':main()

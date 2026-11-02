"""Seen-training-image fitting diagnostic; explicitly not OOF or test scoring."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest
from run_round224_cv5 import device_environment
from audit_semifinal_markers import image_metrics,summarize
from diagnose_semifinal_errors import jpeg,partitions,finalize


def evaluate(recipe):
    import torch
    from torch.utils.data import DataLoader
    from ihc_data import resolve_data_root,image_names,parse_patch_name
    from ihc_torch import PairedPatchDataset,build_model_from_checkpoint,checkpoint_conditioning,batch_condition_tensor,center_crop_like,predict_with_tta
    from ihc_spatial import apply_complete_roi_conditioning
    torch.set_num_threads(4)
    path=Path(f'runs/{recipe}_full_stage2/epochs/epoch_012.pt')
    hashes=json.loads(Path(f'reports/{recipe}_hashes.json').read_text())
    assert digest(path)==hashes[str(path)]
    ck=torch.load(path,map_location='cpu',weights_only=False)
    assert ck['epoch']==12 and ck['args']['train_all'] and ck['args']['seed']==2026
    assert ck['markers'][1]=='CD45RO'
    model=build_model_from_checkpoint(ck,torch.device('cuda:0')).eval()
    root=resolve_data_root(Path(DATASET));names=image_names(root/'train'/'DAPI');assert len(names)==2380
    conditions=apply_complete_roi_conditioning(root/'train'/'DAPI',names,checkpoint_conditioning(ck))
    loader=DataLoader(PairedPatchDataset(root,names,markers=('CD45RO',),augment=False,context_halo=128),batch_size=2,num_workers=0)
    groups=defaultdict(list);sums=defaultdict(lambda:defaultdict(float));seen=[];pixels=0;sse=0.
    with torch.inference_mode():
        for source,target,batch_names in loader:
            condition=batch_condition_tensor(batch_names,conditions,torch.device('cuda:0'))
            prediction=center_crop_like(predict_with_tta(model,source.cuda(),'d4',condition),target)[:,1].cpu().numpy()*255
            truths=np.rint(target.numpy()[:,0]*255)
            dapis=source.numpy()[:,0,128:-128,128:-128]*255
            for name,pred,truth,dapi in zip(batch_names,prediction,truths,dapis):
                row=image_metrics(pred,truth)
                groups['overall'].append(row);groups[parse_patch_name(name).roi].append(row)
                decoded=jpeg(np.clip(np.rint(pred),0,255))
                for region,stats in partitions(decoded,truth,dapi).items():
                    for key,value in stats.items():sums[region][key]+=value
                pixels+=truth.size;sse+=row['mse']*truth.size;seen.append(name)
    assert seen==names
    report={'recipe':recipe,'evaluation_scope':'seen training images; not held-out validation',
            'checkpoint':str(path),'sha256':digest(path),'images':len(names),
            'metrics':{k:summarize(v) for k,v in groups.items()},'regions':finalize(sums,pixels,sse),
            'limitations':'Differences from fold OOF combine memorization, extra training data, full-data pretraining and optimization. No architecture ceiling or hidden-set prediction follows.'}
    output=Path(f'reports/round229_{recipe}_trainfit.json')
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(recipe,'SEEN TRAINING score',report['metrics']['overall']['gaussian11_score'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',choices=('round208','round216'))
    args=parser.parse_args()
    if args.recipe:return evaluate(args.recipe)
    def worker(recipe,gpu):
        print('Diagnosing training fit',recipe,'GPU',gpu,flush=True)
        subprocess.run([sys.executable,__file__,'--recipe',recipe],env=device_environment(gpu),check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,r,g) for r,g in [('round208',0),('round216',1)]]
        for job in concurrent.futures.as_completed(jobs):job.result()
    reference=json.loads(Path('reports/round217_domain_cv5_oof.json').read_text())['markers']['CD45RO']
    result={}
    for recipe in ('round208','round216'):
        report=json.loads(Path(f'reports/round229_{recipe}_trainfit.json').read_text())
        result[recipe]={'train_score':report['metrics']['overall']['gaussian11_score'],
            'teacher_oof_reference':reference['overall']['gaussian11_score'],
            'by_roi_train_minus_teacher_oof':{k:v['gaussian11_score']-reference[k]['gaussian11_score'] for k,v in report['metrics'].items()},
            'bright128':report['regions']['target_128_255']}
    result['warning']='Training fit is not a validation score; teacher OOF is not student OOF.'
    Path('reports/round229_fit_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

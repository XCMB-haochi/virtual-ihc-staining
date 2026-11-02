"""Fixed small-training-set fitting probe, never a submission model."""
import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parent)]
from run_round204_cd45_cv import DATASET
from run_round224_cv5 import device_environment
from run_round210_relational import digest


def select_names(names,activity):
    order=np.argsort(activity,kind='stable');rng=np.random.default_rng(2026)
    selected=[]
    for group in np.array_split(order,4):selected.extend(rng.choice(group,4,replace=False).tolist())
    return [names[i] for i in sorted(selected)]


def weighted_mse(prediction,target,weights):
    return ((prediction.float()-target).square().mean((0,2,3))*weights).sum()


def evaluate(model,source,target,condition,device):
    import torch
    from audit_semifinal_markers import image_metrics,summarize
    from diagnose_semifinal_errors import jpeg
    model.eval();rows=[];bright_error=bright_target=bright_prediction=bright_sse=0.;bright_count=0
    with torch.inference_mode():
        for i in range(0,16,4):
            pred=model(source[i:i+4].to(device),condition[i:i+4].to(device))[:,1,64:-64,64:-64].cpu().numpy()*255
            truths=np.rint(target[i:i+4,1,64:-64,64:-64].numpy()*255)
            for p,y in zip(pred,truths):
                rows.append(image_metrics(p,y));decoded=jpeg(np.rint(p).clip(0,255));mask=y>=128
                bright_error+=float((decoded[mask]-y[mask]).sum());bright_count+=int(mask.sum())
                bright_sse+=float(np.square(decoded[mask]-y[mask]).sum())
                bright_target+=float(y[mask].sum());bright_prediction+=float(decoded[mask].sum())
    return {'seen_patch_metrics':summarize(rows),'bright128_pixels':bright_count,
            'bright128_bias':bright_error/max(1,bright_count),
            'bright128_rmse':float(np.sqrt(bright_sse/max(1,bright_count))),
            'bright128_target_mean':bright_target/max(1,bright_count),
            'bright128_prediction_mean':bright_prediction/max(1,bright_count)}


def worker(objective,steps,smoke):
    import torch
    from ihc_data import resolve_data_root,image_names,load_gray
    from ihc_torch import PairedPatchDataset,build_model_from_checkpoint,checkpoint_conditioning,batch_condition_tensor,CompetitionLoss
    from ihc_spatial import apply_complete_roi_conditioning
    torch.set_num_threads(4);torch.manual_seed(2026)
    root=resolve_data_root(Path(DATASET));names=image_names(root/'train'/'DAPI');assert len(names)==2380
    activity=[float((load_gray(root/'train'/'CD45RO'/n)>=64).mean()) for n in names]
    selected=select_names(names,activity);assert len(set(selected))==16
    checkpoint=Path('runs/round208_full_stage2/epochs/epoch_012.pt')
    hashes=json.loads(Path('reports/round208_hashes.json').read_text());assert digest(checkpoint)==hashes[str(checkpoint)]
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['args']['train_all'] and ck['markers'][1]=='CD45RO'
    device=torch.device('cuda:0');model=build_model_from_checkpoint(ck,device)
    conditions=apply_complete_roi_conditioning(root/'train'/'DAPI',names,checkpoint_conditioning(ck))
    dataset=PairedPatchDataset(root,selected,markers=ck['markers'],augment=False,context_halo=64,target_context_halo=64)
    data=[dataset[i] for i in range(16)]
    source=torch.stack([v[0] for v in data]);target=torch.stack([v[1] for v in data])
    condition=batch_condition_tensor(selected,conditions,torch.device('cpu'))
    prefix=f"round230_{'smoke_' if smoke else ''}{objective}"
    out=Path('runs')/prefix;out.mkdir(exist_ok=False)
    config={'objective':objective,'updates':steps,'learning_rate':1e-4,'seed':2026,'images':selected,
            'initial_checkpoint_sha256':hashes[str(checkpoint)],'scope':'selected seen TRAIN patches and their normal halos; no held-out estimate'}
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    initial=evaluate(model,source,target,condition,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda');weights=torch.tensor([.05,.85,.05,.05],device=device)
    hybrid=CompetitionLoss(ssim_weight=.7,l1_weight=.2,mse_weight=.1,ssim_mode='uniform7',channel_weights=weights)
    milestones={0:initial};measurement_steps={64,256,1024,steps}
    for step in range(1,steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        order=torch.randperm(16,generator=torch.Generator().manual_seed(2026+step))
        for i in range(0,16,4):
            indices=order[i:i+4];x=source[indices].to(device);y=target[indices].to(device);c=condition[indices].to(device)
            with torch.amp.autocast('cuda'):
                prediction=model(x,c)
                loss=hybrid(prediction,y) if objective=='hybrid' else weighted_mse(prediction,y,weights)
            assert bool(torch.isfinite(loss))
            scaler.scale(loss/4).backward()
        scaler.step(optimizer);scaler.update()
        if step in measurement_steps:
            milestones[step]=evaluate(model,source,target,condition,device)
            print(objective,'update',step,json.dumps(milestones[step]),flush=True)
        if step%256==0 or step==steps:
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                        'step':step,'config':config,'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all(),'milestones':milestones},out/'latest.tmp')
            (out/'latest.tmp').replace(out/'latest.pt')
    actual=[float(v['step']) for v in optimizer.state.values() if 'step'in v];assert actual and min(actual)>0
    result={'config':config,'actual_updates':min(actual),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
            'milestones':milestones,'limitation':'Intentional small-set overfitting. No generalization, leaderboard or architecture-ceiling conclusion.'}
    Path(f'reports/{prefix}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--objective',choices=('hybrid','mse'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();steps=8 if args.smoke else 2048
    if args.objective:return worker(args.objective,steps,args.smoke)
    def run(objective,gpu):
        command=[sys.executable,__file__,'--objective',objective]
        if args.smoke:command.append('--smoke')
        print('Fitting probe',objective,'GPU',gpu,flush=True)
        subprocess.run(command,env=device_environment(gpu),check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run,o,g) for o,g in [('hybrid',0),('mse',1)]]
        for job in concurrent.futures.as_completed(jobs):job.result()


if __name__=='__main__':main()

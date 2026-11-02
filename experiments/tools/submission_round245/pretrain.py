"""DAPI-only FCMAE domain pretraining, seed2026,800 epochs, effective batch32.
Reuse the supplied public-only initialization, never a prior-organ checkpoint.
"""
import argparse,json,math,sys,time
from pathlib import Path
import torch
from torch.optim import AdamW
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from ihc_data import image_names,parse_patch_name,resolve_data_root
from ihc_unext2 import DapiFCMAE,unext2_encoder_specification
from training_utils import set_seed,learning_rate_for_epoch,gradient_accumulation_sample_scale,build_epoch_loader,MarkerPatchDataset,cuda_rng_states_for_restore,truncate_metrics
from common import load_checkpoint,sha256,atomic_save,json_write
PUBLIC_SHA='98256dc1aca47cb6e553a1c82d7f343d328e14d2c7a1b1019dc34273669d1578'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--val-rois',default='');p.add_argument('--public-encoder',type=Path,default=ROOT/'assets/public_encoder.pt')
    p.add_argument('--device',default='cuda:0');p.add_argument('--resume',action='store_true')
    p.add_argument('--epochs',type=int,default=800);p.add_argument('--max-images',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(4)
    if args.epochs<=10 and not args.max_images:raise ValueError('Short schedules are only for smoke tests')
    if args.output.exists() and not args.resume:raise FileExistsError(args.output)
    data=resolve_data_root(args.dataset);excluded=set(filter(None,args.val_rois.split(',')))
    names=[n for n in image_names(data/'train/DAPI') if parse_patch_name(n).roi not in excluded]
    if args.max_images:names=[names[i] for i in __import__('numpy').linspace(0,len(names)-1,num=min(len(names),args.max_images),dtype=int)]
    if sha256(args.public_encoder)!=PUBLIC_SHA:raise ValueError('Public initialization hash mismatch')
    args.output.mkdir(parents=True,exist_ok=True);set_seed(2026)
    model=DapiFCMAE(model_size='tiny')
    public=load_checkpoint(args.public_encoder)
    assert public['epoch']==0 and public['config']['data_scope']=='public-only'
    model.encoder.load_state_dict(public['encoder'],strict=True);del public
    device=torch.device(args.device);model=model.to(device);amp=device.type=='cuda'
    opt=AdamW(model.parameters(),lr=.0002,weight_decay=.05)
    scaler=torch.amp.GradScaler('cuda',enabled=amp);start=1
    config={'seed':2026,'epochs':args.epochs,'dataset':str(data),'val_rois':sorted(excluded),
      'train_rois':sorted({parse_patch_name(n).roi for n in names}),'train_images':len(names),'pretraining_markers':['DAPI'],
      'data_scope':'fold-train' if excluded else 'all-train','public_sha256':PUBLIC_SHA,'batch_size':8,'gradient_accumulation_steps':4,
      'effective_batch_size':32,'mask_ratio':.5,'learning_rate':.0002,'minimum_learning_rate':.000001,'warmup_epochs':0 if args.max_images else 10}
    if args.resume:
        if json.loads((args.output/'config.json').read_text(encoding='utf-8'))!=config:raise ValueError('Resume configuration changed')
        ck=load_checkpoint(args.output/'latest.pt');model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        torch.set_rng_state(ck['torch_rng_state'].cpu())
        if amp and ck['cuda_rng_state'] is not None:torch.cuda.set_rng_state_all(cuda_rng_states_for_restore(ck['cuda_rng_state']))
        start=ck['epoch']+1;truncate_metrics(args.output/'metrics.jsonl',start-1);del ck
    else:json_write(args.output/'config.json',config)
    dataset=MarkerPatchDataset(data/'train',[('DAPI',n) for n in names])
    for epoch in range(start,args.epochs+1):
        started=time.perf_counter();lr=learning_rate_for_epoch(epoch,args.epochs,config['warmup_epochs'],.0002,.000001)
        for g in opt.param_groups:g['lr']=lr
        loader=build_epoch_loader(dataset,8,0,2026,epoch,pin_memory=amp)
        model.train();loss_sum=0;count=0;attempts=0;opt.zero_grad(set_to_none=True)
        for index,(x,_) in enumerate(loader):
            x=x.to(device,non_blocking=True)
            with torch.amp.autocast(device.type,enabled=amp):
                reconstruction,mask=model(x,mask_ratio=.5)
                loss=((reconstruction-x).square()*mask).sum()/(mask.sum()*x.shape[1]).clamp_min(1)
            scale=gradient_accumulation_sample_scale(index,8,x.shape[0],4,len(dataset));scaler.scale(loss*scale).backward()
            if (index+1)%4==0 or index+1==len(loader):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);attempts+=1
            loss_sum+=float(loss.detach())*x.shape[0];count+=x.shape[0]
        if not math.isfinite(loss_sum):raise RuntimeError('Nonfinite pretraining loss')
        row={'epoch':epoch,'loss':loss_sum/count,'seconds':time.perf_counter()-started,'optimizer_steps':attempts}
        with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if epoch%25==0 or epoch==args.epochs:
            atomic_save(args.output/'latest.pt',{'epoch':epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
              'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all() if amp else None})
    atomic_save(args.output/'final_encoder.pt',{'version':1,'task':'dapi-fcmae-pretraining','epoch':args.epochs,'val_rois':sorted(excluded),
      'encoder_spec':unext2_encoder_specification(),'encoder':model.encoder.state_dict(),'config':config})

if __name__=='__main__':main()

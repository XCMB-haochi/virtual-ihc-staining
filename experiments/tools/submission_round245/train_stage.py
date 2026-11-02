"""最高分路线的单阶段监督训练；仅保留教师/学生所需分支。
每次运行创建新优化器。stage1使用域内encoder；stage2加载本次stage1的epoch14。
"""
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from PIL import Image
from skimage.metrics import structural_similarity
import io
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from ihc_data import image_names,parse_patch_name,resolve_data_root
from ihc_spatial import build_spatial_thumbnail_conditioning
from ihc_unext2 import UNeXt2VirtualStainer,UNeXt2LocalConditionVirtualStainer,unext2_model_specification,unext2_local_condition_model_specification
from ihc_torch import PairedPatchDataset,CompetitionLoss,batch_condition_tensor,center_crop_like
from training_utils import set_seed,gradient_accumulation_sample_scale
from common import load_checkpoint,sha256,atomic_save,json_write

MARKERS=('CD68','CD45RO','HLA-DR','Vimentin')

def build_optimizer(model,lr,stage_decay):
    # Preserve original group/parameter order exactly for AdamW and AMP.
    groups=[{'params':list(model.decoder_parameters()),'lr':lr,'group_name':'decoder_head'}]
    if stage_decay==1:
        groups.append({'params':list(model.encoder_parameters()),'lr':lr,'group_name':'encoder'})
    else:
        modules=[('stem',model.encoder.stem),*[(f'stage{i}',m) for i,m in enumerate(model.encoder.stages)]]
        groups += [{'params':list(m.parameters()),'lr':lr*stage_decay**(4-i),'group_name':'encoder_'+n} for i,(n,m) in enumerate(modules)]
    return AdamW(groups,weight_decay=1e-4)

def batch_loss(model,source,target,condition,criterion,teacher=None,distillation_criterion=None):
    prediction=center_crop_like(model(source,condition),target)
    loss=criterion(prediction,target)
    if teacher is not None:
        center=center_crop_like(prediction[:,1:2],teacher)
        loss=.8*loss+.2*distillation_criterion(center,teacher)
    return loss

@torch.inference_mode()
def validate(model,loader,conditions,device):
    # Monitoring only; final stage endpoints are fixed, not chosen from these scores.
    model.eval();ssims=[];psnrs=[]
    for source,target,names in loader:
        y=center_crop_like(model(source.to(device),batch_condition_tensor(names,conditions,device)),target)[:,1].cpu().numpy()*255
        t=np.rint(target[:,1].numpy()*255).astype(np.uint8)
        for a,b in zip(y,t):
            buf=io.BytesIO();Image.fromarray(np.rint(a).clip(0,255).astype(np.uint8)).convert('RGB').save(buf,format='JPEG',quality=95);buf.seek(0)
            with Image.open(buf) as im:a=np.array(im.convert('L'),dtype=np.float64)
            mse=np.square(a-b).mean();psnrs.append(min(50,max(0,10*np.log10(255**2/max(mse,1e-10)))))
            ssims.append(structural_similarity(b,a,data_range=255,win_size=11,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))
    return {'images':len(ssims),'gaussian11_ssim':float(np.mean(ssims)),'score':float(70*np.mean(ssims)+.6*np.mean(psnrs))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--role',choices=['teacher','student'],required=True);p.add_argument('--stage',type=int,choices=[1,2],required=True)
    p.add_argument('--init',required=True,type=Path);p.add_argument('--val-rois',default='');p.add_argument('--targets',type=Path)
    p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=14)
    p.add_argument('--max-images',type=int,default=0,help='Only for implementation smoke tests, never scored weights')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.targets and not(args.role=='student' and args.stage==2 and not args.val_rois):raise ValueError('Teacher targets are full-data stage2 only')
    torch.set_num_threads(4);set_seed(2026)
    data=resolve_data_root(args.dataset);names=image_names(data/'train/DAPI');excluded=set(filter(None,args.val_rois.split(',')))
    if args.role=='teacher' and not excluded:raise ValueError('Teacher requires held-out ROI list')
    if args.role=='student' and excluded:raise ValueError('Final student is trained on all permitted training data')
    train=[n for n in names if parse_patch_name(n).roi not in excluded];val=[n for n in names if parse_patch_name(n).roi in excluded]
    conditions,spec=build_spatial_thumbnail_conditioning(data/'train/DAPI',names)
    if args.max_images:
        train=[train[i] for i in np.linspace(0,len(train)-1,num=min(args.max_images,len(train)),dtype=int)]
        val=val[:8]
    if args.targets and any(not(args.targets/n).is_file() for n in train):raise ValueError('Incomplete distillation cache')
    device=torch.device(args.device);amp=device.type=='cuda'
    generator=torch.Generator().manual_seed(2026)
    loader=DataLoader(PairedPatchDataset(data,train,MARKERS,True,64,64,args.targets),batch_size=4,shuffle=True,num_workers=0,pin_memory=amp,persistent_workers=False,generator=generator)
    vloader=DataLoader(PairedPatchDataset(data,val,MARKERS,False,64),batch_size=4,shuffle=False,num_workers=0,pin_memory=amp) if val else None
    cls=UNeXt2LocalConditionVirtualStainer if args.role=='teacher' else UNeXt2VirtualStainer
    model=cls(4,'spatial-thumbnail-film').to(device)
    initial=load_checkpoint(args.init)
    if args.stage==1:
        assert initial['task']=='dapi-fcmae-pretraining' and initial['epoch']==800
        assert set(initial['val_rois'])==excluded
        model.encoder.load_state_dict(initial['encoder'],strict=True)
    else:
        assert initial['epoch']==14 or args.max_images
        assert initial['args']['train_all']==(args.role=='student')
        assert set(filter(None,initial['args']['val_rois'].split(',')))==excluded
        model.load_state_dict(initial['model'],strict=True)
    del initial
    weights=(.05,.85,.05,.05)
    criterion=CompetitionLoss(ssim_weight=.7,l1_weight=.2,mse_weight=.1,ssim_mode='uniform7',channel_weights=weights)
    teacher_criterion=CompetitionLoss(ssim_weight=.7,l1_weight=.2,mse_weight=.1,ssim_mode='uniform7') if args.targets else None
    lr=3e-4 if args.stage==1 else 1e-4;decay=1.0 if args.role=='teacher' else .8
    opt=build_optimizer(model,lr,decay);scheduler=CosineAnnealingLR(opt,T_max=args.epochs,eta_min=0)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    args.output.mkdir(parents=True);(args.output/'epochs').mkdir()
    arch='unext2-fcmae-tiny-local-condition' if args.role=='teacher' else 'unext2-fcmae-tiny'
    metadata={'model_arch':arch,'model_spec':unext2_local_condition_model_specification() if args.role=='teacher' else unext2_model_specification(),
      'condition_mode':'spatial-thumbnail-film','conditioning_spec':spec,'markers':','.join(MARKERS),'seed':2026,
      'train_all':args.role=='student','val_rois':args.val_rois,'train_rois_resolved':sorted({parse_patch_name(n).roi for n in train}),
      'context_halo':64,'train_target_context_halo':64,'batch_size':4,'gradient_accumulation_steps':4,'encoder_stage_decay':decay,
      'base_channels':32,'block_type':'residual','init_checkpoint':str(args.init.resolve()),'init_sha256':sha256(args.init),
      'distillation_weight':.2 if args.targets else 0,'distillation_marker_index':1,'epochs':args.epochs,'learning_rate':lr}
    json_write(args.output/'config.json',{**metadata,'train_images':len(train),'val_images':len(val)})
    for epoch in range(1,args.epochs+1):
        started=time.perf_counter();model.train();loss_sum=0;count=0;attempts=0;opt.zero_grad(set_to_none=True)
        for index,batch in enumerate(loader):
            if args.targets:source,target,teacher,bnames=batch;teacher=teacher.to(device)
            else:source,target,bnames=batch;teacher=None
            source=source.to(device,non_blocking=True);target=target.to(device,non_blocking=True)
            condition=batch_condition_tensor(bnames,conditions,device)
            with torch.amp.autocast(device.type,enabled=amp):loss=batch_loss(model,source,target,condition,criterion,teacher,teacher_criterion)
            scale=gradient_accumulation_sample_scale(index,4,source.shape[0],4,len(train));scaler.scale(loss*scale).backward()
            if (index+1)%4==0 or index+1==len(loader):scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);attempts+=1
            loss_sum+=float(loss.detach())*source.shape[0];count+=source.shape[0]
        validation=validate(model,vloader,conditions,device) if vloader else None
        scheduler.step()
        if not math.isfinite(loss_sum):raise RuntimeError('Nonfinite training loss')
        row={'epoch':epoch,'train_loss':loss_sum/count,'attempted_updates':attempts,'seconds':time.perf_counter()-started,'validation':validation}
        with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        ck={'epoch':epoch,'model':model.state_dict(),'markers':MARKERS,'base_channels':32,'block_type':'residual','args':metadata,
            'conditioning':spec,'model_spec':metadata['model_spec'],'selection_psnr_aggregation':'mean-image','best_score':None}
        atomic_save(args.output/'epochs'/f'epoch_{epoch:03d}.pt',ck)
        atomic_save(args.output/'last.pt',{**ck,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict()})
        print(json.dumps(row),flush=True)

if __name__=='__main__':main()

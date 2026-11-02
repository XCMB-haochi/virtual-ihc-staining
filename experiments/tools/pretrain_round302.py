"""FCMAE over complete organ-qualified Train folds; no Val/Test images."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from round300 import inventory, read_image, tensor, write_json
from ihc_unext2 import DapiFCMAE, initialize_fcmae_encoder_from_official_convnextv2


def rate(epoch, epochs=800):
    if epoch<=10: return .0002*epoch/10
    progress=(epoch-11)/max(epochs-11,1)
    return 1e-6+(.0002-1e-6)*.5*(1+math.cos(math.pi*progress))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/round302_pretrain'))
    p.add_argument('--epochs',type=int,default=800)
    p.add_argument('--benchmark',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.epochs<11: raise ValueError('Schedule requires at least11 epochs')
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    manifest=inventory(Path('dataset'))
    samples=[(organ,name) for organ,spec in manifest.items() for name in spec['train']]
    assert len(samples)==8084 and len(set(samples))==8084
    public=Path('runs/public_checkpoints/convnextv2_tiny_1k_224_fcmae.pt')
    config={'seed':2026,'epochs':args.epochs,'images_per_epoch':len(samples),'batch':1,'accumulation':32,
            'mask_ratio':.5,'lr':.0002,'minimum_lr':1e-6,'warmup':10,'weight_decay':.05,
            'precision':'bfloat16; float32 masked MSE','benchmark':args.benchmark,
            'public_sha256':hashlib.sha256(public.read_bytes()).hexdigest()}
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'last.pt').exists() and not args.resume: raise FileExistsError('Use --resume')
    if args.resume:
        assert json.loads((args.output/'config.json').read_text())==config
        assert json.loads((args.output/'split.json').read_text())==manifest
    else:
        write_json(args.output/'config.json',config); write_json(args.output/'split.json',manifest)
    if args.benchmark:
        random.Random(2026).shuffle(samples); samples=samples[:512]
    print('Loading only training DAPI',len(samples),flush=True)
    cache={(o,n):read_image(Path(manifest[o]['folder'])/'DAPI'/n) for o,n in samples}
    model=DapiFCMAE()
    initialize_fcmae_encoder_from_official_convnextv2(model.encoder,public)
    model=model.cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.05)
    start=1
    if args.resume:
        ck=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start=ck['epoch']+1; del ck
        lines=[json.loads(line) for line in (args.output/'metrics.jsonl').read_text().splitlines()]
        (args.output/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in lines if r['epoch']<start))
    final_epoch=2 if args.benchmark else args.epochs
    for epoch in range(start,final_epoch+1):
        order=list(samples); random.Random(2026+epoch).shuffle(order)
        model.train(); losses=[]; begin=time.perf_counter()
        for group in optimizer.param_groups: group['lr']=rate(epoch,args.epochs)
        for offset in range(0,len(order),32):
            group=order[offset:offset+32]; optimizer.zero_grad(set_to_none=True)
            for key in group:
                array=cache[key]; k=random.randrange(4); flip=random.randrange(2)
                array=np.rot90(array,k)
                if flip: array=np.fliplr(array)
                source=tensor(array,'cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16): reconstruction,mask=model(source,mask_ratio=.5)
                loss=((reconstruction.float()-source).square()*mask).sum()/mask.sum().clamp_min(1)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite FCMAE loss')
                (loss/len(group)).backward(); losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        torch.cuda.synchronize(); seconds=time.perf_counter()-begin
        row={'epoch':epoch,'loss':float(np.mean(losses)),'seconds':seconds,'images':len(order),
             'images_per_second':len(order)/seconds,'updates':math.ceil(len(order)/32),'lr':rate(epoch,args.epochs),
             'peak_gib':torch.cuda.max_memory_allocated()/2**30}
        with (args.output/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if epoch%10==0 or epoch==final_epoch:
            state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                   'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
            torch.save(state,args.output/'last.writing.pt'); (args.output/'last.writing.pt').replace(args.output/'last.pt')
        if epoch%200==0 or epoch==final_epoch:
            payload={'task':'round302-domain-fcmae','epoch':epoch,'encoder':model.encoder.state_dict(),'config':config,'split':manifest}
            torch.save(payload,args.output/f'encoder_{epoch:03d}.pt')
    torch.save(payload,args.output/'final_encoder.pt')
    write_json(args.output/'complete.json',{'epochs':final_epoch,'images_per_epoch':len(samples),'benchmark':args.benchmark})


if __name__=='__main__': main()

"""Grouped validation baseline; no Test access, context, TTA or organ model input."""
import argparse
import hashlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ihc_single import SingleMarkerStainer, MARKERS
from ihc_unext2 import initialize_fcmae_encoder_from_official_convnextv2

ORGANS = ('colon','liver','stomach')


def marker_policy(organ_markers=None):
    if organ_markers is None:
        return {organ:MARKERS for organ in ORGANS}
    if len(organ_markers) != len(ORGANS) or any(m not in MARKERS for m in organ_markers):
        raise ValueError('Specify exactly one valid marker for Colon, Liver and Stomach')
    return {organ:(marker,) for organ,marker in zip(ORGANS,organ_markers)}


def summarize_results(results, policy):
    # Every supervised marker is evaluated and contributes to model selection.
    score=sum(w*float(np.mean([results[o][m]['score'] for m in policy[o]]))
              for o,w in zip(ORGANS,(.1,.2,.7)))
    selected={o:markers[0] if len(markers)==1 else list(markers) for o,markers in policy.items()}
    return {'organs':results,'selected_markers':selected,'weighted_score':score,
            'selection_rule':'mean of all supervised markers per organ; fixed organ weights'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def inventory(root):
    result = {}
    for organ in ORGANS:
        matches = [p for p in root.rglob('train') if p.parent.name.lower().startswith(organ)]
        if len(matches) != 1:
            raise ValueError((organ, matches))
        folder = matches[0]
        names = sorted(p.name for p in (folder/'DAPI').glob('*.jpg'))
        assert names
        for marker in MARKERS:
            assert set(names) == {p.name for p in (folder/marker).glob('*.jpg')}
        groups = sorted({n.split('_')[0] for n in names})
        random.Random(2026).shuffle(groups)
        val = set(groups[:math.ceil(len(groups)*.2)])
        result[organ] = {'folder':str(folder), 'val_rois':sorted(val), 'train_rois':sorted(set(groups)-val),
                         'train':[n for n in names if n.split('_')[0] not in val],
                         'val':[n for n in names if n.split('_')[0] in val]}
    return result


def read_image(path):
    with Image.open(path) as image:
        value = np.asarray(image.convert('L'),dtype=np.uint8).copy()
    if value.shape != (256,256):
        raise ValueError((str(path),value.shape))
    return value


def preload(manifest, policy=None):
    policy=marker_policy() if policy is None else policy
    cache = {}
    for organ, spec in manifest.items():
        folder = Path(spec['folder'])
        for split in ('train','val'):
            for name in spec[split]:
                cache[organ,name] = {m:read_image(folder/m/name) for m in ('DAPI',)+tuple(policy[organ])}
    return cache


def gaussian_ssim(x,y):
    t = torch.arange(11,device=x.device,dtype=torch.float32)-5
    g = torch.exp(-t*t/(2*1.5**2)); g = g/g.sum()
    kernel = (g[:,None]*g[None,:])[None,None]
    a=F.conv2d(x,kernel); b=F.conv2d(y,kernel)
    va=F.conv2d(x*x,kernel)-a*a; vb=F.conv2d(y*y,kernel)-b*b
    cov=F.conv2d(x*y,kernel)-a*b
    return (((2*a*b+.01**2)*(2*cov+.03**2))/((a*a+b*b+.01**2)*(va+vb+.03**2))).mean()


def tensor(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device,dtype=torch.float32)[None,None]/255


def encoded_prediction(pred):
    array = np.rint(pred[0,0].float().cpu().numpy()*255).clip(0,255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).convert('RGB').save(buffer,format='JPEG',quality=95)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return np.asarray(image.convert('L')).copy()


@torch.inference_mode()
def evaluate(model, manifest, cache, limit=0, policy=None, device='cuda'):
    policy=marker_policy() if policy is None else policy
    model.eval(); results = {}
    for organ, spec in manifest.items():
        results[organ] = {}
        names = spec['val'][:limit] if limit else spec['val']
        for marker in policy[organ]:
            index=MARKERS.index(marker)
            rows=[]; rois={}
            for name in names:
                arrays=cache[organ,name]
                x=tensor(arrays['DAPI'],device); y=tensor(arrays[marker],device)
                pred=model(x,torch.tensor([index],device=device))
                assert pred.shape==y.shape and torch.isfinite(pred).all()
                saved=tensor(encoded_prediction(pred),device)
                ssim=gaussian_ssim(saved,y).item(); mse=F.mse_loss(saved,y).item()
                psnr=min(50.,-10*math.log10(max(mse,1e-12)))
                row={'ssim':ssim,'psnr_capped':psnr,'score':70*ssim+.6*psnr}
                rows.append(row); rois.setdefault(name.split('_')[0],[]).append(row)
            mean=lambda values:{key:float(np.mean([v[key] for v in values])) for key in rows[0]}
            results[organ][marker]={**mean(rows),'images':len(rows),'rois':{k:mean(v) for k,v in rois.items()}}
    return summarize_results(results,policy)


def queued_supervised_handoff(args):
    """Redirect the command already held by the live PowerShell launcher.

    The pretraining process and its launcher remain running. The launcher opens
    this Python entry point only after the complete800-epoch encoder audit.
    """
    handoff=args.output/'supervised_handoff.json'
    if not handoff.is_file():
        if args.check_handoff:
            raise FileNotFoundError(handoff)
        return False
    if args.smoke or args.resume or args.audit or args.organ_markers:
        raise ValueError('The queued comparison must use its registered plan')
    request=json.loads(handoff.read_text(encoding='utf-8-sig'))
    if set(request)!={'kind','plan'} or request['kind']!='round302-fixed-marker-comparison':
        raise ValueError('Unrecognized supervised handoff')
    from round302_supervised import run_plan
    run_plan(Path(request['plan']),check_only=args.check_handoff,inherited=args)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('dataset'))
    p.add_argument('--output',type=Path,default=Path('runs/round300'))
    p.add_argument('--public',type=Path,default=Path('runs/public_checkpoints/convnextv2_tiny_1k_224_fcmae.pt'))
    p.add_argument('--audit',action='store_true'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--epochs',type=int,default=48)
    p.add_argument('--validation-every',type=int,default=4)
    p.add_argument('--domain-encoder',type=Path)
    p.add_argument('--organ-markers',nargs=3,choices=MARKERS,
                   metavar=('COLON_MARKER','LIVER_MARKER','STOMACH_MARKER'))
    p.add_argument('--check-handoff',action='store_true',help='Validate the pending phase without running it')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if queued_supervised_handoff(args): return
    if args.epochs < 1 or args.validation_every < 1: raise ValueError('Positive training/validation intervals required')
    policy=marker_policy(args.organ_markers)
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    manifest=inventory(args.data)
    if args.audit:
        write_json(args.output/'split.json',manifest)
        print(json.dumps({o:{'train':len(s['train']),'val':len(s['val']),'val_rois':s['val_rois']} for o,s in manifest.items()}),flush=True)
        return
    if (args.output/'last.pt').exists() and not args.resume:
        raise FileExistsError('Use --resume for an existing trajectory')
    config={'seed':2026,'epochs':args.epochs,'updates_per_epoch':256,'accumulation':8,'batch':1,
            'organ_probabilities':[.1,.2,.7],
            'supervised_markers':{o:list(ms) for o,ms in policy.items()},
            'marker_probabilities_by_organ':{o:[1/len(ms) if m in ms else 0. for m in MARKERS] for o,ms in policy.items()},
            'validation_selection':'mean of every supervised marker, fixed organ weights; no marker selection after training',
            'lr':.0003,'minimum_lr':.000001,'weight_decay':.0001,'encoder_stage_decay':1.,
            'loss':'0.7 Gaussian11 SSIM +0.2 L1 +0.1 MSE','amp':'bfloat16 training; fp32 evaluation',
            'augmentation':'paired D4 on full single train image only','tta':False,'context':False,
            'jpeg_quality':95,'validation_every':args.validation_every,'metric_assumptions':'sigma1.5 population valid-window; mean per-image score; PSNR cap50',
            'public_sha256':hashlib.sha256(args.public.read_bytes()).hexdigest(),'smoke':args.smoke}
    if args.domain_encoder:
        config['domain_encoder_sha256']=hashlib.sha256(args.domain_encoder.read_bytes()).hexdigest()
    if args.resume:
        assert json.loads((args.output/'config.json').read_text())==config
        assert json.loads((args.output/'split.json').read_text())==manifest
    else:
        write_json(args.output/'split.json',manifest); write_json(args.output/'config.json',config)
    print('Supervised markers: '+json.dumps(config['supervised_markers']),flush=True)
    print('Loading Train/Val DAPI and declared targets only',flush=True); cache=preload(manifest,policy)
    model=SingleMarkerStainer()
    init=initialize_fcmae_encoder_from_official_convnextv2(model.encoder,args.public)
    if args.domain_encoder:
        domain=torch.load(args.domain_encoder,map_location='cpu',weights_only=False)
        assert domain['task']=='round302-domain-fcmae' and domain['config']['seed']==2026
        assert domain['split']==manifest, 'Domain pretraining must exclude exactly the same validation ROIs'
        assert domain['config']['public_sha256']==config['public_sha256']
        assert args.smoke or (not domain['config']['benchmark'] and domain['epoch']==domain['config']['epochs'])
        model.encoder.load_state_dict(domain['encoder'],strict=True)
        init['domain_encoder_sha256']=config['domain_encoder_sha256']; init['domain_epochs']=domain['epoch']
        del domain
    write_json(args.output/'initialization.json',init)
    model=model.cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    start=1; best=-float('inf')
    if args.resume:
        ck=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng']); torch.cuda.set_rng_state_all(ck['cuda_rng'])
        start=ck['epoch']+1; best=ck['best']; del ck
        metrics=args.output/'metrics.jsonl'
        if metrics.exists():
            rows=[json.loads(line) for line in metrics.read_text().splitlines()]
            metrics.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['epoch']<start))
    epochs=1 if args.smoke else config['epochs']; updates=4 if args.smoke else config['updates_per_epoch']
    for epoch in range(start,epochs+1):
        model.train(); begin=time.perf_counter(); losses=[]
        sampled_targets={o:{m:0 for m in policy[o]} for o in ORGANS}
        for step in range(updates):
            progress=((epoch-1)*updates+step)/max(1,epochs*updates-1)
            lr=1e-6+(.0003-1e-6)*.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            for _ in range(config['accumulation']):
                organ=random.choices(ORGANS,weights=config['organ_probabilities'])[0]
                name=random.choice(manifest[organ]['train']); target=policy[organ][random.randrange(len(policy[organ]))]
                marker=MARKERS.index(target); sampled_targets[organ][target]+=1
                arrays=cache[organ,name]; x=arrays['DAPI']; y=arrays[target]
                k=random.randrange(4); flip=random.randrange(2)
                x=np.rot90(x,k); y=np.rot90(y,k)
                if flip: x=np.fliplr(x); y=np.fliplr(y)
                x=tensor(x,'cuda'); y=tensor(y,'cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(x,torch.tensor([marker],device='cuda'))
                pred=pred.float()
                loss=.7*(1-gaussian_ssim(pred,y))+.2*F.l1_loss(pred,y)+.1*F.mse_loss(pred,y)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                (loss/config['accumulation']).backward(); losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        torch.cuda.synchronize(); seconds=time.perf_counter()-begin
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'train_seconds':seconds,
             'images_per_second':updates*config['accumulation']/seconds,'peak_gib':torch.cuda.max_memory_allocated()/2**30,
             'sampled_targets':sampled_targets}
        if args.smoke or epoch%config['validation_every']==0 or epoch==epochs:
            begin=time.perf_counter(); result=evaluate(model,manifest,cache,limit=1 if args.smoke else 0,policy=policy)
            row['validation_seconds']=time.perf_counter()-begin; row['validation_score']=result['weighted_score']
            write_json(args.output/f'validation_{epoch:03d}.json',result)
            if result['weighted_score']>best:
                best=result['weighted_score']
                torch.save({'model':model.state_dict(),'epoch':epoch,'config':config,'markers':result['selected_markers']},args.output/'best.pt')
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,
               'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
        torch.save(state,args.output/'last.writing.pt'); (args.output/'last.writing.pt').replace(args.output/'last.pt')
        with (args.output/'metrics.jsonl').open('a',encoding='utf-8') as handle: handle.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    write_json(args.output/'complete.json',{'epochs':epochs,'best_validation_score':best,'test_inference':False})


if __name__=='__main__': main()

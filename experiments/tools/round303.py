"""Train-only two-stage recipe comparison on the unchanged semifinal model."""
import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from round300 import (ORGANS, MARKERS, SingleMarkerStainer, inventory, marker_policy,
                      preload, tensor, evaluate, write_json)
from ihc_torch import CompetitionLoss

ROOT = Path(__file__).resolve().parents[1]


def optimizer_for(model, stage):
    lr = 3e-4 if stage == 1 else 1e-4
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    groups = [{'params': [p for p in model.parameters() if id(p) not in encoder_ids],
               'lr': lr, 'name': 'decoder_head'}]
    modules = [model.encoder.stem, *model.encoder.stages]
    for i, module in enumerate(modules):
        groups.append({'params': list(module.parameters()), 'lr': lr * .8**(4-i),
                       'name': f'encoder_{i}'})
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def epoch_order(manifest, epoch):
    order = [(o,n) for o in ORGANS for n in manifest[o]['train']]
    random.Random(2026 + epoch).shuffle(order)
    return order


def all_train_manifest(grouped):
    """Include the held-out Train ROIs for final fitting; never include Test."""
    result = copy.deepcopy(grouped)
    for spec in result.values():
        assert not set(spec['train']) & set(spec['val'])
        spec['train'] = sorted(spec['train'] + spec['val'])
        spec['train_rois'] = sorted(spec['train_rois'] + spec['val_rois'])
        spec['val'] = []; spec['val_rois'] = []
    return result


def organ_loss_weights(manifest):
    """Make the mean per-image loss represent the fixed per-organ score weights."""
    counts={o:len(manifest[o]['train']) for o in ORGANS}
    if any(n<=0 for n in counts.values()): raise ValueError('Every organ needs training images')
    total=sum(counts.values())
    return {o:total*w/counts[o] for o,w in zip(ORGANS,(.1,.2,.7))}


def run(args):
    full_train = getattr(args, 'full_train', False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    manifest = inventory(Path('dataset'))
    policy = marker_policy(getattr(args, 'organ_markers', ['CD68','CD45RO','CD68']))
    domain_path = Path('runs/round302_pretrain/final_encoder.pt')
    domain_hash = hashlib.sha256(domain_path.read_bytes()).hexdigest()
    audit = json.loads(Path('reports/round302_pretrain_audit.json').read_text())
    assert audit['epochs'] == 800 and audit['optimizer_steps'] == 202400
    assert domain_hash == audit['encoder_sha256']
    domain = torch.load(domain_path, map_location='cpu', weights_only=False)
    assert domain['split'] == manifest and domain['epoch'] == 800
    if full_train:
        grouped_manifest = copy.deepcopy(manifest)
        manifest = all_train_manifest(manifest)
        assert {o:len(s['train']) for o,s in manifest.items()} == {'colon':6296,'liver':2380,'stomach':1680}
    loss_weights=organ_loss_weights(manifest) if getattr(args,'score_weighted_loss',False) else None
    config = dict(seed=2026, stages=2, epochs_per_stage=14, accumulation=16, batch=1,
                  stage_lrs=[3e-4,1e-4], encoder_stage_decay=.8, minimum_lr=0.,
                  weight_decay=1e-4, clip=1., precision='bfloat16 forward; float32 loss',
                  loss='existing CompetitionLoss uniform7: 0.7 SSIM + 0.2 L1 + 0.1 MSE',
                  sampling='each Train image exactly once per epoch; no replacement',
                  supervised_markers={o:list(policy[o]) for o in ORGANS},
                  domain_encoder_sha256=domain_hash, smoke=args.smoke,
                  model='unchanged SingleMarkerStainer', validation='Gaussian11 JPEG95 every epoch',
                  context=False, tta=False, test_inference=False)
    if loss_weights is not None:
        config.update(organ_loss_weights=loss_weights,organ_loss_objective=[.1,.2,.7])
    endpoint = getattr(args, 'endpoint', 27)
    assert 1 <= endpoint <= 28
    if full_train:
        config.update(full_train=True, endpoint=endpoint,
                      validation='none during full-data fitting; endpoint fixed by Round303 grouped Val',
                      pretraining_scope='Round302 grouped Train only; encoder reused unchanged',
                      selected_by=getattr(args, 'selected_by', 'Round303 best epoch27: stage1 14 + stage2 13; both cosine horizons14'))
    prefix = getattr(args, 'run_prefix', 'round304' if full_train else 'round303')
    if prefix not in ('round303','round304','round305','round306','round307'):
        raise ValueError('Unregistered output prefix')
    output = Path(f'runs/{prefix}_smoke' if args.smoke else f'runs/{prefix}')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'last.pt').exists() and not args.resume:
        raise FileExistsError('Use --resume to preserve the existing run')
    if (output/'config.json').exists():
        assert json.loads((output/'config.json').read_text()) == config
    write_json(output/'config.json',config); write_json(output/'split.json',manifest)
    if full_train: write_json(output/'selection_split.json',grouped_manifest)
    model = SingleMarkerStainer()
    model.encoder.load_state_dict(domain['encoder'],strict=True)
    del domain
    model = model.cuda()
    criterion = CompetitionLoss(ssim_mode='uniform7')
    cache = preload(manifest,policy)
    print('Loaded Train/Val and fixed targets',flush=True)
    stage_length = 1 if args.smoke else 14
    total = 2*stage_length if args.smoke or not full_train else endpoint
    start, best, saved = 1, -math.inf, None
    if args.resume and (output/'last.pt').exists():
        saved = torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model']); start=saved['epoch']+1; best=saved['best']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
        rows = [json.loads(s) for s in (output/'metrics.jsonl').read_text().splitlines()]
        (output/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<start))
    optimizer = scheduler = None
    for epoch in range(start,total+1):
        stage = 1 + (epoch-1)//stage_length
        stage_epoch = 1 + (epoch-1)%stage_length
        if optimizer is None or stage_epoch == 1:
            optimizer = optimizer_for(model,stage)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=stage_length,eta_min=0.)
            if saved is not None and stage_epoch != 1:
                optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            saved = None
        rates = [g['lr'] for g in optimizer.param_groups]
        order = epoch_order(manifest,epoch)
        if args.smoke: order=order[:32]
        assert len(order)==len(set(order))
        model.train(); begin=time.perf_counter(); losses=[]; counts={o:0 for o in ORGANS}
        for offset in range(0,len(order),16):
            group = order[offset:offset+16]; optimizer.zero_grad(set_to_none=True)
            for organ,name in group:
                marker=policy[organ][0]; arrays=cache[organ,name]
                k=random.randrange(4); flip=random.randrange(2)
                x=np.rot90(arrays['DAPI'],k); y=np.rot90(arrays[marker],k)
                if flip: x=np.fliplr(x); y=np.fliplr(y)
                x=tensor(x,'cuda'); y=tensor(y,'cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(x,torch.tensor([MARKERS.index(marker)],device='cuda'))
                loss=criterion(pred.float(),y)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                objective=loss if loss_weights is None else loss*loss_weights[organ]
                (objective/len(group)).backward(); losses.append(loss.item()); counts[organ]+=1
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize(); seconds=time.perf_counter()-begin
        if not args.smoke: assert counts=={o:len(manifest[o]['train']) for o in ORGANS}
        result=None if full_train else evaluate(model,manifest,cache,policy=policy,limit=1 if args.smoke else 0)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if result is not None: write_json(output/f'validation_{epoch:03d}.json',result)
        if result is not None and result['weighted_score']>best:
            best=result['weighted_score']
            torch.save(dict(model=model.state_dict(),epoch=epoch,stage=stage,stage_epoch=stage_epoch,
                            config=config,markers=result['selected_markers']),output/'best.pt')
        scheduler.step()
        row=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch,train_seconds=seconds,
                 loss=float(np.mean(losses)),samples=counts,updates=math.ceil(len(order)/16),
                 group_lrs=rates,validation_score=None if result is None else result['weighted_score'])
        with (output/'metrics.jsonl').open('a') as handle: handle.write(json.dumps(row)+'\n')
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   epoch=epoch,best=best,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(state,output/'last.writing.pt'); (output/'last.writing.pt').replace(output/'last.pt')
        if stage_epoch==stage_length: torch.save(state,output/f'stage{stage}_final.pt')
        if full_train and epoch==total:
            torch.save(dict(model=model.state_dict(),epoch=epoch,stage=stage,stage_epoch=stage_epoch,
                            config=config,markers={o:policy[o][0] for o in ORGANS}),output/'final.pt')
        print(json.dumps(row),flush=True)
    write_json(output/'complete.json',dict(epochs=total,best_validation_score=None if full_train else best,
                                          full_train=full_train,test_inference=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--log',action='store_true')
    args=parser.parse_args()
    import os
    os.chdir(ROOT)
    if args.log:
        sys.stdout=open('reports/round303.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open('reports/round303.err','a',encoding='utf-8',buffering=1)
    try: run(args)
    except Exception:
        traceback.print_exc(); sys.exit(1)

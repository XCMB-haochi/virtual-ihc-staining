"""Queued five-fold label-excluded distillation with an untouched outer Val."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from round300 import (ORGANS, MARKERS, SingleMarkerStainer, inventory, marker_policy,
                      preload, read_image, tensor, evaluate, write_json)
from round303 import ROOT, optimizer_for, epoch_order, organ_loss_weights, all_train_manifest
from round305 import control_record
from ihc_torch import CompetitionLoss, apply_d4_transform, invert_d4_transform

POLICY=marker_policy(['CD68','CD45RO','CD68'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distilled_loss(prediction, target, teacher, criterion, weight=.2):
    hard=criterion(prediction,target)
    return hard if teacher is None else (1-weight)*hard+weight*criterion(prediction,teacher)


def training_input_gain(image, generator, bounds):
    return (image*generator.uniform(*bounds)).clamp(0,1)


def geometry_consistency_loss(model, source, marker, prediction, index, criterion):
    # Training-only second pass; neither branch is detached or averaged.
    transformed=apply_d4_transform(source,index)
    with torch.autocast(source.device.type,dtype=torch.bfloat16,enabled=source.is_cuda):
        alternative=model(transformed,marker)
    restored=invert_d4_transform(alternative,index)
    return criterion(prediction.float(),restored.float())


def fixed_folds(grouped, number=5):
    """Partition each organ's Train ROIs, greedily balancing image counts."""
    folds=[{o:[] for o in ORGANS} for _ in range(number)]
    for organ in ORGANS:
        by_roi={roi:[n for n in grouped[organ]['train'] if n.split('_')[0]==roi]
                for roi in grouped[organ]['train_rois']}
        rois=sorted(by_roi); random.Random(2026).shuffle(rois)
        rois.sort(key=lambda roi:len(by_roi[roi]),reverse=True)
        loads=[0]*number
        for roi in rois:
            slot=min(range(number),key=lambda i:(loads[i],i))
            folds[slot][organ].extend(by_roi[roi]); loads[slot]+=len(by_roi[roi])
        assert all(loads)
        flattened=[n for fold in folds for n in fold[organ]]
        assert len(flattened)==len(set(flattened))==len(grouped[organ]['train'])
        assert set(flattened)==set(grouped[organ]['train'])
    return [{o:sorted(f[o]) for o in ORGANS} for f in folds]


def assign_heldout_rois(grouped, number=5):
    """Assign each outer-Val ROI to one teacher; all teachers excluded outer Val."""
    assignments=[{o:[] for o in ORGANS} for _ in range(number)]
    for organ in ORGANS:
        rois=sorted(grouped[organ]['val_rois'])
        for index,roi in enumerate(rois):
            slot=index%number
            assignments[slot][organ].extend(n for n in grouped[organ]['val'] if n.split('_')[0]==roi)
        emitted=[n for fold in assignments for n in fold[organ]]
        assert len(emitted)==len(set(emitted))==len(grouped[organ]['val'])
        assert set(emitted)==set(grouped[organ]['val'])
        for fold in assignments:
            roi_ids={n.split('_')[0] for n in fold[organ]}
            assert all(sum(roi in {x.split('_')[0] for x in other[organ]} for other in assignments)==1 for roi in roi_ids)
    return assignments


def teacher_split(grouped, excluded):
    split=copy.deepcopy(grouped)
    for o,spec in split.items():
        withheld=set(excluded[o]); assert withheld<=set(spec['train'])
        spec['train']=[n for n in spec['train'] if n not in withheld]
        spec['train_rois']=sorted({n.split('_')[0] for n in spec['train']})
        spec['val']=[]; spec['val_rois']=[]
        assert not set(spec['train_rois']) & {n.split('_')[0] for n in withheld}
    return split


def seed():
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark=False


def fit(output, split, base_config, endpoint, encoder, initial=None, targets=None,
        validate=False, smoke=False, dapi_gain_range=None, training_ssim='uniform7', consistency_weight=0., distillation_weight=.2, organ_objective=None, stage2_lr_scale=1., stratified_updates=False, marker_heads=False, marker_heads_from_start=False, paired_crop=False):
    """Fixed horizons14. Student replays stage2 from exact saved stage1 state."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    cfg=dict(base_config)
    if paired_crop:
        cfg['training_paired_crop']={'size':240,'probability':.5,'stage':2,
            'seed_rule':'2026+200000*epoch','validation_crop':False}
    if marker_heads or marker_heads_from_start:
        if marker_heads_from_start:
            assert initial is None, 'From-start heads must use the fixed encoder initialization'
        else:
            assert initial is not None, 'Stage2 head split requires a fixed stage1 checkpoint'
        cfg['model']='SingleMarkerStainer with CD68/CD45RO output heads split at '+('stage1' if marker_heads_from_start else 'stage2')
        if marker_heads_from_start:
            cfg['consistency_start_epoch']=15
            cfg['selection_start_epoch']=15
    if stratified_updates: cfg['update_order']='proportional organ interleaving; same per-image geometry and consistency view'
    assert stage2_lr_scale in (1.,.5)
    if stage2_lr_scale!=1.:
        cfg['stage2_lr_scale']=stage2_lr_scale
        cfg['stage_lrs']=[base_config['stage_lrs'][0],base_config['stage_lrs'][1]*stage2_lr_scale]
    if distillation_weight not in (.2,.4): raise ValueError('Unregistered distillation weight')
    if consistency_weight not in (0.,.1,.2,.4): raise ValueError('Unregistered consistency strength')
    if consistency_weight:
        cfg['training_geometry_consistency_weight']=consistency_weight
    if training_ssim not in ('uniform7','gaussian11_valid'): raise ValueError('Unregistered SSIM objective')
    if training_ssim != 'uniform7':
        cfg['training_ssim']=training_ssim
        cfg['loss']='CompetitionLoss Gaussian11 valid: 0.7 SSIM + 0.2 L1 + 0.1 MSE'
    weighted='organ_loss_objective' in base_config
    weights=organ_loss_weights(split) if weighted else None
    if organ_objective is not None:
        assert weighted and tuple(organ_objective)==(.1,.1,.8)
        total=sum(len(split[o]['train']) for o in ORGANS)
        weights={o:total*w/len(split[o]['train']) for o,w in zip(ORGANS,organ_objective)}
        cfg['organ_loss_objective']=list(organ_objective)
    if weights is not None: cfg['organ_loss_weights']=weights
    cfg.update(endpoint=endpoint,smoke=smoke,full_train=not validate,
               distillation_weight=distillation_weight if targets is not None else 0.,
               initial_checkpoint_sha256=digest(initial) if initial else None,
               teacher_cache=str(targets) if targets else None,
               pretraining_scope='shared Round302 outer-Train DAPI; not inner-fold-excluded pretraining',
               validation='Gaussian11 JPEG95 each epoch' if validate else 'no validation during fitting')
    if dapi_gain_range is not None:
        assert 0 < dapi_gain_range[0] <= dapi_gain_range[1]
        cfg['training_dapi_gain_range']=list(dapi_gain_range)
    config_path=output/'config.json'
    if config_path.exists(): assert json.loads(config_path.read_text(encoding='utf-8'))==cfg
    write_json(config_path,cfg)
    if (output/'split.json').exists(): assert json.loads((output/'split.json').read_text(encoding='utf-8'))==split
    write_json(output/'split.json',split)
    if (output/'complete.json').exists(): return
    from ihc_single import MarkerHeadStainer
    seed(); model=MarkerHeadStainer() if (marker_heads or marker_heads_from_start) else SingleMarkerStainer()
    if marker_heads_from_start: model.split_output_head()
    model.encoder.load_state_dict(encoder,strict=True)
    model=model.cuda(); best=-math.inf; start=1; saved=None
    last_path=output/'last.pt'
    if last_path.exists():
        saved=torch.load(last_path,map_location='cpu',weights_only=False)
        if marker_heads and not marker_heads_from_start: model.split_output_head()
        model.load_state_dict(saved['model']); start=saved['epoch']+1; best=saved['best']
    elif initial:
        saved=torch.load(initial,map_location='cpu',weights_only=False)
        assert saved['epoch']==14
        model.load_state_dict(saved['model']); start=15
        if marker_heads and not marker_heads_from_start: model.split_output_head()
    if saved is not None:
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    trajectory=output/'metrics.jsonl'
    if trajectory.exists():
        rows=[json.loads(s) for s in trajectory.read_text(encoding='utf-8').splitlines()]
        trajectory.write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<start),encoding='utf-8')
    criterion=CompetitionLoss(ssim_mode=training_ssim)
    cache=preload(split,POLICY)
    teacher_cache={}
    if targets:
        manifest=json.loads((Path(targets)/'manifest.json').read_text(encoding='utf-8'))
        expected={(o,n) for o in ORGANS for n in split[o]['train']}
        assert expected=={(r['organ'],r['name']) for r in manifest['items']}
        assert len(manifest['items'])==len(expected)
        teacher_cache={(o,n):read_image(Path(targets)/o/(n+'.png')) for o,n in expected}
    optimizer=scheduler=None
    for epoch in range(start,endpoint+1):
        stage=1+(epoch-1)//14; stage_epoch=1+(epoch-1)%14
        if optimizer is None or stage_epoch==1:
            optimizer=optimizer_for(model,stage)
            if stage==2:
                for group in optimizer.param_groups: group['lr']*=stage2_lr_scale
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,14,eta_min=0.)
            if last_path.exists() and saved is not None and stage_epoch!=1:
                optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
            saved=None
        rates=[g['lr'] for g in optimizer.param_groups]
        gain_rng=random.Random(2026+100000*epoch)
        crop_rng=random.Random(2026+200000*epoch) if paired_crop else None
        cropped_counts={o:0 for o in ORGANS}
        order=epoch_order(split,epoch)
        geometry_by_image=None
        if stratified_updates:
            from organ_order import stratified_order
            original_positions={item:i for i,item in enumerate(order)}
            geometry_rng=random.Random(); geometry_rng.setstate(random.getstate())
            geometry_by_image={item:(geometry_rng.randrange(4),geometry_rng.randrange(2)) for item in order}
            order=stratified_order(order,epoch)
        if smoke: order=order[:32]
        model.train(); begin=time.perf_counter(); losses=[]; consistency_values=[]; counts={o:0 for o in ORGANS}
        for offset in range(0,len(order),16):
            group=order[offset:offset+16]; optimizer.zero_grad(set_to_none=True)
            for organ,name in group:
                arrays=cache[organ,name]; k=random.randrange(4); flip=random.randrange(2)
                if geometry_by_image is not None: k,flip=geometry_by_image[organ,name]
                def geometry(a):
                    a=np.rot90(a,k)
                    return np.fliplr(a) if flip else a
                x=tensor(geometry(arrays['DAPI']),'cuda'); y=tensor(geometry(arrays[POLICY[organ][0]]),'cuda')
                if dapi_gain_range is not None:
                    x=training_input_gain(x,gain_rng,dapi_gain_range)
                crop_box=None
                if paired_crop and stage==2:
                    from paired_crop import draw_crop,apply_crop
                    crop_box=draw_crop(crop_rng,*x.shape[-2:])
                    x=apply_crop(x,crop_box); y=apply_crop(y,crop_box)
                    cropped_counts[organ]+=int(crop_box is not None)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(x,torch.tensor([MARKERS.index(POLICY[organ][0])],device='cuda'))
                soft=None
                if targets is not None and stage==2:
                    soft=tensor(geometry(teacher_cache[organ,name]),'cuda')
                    if paired_crop: soft=apply_crop(soft,crop_box)
                loss=distilled_loss(pred.float(),y,soft,criterion,distillation_weight)
                if consistency_weight and epoch>=cfg.get('consistency_start_epoch',1):
                    position=original_positions[organ,name] if geometry_by_image is not None else sum(counts.values())
                    index=1+(((epoch-1)*len(order)+position)%7)
                    consistency=geometry_consistency_loss(model,x,torch.tensor([MARKERS.index(POLICY[organ][0])],device='cuda'),pred,index,criterion)
                    loss=loss+consistency_weight*consistency
                    consistency_values.append(consistency.item())
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite distillation loss')
                objective=loss if weights is None else loss*weights[organ]
                (objective/len(group)).backward(); losses.append(loss.item()); counts[organ]+=1
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        torch.cuda.synchronize(); elapsed=time.perf_counter()-begin
        if not smoke: assert counts=={o:len(split[o]['train']) for o in ORGANS}
        result=evaluate(model,split,cache,policy=POLICY,limit=1 if smoke else 0) if validate else None
        if result:
            write_json(output/f'validation_{epoch:03d}.json',result)
            if epoch>=cfg.get('selection_start_epoch',1) and result['weighted_score']>best:
                best=result['weighted_score']
                torch.save(dict(model=model.state_dict(),config=cfg,epoch=epoch,stage=stage,
                                stage_epoch=stage_epoch,markers={o:POLICY[o][0] for o in ORGANS}),output/'best.pt')
        scheduler.step()
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   epoch=epoch,best=best,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        row=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch,train_seconds=elapsed,loss=float(np.mean(losses)),
                 samples=counts,updates=math.ceil(len(order)/16),group_lrs=rates,
                 validation_score=result['weighted_score'] if result else None)
        if consistency_values: row['geometry_consistency_loss']=float(np.mean(consistency_values))
        if paired_crop: row['cropped_samples']=cropped_counts
        with trajectory.open('a',encoding='utf-8') as handle: handle.write(json.dumps(row)+'\n')
        torch.save(state,output/'last.writing.pt'); (output/'last.writing.pt').replace(last_path)
        if stage_epoch==14: torch.save(state,output/'stage1_final.pt' if stage==1 else output/'stage2_final.pt')
        print(json.dumps({'run':str(output),**row}),flush=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    torch.save(dict(model=model.state_dict(),config=cfg,epoch=endpoint,stage=1+(endpoint-1)//14,
                    stage_epoch=1+(endpoint-1)%14,markers={o:POLICY[o][0] for o in ORGANS}),output/'final.pt')
    control_record(output/'complete.json',dict(epochs=endpoint,best_validation_score=best if validate else None))


def predict_targets(checkpoint, grouped, names, folder, fold, existing):
    """One single teacher prediction per training image; fixed lossless uint8."""
    model=SingleMarkerStainer()
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True); model=model.cuda().eval()
    assert ck['markers']=={o:POLICY[o][0] for o in ORGANS}
    teacher_hash=digest(checkpoint)
    with torch.inference_mode():
        for organ in ORGANS:
            target_folder=folder/organ; target_folder.mkdir(parents=True,exist_ok=True)
            for name in names[organ]:
                key=(organ,name); output=target_folder/(name+'.png')
                image=read_image(Path(grouped[organ]['folder'])/'DAPI'/name)
                pred=model(tensor(image,'cuda'),torch.tensor([MARKERS.index(POLICY[organ][0])],device='cuda'))
                assert pred.shape==(1,1,256,256) and torch.isfinite(pred).all()
                pixels=np.rint(pred[0,0].cpu().numpy()*255).clip(0,255).astype(np.uint8)
                Image.fromarray(pixels).save(output,format='PNG')
                if key in existing: raise ValueError('Duplicate target')
                existing[key]={'organ':organ,'name':name,'fold':fold,'teacher_sha256':teacher_hash}
    del model,ck; torch.cuda.empty_cache()


def audit_fit(folder, split, endpoint, initial=False, smoke=False):
    folder=Path(folder)
    rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['epoch'] for r in rows]==list(range(15 if initial else 1,endpoint+1))
    counts={o:len(split[o]['train']) for o in ORGANS}
    updates=2 if smoke else math.ceil(sum(counts.values())/16)
    assert all(math.isfinite(r['loss']) and r['updates']==updates for r in rows)
    if not smoke: assert all(r['samples']==counts for r in rows)
    last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    steps={int(s['step']) for s in last['optimizer']['state'].values() if 'step' in s}
    expected_steps=updates*(1+(endpoint-1)%14)
    assert last['epoch']==endpoint
    cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    if 'training_paired_crop' in cfg:
        from paired_crop import draw_crop
        for row in rows:
            epoch=row['epoch']; generator=random.Random(2026+200000*epoch)
            order=epoch_order(split,epoch)
            if smoke: order=order[:32]
            expected={o:0 for o in ORGANS}
            if epoch>14:
                for organ,name in order: expected[organ]+=int(draw_crop(generator,256,256) is not None)
            assert row['cropped_samples']==expected, 'Training crop schedule mismatch'
    if 'output heads split' in cfg['model']:
        from ihc_single import MarkerHeadStainer
        with torch.random.fork_rng(devices=[]):
            audit_model=MarkerHeadStainer(); audit_model.split_output_head()
            audit_optimizer=optimizer_for(audit_model,2)
        by_id={id(p):name for name,p in audit_model.named_parameters()}
        head_steps={0:0,1:0}
        # AdamW is reset at stage2, so the saved optimizer state only contains
        # stage2 head steps even when heads were active during stage1.
        for epoch in range(15,endpoint+1):
            order=epoch_order(split,epoch)
            if smoke: order=order[:32]
            for offset in range(0,len(order),16):
                present={MARKERS.index(POLICY[o][0]) for o,n in order[offset:offset+16]}
                for index in present: head_steps[index]+=1
        for group,saved_group in zip(audit_optimizer.param_groups,last['optimizer']['param_groups']):
            for param,key in zip(group['params'],saved_group['params']):
                name=by_id[id(param)]
                expected=head_steps[int(name.split('.')[1])] if name.startswith('marker_heads.') else expected_steps
                assert int(last['optimizer']['state'].get(key,{}).get('step',0))==expected,(name,expected)
    else:
        assert steps=={expected_steps}
    final=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    assert final['epoch']==endpoint and final['markers']=={o:POLICY[o][0] for o in ORGANS}
    assert all(torch.isfinite(v).all() and torch.equal(v,last['model'][k]) for k,v in final['model'].items())
    control_record(folder/'audit.json',{'endpoint':endpoint,'updates':sum(r['updates'] for r in rows),
                                      'initial_stage1_reused':initial,'finite_weights':True})


def wait_for_round307():
    deadline=time.monotonic()+6*3600
    while not Path('reports/round307_complete.json').exists():
        error=Path('reports/round307.err')
        if error.exists() and 'Traceback (most recent call last)' in error.read_text(encoding='utf-8',errors='replace'):
            raise RuntimeError('Round307 failed before dependency completion')
        if time.monotonic()>deadline: raise TimeoutError('Round307 dependency exceeded6h')
        time.sleep(30)


def audit_targets(items, folds, original, root, include_outer_val=False):
    expected={(o,n) for o in ORGANS for n in original[o]['train']}
    if include_outer_val: expected|={(o,n) for o in ORGANS for n in original[o]['val']}
    assert set(items)==expected
    for i,excluded in enumerate(folds):
        split=json.loads((root/f'teacher{i}/split.json').read_text(encoding='utf-8'))
        sha=digest(root/f'teacher{i}/final.pt')
        for organ in ORGANS:
            assert not set(split[organ]['train']) & (set(excluded[organ])|set(original[organ]['val']))
            for name in excluded[organ]:
                assert items[organ,name]['fold']==i and items[organ,name]['teacher_sha256']==sha
            if include_outer_val:
                for name in original[organ]['val']:
                    row=items[organ,name]
                    assert row['fold'] in range(len(folds))
                    teacher_split=json.loads((root/f"teacher{row['fold']}/split.json").read_text(encoding='utf-8'))
                    assert name not in teacher_split[organ]['train']
                    assert row['teacher_sha256']==digest(root/f"teacher{row['fold']}/final.pt")


def pipeline(smoke=False):
    torch.set_num_threads(4)
    if not smoke: wait_for_round307()
    selected=303
    if not smoke:
        decision=json.loads(Path('reports/round307_decision.json').read_text(encoding='utf-8'))
        selected=307 if decision['candidate_round307']>decision['baseline_round303'] else 303
    base=Path(f'runs/round{selected}')
    base_cfg=json.loads((base/'config.json').read_text(encoding='utf-8'))
    original=inventory(Path('dataset'))
    assert original==json.loads((base/'split.json').read_text(encoding='utf-8'))
    domain=torch.load('runs/round302_pretrain/final_encoder.pt',map_location='cpu',weights_only=False)
    assert domain['split']==original and domain['epoch']==800
    assert digest('runs/round302_pretrain/final_encoder.pt')==base_cfg['domain_encoder_sha256']
    encoder=domain['encoder']; del domain
    chosen=torch.load(base/'best.pt',map_location='cpu',weights_only=False)
    teacher_endpoint=chosen['epoch']; del chosen
    root=Path('runs/round308_smoke' if smoke else 'runs/round308'); root.mkdir(parents=True,exist_ok=True)
    config=dict(base_round=selected,base_config=base_cfg,teacher_endpoint=teacher_endpoint,
                teacher_folds=5,weight=.2,smoke=smoke,shared_domain_encoder=True)
    if (root/'plan.json').exists(): assert json.loads((root/'plan.json').read_text(encoding='utf-8'))==config
    write_json(root/'plan.json',config)
    grouped=copy.deepcopy(original)
    if smoke:
        for spec in grouped.values():
            spec['train']=spec['train'][:32]
            spec['train_rois']=sorted({n.split('_')[0] for n in spec['train']})
        # Execution smoke is separate from the real five-fold grouping test.
        folds=[{o:grouped[o]['train'][::2] for o in ORGANS},{o:grouped[o]['train'][1::2] for o in ORGANS}]
    else: folds=fixed_folds(grouped)
    write_json(root/'folds.json',folds)
    targets=root/'targets'; targets.mkdir(exist_ok=True); items={}
    for i,excluded in enumerate(folds):
        if smoke:
            split=copy.deepcopy(grouped)
            for o in ORGANS:
                split[o]['train']=[n for n in grouped[o]['train'] if n not in set(excluded[o])]
                split[o]['val']=[]; split[o]['val_rois']=[]
        else: split=teacher_split(grouped,excluded)
        for o in ORGANS:
            assert not set(split[o]['train']) & set(original[o]['val'])
            assert not set(split[o]['train']) & set(excluded[o])
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if int(hardware.strip())<4096: raise RuntimeError('Insufficient GPU memory at teacher boundary')
        checkpoint=root/f'teacher{i}'/'final.pt'
        fit(checkpoint.parent,split,base_cfg,1 if smoke else teacher_endpoint,encoder,smoke=smoke)
        audit_fit(checkpoint.parent,split,1 if smoke else teacher_endpoint,smoke=smoke)
        predict_targets(checkpoint,grouped,excluded,targets,i,items)
    audit_targets(items,folds,grouped,root)
    control_record(targets/'manifest.json',{'items':list(items.values()),'scope':'outer Train only; inner supervision label-excluded',
                                         'original_split':original,'shared_domain_encoder':True})
    student=root/'student'
    fit(student,grouped,base_cfg,15 if smoke else 28,encoder,
        initial=base/'stage1_final.pt',targets=targets,validate=True,smoke=smoke)
    audit_fit(student,grouped,15 if smoke else 28,initial=True,smoke=smoke)
    checkpoint=torch.load(student/'best.pt',map_location='cpu',weights_only=False)
    result=json.loads((student/f"validation_{checkpoint['epoch']:03d}.json").read_text(encoding='utf-8'))
    baseline=json.loads((base/'complete.json').read_text(encoding='utf-8'))['best_validation_score']
    promote=result['weighted_score']>baseline
    decision={'base_round':selected,'base_score':baseline,'distilled_score':result['weighted_score'],
              'difference':result['weighted_score']-baseline,'use_distillation':promote,
              'selected_epoch':checkpoint['epoch'] if promote else teacher_endpoint,
              'organs':result['organs'],'scope':'five inner supervision folds; outer Val excluded throughout training',
              'limitation':'shared DAPI pretraining; one outer split with3 Stomach ROIs',
              'needs_full_refit':promote or selected==307}
    del checkpoint
    if smoke:
        control_record(root/'smoke_complete.json',{'targets':len(items),'student_epoch':15,'finite_training':True})
        print('Teacher/cache/student smoke passed',flush=True); return
    control_record(Path('reports/round308_decision.json'),decision)
    print(json.dumps(decision),flush=True)
    if decision['needs_full_refit']:
        full=all_train_manifest(original)
        # Only after Val selection, additional official Train rows become fitting data.
        if promote:
            for o in ORGANS:
                assert not set(original[o]['val']) & {n for org,n in items if org==o}
            assignments=[{o:[] for o in ORGANS} for _ in folds]
            for o in ORGANS:
                for index,roi in enumerate(sorted(original[o]['val_rois'])):
                    assignments[index%len(folds)][o].extend(n for n in original[o]['val'] if n.split('_')[0]==roi)
            full_targets=root/'full_targets'; full_targets.mkdir(exist_ok=True)
            import shutil
            for o,n in items:
                (full_targets/o).mkdir(exist_ok=True)
                shutil.copyfile(targets/o/(n+'.png'),full_targets/o/(n+'.png'))
            full_items=dict(items)
            for i,names in enumerate(assignments): predict_targets(root/f'teacher{i}/final.pt',original,names,full_targets,i,full_items)
            control_record(full_targets/'manifest.json',{'items':list(full_items.values()),'scope':'all official Train after fixed Val selection'})
        else: full_targets=None
        final_root=Path('runs/round309')
        fit(final_root,full,base_cfg,decision['selected_epoch'],encoder,targets=full_targets)
        audit_fit(final_root,full,decision['selected_epoch'])
        write_json(final_root/'selection_split.json',original)
        from package_round302 import main
        manifest=Path('reports/round309_submission.json')
        if manifest.exists():
            packaged=json.loads(manifest.read_text(encoding='utf-8'))
            assert digest(packaged['package'])==packaged['sha256']
        else: main(309)
    control_record(Path('reports/round308_complete.json'),{'full_refit':decision['needs_full_refit']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true'); parser.add_argument('--log',action='store_true')
    args=parser.parse_args(); os.chdir(ROOT)
    if args.log:
        sys.stdout=open('reports/round308.log','a',encoding='utf-8',buffering=1)
        sys.stderr=open('reports/round308.err','a',encoding='utf-8',buffering=1)
    try: pipeline(args.smoke)
    except Exception:
        traceback.print_exc(); sys.exit(1)

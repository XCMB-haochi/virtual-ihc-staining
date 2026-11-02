"""Read-only model/teacher audit and independent relation-gradient checks."""
import hashlib
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from ihc_data import image_names, parse_patch_name, resolve_data_root
from ihc_feature_guidance import (load_frozen_multimarker_fcmae_teacher,
    FrozenFCMAERelationalLoss, center_crop_square, relational_distance_loss)
from ihc_spatial import apply_complete_roi_conditioning
from ihc_torch import (PairedPatchDataset, build_model_from_checkpoint,
    checkpoint_conditioning, batch_condition_tensor, CompetitionLoss)
from train_unet import training_batch_losses
from run_round204_cd45_cv import DATASET


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def independent_relation(student, teacher):
    pairs = torch.triu_indices(len(student), len(student), offset=1, device=student.device)
    s = torch.linalg.vector_norm(student[pairs[0]]-student[pairs[1]], dim=1)
    t = torch.linalg.vector_norm(teacher[pairs[0]]-teacher[pairs[1]], dim=1)
    s = s / s.sort().values[(len(s)-1)//2].clamp_min(1e-6)
    t = t / t.sort().values[(len(t)-1)//2].clamp_min(1e-6)
    error = (s-t).abs()
    return torch.where(error < 1, .5*error.square(), error-.5).mean()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-batches-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    data = resolve_data_root(Path(DATASET))
    names = image_names(data/'train'/'DAPI')
    receipts = json.loads(Path('reports/round211_source_hashes.json').read_text())
    report = {'purpose':'implementation audit, no optimizer step or checkpoint selection', 'folds':{}}
    if args.real_batches_only:
        report=json.loads(Path('reports/round211_implementation_audit.json').read_text())
    permitted = {'output_dir','init_checkpoint','feature_guidance_checkpoint',
        'feature_guidance_checkpoint_sha256','feature_guidance_mode','feature_guidance_marker_index',
        'feature_guidance_report','feature_guidance_marker','feature_guidance_batch_samples','feature_guidance_weight'}
    for fold in ('fold0','fold3'):
        record = {'configuration_pairs':[], 'trajectories':[], 'real_batches':[]}
        if args.real_batches_only:
            record=report['folds'][fold]
            record['real_batches']=[]
        for stage in (() if args.real_batches_only else (1,2)):
            configs = [json.loads(Path(f'runs/round211_{fold}_{r}_stage{stage}/config.json').read_text()) for r in ('control','relational')]
            differences = sorted(k for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k))
            assert set(differences) <= permitted, differences
            record['configuration_pairs'].append({'stage':stage, 'different_fields':differences})
            for recipe, config in zip(('control','relational'),configs):
                assert config['batch_size']==16 and config['gradient_accumulation_steps']==1
                assert config['seed']==2026 and config['epochs']==14 and config['distillation_weight']==0
                assert config['markers']==['CD68','CD45RO','HLA-DR','Vimentin']
                if stage==1:
                    assert config['init_checkpoint'] is None
                    assert sha(config['fcmae_encoder_init_checkpoint'])==receipts[fold]['dapi_encoder_sha256']
                else:
                    expected=ROOT/f'runs/round211_{fold}_{recipe}_stage1/epochs/epoch_014.pt'
                    assert Path(config['init_checkpoint']).resolve()==expected.resolve()
                if recipe=='relational':
                    assert config['feature_guidance_marker_index']==1
                    assert config['feature_guidance_marker']=='CD45RO' and config['feature_guidance_batch_samples']==16
                    assert config['feature_guidance_mode']=='encoder-relational-robust'
                root=Path(f'runs/round211_{fold}_{recipe}_stage{stage}')
                rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
                assert [r['epoch'] for r in rows]==list(range(1,15))
                assert all(math.isfinite(r['train_loss']) for r in rows)
                last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
                updates=[float(s['step']) for s in last['optimizer']['state'].values() if 'step' in s]
                assert updates and min(updates)>0
                record['trajectories'].append({'recipe':recipe,'stage':stage,'actual_adam_steps':min(updates),
                    'attempted_steps':sum(r['optimizer_steps'] for r in rows),
                    'initial_supervised_loss':rows[0]['train_supervised_loss'],
                    'final_supervised_loss':rows[-1]['train_supervised_loss'],
                    'initial_guidance_loss':rows[0]['train_feature_guidance_loss'],
                    'final_guidance_loss':rows[-1]['train_feature_guidance_loss']})
                del last
        path=Path(f'runs/round211_{fold}_relational_stage2/epochs/epoch_012.pt')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        config=ck['args']; rois=set(config['val_rois'].split(','))
        teacher_path=Path(config['feature_guidance_checkpoint'])
        teacher_payload=torch.load(teacher_path,map_location='cpu',weights_only=False)
        assert teacher_payload['epoch']==160
        assert not set(teacher_payload['config']['train_rois']) & rois
        assert teacher_payload['config']['data_scope']=='fold-train'
        teacher, _ = load_frozen_multimarker_fcmae_teacher(teacher_path,receipts[fold]['teacher_sha256'],rois,device)
        record['teacher']={'epoch':160,'excluded_rois':sorted(rois),'sha256':sha(teacher_path),
                           'all_parameters_frozen':all(not p.requires_grad for p in teacher.parameters())}
        model=build_model_from_checkpoint(ck,device).train()
        guidance=FrozenFCMAERelationalLoss(teacher,robust_scale=True,stage=4,crop_size=256,batch_samples=16)
        train_names=[n for n in names if parse_patch_name(n).roi not in rois]
        conditions=apply_complete_roi_conditioning(data/'train'/'DAPI',train_names,checkpoint_conditioning(ck))
        chosen=np.random.default_rng(2026).choice(train_names,32,replace=False).tolist()
        loader=DataLoader(PairedPatchDataset(data,chosen,markers=ck['markers'],augment=False,
                        context_halo=64,target_context_halo=64),batch_size=16,shuffle=False,num_workers=0)
        criterion=CompetitionLoss(ssim_mode='uniform7',ssim_weight=.7,l1_weight=.2,mse_weight=.1,
                                  channel_weights=(.05,.85,.05,.05))
        captured={}
        def encoder_hook(module, inputs, outputs): captured['student_features']=outputs[0][-1]
        def teacher_hook(module, inputs, outputs):
            captured['teacher_input']=inputs[0].detach()
            captured['teacher_features']=outputs[0][3].detach()
        hooks=[model.encoder.register_forward_hook(encoder_hook),teacher.register_forward_hook(teacher_hook)]
        for source,target,batch_names in loader:
            source,target=source.to(device),target.to(device)
            condition=batch_condition_tensor(batch_names,conditions,device)
            with torch.amp.autocast('cuda',enabled=True):
                total,supervised,_,relation=training_batch_losses(model,source,target,condition,criterion,0,
                    feature_guidance=guidance,feature_guidance_weight=.05,feature_guidance_marker_index=1)
            assert torch.equal(captured['teacher_input'],center_crop_square(target[:,1:2],256))
            features=captured['student_features']
            assert tuple(features.shape[:2])==(16,768)
            student=center_crop_square(features,16).float().mean((-2,-1))
            reference=captured['teacher_features'].float().mean((-2,-1))
            expected=independent_relation(student,reference)
            torch.testing.assert_close(relation,expected,rtol=2e-4,atol=2e-5)
            s=student.detach().clone().requires_grad_(True)
            a=relational_distance_loss(s,reference,robust_scale=True)
            b=independent_relation(s,reference)
            ga=torch.autograd.grad(a,s)[0];gb=torch.autograd.grad(b,s)[0]
            torch.testing.assert_close(ga,gb,rtol=2e-3,atol=2e-5)
            # Scale before backpropagating through half-precision activations,
            # just as GradScaler does; raw FP16 gradients can underflow.
            gradient_scale=32768.
            gs=torch.autograd.grad(supervised*gradient_scale,features,retain_graph=True)[0].float()/gradient_scale
            gr=torch.autograd.grad(.05*relation*gradient_scale,features)[0].float()/gradient_scale
            assert torch.isfinite(gs).all() and torch.isfinite(gr).all()
            assert gs.norm()>0 and gr.norm()>0
            distances=torch.pdist(reference)
            record['real_batches'].append({'batch_size':16,'pair_count':120,
                'teacher_input_matches_cd45ro':True,'formula_scalar_error':float((relation-expected).abs().detach()),
                'formula_gradient_max_error':float((ga-gb).abs().max()),
                'teacher_pair_distance_median':float(distances.median()),
                'student_pair_distance_median':float(torch.pdist(student.detach()).median()),
                'teacher_near_zero_pair_fraction':float((distances<1e-6).float().mean()),
                'supervised_loss':float(supervised.detach()),'relation_loss':float(relation.detach()),
                'diagnostic_gradient_scale':gradient_scale,
                'supervised_feature_gradient_norm':float(gs.norm()),
                'weighted_relation_feature_gradient_norm':float(gr.norm()),
                'weighted_relation_to_supervised_feature_gradient_norm':float(gr.norm()/gs.norm().clamp_min(1e-12)),
                'feature_gradient_cosine':float(torch.nn.functional.cosine_similarity(gs.flatten(),gr.flatten(),dim=0))})
            del source,target,total,supervised,relation,features,student,expected,gs,gr
            captured.clear()
        for hook in hooks: hook.remove()
        report['folds'][fold]=record
        print(fold,json.dumps(record['real_batches']),flush=True)
        del model,teacher,guidance,ck,teacher_payload
        torch.cuda.empty_cache()
        control_ck=torch.load(f'runs/round211_{fold}_control_stage2/epochs/epoch_012.pt',map_location='cpu',weights_only=False)
        control=build_model_from_checkpoint(control_ck,device).eval()
        control_medians=[]
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            for source,target,batch_names in loader:
                _,feature=control.forward_with_encoder_features(source.to(device),batch_condition_tensor(batch_names,conditions,device))
                control_medians.append(float(torch.pdist(center_crop_square(feature,16).float().mean((-2,-1))).median()))
        record['control_same_batch_feature_distance_medians']=control_medians
        del control,control_ck
        torch.cuda.empty_cache()
    Path('reports/round211_implementation_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Round211 implementation checks passed; no training or weights changed.',flush=True)


if __name__=='__main__': main()

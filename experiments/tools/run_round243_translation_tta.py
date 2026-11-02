"""One fixed full-ROI translation plus D4, equal-weight same-checkpoint TTA."""
import argparse,json,subprocess,sys,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ihc_data import image_names,load_gray,load_context_gray,parse_patch_name
from ihc_spatial import apply_complete_roi_conditioning
from ihc_torch import build_model_from_checkpoint,checkpoint_conditioning,batch_condition_tensor,predict_with_tta,shift_array_edge,save_grayscale_rgb
from audit_semifinal_markers import image_metrics,summarize
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest


def assemble(names,values):
    coords=[parse_patch_name(n) for n in names];rows=sorted({p.row for p in coords});cols=sorted({p.col for p in coords})
    assert rows==list(range(rows[0],rows[-1]+1)) and cols==list(range(cols[0],cols[-1]+1))
    assert len(names)==len(rows)*len(cols)
    out=np.empty((len(rows)*256,len(cols)*256),dtype=np.float32)
    for n,p in zip(names,coords):out[(p.row-rows[0])*256:(p.row-rows[0]+1)*256,(p.col-cols[0])*256:(p.col-cols[0]+1)*256]=values[n]
    return out


def tiles(names,mosaic):
    coords=[parse_patch_name(n) for n in names];r0=min(p.row for p in coords);c0=min(p.col for p in coords)
    return {n:mosaic[(p.row-r0)*256:(p.row-r0+1)*256,(p.col-c0)*256:(p.col-c0+1)*256] for n,p in zip(names,coords)}


def materialize(source,names,destination):
    destination.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(list)
    for n in names:groups[parse_patch_name(n).roi].append(n)
    for items in groups.values():
        mosaic=assemble(items,{n:load_gray(source/n) for n in items})
        # Content moves down/right8. Finite-ROI boundary uses fixed replication.
        moved=shift_array_edge(mosaic,(-8,-8))
        for n,a in tiles(items,moved).items():Image.fromarray(a.astype(np.uint8)).save(destination/n,format='PNG')


@torch.inference_mode()
def predictions(checkpoint,source,shifted,names):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['args']['seed']==2026
    model=build_model_from_checkpoint(ck,torch.device('cuda:0')).eval()
    marker=list(ck['markers']).index('CD45RO');spec=checkpoint_conditioning(ck)
    outputs=[]
    for directory in (source,shifted):
        conditions=apply_complete_roi_conditioning(directory,names,spec)
        values={}
        for i in range(0,len(names),2):
            batch=names[i:i+2]
            x=torch.from_numpy(np.stack([load_context_gray(directory,n,128) for n in batch])[:,None]/255.).float().cuda()
            condition=batch_condition_tensor(batch,conditions,torch.device('cuda:0'))
            y=predict_with_tta(model,x,'d4',condition)[:,marker,128:384,128:384].cpu().numpy()*255.
            for n,value in zip(batch,y):values[n]=value
        outputs.append(values)
    aligned={};groups=defaultdict(list)
    for n in names:groups[parse_patch_name(n).roi].append(n)
    for items in groups.values():
        restored=shift_array_edge(assemble(items,outputs[1]),(8,8))
        aligned.update(tiles(items,restored))
    result={n:(outputs[0][n]+aligned[n])*.5 for n in names}
    assert all(np.isfinite(x).all() for x in result.values())
    del model;torch.cuda.empty_cache()
    return outputs[0],result


def main():
    torch.set_num_threads(4);torch.manual_seed(2026)
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--user-confirmation',action='store_true',help='Generate the unchanged candidate following explicit user confirmation; keep original gate unchanged.')
    args=parser.parse_args()
    if args.user_confirmation:
        report=json.loads(Path('reports/round243_oof.json').read_text())
        decision=json.loads(Path('reports/round243_decision.json').read_text())
        assert report['base']['overall']['images']==report['candidate']['overall']['images']==2380
        assert abs(report['candidate']['overall']['gaussian11_score']-report['base']['overall']['gaussian11_score']-decision['gains']['overall'])<1e-9
        print('User-authorized fixed confirmation; original automatic gate remains',decision['package'],flush=True)
        package_candidate(Path(DATASET))
        return
    root=Path(DATASET);tag='round243_smoke' if args.smoke else 'round243'
    manifest=json.loads(Path('reports/round239_oof.json').read_text());folds=manifest['fold_sources']
    allnames=image_names(root/'train/DAPI');assert len(allnames)==2380
    if args.smoke:folds=folds[:1];allnames=[n for n in allnames if parse_patch_name(n).roi==folds[0]['val_rois'][0]]
    shifted=Path('runs')/tag/'shifted_train_DAPI';materialize(root/'train/DAPI',allnames,shifted)
    rows={'base':defaultdict(list),'candidate':defaultdict(list)}
    for fold in folds:
        ckpath=Path(fold['checkpoint']);assert digest(ckpath)==fold['checkpoint_sha256']
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        assert not ck['args']['train_all'] and set(ck['args']['val_rois'].split(','))==set(fold['val_rois'])
        names=[n for n in allnames if parse_patch_name(n).roi in fold['val_rois']]
        print('Evaluate',fold['name'],len(names),flush=True)
        base,candidate=predictions(ckpath,root/'train/DAPI',shifted,names)
        for n in names:
            target=load_gray(root/'train/CD45RO'/n)
            for label,values in [('base',base),('candidate',candidate)]:
                row=image_metrics(values[n],target)
                for key in ('overall',fold['name'],parse_patch_name(n).roi):rows[label][key].append(row)
    report={label:{s:summarize(v) for s,v in groups.items()} for label,groups in rows.items()}
    reference=manifest['markers']['CD45RO']
    for key,value in report['base'].items():
        if args.smoke and key in ('overall','fold0'):continue
        assert abs(value['gaussian11_score']-reference[key]['gaussian11_score'])<1e-4,('baseline drift',key)
    Path(f'reports/{tag}_oof.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.smoke:
        print('Full-ROI transform and unchanged baseline replay passed; smoke is not selection evidence',flush=True);return
    assert report['base']['overall']['images']==report['candidate']['overall']['images']==2380
    gains={s:v['gaussian11_score']-report['base'][s]['gaussian11_score'] for s,v in report['candidate'].items()}
    decision={'gains':gains,'package':gains['overall']>=.02 and sum(gains[f['name']]>0 for f in folds)>=3}
    Path('reports/round243_decision.json').write_text(json.dumps(decision,indent=2)+'\n');print(json.dumps(decision),flush=True)
    if not decision['package']:return
    package_candidate(root)


def package_candidate(root):
    art=json.loads(Path('reports/round240_artifacts.json').read_text());model=Path('runs/round240_full_stage2/epochs/epoch_012.pt')
    assert digest(model)==art['checkpoint_sha256']
    names=image_names(root/'test/DAPI');assert len(names)==560
    shifted=Path('runs/round243/shifted_test_DAPI');materialize(root/'test/DAPI',names,shifted)
    _,candidate=predictions(model,root/'test/DAPI',shifted,names)
    pred=Path('predictions/round243_cd45ro_q95');pred.mkdir(exist_ok=False)
    for n,value in candidate.items():save_grayscale_rgb(value / 255.0,pred/n,quality=95)
    package=Path('submissions/semifinal_round243_cd45ro_q95.zip')
    if package.exists():raise FileExistsError(package)
    subprocess.run([sys.executable,'tools/package_submission.py',str(pred),str(root/'test/DAPI'),str(package),'--marker','CD45RO'],check=True)
    with zipfile.ZipFile(package) as z:assert z.testzip() is None and len(z.namelist())==560
    Path('reports/round243_hashes.json').write_text(json.dumps({'model_sha256':digest(model),'package_sha256':digest(package),'views':16,'translation':[8,8],'boundary':'replicate','inverse_translation':[-8,-8]},indent=2)+'\n')

if __name__=='__main__':main()

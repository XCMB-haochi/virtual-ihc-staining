"""One fixed same-trajectory average, grouped validation and conditional package."""
import argparse,json,subprocess,sys,zipfile
from pathlib import Path
from average_checkpoints import average_checkpoints
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest

EPOCHS=(11,12,13)
WEIGHTS=(1,2,1)

def make_average(source,output):
    if output.exists():raise FileExistsError(output)
    return average_checkpoints([source/'epochs'/f'epoch_{e:03d}.pt' for e in EPOCHS],WEIGHTS,output)

def main():
    import torch
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    sources=[];averages=[]
    for fold in folds[:1] if args.preflight else folds:
        prefix='round207' if fold['name'] in ('fold0','fold3') else 'round216'
        root=Path(f"runs/{prefix}_{fold['name']}_stage2")
        out=Path('runs/round238_preflight/model.pt') if args.preflight else Path(f"runs/round238_{fold['name']}/model.pt")
        report=make_average(root,out);averages.append(report)
        ck=torch.load(out,map_location='cpu',weights_only=False)
        assert not ck['args']['train_all'] and set(ck['args']['val_rois'].split(','))==set(fold['val_rois'])
        assert ck['args']['distillation_weight']==0 and ck['args']['seed']==2026
        assert [x['epoch'] for x in ck['soup']['inputs']]==list(EPOCHS)
        sources.append({**fold,'images':140*len(fold['val_rois']),'checkpoint':str(out.resolve()),'checkpoint_sha256':digest(out)})
    if args.preflight:
        print('Strict same-trajectory average and fold metadata preflight passed',flush=True)
        return
    root=Path('runs/round238_oof_sources');root.mkdir(exist_ok=False)
    (root/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher','output_source_offset_applied':False,'images':2380,'folds':sources},indent=2)+'\n')
    output=Path('reports/round238_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(root),str(output),'--mode','models','--device','cuda:0'],check=True)
    candidate=json.loads(output.read_text())['markers']['CD45RO']
    base=json.loads(Path('reports/round217_domain_cv5_oof.json').read_text())['markers']['CD45RO']
    assert candidate['overall']['images']==2380
    gains={s:r['gaussian11_score']-base[s]['gaussian11_score'] for s,r in candidate.items()}
    positive=sum(gains[f['name']]>0 for f in folds)
    decision={'gains':gains,'positive_folds':positive,'package':gains['overall']>0 and positive>=3,
              'rule':'Positive mean and at least3 positive folds authorize one fixed platform confirmation, not promotion.'}
    Path('reports/round238_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(decision),flush=True)
    if not decision['package']:return
    model=Path('runs/round238_full/model.pt')
    fullreport=make_average(Path('runs/round216_full_stage2'),model)
    ck=torch.load(model,map_location='cpu',weights_only=False)
    assert ck['args']['train_all'] and ck['args']['distillation_weight']==.2 and ck['args']['distillation_marker_index']==1
    pred='predictions/round238_cd45ro_q95';package='submissions/semifinal_round238_cd45ro_q95.zip'
    if Path(pred).exists() or Path(package).exists():raise FileExistsError('Existing package output')
    subprocess.run([sys.executable,'tools/predict_unet.py',DATASET,str(model),pred,'--marker','CD45RO','--jpeg-quality','95','--tta','d4','--context-halo','128','--output-source-offset','0,0','--batch-size','2','--device','cuda:0'],check=True)
    subprocess.run([sys.executable,'tools/package_submission.py',pred,DATASET+'/test/DAPI',package,'--marker','CD45RO'],check=True)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==560
    Path('reports/round238_artifacts.json').write_text(json.dumps({'fold_averages':averages,'full_average':fullreport,'package':package,'package_sha256':digest(Path(package))},indent=2)+'\n')

if __name__=='__main__':main()

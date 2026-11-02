"""Training/OOF-only spatial error audit; no fitting, label edits or test access."""
import io
import json
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import sobel, distance_transform_edt
from skimage.metrics import structural_similarity

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'tools')]
from ihc_data import image_names, load_gray, parse_patch_name, resolve_data_root
from run_round204_cd45_cv import DATASET


def partitions(pred, truth, dapi):
    error=pred.astype(np.float64)-truth
    gx=sobel(truth,axis=1,mode='reflect')/8
    gy=sobel(truth,axis=0,mode='reflect')/8
    edge=np.hypot(gx,gy)>=8
    # Intensity/proximity strata are diagnostic proxies, not biological labels.
    masks={'target_0_15':truth<16,'target_16_31':(truth>=16)&(truth<32),
           'target_32_63':(truth>=32)&(truth<64),'target_64_127':(truth>=64)&(truth<128),
           'target_128_255':truth>=128,'target_edge':edge,'target_nonedge':~edge}
    if (dapi>=32).any():
        dist=distance_transform_edt(dapi<32)
        masks.update({'dapi_signal':dist==0,'dapi_near_1_8':(dist>0)&(dist<=8),'dapi_far_gt8':dist>8})
    else:
        masks['dapi_absent']=np.ones_like(truth,dtype=bool)
    return {key:{'pixels':int(mask.sum()),'absolute_error_sum':float(np.abs(error[mask]).sum()),
                 'squared_error_sum':float(np.square(error[mask]).sum()),'bias_sum':float(error[mask].sum()),
                 'target_sum':float(truth[mask].sum()),'prediction_sum':float(pred[mask].sum())}
            for key,mask in masks.items()}


def jpeg(pred):
    stream=io.BytesIO()
    Image.fromarray(pred.astype(np.uint8)).convert('RGB').save(stream,format='JPEG',quality=95)
    stream.seek(0)
    with Image.open(stream) as image:return np.array(image.convert('L'),dtype=np.float64)


def metrics(pred,truth):
    mse=float(np.mean((pred-truth)**2))
    psnr=min(50.,10*np.log10(255**2/max(mse,1e-10)))
    ssim=float(structural_similarity(truth.astype(np.float64),pred.astype(np.float64),data_range=255,
             gaussian_weights=True,sigma=1.5,win_size=11,use_sample_covariance=False))
    return {'mse':mse,'ssim':ssim,'score':70*ssim+.6*max(0.,psnr)}


def finalize(sums,total_pixels,total_sse):
    return {key:{**vals,'pixel_fraction':vals['pixels']/total_pixels,
        'mae':vals['absolute_error_sum']/max(1,vals['pixels']),
        'rmse':float(np.sqrt(vals['squared_error_sum']/max(1,vals['pixels']))),
        'bias':vals['bias_sum']/max(1,vals['pixels']),
        'fraction_of_total_squared_error':vals['squared_error_sum']/max(1e-12,total_sse)} for key,vals in sums.items()}


def main():
    root=resolve_data_root(Path(DATASET))
    cache=Path('runs/round212_distill_targets')
    manifest=json.loads((cache/'manifest.json').read_text())
    assert manifest['method']=='fold-excluded-cd45ro-domain-d4-teacher'
    rois={roi for f in manifest['folds'] for roi in f['fold']['val_rois']}
    names=[n for n in image_names(root/'train'/'DAPI') if parse_patch_name(n).roi in rois]
    assert len(names)==980 and set(image_names(cache/'targets'))==set(names)
    sources=json.loads(Path('runs/round209_domain_audit_sources/manifest.json').read_text())
    for f in manifest['folds']:
        with Path(f['checkpoint']).open('rb') as handle:
            assert hashlib.file_digest(handle,'sha256').hexdigest()==f['sha256']
        match=next(x for x in sources['folds'] if x['name']==f['fold']['name'])
        assert match['checkpoint_sha256']==f['sha256'] and set(match['val_rois'])==set(f['fold']['val_rois'])
    sums=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    rows=[]; worst={}; total_pixels=0; total_sse=0.
    for name in names:
        roi=parse_patch_name(name).roi
        dapi=load_gray(root/'train'/'DAPI'/name)
        truth=load_gray(root/'train'/'CD45RO'/name).astype(np.float64)
        with Image.open(cache/'targets'/name) as im:
            assert im.format=='PNG' and im.size==(256,256)
            pred=jpeg(np.asarray(im.convert('L')))
        current=metrics(pred,truth);black=metrics(np.zeros_like(pred),truth)
        item={'name':name,'roi':roi,**current,'black_score':black['score'],
              'fraction_target_ge32':float((truth>=32).mean()),'fraction_dapi_ge32':float((dapi>=32).mean()),
              'target_mean':float(truth.mean()),'prediction_mean':float(pred.mean())}
        rows.append(item)
        stats=partitions(pred,truth,dapi)
        for scope in ('overall',roi):
            for key,values in stats.items():
                for field,value in values.items():sums[scope][key][field]+=value
        total_pixels+=truth.size;total_sse+=current['mse']*truth.size
        if roi not in worst or current['mse']>worst[roi][0]:
            worst[roi]=(current['mse'],name,dapi,truth,pred)
    reference=json.loads(Path('reports/round207_d4_oof.json').read_text())['markers']['CD45RO']['overall']['gaussian11_score']
    score=float(np.mean([r['score'] for r in rows]))
    assert abs(score-reference)<1e-5,(score,reference)
    quartiles=[]
    for indices in np.array_split(np.argsort([r['fraction_target_ge32'] for r in rows],kind='stable'),4):
        group=[rows[i] for i in indices]
        quartiles.append({key:float(np.mean([r[key] for r in group])) for key in ['score','black_score','mse','fraction_target_ge32','fraction_dapi_ge32']})
    artifact=Path('reports/round213_error_examples.jpg')
    board=Image.new('RGB',(256*4,282*len(worst)),'white');draw=ImageDraw.Draw(board)
    for row,roi in enumerate(sorted(worst)):
        _,name,dapi,truth,pred=worst[roi]
        for col,(label,array) in enumerate([('DAPI',dapi),('CD45RO target',truth),('OOF prediction',pred),('absolute error x2',np.abs(pred-truth)*2)]):
            board.paste(Image.fromarray(np.clip(array,0,255).astype('uint8')).convert('RGB'),(256*col,282*row+26))
            draw.text((256*col+4,282*row+5),f'{name} {label}',fill='black')
    board.save(artifact,quality=95)
    result={'images':len(rows),'rois':sorted(rois),'reference_q95_score':reference,'recomputed_q95_score':score,
            'black_baseline_score':float(np.mean([r['black_score'] for r in rows])),
            'target_intensity32_activity_quartiles':quartiles,
            'regions':{scope:finalize(value,total_pixels if scope=='overall' else 140*256**2,
                       total_sse if scope=='overall' else sum(r['mse']*256**2 for r in rows if r['roi']==scope)) for scope,value in sums.items()},
            'limitations':'Seven ROI development subset, fixed thresholds are intensity/proximity proxies not cell or positivity annotations. No test data, no label edits, no training or output correction. Sparse bright errors alone do not prove reweighting helps.',
            'examples':str(artifact)}
    Path('reports/round213_error_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('images','recomputed_q95_score','black_baseline_score','target_intensity32_activity_quartiles')},indent=2),flush=True)
    print(json.dumps(result['regions']['overall'],indent=2),flush=True)


if __name__=='__main__':main()

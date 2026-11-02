"""TRAIN-only correspondence evidence; no registration correction or label edits."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter
from run_round204_cd45_cv import DATASET

MARKERS=('CD45RO','CD68','HLA-DR','Vimentin')


def nmi(a,b):
    # Fixed 16-bin mutual information, symmetric entropy normalization.
    a=np.minimum(a.astype(np.int32)//16,15).ravel()
    b=np.minimum(b.astype(np.int32)//16,15).ravel()
    histogram=np.bincount(a*16+b,minlength=256).reshape(16,16).astype(float)
    histogram/=histogram.sum();pa=histogram.sum(1);pb=histogram.sum(0)
    ha=-np.sum(pa[pa>0]*np.log(pa[pa>0]));hb=-np.sum(pb[pb>0]*np.log(pb[pb>0]))
    if min(ha,hb)<1e-8:return 0.
    expected=pa[:,None]*pb[None,:];mask=histogram>0
    return float(2*np.sum(histogram[mask]*np.log(histogram[mask]/expected[mask]))/(ha+hb))


def load(path):
    with Image.open(path) as im:return np.asarray(im.convert('L'),dtype=np.float32)


def coarse(array):
    return gaussian_filter(array,3)[::4,::4]


def one_roi(roi,rois,images):
    dapi=images['DAPI'][roi];keys=sorted(dapi);result={}
    for marker in MARKERS:
        targets=images[marker][roi]
        eligible=[i for i,k in enumerate(keys) if np.std(dapi[k])>=4 and np.std(targets[k])>=4]
        if not eligible:
            result[marker]={'informative_pairs':0};continue
        matched=[nmi(dapi[keys[i]],targets[keys[i]]) for i in eligible]
        within={str(shift):[nmi(dapi[keys[i]],targets[keys[(i+shift)%len(keys)]]) for i in eligible] for shift in (7,31,73)}
        other={r:float(np.mean([nmi(dapi[keys[i]],images[marker][r][keys[i]]) for i in eligible])) for r in rois if r!=roi}
        mean=float(np.mean(matched));best_other=max(other,key=other.get)
        per_pair_control=np.mean(np.asarray(list(within.values())),axis=0)
        result[marker]={'informative_pairs':len(eligible),'same_name_nmi':mean,
            'wrong_tile_nmi_means':{s:float(np.mean(v)) for s,v in within.items()},
            'same_pair_beats_mean_wrong_tile_fraction':float(np.mean(np.asarray(matched)>per_pair_control)),
            'same_coordinate_other_roi_means':other,'strongest_other_roi':best_other,
            'strongest_other_roi_nmi':other[best_other],
            'same_roi_ranks_first':mean>=max(other.values())}
    return roi,result


def local_offsets(roi,marker,root):
    files=sorted((root/'DAPI').glob(roi+'_*.jpg'))
    eligible=[p for p in files if np.std(load(p))>=8 and np.std(load(root/marker/p.name))>=8]
    if not eligible:return {'patches':0}
    chosen=[eligible[i] for i in np.unique(np.linspace(0,len(eligible)-1,min(6,len(eligible)),dtype=int))]
    offsets=[(dy,dx) for dy in range(-8,9,2) for dx in range(-8,9,2)]
    scores=[]
    for p in chosen:
        a=gaussian_filter(load(p),2);b=gaussian_filter(load(root/marker/p.name),2)
        # Same central support at every offset; no padding-induced ranking.
        aa=a[16:-16:2,16:-16:2]
        scores.append([nmi(aa,b[16+dy:240+dy:2,16+dx:240+dx:2]) for dy,dx in offsets])
    scores=np.asarray(scores);means=scores.mean(0);zero=offsets.index((0,0));best=int(means.argmax())
    return {'patches':len(chosen),'names':[p.name for p in chosen],'zero_nmi':float(means[zero]),
            'best_sampled_offset':offsets[best],'best_nmi':float(means[best]),
            'per_patch_best_offsets':[offsets[int(row.argmax())] for row in scores],
            'best_offset_at_search_boundary':max(abs(v) for v in offsets[best])==8,
            'warning':'Different modalities need not maximize pixelwise MI at true alignment; this is not a correction estimate.'}


def main():
    root=Path(DATASET)/'train';output=Path('reports/round236_pairing_audit.json')
    if output.exists():raise FileExistsError(output)
    images={};reference=None
    for marker in ('DAPI',*MARKERS):
        files=sorted((root/marker).glob('*.jpg'));names={p.name for p in files}
        if reference is None:reference=names
        assert names==reference and len(names)==2380
        groups={}
        for p in files:
            roi,row,col=p.stem.split('_');groups.setdefault(roi,{})[(int(row),int(col))]=coarse(load(p))
        images[marker]=groups
    rois=sorted(images['DAPI']);assert len(rois)==17
    with ThreadPoolExecutor(max_workers=4) as pool:
        correspondence=dict(pool.map(lambda roi:one_roi(roi,rois,images),rois))
    summary={m:{'same_roi_ranks_first':sum(r[m].get('same_roi_ranks_first',False) for r in correspondence.values()),
        'informative_rois':sum(r[m]['informative_pairs']>0 for r in correspondence.values()),
        'mean_same_name_nmi':float(np.mean([r[m]['same_name_nmi'] for r in correspondence.values() if r[m]['informative_pairs']])),
        'mean_wrong_tile_nmi':float(np.mean([np.mean(list(r[m]['wrong_tile_nmi_means'].values())) for r in correspondence.values() if r[m]['informative_pairs']]))}
        for m in MARKERS}
    print('Coarse correspondence',json.dumps(summary),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={(roi,m):pool.submit(local_offsets,roi,m,root) for roi in rois for m in MARKERS}
        offsets={roi:{m:jobs[(roi,m)].result() for m in MARKERS} for roi in rois}
    # Fixed ROI selection; choose a central informative patch for visibility.
    examples=[]
    for roi in (rois[0],rois[len(rois)//2],rois[-1]):
        for m in MARKERS:
            candidate=offsets[roi][m].get('names',[])
            if candidate:examples.append((roi,m,candidate[len(candidate)//2]))
    board=Image.new('RGB',(768,280*len(examples)),'white');draw=ImageDraw.Draw(board)
    for i,(_,marker,name) in enumerate(examples):
        a=load(root/'DAPI'/name).astype(np.uint8);b=load(root/marker/name).astype(np.uint8)
        overlay=np.stack((a,b,np.zeros_like(a)),axis=-1)
        for j,(label,arr) in enumerate([('DAPI',a),(marker,b),('DAPI red / marker green',overlay)]):
            board.paste(Image.fromarray(arr).convert('RGB'),(j*256,i*280+24))
            draw.text((j*256+3,i*280+4),name+' '+label,fill='black')
    board.save('reports/round236_pairing_examples.jpg',quality=95)
    report={'scope':'TRAIN only; immutable labels; no parameter changes or inferred correction',
      'method':'fixed Gaussian coarse features,16-bin symmetric MI, wrong tiles and all other ROI controls',
      'summary':summary,'correspondence':correspondence,'sampled_offsets':offsets,
      'limitations':'Shared background and different stain structures limit MI; off-zero maxima do not prove misregistration. No TEST marker labels available.'}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Complete training-only correspondence audit',flush=True)


if __name__=='__main__':main()

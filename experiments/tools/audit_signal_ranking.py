"""Quantized global-OOF signal ranking; diagnostic thresholds, not cell labels."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_round204_cd45_cv import DATASET


def summarize(histogram):
    pos,neg=histogram;p=int(pos.sum());n=int(neg.sum())
    if not p or not n:return {'positive':p,'negative':n}
    auc=float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(p*n))
    tp=np.cumsum(pos[::-1]);fp=np.cumsum(neg[::-1])
    ap=float(np.sum(pos[::-1]*(tp/np.maximum(1,tp+fp)))/p)
    return {'positive':p,'negative':n,'prevalence':p/(p+n),'auc':auc,'average_precision':ap,
            'mean_at_positive':float(np.dot(pos,np.arange(256))/p),
            'mean_at_negative':float(np.dot(neg,np.arange(256))/n)}


def main():
    root=Path(DATASET);cache=Path('runs/round216_oof_targets/targets')
    manifest=json.loads((cache.parent/'manifest.json').read_text());assert manifest['marker']=='CD45RO'
    files=sorted(cache.glob('*.jpg'))
    assert len(files)==2380 and {p.name for p in files}=={p.name for p in (root/'train'/'DAPI').glob('*.jpg')}
    stats={}
    def gray(path):
        with Image.open(path) as im:return np.asarray(im.convert('L'))
    for p in files:
        roi=p.name.split('_')[0];target=gray(root/'train'/'CD45RO'/p.name)
        values={'prediction':gray(p),'dapi':gray(root/'train'/'DAPI'/p.name)}
        for cut in (64,128):
            positive=target>=cut
            for kind,x in values.items():
                histogram=stats.setdefault((roi,cut,kind),np.zeros((2,256),dtype=np.int64))
                histogram[0]+=np.bincount(x[positive],minlength=256)
                histogram[1]+=np.bincount(x[~positive],minlength=256)
    result={str(cut):{kind:summarize(sum(v for (r,c,k),v in stats.items() if c==cut and k==kind))
                       for kind in ('prediction','dapi')} for cut in (64,128)}
    result['by_roi']={roi:{str(cut):{kind:summarize(stats[(roi,cut,kind)]) for kind in ('prediction','dapi')}
                          for cut in (64,128)} for roi in sorted({k[0] for k in stats})}
    output=Path('reports/round222_signal_ranking.json')
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

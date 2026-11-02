"""TRAIN-only tile-continuity diagnostic with fixed within-ROI random reference."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from run_round204_cd45_cv import DATASET


def main():
    root=Path(DATASET)/'train';rng=np.random.default_rng(2026);result={}
    for marker in ('DAPI','CD45RO'):
        groups={}
        for p in sorted((root/marker).glob('*.jpg')):
            roi,row,col=p.stem.split('_')
            with Image.open(p) as im:array=np.asarray(im.convert('L'),dtype=np.float32)
            groups.setdefault(roi,{})[(int(row),int(col))]=array
        stats={}
        for roi,tiles in groups.items():
            keys=sorted(tiles);values={k:[] for k in ['horizontal','vertical','random_h','random_v','interior']}
            for r,c in keys:
                a=tiles[(r,c)]
                values['interior'].append(float(np.mean(np.abs(a[:,127]-a[:,128]))))
                if (r,c+1) in tiles:
                    values['horizontal'].append(float(np.mean(np.abs(a[:,-1]-tiles[(r,c+1)][:,0]))))
                    b=tiles[keys[int(rng.integers(len(keys)))]]
                    values['random_h'].append(float(np.mean(np.abs(a[:,-1]-b[:,0]))))
                if (r+1,c) in tiles:
                    values['vertical'].append(float(np.mean(np.abs(a[-1,:]-tiles[(r+1,c)][0,:]))))
                    b=tiles[keys[int(rng.integers(len(keys)))]]
                    values['random_v'].append(float(np.mean(np.abs(a[-1,:]-b[0,:]))))
            stats[roi]={k:float(np.mean(v)) for k,v in values.items()}
        result[marker]=stats
    output=Path('reports/round233_train_seams.json')
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

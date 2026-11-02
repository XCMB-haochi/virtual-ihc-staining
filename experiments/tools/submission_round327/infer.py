"""单张完整DAPI、指定marker、一次前向、固定JPEG95输出。"""
import argparse
import io
import random
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from model import SingleMarkerStainer,MARKERS
from data import read_image,tensor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--marker',choices=('CD68','CD45RO'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=SingleMarkerStainer();model.load_state_dict(ck['model'],strict=True)
    model=model.to(args.device).eval()
    names=sorted(args.input_dir.glob('*.jpg'))
    if not names:raise ValueError('No JPG inputs found')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    marker=torch.tensor([MARKERS.index(args.marker)],device=args.device)
    with torch.inference_mode():
        for path in names:
            out=args.output_dir/path.name
            if out.exists():raise FileExistsError(out)
            pred=model(tensor(read_image(path),args.device),marker)
            assert pred.shape==(1,1,256,256) and torch.isfinite(pred).all()
            pixels=np.rint(pred[0,0].float().cpu().numpy()*255).clip(0,255).astype(np.uint8)
            Image.fromarray(pixels).convert('RGB').save(out,format='JPEG',quality=95)
    print(f'Saved {len(names)} images to {args.output_dir}')


if __name__=='__main__':main()

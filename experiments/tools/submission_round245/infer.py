"""最高分推理：完整 ROI 整数平移、逆变换与16视角等权平均。
原始 DAPI 文件只读；平移副本是无损 PNG，不改变原数据。
"""
from __future__ import annotations
import argparse,hashlib,json,sys,tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import torch
CODE_ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE_ROOT/'src'))
from ihc_data import image_names,load_gray,load_context_gray,parse_patch_name
from ihc_spatial import apply_complete_roi_conditioning
from ihc_torch import build_model_from_checkpoint,checkpoint_conditioning,batch_condition_tensor,predict_with_tta,shift_array_edge,save_grayscale_rgb
from common import load_checkpoint,sha256


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
def predictions(checkpoint,source,shifted,names,device="cuda:0"):
    ck=load_checkpoint(checkpoint)
    assert ck['args']['seed']==2026
    model=build_model_from_checkpoint(ck,torch.device(device)).eval()
    marker=list(ck['markers']).index('CD45RO');spec=checkpoint_conditioning(ck)
    outputs=[]
    for directory in (source,shifted):
        conditions=apply_complete_roi_conditioning(directory,names,spec)
        values={}
        for i in range(0,len(names),2):
            batch=names[i:i+2]
            x=torch.from_numpy(np.stack([load_context_gray(directory,n,128) for n in batch])[:,None]/255.).float().to(device)
            condition=batch_condition_tensor(batch,conditions,torch.device(device))
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
    del model
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return outputs[0],result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True,help='官方 test/DAPI 文件夹，包含完整 ROI')
    parser.add_argument('--checkpoint',type=Path,default=CODE_ROOT.parent/'AIC-2026-51865112-模型文件'/'model.pt')
    parser.add_argument('--output-dir',type=Path,required=True,help='新建输出根目录，将写入 results/test/CD45RO')
    parser.add_argument('--work-dir',type=Path,required=True,help='临时输入副本存放位置，不能与原数据相同')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError('Use a new output directory to protect submitted results')
    torch.set_num_threads(4);torch.manual_seed(2026)
    names=image_names(args.input_dir)
    if not names:raise ValueError('No DAPI jpg tiles')
    args.work_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='roi_tta_',dir=args.work_dir) as folder:
        shifted=Path(folder)/'DAPI';materialize(args.input_dir,names,shifted)
        print(f'Predicting {len(names)} images with one checkpoint and16 fixed equal-weight views',flush=True)
        _,result=predictions(args.checkpoint,args.input_dir,shifted,names,args.device)
        out=args.output_dir/'results/test/CD45RO';out.mkdir(parents=True)
        for name,values in result.items():save_grayscale_rgb(values/255.,out/name,quality=95)
    records={name:sha256(out/name) for name in names}
    report={'checkpoint_sha256':sha256(args.checkpoint),'images':len(names),'marker':'CD45RO','views':16,
      'input_translation':[8,8],'inverse_translation':[-8,-8],'boundary':'replicate','jpeg_quality':95,'files':records}
    (args.output_dir/'inference_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('Completed',out,flush=True)

if __name__=='__main__':main()

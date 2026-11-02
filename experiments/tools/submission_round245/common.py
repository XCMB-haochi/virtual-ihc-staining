"""Checkpoint IO and integrity helpers; no downloads or model selection."""
import hashlib,json,contextlib,pathlib,os
from pathlib import Path
import torch

def sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_checkpoint(path):
    # Released original PyTorch checkpoint contains Linux PosixPath metadata.
    if os.name!='nt':return torch.load(path,map_location='cpu',weights_only=False)
    original=pathlib.PosixPath
    try:
        pathlib.PosixPath=pathlib.WindowsPath
        return torch.load(path,map_location='cpu',weights_only=False)
    finally:pathlib.PosixPath=original

def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

def atomic_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.writing');torch.save(value,tmp);tmp.replace(path)

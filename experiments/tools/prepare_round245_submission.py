"""Assemble the named Round245 delivery folder from verified local artifacts."""
import argparse,hashlib,shutil,zipfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
TEAM='AIC-2026-51865112-通信速通生物-复赛-团队成员'
ID='AIC-2026-51865112'
EXPECTED={'model':'16a530ac9b0b8396592668bcaad4405b697083ad09a7b2e176b1e33ed36a17b1',
'public':'98256dc1aca47cb6e553a1c82d7f343d328e14d2c7a1b1019dc34273669d1578',
'zip':'7731e7c048974a275314dfbdb2a052c705e6831fb5321bdf9141c372f3db1d7c'}

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=REPO/'deliverables'/TEAM);args=p.parse_args()
 root=args.output;modeldir=root/(ID+'-模型文件');code=root/(ID+'-代码与数据');zipfile_path=REPO/'submissions/semifinal_round245_cd45ro_q95.zip'
 for path,expected in [(modeldir/'model.pt',EXPECTED['model']),(code/'assets/public_encoder.pt',EXPECTED['public']),(zipfile_path,EXPECTED['zip'])]:
  if sha(path)!=expected:raise ValueError(f'Hash mismatch: {path}')
 shutil.copytree(REPO/'tools/submission_round245',code,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 with zipfile.ZipFile(zipfile_path) as z:
  names=z.namelist();assert len(names)==560 and len(set(names))==560 and z.testzip() is None
  for n in names:
   if not n.startswith('results/test/CD45RO/') or '..' in Path(n).parts:raise ValueError(n)
  outer='AIC-2026-51865112-赛题结果文档'
  with zipfile.ZipFile(root/'results.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as packed:
   for n in names:packed.writestr(outer+'/results/test/cd45ro/'+Path(n).name,z.read(n))
  with zipfile.ZipFile(root/'results.zip') as packed:
   assert packed.testzip() is None and len(packed.namelist())==560
   assert all(packed.read(outer+'/results/test/cd45ro/'+Path(n).name)==z.read(n) for n in names)
 (root/'提交说明.md').write_text(f"""# 复赛提交文件

- 参赛编号：{ID}
- 团队名称：通信速通生物
- 队长：团队成员
- 最高成绩：79.1052（SSIM0.899，PSNR26.932）
- 输出Marker：CD45RO

```text
{TEAM}/
├── results.zip                           内部为 AIC-2026-51865112-赛题结果文档/results/test/cd45ro/，共560张结果图
├── {ID}-模型文件/model.pt   最终模型
├── {ID}-代码与数据/          代码、训练初始化和执行说明
└── 提交说明.md
```

执行命令见 `{ID}-代码与数据/README.md`。结果图对应最高分提交ZIP的原文件。
技术报告见 `{ID}-技术方案.pdf`。
""",encoding='utf-8')
 (modeldir/'README.md').write_text(f"""# 模型文件

文件：`model.pt`。模型参数量：31,753,556。输入：DAPI。
输出顺序：CD68、CD45RO、HLA-DR、Vimentin；提交使用CD45RO。
权重节点：全量学生第二阶段epoch12。
SHA256：`{EXPECTED['model']}`。

推理入口：`../{ID}-代码与数据/infer.py`，命令见代码目录README。
""",encoding='utf-8')
 files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
 print(root);print('Files:',len(files),'bytes:',sum(p.stat().st_size for p in files))

if __name__=='__main__':main()

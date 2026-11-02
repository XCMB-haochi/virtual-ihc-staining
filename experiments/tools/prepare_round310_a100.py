"""Detached data-ready handoff, smoke checks, and resource-aware A100 launch."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent
EXPECTED='4acc45b53e5fc98d271599384a9a08f0a1abb926df04f36f7cb69e4f41d11d75'


def state(phase,**kwargs):
    target=ROOT/'reports/round310_pipeline_status.json'
    temporary=target.with_suffix('.writing')
    temporary.write_text(json.dumps({'phase':phase,**kwargs},indent=2),encoding='utf-8'); temporary.replace(target)


def resources():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(a):{'free':int(b),'util':int(c)} for a,b,c in (line.split(',') for line in output.strip().splitlines())}


def arm(name,gpu,smoke=False):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    label=name+('_smoke' if smoke else '')
    with (ROOT/f'reports/round310_{label}.log').open('w') as out, (ROOT/f'reports/round310_{label}.err').open('w') as err:
        result=subprocess.run([sys.executable,'-u','tools/round310.py','--arm',name]+(['--smoke'] if smoke else []),
                              cwd=ROOT,env=env,stdout=out,stderr=err)
    (ROOT/f'reports/round310_{label}.exit').write_text(str(result.returncode))
    if result.returncode: raise RuntimeError(f'{label} failed; inspect its log')


def main():
    os.chdir(ROOT); state('waiting-for-data')
    deadline=time.monotonic()+3*3600
    while not (PARENT/'round310_data.ready').exists():
        if time.monotonic()>deadline: raise TimeoutError('Input transfer exceeded3h')
        time.sleep(10)
    state('verifying-data')
    archive=PARENT/'round310_data.tar'
    with archive.open('rb') as handle: assert hashlib.file_digest(handle,'sha256').hexdigest()==EXPECTED
    with tarfile.open(archive) as tar: tar.extractall(ROOT,filter='data')
    subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_round310.py','-v'],check=True)
    state('waiting-for-gpu1')
    while True:
        info=resources()
        if info[1]['free']>=8192 and info[1]['util']<=20: break
        if time.monotonic()>deadline: raise TimeoutError('GPU1 unavailable within staging deadline')
        time.sleep(30)
    state('smoke')
    arm('control',1,True); arm('gain',1,True)
    samples=[]
    for _ in range(3): samples.append(resources()); time.sleep(5)
    dual=all(s[0]['free']>=8192 and s[0]['util']<=20 and s[1]['free']>=8192 and s[1]['util']<=20 for s in samples)
    state('training',schedule='GPU0 control + GPU1 gain' if dual else 'GPU1 control then gain')
    if dual:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(arm,'control',0),pool.submit(arm,'gain',1)]
            for job in jobs: job.result()
    else:
        arm('control',1); arm('gain',1)
    state('complete',comparison='reports/round311_comparison.json')


if __name__=='__main__':
    try: main()
    except Exception as error:
        state('failed',error=repr(error)); raise

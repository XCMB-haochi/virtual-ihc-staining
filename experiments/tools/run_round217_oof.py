"""Complete five-fold domain-teacher evidence; no student validation claim."""
import json
import subprocess
import sys
from pathlib import Path
from run_round204_cd45_cv import DATASET

def main():
    source=json.loads(Path('runs/round216_oof_targets/manifest.json').read_text())
    assert source['marker']=='CD45RO' and source['images']==2380
    folds=[{'name':f['name'],'val_rois':f['val_rois'],'images':f['images'],
            'checkpoint':f['checkpoint'],'checkpoint_sha256':f['sha256']} for f in source['folds']]
    assert len(folds)==5 and sum(f['images'] for f in folds)==2380
    directory=Path('runs/round217_oof_sources');directory.mkdir(exist_ok=False)
    (directory/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher',
        'output_source_offset_applied':False,'images':2380,'folds':folds},indent=2)+'\n')
    output=Path('reports/round217_domain_cv5_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(directory),str(output),
                    '--mode','models','--device','cuda:0'],check=True)
    current=json.loads(output.read_text())['markers']
    baseline=json.loads(Path('reports/round205_round204_d4_oof.json').read_text())['markers']
    gains={m:{scope:value['gaussian11_score']-baseline[m][scope]['gaussian11_score']
              for scope,value in scopes.items()} for m,scopes in current.items()}
    Path('reports/round217_domain_cv5_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('CD45RO full-fold domain changes',json.dumps(gains['CD45RO']),flush=True)

if __name__=='__main__':main()

"""Expand fixed texture conditioning to the three remaining folds on GPU0."""
import json
import math
import subprocess
import sys
from pathlib import Path
from run_round219_texture import command as screen_command
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest, run_logged


def command(fold,stage):
    cmd=screen_command(fold,stage)
    cmd[cmd.index('--output-dir')+1]=f"runs/round220_{fold['name']}_stage{stage}"
    if stage==1:
        cmd[cmd.index('--fcmae-encoder-init-checkpoint')+1]=f"runs/round216_{fold['name']}_fcmae800/final_encoder.pt"
    else:
        cmd[cmd.index('--init-checkpoint')+1]=f"runs/round220_{fold['name']}_stage1/epochs/epoch_014.pt"
    return cmd


def main():
    import torch
    torch.set_num_threads(4)
    folds=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
    for fold in folds:
        if fold['name'] in ('fold0','fold3'):continue
        encoder=Path(f"runs/round216_{fold['name']}_fcmae800/final_encoder.pt")
        p=torch.load(encoder,map_location='cpu',weights_only=False)
        assert p['epoch']==800 and set(p['val_rois'])==set(fold['val_rois'])
        baseline=json.loads(Path(f"runs/round216_{fold['name']}_stage1/config.json").read_text())
        assert Path(baseline['fcmae_encoder_init_checkpoint']).resolve()==encoder.resolve()
        for stage in (1,2):
            cmd=command(fold,stage);root=Path(cmd[cmd.index('--output-dir')+1])
            if root.exists():raise FileExistsError(root)
            print('Training',fold['name'],stage,flush=True)
            run_logged(cmd,f"reports/round220_{fold['name']}_stage{stage}.log")
    sources=[]
    for fold in folds:
        prefix='round219' if fold['name'] in ('fold0','fold3') else 'round220'
        for stage in (1,2):
            root=Path(f"runs/{prefix}_{fold['name']}_stage{stage}")
            rows=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
            assert [r['epoch'] for r in rows]==list(range(1,15))
            assert all(math.isfinite(r['train_loss']) and math.isfinite(r['selection_score']) for r in rows)
        ck=root/'epochs'/'epoch_012.pt'
        p=torch.load(ck,map_location='cpu',weights_only=False)
        assert p['args']['condition_mode']=='spatial-thumbnail-texture-film'
        assert p['args']['seed']==2026 and p['args']['selection_marker']=='CD45RO'
        sources.append({**fold,'images':140*len(fold['val_rois']),
                        'checkpoint':str(ck.resolve()),'checkpoint_sha256':digest(ck)})
    directory=Path('runs/round220_oof_sources');directory.mkdir(exist_ok=False)
    (directory/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher',
       'images':2380,'output_source_offset_applied':False,'folds':sources},indent=2)+'\n')
    output=Path('reports/round220_texture_cv5_oof.json')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(directory),str(output),
                    '--mode','models','--device','cuda:0'],check=True)
    candidate=json.loads(output.read_text())['markers']
    control=json.loads(Path('reports/round217_domain_cv5_oof.json').read_text())['markers']
    gains={m:{scope:values['gaussian11_score']-control[m][scope]['gaussian11_score']
              for scope,values in scopes.items()} for m,scopes in candidate.items()}
    Path('reports/round220_texture_cv5_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('CD45RO full-fold gains',json.dumps(gains['CD45RO']),flush=True)


if __name__=='__main__':main()

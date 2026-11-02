"""Finish domain D4 audit and compare CD68-primary public/domain initialization."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from run_round204_cd45_cv import DATASET, ENCODER, ENCODER_SHA, training_command


def command_for(fold, stage, recipe):
    if recipe not in ('public', 'domain'):
        raise ValueError(recipe)
    command, _ = training_command(fold, stage, 14)
    root = f"runs/round209_{fold['name']}_{recipe}"
    command[command.index('--output-dir') + 1] = f'{root}_stage{stage}'
    command[command.index('--marker-loss-weights') + 1] = '0.85,0.05,0.05,0.05'
    command[command.index('--selection-marker') + 1] = 'CD68'
    if stage == 1 and recipe == 'domain':
        command[command.index('--fcmae-encoder-init-checkpoint') + 1] = f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt"
    if stage == 2:
        command[command.index('--init-checkpoint') + 1] = f'{root}_stage1/epochs/epoch_014.pt'
    return command


def main():
    import torch
    folds = [f for f in json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds'] if f['name'] in ('fold0', 'fold3')]
    with Path(ENCODER).open('rb') as handle:
        assert hashlib.file_digest(handle, 'sha256').hexdigest() == ENCODER_SHA
    sources = []
    encoders = {}
    for fold in folds:
        encoder = Path(f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt")
        payload = torch.load(encoder, map_location='cpu', weights_only=False)
        assert payload['epoch'] == 800 and set(payload['val_rois']) == set(fold['val_rois'])
        assert not set(payload['config']['train_rois']) & set(fold['val_rois'])
        with encoder.open('rb') as handle:
            encoders[str(encoder)] = hashlib.file_digest(handle, 'sha256').hexdigest()
        path = Path(f"runs/round207_{fold['name']}_stage2/epochs/epoch_012.pt")
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        sources.append({**fold, 'images': len(fold['val_rois'])*140, 'checkpoint': str(path.resolve()), 'checkpoint_sha256': digest})
        for recipe in ('public', 'domain'):
            for stage in (1, 2):
                command = command_for(fold, stage, recipe)
                if Path(command[command.index('--output-dir')+1]).exists():
                    raise FileExistsError('Refusing existing trajectory')
    root = Path('runs/round209_domain_audit_sources')
    root.mkdir(exist_ok=False)
    (root/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher', 'output_source_offset_applied':False, 'images':980,'folds':sources},indent=2))
    Path('reports/round209_encoder_hashes.json').write_text(json.dumps(encoders,indent=2)+'\n')
    subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(root),'reports/round207_d4_oof.json','--mode','models','--subset-rois',','.join(r for f in folds for r in f['val_rois']),'--device','cuda:0'],check=True)
    current=json.loads(Path('reports/round207_d4_oof.json').read_text())['markers']
    baseline=json.loads(Path('reports/round205_round204_d4_oof.json').read_text())['markers']
    gains={m:{f['name']:current[m][f['name']]['gaussian11_score']-baseline[m][f['name']]['gaussian11_score'] for f in folds} for m in current}
    Path('reports/round207_d4_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('Round207 D4 changes',json.dumps(gains),flush=True)
    for fold in folds:
        for recipe in ('public','domain'):
            for stage in (1,2):
                print('Training',fold['name'],recipe,stage,flush=True)
                with Path(f"reports/round209_{fold['name']}_{recipe}_stage{stage}.log").open('w') as log:
                    subprocess.run(command_for(fold,stage,recipe),stdout=log,stderr=subprocess.STDOUT,check=True)
    result=[]
    for fold in folds:
        scores={}
        for recipe in ('public','domain'):
            rows=[json.loads(l) for l in Path(f"runs/round209_{fold['name']}_{recipe}_stage2/metrics.jsonl").read_text().splitlines()]
            assert [r['epoch'] for r in rows]==list(range(1,15))
            scores[recipe]={m:rows[11]['validation']['markers'][m]['competition_score_mean_image_psnr'] for m in ('CD68','CD45RO')}
        result.append({'fold':fold['name'],'scores':scores,'domain_minus_public':{m:scores['domain'][m]-scores['public'][m] for m in scores['public']}})
    Path('reports/round209_training_screen.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()

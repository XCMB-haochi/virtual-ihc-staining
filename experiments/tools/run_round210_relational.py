"""Fold-local multimarker teacher and batch-matched CD45RO relational screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from run_round204_cd45_cv import DATASET, training_command
from run_round207_domain import pretrain_command

MODALITIES = 'DAPI,CD68,CD45RO,HLA-DR,Vimentin'


def teacher_command(fold, smoke=False):
    command, _ = pretrain_command(fold, smoke)
    output = Path('runs/round210_smoke_teacher' if smoke else f"runs/round210_{fold['name']}_teacher160")
    command[command.index('--output-dir') + 1] = output.as_posix()
    command[command.index('--pretraining-markers') + 1] = MODALITIES
    command[command.index('--epochs') + 1] = '2' if smoke else '160'
    command[command.index('--warmup-epochs') + 1] = '0' if smoke else '2'
    command[command.index('--checkpoint-interval') + 1] = '1' if smoke else '10'
    return command, output


def student_command(fold, stage, recipe, teacher=None, teacher_sha=None, smoke=False):
    if recipe not in ('control', 'relational'):
        raise ValueError(recipe)
    command, _ = training_command(fold, stage, 14, smoke)
    prefix = 'runs/round210_smoke' if smoke else f"runs/round210_{fold['name']}_{recipe}"
    command[command.index('--output-dir') + 1] = f'{prefix}_stage{stage}'
    command[command.index('--batch-size') + 1] = '8'
    command[command.index('--gradient-accumulation-steps') + 1] = '1'
    if stage == 1:
        command[command.index('--fcmae-encoder-init-checkpoint') + 1] = f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt"
    else:
        epoch = 1 if smoke else 14
        command[command.index('--init-checkpoint') + 1] = f'{prefix}_stage1/epochs/epoch_{epoch:03d}.pt'
    if recipe == 'relational':
        if not teacher or not teacher_sha:
            raise ValueError('Relational training requires a validated teacher')
        command += ['--feature-guidance-checkpoint', str(teacher),
                    '--feature-guidance-checkpoint-sha256', teacher_sha,
                    '--feature-guidance-mode', 'encoder-relational-robust',
                    '--feature-guidance-marker', 'CD45RO', '--feature-guidance-weight', '0.05',
                    '--feature-guidance-stage', '4', '--feature-guidance-crop-size', '256',
                    '--feature-guidance-batch-samples', '8']
    return command


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify_teacher(path, fold, smoke=False):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    assert payload['epoch'] == (2 if smoke else 160)
    assert set(payload['val_rois']) == set(fold['val_rois'])
    config = payload['config']
    assert config['data_scope'] == 'fold-train' and config['seed'] == 2026
    assert config['pretraining_markers'] == MODALITIES.split(',')
    assert not set(config['train_rois']) & set(fold['val_rois'])
    return digest(path)


def run_logged(command, path):
    with Path(path).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def paired_d4(folds, round_name='round210'):
    reports = {}
    for recipe in ('control', 'relational'):
        root = Path(f'runs/{round_name}_{recipe}_oof_sources')
        root.mkdir(exist_ok=False)
        sources = []
        for fold in folds:
            path = Path(f"runs/{round_name}_{fold['name']}_{recipe}_stage2/epochs/epoch_012.pt")
            sources.append({**fold, 'images': len(fold['val_rois'])*140,
                            'checkpoint': str(path.resolve()), 'checkpoint_sha256': digest(path)})
        (root/'manifest.json').write_text(json.dumps({
            'method': 'fold-excluded-single-recipe-d4-teacher', 'output_source_offset_applied': False,
            'images': 980, 'folds': sources}, indent=2) + '\n')
        output = Path(f'reports/{round_name}_{recipe}_d4_oof.json')
        subprocess.run([sys.executable, 'tools/audit_semifinal_markers.py', DATASET, str(root), str(output),
                        '--mode', 'models', '--subset-rois', ','.join(r for f in folds for r in f['val_rois']),
                        '--device', 'cuda:0'], check=True)
        reports[recipe] = json.loads(output.read_text())['markers']
    prior = json.loads(Path('reports/round207_d4_oof.json').read_text())['markers']
    result = {}
    for marker in reports['control']:
        result[marker] = {scope: {
            'relational_minus_control': row['gaussian11_score']-reports['control'][marker][scope]['gaussian11_score'],
            'control_minus_prior': reports['control'][marker][scope]['gaussian11_score']-prior[marker][scope]['gaussian11_score'],
            'relational_minus_prior': row['gaussian11_score']-prior[marker][scope]['gaussian11_score']}
            for scope, row in reports['relational'][marker].items()}
    Path(f'reports/{round_name}_paired_gains.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Matched relational gain', json.dumps(result['CD45RO']['overall']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    folds = [f for f in json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds'] if f['name'] in ('fold0','fold3')]
    receipts = {}
    for fold in (folds[:1] if args.smoke else folds):
        init = Path(f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt")
        expected = json.loads(Path('reports/round209_encoder_hashes.json').read_text())[str(init)]
        if digest(init) != expected:
            raise ValueError('DAPI initialization changed')
        teacher_cmd, output = teacher_command(fold, args.smoke)
        if output.exists():
            raise FileExistsError(output)
        for recipe in (('relational',) if args.smoke else ('control','relational')):
            for stage in (1,2):
                cmd = student_command(fold, stage, recipe, 'preflight.pt', '0'*64, args.smoke)
                if Path(cmd[cmd.index('--output-dir')+1]).exists():
                    raise FileExistsError('Refusing existing supervised trajectory')
        print('Training multimarker teacher', fold['name'], flush=True)
        run_logged(teacher_cmd, f"reports/round210_{'smoke' if args.smoke else fold['name']}_teacher.log")
        teacher = output/'final_encoder.pt'
        teacher_sha = verify_teacher(teacher, fold, args.smoke)
        receipts[fold['name']] = {'teacher_sha256': teacher_sha, 'dapi_encoder_sha256': expected}
        for recipe in (('relational',) if args.smoke else ('control','relational')):
            for stage in (1,2):
                print('Training student', fold['name'], recipe, stage, flush=True)
                command = student_command(fold, stage, recipe, teacher, teacher_sha, args.smoke)
                run_logged(command, f"reports/round210_{'smoke' if args.smoke else fold['name']}_{recipe}_stage{stage}.log")
                rows = [json.loads(l) for l in (Path(command[command.index('--output-dir')+1])/'metrics.jsonl').read_text().splitlines()]
                assert len(rows) == (1 if args.smoke else 14)
                assert all(math.isfinite(r['train_loss']) and math.isfinite(r['train_feature_guidance_loss']) for r in rows)
                if recipe == 'relational':
                    assert any(r['train_feature_guidance_loss'] > 0 for r in rows)
                if args.smoke:
                    import torch
                    root = Path(command[command.index('--output-dir')+1])
                    ck = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
                    steps = [float(s['step']) for s in ck['optimizer']['state'].values() if 'step' in s]
                    assert steps and min(steps)>0
                    assert ck['args']['feature_guidance_marker_index'] == 1
                    assert not any('teacher' in k for k in ck['model'])
                    print('Smoke', stage, 'actual updates', min(steps), 'peak GiB', rows[-1]['peak_gpu_gib'],
                          'guidance loss', rows[-1]['train_feature_guidance_loss'], flush=True)
    if not args.smoke:
        Path('reports/round210_source_hashes.json').write_text(json.dumps(receipts, indent=2)+'\n')
        paired_d4(folds)


if __name__ == '__main__':
    main()

"""Full-data domain FCMAE800 followed by one CD45RO-primary student."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
from run_round204_cd45_cv import DATASET, ENCODER, ENCODER_SHA, training_command

def pretrain_command():
    return [sys.executable, 'tools/train_dapi_fcmae.py', DATASET,
        '--output-dir', 'runs/round208_full_fcmae800', '--val-rois', '',
        '--data-scope', 'all-train', '--pretraining-markers', 'DAPI',
        '--epochs', '800', '--batch-size', '8', '--gradient-accumulation-steps', '4',
        '--workers', '0', '--model-size', 'tiny', '--learning-rate', '0.0002',
        '--minimum-learning-rate', '0.000001', '--warmup-epochs', '10',
        '--weight-decay', '0.05', '--mask-ratio', '0.5', '--checkpoint-interval', '25',
        '--seed', '2026', '--device', 'cuda:0', '--encoder-init-checkpoint', ENCODER,
        '--encoder-init-mode', 'packaged-public-encoder', '--encoder-init-sha256', ENCODER_SHA]

def supervised_command(stage):
    command, _ = training_command({'name': 'fold0', 'val_rois': []}, stage, 14)
    command[command.index('--output-dir') + 1] = f'runs/round208_full_stage{stage}'
    command[command.index('--val-rois') + 1] = ''
    command.append('--train-all')
    if stage == 1:
        command[command.index('--fcmae-encoder-init-checkpoint') + 1] = 'runs/round208_full_fcmae800/final_encoder.pt'
    else:
        command[command.index('--init-checkpoint') + 1] = 'runs/round208_full_stage1/epochs/epoch_014.pt'
    return command

def main():
    import torch
    for path in ('runs/round208_full_fcmae800', 'runs/round208_full_stage1', 'runs/round208_full_stage2',
                 'predictions/round208_cd45ro_q95', 'submissions/semifinal_round208_cd45ro_q95.zip'):
        if Path(path).exists(): raise FileExistsError(path)
    with Path(ENCODER).open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != ENCODER_SHA: raise ValueError('Encoder hash changed')
    with Path('reports/round208_full_fcmae.log').open('w') as log:
        subprocess.run(pretrain_command(), stdout=log, stderr=subprocess.STDOUT, check=True)
    enc = torch.load('runs/round208_full_fcmae800/final_encoder.pt', map_location='cpu', weights_only=False)
    assert enc['epoch'] == 800 and enc['val_rois'] == [] and enc['config']['data_scope'] == 'all-train'
    for stage in (1, 2):
        with Path(f'reports/round208_full_stage{stage}.log').open('w') as log:
            subprocess.run(supervised_command(stage), stdout=log, stderr=subprocess.STDOUT, check=True)
    checkpoint = 'runs/round208_full_stage2/epochs/epoch_012.pt'
    subprocess.run([sys.executable, 'tools/predict_unet.py', DATASET, checkpoint,
        'predictions/round208_cd45ro_q95', '--marker', 'CD45RO', '--jpeg-quality', '95',
        '--tta', 'd4', '--context-halo', '128', '--output-source-offset', '0,0',
        '--batch-size', '2', '--device', 'cuda:0'], check=True)
    package = 'submissions/semifinal_round208_cd45ro_q95.zip'
    subprocess.run([sys.executable, 'tools/package_submission.py', 'predictions/round208_cd45ro_q95',
        f'{DATASET}/test/DAPI', package, '--marker', 'CD45RO'], check=True)
    subprocess.run([sys.executable, '-m', 'zipfile', '-t', package], check=True)
    hashes = {}
    for path in (checkpoint, package):
        with Path(path).open('rb') as f: hashes[path] = hashlib.file_digest(f, 'sha256').hexdigest()
    Path('reports/round208_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')

if __name__ == '__main__': main()

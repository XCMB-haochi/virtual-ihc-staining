"""Fold-excluded 800-epoch DAPI domain pretraining and matched supervision."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from run_round204_cd45_cv import DATASET, ENCODER, ENCODER_SHA, training_command


def pretrain_command(fold, smoke=False):
    output = 'runs/round207_smoke' if smoke else f"runs/round207_{fold['name']}_fcmae800"
    cmd = [sys.executable, 'tools/train_dapi_fcmae.py', DATASET,
           '--output-dir', output, '--val-rois', ','.join(fold['val_rois']),
           '--data-scope', 'fold-train', '--pretraining-markers', 'DAPI',
           '--epochs', '2' if smoke else '800', '--batch-size', '8',
           '--gradient-accumulation-steps', '4', '--workers', '0', '--model-size', 'tiny',
           '--learning-rate', '0.0002', '--minimum-learning-rate', '0.000001',
           '--warmup-epochs', '0' if smoke else '10', '--weight-decay', '0.05',
           '--mask-ratio', '0.5', '--checkpoint-interval', '1' if smoke else '25',
           '--seed', '2026', '--device', 'cuda:0', '--encoder-init-checkpoint', ENCODER,
           '--encoder-init-mode', 'packaged-public-encoder', '--encoder-init-sha256', ENCODER_SHA]
    if smoke:
        cmd += ['--max-train-images', '256']
    return cmd, Path(output)


def audit206(folds):
    folder = Path('runs/round206_d4_sources')
    folder.mkdir(exist_ok=False)
    sources = []
    for fold in folds:
        path = Path(f"runs/round206_{fold['name']}_stage2/epochs/epoch_012.pt")
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        sources.append({**fold, 'checkpoint': str(path.resolve()), 'checkpoint_sha256': digest,
                        'images': 140 * len(fold['val_rois'])})
    manifest = {'method': 'fold-excluded-single-recipe-d4-teacher',
                'output_source_offset_applied': False, 'images': 980, 'folds': sources}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    output = 'reports/round206_d4_oof.json'
    rois = ','.join(r for f in folds for r in f['val_rois'])
    subprocess.run([sys.executable, 'tools/audit_semifinal_markers.py', DATASET, str(folder), output,
                    '--mode', 'models', '--subset-rois', rois, '--device', 'cuda:0'], check=True)
    current = json.loads(Path(output).read_text())['markers']
    baseline = json.loads(Path('reports/round205_round204_d4_oof.json').read_text())['markers']
    gains = {m: {f['name']: current[m][f['name']]['gaussian11_score'] - baseline[m][f['name']]['gaussian11_score'] for f in folds} for m in current}
    Path('reports/round206_d4_gains.json').write_text(json.dumps(gains, indent=2) + '\n')
    print('Round206 D4 deltas', json.dumps(gains), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    folds = [f for f in json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds'] if f['name'] in ('fold0', 'fold3')]
    if args.smoke:
        cmd, output = pretrain_command(folds[0], True)
        subprocess.run(cmd, check=True)
        import torch
        p = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        steps = [float(s['step']) for s in p['optimizer']['state'].values() if 'step' in s]
        assert steps and min(steps) > 0
        final = torch.load(output / 'final_encoder.pt', map_location='cpu', weights_only=False)
        assert set(final['val_rois']) == set(folds[0]['val_rois'])
        print('Verified optimizer updates and excluded ROIs', min(steps), final['val_rois'], flush=True)
        return
    audit206(folds)
    for fold in folds:
        command, output = pretrain_command(fold)
        print('Starting domain FCMAE800', fold['name'], flush=True)
        with Path(f"reports/round207_{fold['name']}_fcmae.log").open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        for stage in (1, 2):
            command, _ = training_command(fold, stage, 14)
            command[command.index('--output-dir') + 1] = f"runs/round207_{fold['name']}_stage{stage}"
            if stage == 1:
                command[command.index('--fcmae-encoder-init-checkpoint') + 1] = str(output / 'final_encoder.pt')
            else:
                command[command.index('--init-checkpoint') + 1] = f"runs/round207_{fold['name']}_stage1/epochs/epoch_014.pt"
            if Path(command[command.index('--output-dir') + 1]).exists():
                raise FileExistsError('Refusing existing supervised trajectory')
            print('Starting supervision', fold['name'], stage, flush=True)
            with Path(f"reports/round207_{fold['name']}_stage{stage}.log").open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Domain pretraining and matched two-fold supervision complete; review fixed epoch12.', flush=True)


if __name__ == '__main__':
    main()

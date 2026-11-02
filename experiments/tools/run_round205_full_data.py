"""Fixed Round204 D4 confirmation followed by one full-data student if positive."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from run_round204_cd45_cv import DATASET, ENCODER, ENCODER_SHA, training_command


def full_command(stage):
    command, _ = training_command({'name': 'fold0', 'val_rois': []}, stage, 14)
    command[command.index('--output-dir') + 1] = f'runs/round205_full_stage{stage}'
    command.append('--train-all')
    if stage == 2:
        command[command.index('--init-checkpoint') + 1] = 'runs/round205_full_stage1/epochs/epoch_014.pt'
    # Keep T_max=14 even though the selected stage2 endpoint is epoch12.
    return command


def main():
    import torch
    with Path(ENCODER).open('rb') as handle:
        assert hashlib.file_digest(handle, 'sha256').hexdigest() == ENCODER_SHA
    clocks = [json.loads(Path(f'reports/round204_stage{s}_clock.json').read_text())['selected_epoch'] for s in (1, 2)]
    if clocks != [14, 12]:
        raise ValueError(f'Unexpected selection: {clocks}')
    manifest_dir = Path('runs/round205_oof_sources')
    manifest_dir.mkdir(exist_ok=False)
    protocol = json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())
    folds = []
    for fold in protocol['folds']:
        path = Path(f"runs/round204_{fold['name']}_stage2/epochs/epoch_012.pt")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 12
        assert checkpoint['args']['seed'] == 2026
        assert checkpoint['args']['selection_marker'] == 'CD45RO'
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        folds.append({**fold, 'images': len(fold['val_rois']) * 140,
                      'checkpoint': str(path.resolve()), 'checkpoint_sha256': digest})
    # Manifest selects fold-excluded model inference; no cached targets are created.
    manifest = {'method': 'fold-excluded-single-recipe-d4-teacher',
                'output_source_offset_applied': False, 'images': 2380, 'folds': folds}
    (manifest_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    oof = Path('reports/round205_round204_d4_oof.json')
    subprocess.run([sys.executable, 'tools/audit_semifinal_markers.py', DATASET,
                    str(manifest_dir), str(oof), '--mode', 'models', '--device', 'cuda:0'], check=True)
    current = json.loads(oof.read_text())['markers']['CD45RO']
    baseline = json.loads(Path('reports/round202_all_marker_metrics.json').read_text())['markers']['CD45RO']
    gains = {scope: row['gaussian11_score'] - baseline[scope]['gaussian11_score'] for scope, row in current.items()}
    # A zero mean gain is the minimum reason to spend a full-data training run.
    # ROI/fold declines are reported, not used as a Colon-style automatic veto.
    proceed = gains['overall'] > 0
    decision = {'mean_gain': gains['overall'], 'paired_gains': gains,
                'full_training_allowed': proceed,
                'rule': 'Positive matched mean score permits one full-data candidate; this is not statistical significance or automatic promotion.'}
    Path('reports/round205_decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps(decision), flush=True)
    if not proceed:
        print('No mean improvement; stop before full-data training for direction review.', flush=True)
        return
    for stage in (1, 2):
        if Path(f'runs/round205_full_stage{stage}').exists():
            raise FileExistsError('Refusing existing full-data run')
    for stage in (1, 2):
        print(f'Starting full-data stage{stage}', flush=True)
        with Path(f'reports/round205_full_stage{stage}.log').open('w') as log:
            subprocess.run(full_command(stage), stdout=log, stderr=subprocess.STDOUT, check=True)
    checkpoint = 'runs/round205_full_stage2/epochs/epoch_012.pt'
    pred = 'predictions/round205_cd45ro_q95'
    package = 'submissions/semifinal_round205_cd45ro_q95.zip'
    if Path(pred).exists() or Path(package).exists():
        raise FileExistsError('Refusing existing submission artifacts')
    subprocess.run([sys.executable, 'tools/predict_unet.py', DATASET, checkpoint, pred,
                    '--marker', 'CD45RO', '--jpeg-quality', '95', '--tta', 'd4',
                    '--context-halo', '128', '--output-source-offset', '0,0',
                    '--batch-size', '2', '--device', 'cuda:0'], check=True)
    subprocess.run([sys.executable, 'tools/package_submission.py', pred, f'{DATASET}/test/DAPI',
                    package, '--marker', 'CD45RO'], check=True)
    subprocess.run([sys.executable, '-m', 'zipfile', '-t', package], check=True)
    hashes = {}
    for file in (checkpoint, package):
        with Path(file).open('rb') as handle:
            hashes[file] = hashlib.file_digest(handle, 'sha256').hexdigest()
    Path('reports/round205_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')


if __name__ == '__main__':
    main()

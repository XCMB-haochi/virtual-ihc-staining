"""Matched two-fold Gaussian loss screen; no checkpoint or loss-weight search."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from run_round204_cd45_cv import ENCODER, ENCODER_SHA, training_command


def command_for(fold, stage, smoke=False):
    command, _ = training_command(fold, stage, 14, smoke)
    output = Path('runs/round206_smoke' if smoke else f"runs/round206_{fold['name']}_stage{stage}")
    command[command.index('--output-dir') + 1] = output.as_posix()
    command[command.index('--ssim-mode') + 1] = 'gaussian11'
    if stage == 2:
        command[command.index('--init-checkpoint') + 1] = f"runs/round206_{fold['name']}_stage1/epochs/epoch_014.pt"
    return command, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    with Path(ENCODER).open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == ENCODER_SHA
    folds = [f for f in json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds'] if f['name'] in ('fold0', 'fold3')]
    if args.smoke:
        command, output = command_for(folds[0], 1, True)
        if output.exists():
            raise FileExistsError(output)
        subprocess.run(command, check=True)
        import torch
        checkpoint = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
        steps = [float(s['step']) for s in checkpoint['optimizer']['state'].values() if 'step' in s]
        assert steps and min(steps) > 0
        print('Verified actual optimizer updates', min(steps), flush=True)
        return
    for fold in folds:
        for stage in (1, 2):
            _, output = command_for(fold, stage)
            if output.exists():
                raise FileExistsError(output)
    result = []
    for fold in folds:
        for stage in (1, 2):
            command, output = command_for(fold, stage)
            print(f"Starting {fold['name']} stage{stage}", flush=True)
            with Path(f"reports/round206_{fold['name']}_stage{stage}.log").open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(l) for l in (output / 'metrics.jsonl').read_text().splitlines()]
        baseline = [json.loads(l) for l in Path(f"runs/round204_{fold['name']}_stage2/metrics.jsonl").read_text().splitlines()]
        def score(row):
            return row['validation']['markers']['CD45RO']['competition_score_mean_image_psnr']
        assert [r['epoch'] for r in rows] == list(range(1, 15))
        result.append({'fold': fold['name'], 'images': rows[11]['validation']['count'],
                       'epoch12_gain': score(rows[11]) - score(baseline[11]),
                       'late_gains': [score(rows[i])-score(baseline[i]) for i in range(10, 14)]})
    Path('reports/round206_training_screen.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    print('Training screen complete; review matched D4 OOF before expansion. No test package.', flush=True)


if __name__ == '__main__':
    main()

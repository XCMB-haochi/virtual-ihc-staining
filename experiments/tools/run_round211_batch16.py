"""Restore full batch16 relations with matched control and reused teachers."""
import argparse
import json
import math
from pathlib import Path

from run_round210_relational import student_command, digest, verify_teacher, run_logged, paired_d4


def command_for(fold, stage, recipe, teacher, teacher_sha, smoke=False):
    command = student_command(fold, stage, recipe, teacher, teacher_sha, smoke)
    command = [value.replace('runs/round210_', 'runs/round211_')
               if value.startswith('runs/round210_') and 'teacher160' not in value else value
               for value in command]
    command[command.index('--batch-size')+1] = '16'
    if recipe == 'relational':
        command[command.index('--feature-guidance-batch-samples')+1] = '16'
    return command


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    folds = [f for f in json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())['folds']
             if f['name'] in ('fold0','fold3')]
    receipts = {}
    for fold in (folds[:1] if args.smoke else folds):
        teacher = Path(f"runs/round210_{fold['name']}_teacher160/final_encoder.pt")
        teacher_sha = verify_teacher(teacher, fold)
        if not args.smoke and Path('reports/round210_source_hashes.json').exists():
            prior = json.loads(Path('reports/round210_source_hashes.json').read_text())
            if teacher_sha != prior[fold['name']]['teacher_sha256']:
                raise ValueError('Completed teacher changed since Round210')
        encoder = Path(f"runs/round207_{fold['name']}_fcmae800/final_encoder.pt")
        expected = json.loads(Path('reports/round209_encoder_hashes.json').read_text())[str(encoder)]
        if digest(encoder) != expected:
            raise ValueError('DAPI encoder changed')
        receipts[fold['name']] = {'teacher_sha256':teacher_sha,'dapi_encoder_sha256':expected}
        for recipe in (('relational',) if args.smoke else ('control','relational')):
            for stage in (1,2):
                command = command_for(fold, stage, recipe, teacher, teacher_sha, args.smoke)
                root = Path(command[command.index('--output-dir')+1])
                if root.exists():
                    raise FileExistsError(root)
                print('Training batch16',fold['name'],recipe,stage,flush=True)
                run_logged(command,f"reports/round211_{'smoke' if args.smoke else fold['name']}_{recipe}_stage{stage}.log")
                rows = [json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()]
                assert len(rows)==(1 if args.smoke else 14)
                assert all(math.isfinite(r['train_loss']) and math.isfinite(r['train_feature_guidance_loss']) for r in rows)
                if recipe=='relational':
                    assert any(r['train_feature_guidance_loss']>0 for r in rows)
                if args.smoke:
                    ck=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
                    steps=[float(s['step']) for s in ck['optimizer']['state'].values() if 'step' in s]
                    assert steps and min(steps)>0
                    assert ck['args']['batch_size']==16 and ck['args']['gradient_accumulation_steps']==1
                    assert ck['args']['feature_guidance_batch_samples']==16
                    print('Smoke verified',stage,'updates',min(steps),'peak GiB',rows[-1]['peak_gpu_gib'],flush=True)
    if not args.smoke:
        Path('reports/round211_source_hashes.json').write_text(json.dumps(receipts,indent=2)+'\n')
        paired_d4(folds, 'round211')


if __name__=='__main__':
    main()

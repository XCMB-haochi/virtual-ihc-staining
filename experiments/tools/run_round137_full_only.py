"""Single-GPU Colon full-data paired comparison, without fold training/distillation."""
import argparse
import json
import os
from pathlib import Path

import run_round137_colon as original


def student_command(arm, stage, teacher_sha):
    cmd = original.student_command('full', arm, stage, teacher_sha)
    # This comparison has no newly trained OOF teachers or distillation targets.
    for flag in ('--distillation-target-dir', '--distillation-marker', '--distillation-weight'):
        if flag in cmd:
            index = cmd.index(flag)
            del cmd[index:index + 2]
    old = str(original.ROOT / 'full' / arm)
    new = str(original.ROOT / 'full_only' / arm)
    return [arg.replace(old, new) for arg in cmd]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    import torch
    from average_checkpoints import average_checkpoints
    torch.set_num_threads(4)
    original.preflight()
    latest = original.ROOT / 'full/dapi/latest.pt'
    state = torch.load(latest, map_location='cpu', weights_only=False)
    print('DAPI resume epoch:', state['epoch'], flush=True)
    del state
    for arm in original.C['arms']:
        for stage in (1, 2):
            cmd = student_command(arm, stage, 'preflight')
            assert '--distillation-target-dir' not in cmd
            assert '--distillation-weight' not in cmd
    if args.preflight:
        return
    report = original.REPORTS / 'full_only'
    report.mkdir(parents=True, exist_ok=True)
    # A single process runs every job sequentially on the selected physical GPU.
    for kind in ('dapi', 'teacher'):
        original.write_json(report / 'status.json', {'phase': kind, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES')})
        out = original.ROOT / 'full' / kind
        if kind == 'teacher' and out.exists() and not (out / 'latest.pt').exists() and not (out / 'final_encoder.pt').exists():
            archive = out.with_name('teacher_incomplete_20261021')
            if archive.exists():
                raise FileExistsError(archive)
            out.rename(archive)
        if not (out / 'final_encoder.pt').exists():
            original.execute_task({'id': f'full_only.{kind}', 'kind': kind, 'scope': 'full'})
        else:
            epochs = original.C['dapi_epochs_full'] if kind == 'dapi' else original.C['feature_teacher_epochs']
            original.audit_training(out, 'full', epochs, kind)
    teacher_sha = original.digest(original.ROOT / 'full/teacher/final_encoder.pt')
    for arm in original.C['arms']:
        for stage in (1, 2):
            cmd = student_command(arm, stage, teacher_sha)
            out = Path(cmd[cmd.index('--output-dir') + 1])
            if out.exists():
                raise FileExistsError(f'Review partial supervised run before resuming: {out}')
            original.write_json(report / 'status.json', {'phase': f'{arm}.stage{stage}'})
            original.launch(cmd)
            ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
            settings = ck['args']
            assert ck['epoch'] == 14 and settings['seed'] == 2026
            assert settings['batch_size'] == 16 and settings['gradient_accumulation_steps'] == 1
            assert settings['distillation_weight'] == 0 and settings['train_all']
            assert settings['encoder_stage_decay'] == original.C['arms'][arm]
            assert all(torch.isfinite(v).all() for v in ck['model'].values())
            steps = [float(v['step']) for v in ck['optimizer']['state'].values() if 'step' in v]
            assert steps and min(steps) > 0
            original.write_json(report / f'{arm}.stage{stage}.json', {'command': cmd, 'epoch': 14, 'updates': min(steps)})
            del ck
        base = original.ROOT / 'full_only' / arm
        model = base / 'model.pt'
        average_checkpoints([base / 'stage2/epochs' / f'epoch_{e:03d}.pt' for e in original.C['final_epochs']], original.C['final_weights'], model, buffer_policy='latest')
        pred = Path(f'predictions/round137_full_only_{arm}_zero')
        package = Path(f'submissions/colon_round137_full_only_{arm}_seed2026_zero_q95.zip')
        original.launch([sys.executable, 'tools/predict_unet.py', original.C['dataset'], str(model), str(pred), '--marker', 'CD68', '--jpeg-quality', '95', '--tta', 'd4', '--context-halo', '128', '--output-source-offset', '0,0', '--batch-size', '2', '--device', 'cuda:0'])
        original.launch([sys.executable, 'tools/package_submission.py', str(pred), original.C['dataset'] + '/test/DAPI', str(package), '--marker', 'CD68'])
        import zipfile
        with zipfile.ZipFile(package) as z:
            assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == original.C['test_images']
        original.write_json(report / f'{arm}.artifacts.json', {'model': str(model), 'model_sha256': original.digest(model), 'package': str(package), 'package_sha256': original.digest(package)})
    original.write_json(report / 'status.json', {'phase': 'complete', 'independent_validation': False})


if __name__ == '__main__':
    main()

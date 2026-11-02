from __future__ import annotations
from pathlib import Path
from data import write_json
import random
import torch
import numpy as np
from data import SingleMarkerStainer
from losses import CompetitionLoss
from data import preload
from data import marker_policy
from io_utils import control_record
from training_utils import organ_loss_weights
import math
from training_utils import epoch_order
import json
from data import ORGANS
from data import read_image
from training_utils import optimizer_for
import time
from data import evaluate
import hashlib
from data import tensor
from losses import apply_d4_transform
from losses import invert_d4_transform
from data import MARKERS
import copy
from PIL import Image
POLICY = marker_policy(['CD68', 'CD45RO', 'CD68'])

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def distilled_loss(prediction, target, teacher, criterion, weight=0.2):
    hard = criterion(prediction, target)
    return hard if teacher is None else (1 - weight) * hard + weight * criterion(prediction, teacher)

def geometry_consistency_loss(model, source, marker, prediction, index, criterion):
    transformed = apply_d4_transform(source, index)
    with torch.autocast(source.device.type, dtype=torch.bfloat16, enabled=source.is_cuda):
        alternative = model(transformed, marker)
    restored = invert_d4_transform(alternative, index)
    return criterion(prediction.float(), restored.float())

def fixed_folds(grouped, number=5):
    """Partition each organ's Train ROIs, greedily balancing image counts."""
    folds = [{o: [] for o in ORGANS} for _ in range(number)]
    for organ in ORGANS:
        by_roi = {roi: [n for n in grouped[organ]['train'] if n.split('_')[0] == roi] for roi in grouped[organ]['train_rois']}
        rois = sorted(by_roi)
        random.Random(2026).shuffle(rois)
        rois.sort(key=lambda roi: len(by_roi[roi]), reverse=True)
        loads = [0] * number
        for roi in rois:
            slot = min(range(number), key=lambda i: (loads[i], i))
            folds[slot][organ].extend(by_roi[roi])
            loads[slot] += len(by_roi[roi])
        assert all(loads)
        flattened = [n for fold in folds for n in fold[organ]]
        assert len(flattened) == len(set(flattened)) == len(grouped[organ]['train'])
        assert set(flattened) == set(grouped[organ]['train'])
    return [{o: sorted(f[o]) for o in ORGANS} for f in folds]

def assign_heldout_rois(grouped, number=5):
    """Assign each outer-Val ROI to one teacher; all teachers excluded outer Val."""
    assignments = [{o: [] for o in ORGANS} for _ in range(number)]
    for organ in ORGANS:
        rois = sorted(grouped[organ]['val_rois'])
        for index, roi in enumerate(rois):
            slot = index % number
            assignments[slot][organ].extend((n for n in grouped[organ]['val'] if n.split('_')[0] == roi))
        emitted = [n for fold in assignments for n in fold[organ]]
        assert len(emitted) == len(set(emitted)) == len(grouped[organ]['val'])
        assert set(emitted) == set(grouped[organ]['val'])
        for fold in assignments:
            roi_ids = {n.split('_')[0] for n in fold[organ]}
            assert all((sum((roi in {x.split('_')[0] for x in other[organ]} for other in assignments)) == 1 for roi in roi_ids))
    return assignments

def teacher_split(grouped, excluded):
    split = copy.deepcopy(grouped)
    for o, spec in split.items():
        withheld = set(excluded[o])
        assert withheld <= set(spec['train'])
        spec['train'] = [n for n in spec['train'] if n not in withheld]
        spec['train_rois'] = sorted({n.split('_')[0] for n in spec['train']})
        spec['val'] = []
        spec['val_rois'] = []
        assert not set(spec['train_rois']) & {n.split('_')[0] for n in withheld}
    return split

def seed():
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False

def fit(output, split, base_config, endpoint, encoder, initial=None, targets=None, validate=False, smoke=False, consistency_weight=0.0, distillation_weight=0.2):
    """Fixed horizons14. Student replays stage2 from exact saved stage1 state."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = dict(base_config)
    if distillation_weight not in (0.2, 0.4):
        raise ValueError('Unregistered distillation weight')
    if consistency_weight not in (0.0, 0.1, 0.2, 0.4):
        raise ValueError('Unregistered consistency strength')
    if consistency_weight:
        cfg['training_geometry_consistency_weight'] = consistency_weight
    weighted = 'organ_loss_objective' in base_config
    weights = organ_loss_weights(split) if weighted else None
    if weights is not None:
        cfg['organ_loss_weights'] = weights
    cfg.update(endpoint=endpoint, smoke=smoke, full_train=not validate, distillation_weight=distillation_weight if targets is not None else 0.0, initial_checkpoint_sha256=digest(initial) if initial else None, teacher_cache=str(targets) if targets else None, pretraining_scope='shared Round302 outer-Train DAPI; not inner-fold-excluded pretraining', validation='Gaussian11 JPEG95 each epoch' if validate else 'no validation during fitting')
    config_path = output / 'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding='utf-8')) == cfg
    write_json(config_path, cfg)
    if (output / 'split.json').exists():
        assert json.loads((output / 'split.json').read_text(encoding='utf-8')) == split
    write_json(output / 'split.json', split)
    if (output / 'complete.json').exists():
        return
    seed()
    model = SingleMarkerStainer()
    model.encoder.load_state_dict(encoder, strict=True)
    model = model.cuda()
    best = -math.inf
    start = 1
    saved = None
    last_path = output / 'last.pt'
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        start = saved['epoch'] + 1
        best = saved['best']
    elif initial:
        saved = torch.load(initial, map_location='cpu', weights_only=False)
        assert saved['epoch'] == 14
        model.load_state_dict(saved['model'])
        start = 15
    if saved is not None:
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    trajectory = output / 'metrics.jsonl'
    if trajectory.exists():
        rows = [json.loads(s) for s in trajectory.read_text(encoding='utf-8').splitlines()]
        trajectory.write_text(''.join((json.dumps(r) + '\n' for r in rows if r['epoch'] < start)), encoding='utf-8')
    criterion = CompetitionLoss(ssim_mode='uniform7')
    cache = preload(split, POLICY)
    teacher_cache = {}
    if targets:
        manifest = json.loads((Path(targets) / 'manifest.json').read_text(encoding='utf-8'))
        expected = {(o, n) for o in ORGANS for n in split[o]['train']}
        assert expected == {(r['organ'], r['name']) for r in manifest['items']}
        assert len(manifest['items']) == len(expected)
        teacher_cache = {(o, n): read_image(Path(targets) / o / (n + '.png')) for o, n in expected}
    optimizer = scheduler = None
    for epoch in range(start, endpoint + 1):
        stage = 1 + (epoch - 1) // 14
        stage_epoch = 1 + (epoch - 1) % 14
        if optimizer is None or stage_epoch == 1:
            optimizer = optimizer_for(model, stage)
            if stage == 2:
                for group in optimizer.param_groups:
                    group['lr'] *= 1.0
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 14, eta_min=0.0)
            if last_path.exists() and saved is not None and (stage_epoch != 1):
                optimizer.load_state_dict(saved['optimizer'])
                scheduler.load_state_dict(saved['scheduler'])
            saved = None
        rates = [g['lr'] for g in optimizer.param_groups]
        order = epoch_order(split, epoch)
        if smoke:
            order = order[:32]
        model.train()
        begin = time.perf_counter()
        losses = []
        consistency_values = []
        counts = {o: 0 for o in ORGANS}
        for offset in range(0, len(order), 16):
            group = order[offset:offset + 16]
            optimizer.zero_grad(set_to_none=True)
            for organ, name in group:
                arrays = cache[organ, name]
                k = random.randrange(4)
                flip = random.randrange(2)

                def geometry(a):
                    a = np.rot90(a, k)
                    return np.fliplr(a) if flip else a
                x = tensor(geometry(arrays['DAPI']), 'cuda')
                y = tensor(geometry(arrays[POLICY[organ][0]]), 'cuda')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(x, torch.tensor([MARKERS.index(POLICY[organ][0])], device='cuda'))
                soft = None
                if targets is not None and stage == 2:
                    soft = tensor(geometry(teacher_cache[organ, name]), 'cuda')
                loss = distilled_loss(pred.float(), y, soft, criterion, distillation_weight)
                if consistency_weight and epoch >= cfg.get('consistency_start_epoch', 1):
                    position = sum(counts.values())
                    index = 1 + ((epoch - 1) * len(order) + position) % 7
                    consistency = geometry_consistency_loss(model, x, torch.tensor([MARKERS.index(POLICY[organ][0])], device='cuda'), pred, index, criterion)
                    loss = loss + consistency_weight * consistency
                    consistency_values.append(consistency.item())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite distillation loss')
                objective = loss if weights is None else loss * weights[organ]
                (objective / len(group)).backward()
                losses.append(loss.item())
                counts[organ] += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        if not smoke:
            assert counts == {o: len(split[o]['train']) for o in ORGANS}
        result = evaluate(model, split, cache, policy=POLICY, limit=1 if smoke else 0) if validate else None
        if result:
            write_json(output / f'validation_{epoch:03d}.json', result)
            if epoch >= cfg.get('selection_start_epoch', 1) and result['weighted_score'] > best:
                best = result['weighted_score']
                torch.save(dict(model=model.state_dict(), config=cfg, epoch=epoch, stage=stage, stage_epoch=stage_epoch, markers={o: POLICY[o][0] for o in ORGANS}), output / 'best.pt')
        scheduler.step()
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch, best=best, python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        row = dict(epoch=epoch, stage=stage, stage_epoch=stage_epoch, train_seconds=elapsed, loss=float(np.mean(losses)), samples=counts, updates=math.ceil(len(order) / 16), group_lrs=rates, validation_score=result['weighted_score'] if result else None)
        if consistency_values:
            row['geometry_consistency_loss'] = float(np.mean(consistency_values))
        with trajectory.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row) + '\n')
        torch.save(state, output / 'last.writing.pt')
        (output / 'last.writing.pt').replace(last_path)
        if stage_epoch == 14:
            torch.save(state, output / 'stage1_final.pt' if stage == 1 else output / 'stage2_final.pt')
        print(json.dumps({'run': str(output), **row}), flush=True)
    assert all((torch.isfinite(p).all() for p in model.parameters()))
    torch.save(dict(model=model.state_dict(), config=cfg, epoch=endpoint, stage=1 + (endpoint - 1) // 14, stage_epoch=1 + (endpoint - 1) % 14, markers={o: POLICY[o][0] for o in ORGANS}), output / 'final.pt')
    control_record(output / 'complete.json', dict(epochs=endpoint, best_validation_score=best if validate else None))

def predict_targets(checkpoint, grouped, names, folder, fold, existing):
    """One single teacher prediction per training image; fixed lossless uint8."""
    model = SingleMarkerStainer()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'], strict=True)
    model = model.cuda().eval()
    assert ck['markers'] == {o: POLICY[o][0] for o in ORGANS}
    teacher_hash = digest(checkpoint)
    with torch.inference_mode():
        for organ in ORGANS:
            target_folder = folder / organ
            target_folder.mkdir(parents=True, exist_ok=True)
            for name in names[organ]:
                key = (organ, name)
                output = target_folder / (name + '.png')
                image = read_image(Path(grouped[organ]['folder']) / 'DAPI' / name)
                pred = model(tensor(image, 'cuda'), torch.tensor([MARKERS.index(POLICY[organ][0])], device='cuda'))
                assert pred.shape == (1, 1, 256, 256) and torch.isfinite(pred).all()
                pixels = np.rint(pred[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(pixels).save(output, format='PNG')
                if key in existing:
                    raise ValueError('Duplicate target')
                existing[key] = {'organ': organ, 'name': name, 'fold': fold, 'teacher_sha256': teacher_hash}
    del model, ck
    torch.cuda.empty_cache()

def audit_fit(folder, split, endpoint, initial=False, smoke=False):
    folder = Path(folder)
    rows = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['epoch'] for r in rows] == list(range(15 if initial else 1, endpoint + 1))
    counts = {o: len(split[o]['train']) for o in ORGANS}
    updates = 2 if smoke else math.ceil(sum(counts.values()) / 16)
    assert all((math.isfinite(r['loss']) and r['updates'] == updates for r in rows))
    if not smoke:
        assert all((r['samples'] == counts for r in rows))
    last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    steps = {int(s['step']) for s in last['optimizer']['state'].values() if 'step' in s}
    expected_steps = updates * (1 + (endpoint - 1) % 14)
    assert last['epoch'] == endpoint
    cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    assert steps == {expected_steps}
    final = torch.load(folder / 'final.pt', map_location='cpu', weights_only=False)
    assert final['epoch'] == endpoint and final['markers'] == {o: POLICY[o][0] for o in ORGANS}
    assert all((torch.isfinite(v).all() and torch.equal(v, last['model'][k]) for k, v in final['model'].items()))
    control_record(folder / 'audit.json', {'endpoint': endpoint, 'updates': sum((r['updates'] for r in rows)), 'initial_stage1_reused': initial, 'finite_weights': True})

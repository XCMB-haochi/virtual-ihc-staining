from __future__ import annotations
from data import ORGANS
import random
import torch
import copy

def optimizer_for(model, stage):
    lr = 0.0003 if stage == 1 else 0.0001
    encoder_ids = {id(p) for p in model.encoder.parameters()}
    groups = [{'params': [p for p in model.parameters() if id(p) not in encoder_ids], 'lr': lr, 'name': 'decoder_head'}]
    modules = [model.encoder.stem, *model.encoder.stages]
    for i, module in enumerate(modules):
        groups.append({'params': list(module.parameters()), 'lr': lr * 0.8 ** (4 - i), 'name': f'encoder_{i}'})
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) and set(ids) == {id(p) for p in model.parameters()}
    return torch.optim.AdamW(groups, weight_decay=0.0001)

def epoch_order(manifest, epoch):
    order = [(o, n) for o in ORGANS for n in manifest[o]['train']]
    random.Random(2026 + epoch).shuffle(order)
    return order

def all_train_manifest(grouped):
    """Include the held-out Train ROIs for final fitting; never include Test."""
    result = copy.deepcopy(grouped)
    for spec in result.values():
        assert not set(spec['train']) & set(spec['val'])
        spec['train'] = sorted(spec['train'] + spec['val'])
        spec['train_rois'] = sorted(spec['train_rois'] + spec['val_rois'])
        spec['val'] = []
        spec['val_rois'] = []
    return result

def organ_loss_weights(manifest):
    """Make the mean per-image loss represent the fixed per-organ score weights."""
    counts = {o: len(manifest[o]['train']) for o in ORGANS}
    if any((n <= 0 for n in counts.values())):
        raise ValueError('Every organ needs training images')
    total = sum(counts.values())
    return {o: total * w / counts[o] for o, w in zip(ORGANS, (0.1, 0.2, 0.7))}

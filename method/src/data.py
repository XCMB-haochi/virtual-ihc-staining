from __future__ import annotations
import json
from model import SingleMarkerStainer
from model import MARKERS
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import io
import torch.nn.functional as F
import math
import random
ORGANS = ('colon', 'liver', 'stomach')

def marker_policy(organ_markers=None):
    if organ_markers is None:
        return {organ: MARKERS for organ in ORGANS}
    if len(organ_markers) != len(ORGANS) or any((m not in MARKERS for m in organ_markers)):
        raise ValueError('Specify exactly one valid marker for Colon, Liver and Stomach')
    return {organ: (marker,) for organ, marker in zip(ORGANS, organ_markers)}

def summarize_results(results, policy):
    score = sum((w * float(np.mean([results[o][m]['score'] for m in policy[o]])) for o, w in zip(ORGANS, (0.1, 0.2, 0.7))))
    selected = {o: markers[0] if len(markers) == 1 else list(markers) for o, markers in policy.items()}
    return {'organs': results, 'selected_markers': selected, 'weighted_score': score, 'selection_rule': 'mean of all supervised markers per organ; fixed organ weights'}

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def inventory(root):
    result = {}
    for organ in ORGANS:
        matches = [p for p in root.rglob('train') if p.parent.name.lower().startswith(organ)]
        if len(matches) != 1:
            raise ValueError((organ, matches))
        folder = matches[0]
        names = sorted((p.name for p in (folder / 'DAPI').glob('*.jpg')))
        assert names
        for marker in MARKERS:
            assert set(names) == {p.name for p in (folder / marker).glob('*.jpg')}
        groups = sorted({n.split('_')[0] for n in names})
        random.Random(2026).shuffle(groups)
        val = set(groups[:math.ceil(len(groups) * 0.2)])
        result[organ] = {'folder': str(folder), 'val_rois': sorted(val), 'train_rois': sorted(set(groups) - val), 'train': [n for n in names if n.split('_')[0] not in val], 'val': [n for n in names if n.split('_')[0] in val]}
    return result

def read_image(path):
    with Image.open(path) as image:
        value = np.asarray(image.convert('L'), dtype=np.uint8).copy()
    if value.shape != (256, 256):
        raise ValueError((str(path), value.shape))
    return value

def preload(manifest, policy=None):
    policy = marker_policy() if policy is None else policy
    cache = {}
    for organ, spec in manifest.items():
        folder = Path(spec['folder'])
        for split in ('train', 'val'):
            for name in spec[split]:
                cache[organ, name] = {m: read_image(folder / m / name) for m in ('DAPI',) + tuple(policy[organ])}
    return cache

def gaussian_ssim(x, y):
    t = torch.arange(11, device=x.device, dtype=torch.float32) - 5
    g = torch.exp(-t * t / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :])[None, None]
    a = F.conv2d(x, kernel)
    b = F.conv2d(y, kernel)
    va = F.conv2d(x * x, kernel) - a * a
    vb = F.conv2d(y * y, kernel) - b * b
    cov = F.conv2d(x * y, kernel) - a * b
    return ((2 * a * b + 0.01 ** 2) * (2 * cov + 0.03 ** 2) / ((a * a + b * b + 0.01 ** 2) * (va + vb + 0.03 ** 2))).mean()

def tensor(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.float32)[None, None] / 255

def encoded_prediction(pred):
    array = np.rint(pred[0, 0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).convert('RGB').save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return np.asarray(image.convert('L')).copy()

@torch.inference_mode()
def evaluate(model, manifest, cache, limit=0, policy=None, device='cuda'):
    policy = marker_policy() if policy is None else policy
    model.eval()
    results = {}
    for organ, spec in manifest.items():
        results[organ] = {}
        names = spec['val'][:limit] if limit else spec['val']
        for marker in policy[organ]:
            index = MARKERS.index(marker)
            rows = []
            rois = {}
            for name in names:
                arrays = cache[organ, name]
                x = tensor(arrays['DAPI'], device)
                y = tensor(arrays[marker], device)
                pred = model(x, torch.tensor([index], device=device))
                assert pred.shape == y.shape and torch.isfinite(pred).all()
                saved = tensor(encoded_prediction(pred), device)
                ssim = gaussian_ssim(saved, y).item()
                mse = F.mse_loss(saved, y).item()
                psnr = min(50.0, -10 * math.log10(max(mse, 1e-12)))
                row = {'ssim': ssim, 'psnr_capped': psnr, 'score': 70 * ssim + 0.6 * psnr}
                rows.append(row)
                rois.setdefault(name.split('_')[0], []).append(row)
            mean = lambda values: {key: float(np.mean([v[key] for v in values])) for key in rows[0]}
            results[organ][marker] = {**mean(rows), 'images': len(rows), 'rois': {k: mean(v) for k, v in rois.items()}}
    return summarize_results(results, policy)

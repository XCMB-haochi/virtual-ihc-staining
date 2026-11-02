"""Count the submitted network's disjoint modules and verify forward shapes."""
import hashlib
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/submission_round245/src'))
from ihc_unext2 import UNeXt2VirtualStainer


def main():
    torch.set_num_threads(4)
    model = UNeXt2VirtualStainer(4, 'spatial-thumbnail-film')
    checkpoint = ROOT / 'deliverables/AIC-2026-51865112-通信速通生物-复赛-团队成员/AIC-2026-51865112-模型文件/model.pt'
    original = pathlib.PosixPath
    try:
        if sys.platform == 'win32':
            pathlib.PosixPath = pathlib.WindowsPath
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    finally:
        pathlib.PosixPath = original
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    count = lambda module: sum(p.numel() for p in module.parameters())
    modules = [('encoder.stem', model.encoder.stem)]
    modules += [(f'encoder.stage{i+1}', m) for i, m in enumerate(model.encoder.stages)]
    modules += [(f'decoder.stage{3-i}', m) for i, m in enumerate(model.decoder.stages)]
    modules += [(f'condition.{n}', m) for n, m in model.condition_film.named_children()]
    rows = [{'module': n, 'parameters': count(m)} for n, m in modules]
    rows.append({'module': 'output_bias', 'parameters': model.output_bias.numel()})
    total = count(model)
    assert sum(r['parameters'] for r in rows) == total == 31753556
    shapes = {}
    handles = [m.register_forward_hook(lambda mod, args, out, n=n: shapes.update({n: list(out.shape)}))
               for n, m in modules if n.startswith(('encoder.', 'decoder.'))]
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 512, 512), torch.zeros(1, model.condition_dimension))
    assert list(output.shape) == [1, 4, 512, 512]
    for h in handles:
        h.remove()
    with checkpoint.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    report = {'checkpoint_sha256': digest, 'total_parameters': total,
              'condition_dimension': model.condition_dimension, 'modules': rows,
              'forward_shapes_512': shapes, 'output_shape': list(output.shape)}
    destination = pathlib.Path(__file__).with_name('model_audit.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

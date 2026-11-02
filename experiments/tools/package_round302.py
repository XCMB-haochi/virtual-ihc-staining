"""Fixed Val-selected Round302 mixed checkpoint; single-image JPEG95 inference."""
import hashlib
import argparse
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from round300 import (SingleMarkerStainer, MARKERS, ORGANS, inventory, read_image,
                      tensor, encoded_prediction, write_json)


def jpeg_bytes(prediction):
    pixels=np.rint(prediction[0,0].float().cpu().numpy()*255).clip(0,255).astype(np.uint8)
    buffer=io.BytesIO()
    Image.fromarray(pixels).convert('RGB').save(buffer,format='JPEG',quality=95)
    return buffer.getvalue()


def main(round_id=302):
    full_train = round_id in (304,306,309,314,319,321,324,327)
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    torch.set_num_threads(2); torch.backends.cudnn.benchmark=False
    run=Path('runs/round302_supervised/mixed' if round_id==302 else f'runs/round{round_id}')
    source=run/('final.pt' if full_train else 'best.pt')
    endpoint=27
    if round_id==306:
        decision=json.loads(Path('reports/round305_decision.json').read_text(encoding='utf-8'))
        assert decision['refit_cd68']
        endpoint=decision['selected_epoch']
    if round_id==309:
        decision=json.loads(Path('reports/round308_decision.json').read_text(encoding='utf-8'))
        assert decision['needs_full_refit']
        endpoint=decision['selected_epoch']
    if round_id in (324,327):
        selection_round=323 if round_id==324 else 326
        decision=json.loads(Path(f'reports/round{selection_round}_decision.json').read_text(encoding='utf-8'))
        assert decision['difference']>0
        endpoint=decision['selected_epoch']
    if round_id in (314,319,321):
        provenance=json.loads((run/'stage1_provenance.json').read_text(encoding='utf-8'))
        assert provenance['full_train_images']==10356 and provenance['stage1_epochs']==14
        assert provenance['sha256']==hashlib.sha256(Path('runs/round309/stage1_final.pt').read_bytes()).hexdigest()
        selection_round={314:313,319:318,321:320}[round_id]
        decision=json.loads(Path(f'reports/round{selection_round}_decision.json').read_text(encoding='utf-8'))
        assert (decision['candidate_for_review'] if round_id==314 else decision['difference']>0)
        endpoint=decision['selected_epoch']
    if round_id==302:
        comparison=json.loads(Path('runs/round302_supervised/comparison.json').read_text(encoding='utf-8'))
        assert comparison['selected_candidate']=='mixed'
        chosen=next(c for c in comparison['candidates'] if c['name']=='mixed')
    else:
        complete=json.loads((run/'complete.json').read_text(encoding='utf-8'))
        assert complete['epochs']==(endpoint if full_train else 28)
        chosen={'weighted_score':complete['best_validation_score']}
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    if round_id in (314,319,321):
        assert checkpoint['config']['initial_checkpoint_sha256']==provenance['sha256']
        assert checkpoint['config']['distillation_weight']==.2
        assert checkpoint['config']['training_geometry_consistency_weight']==(.1 if round_id==314 else .2)
        if round_id==321:
            assert checkpoint['config']['update_order']=='proportional organ interleaving; same per-image geometry and consistency view'
    if round_id==302: assert checkpoint['epoch']==chosen['best_epoch']==32
    elif round_id in (303,307):
        validation=json.loads((run/f"validation_{checkpoint['epoch']:03d}.json").read_text(encoding='utf-8'))
        assert validation['weighted_score']==chosen['weighted_score']
    else:
        assert checkpoint['epoch']==endpoint and checkpoint['stage']==1+(endpoint-1)//14 and checkpoint['stage_epoch']==1+(endpoint-1)%14
        assert checkpoint['config']['full_train'] and not checkpoint['config']['smoke']
        assert checkpoint['config']['epochs_per_stage']==14
    markers={'colon':'CD68','liver':'CD68' if round_id==306 else 'CD45RO','stomach':'CD68'}
    assert checkpoint['markers']==markers
    if round_id==302: assert markers==chosen['organ_markers']
    assert checkpoint['config']['supervised_markers']=={o:[m] for o,m in markers.items()}
    audit=json.loads(Path('reports/round302_pretrain_audit.json').read_text(encoding='utf-8'))
    assert checkpoint['config']['domain_encoder_sha256']==audit['encoder_sha256']
    manifest=inventory(Path('dataset'))
    if full_train:
        from round303 import all_train_manifest
        assert manifest==json.loads((run/'selection_split.json').read_text(encoding='utf-8'))
        manifest=all_train_manifest(manifest)
    assert manifest==json.loads((run/'split.json').read_text(encoding='utf-8'))
    if round_id in (324,327):
        if round_id==324:
            from round324 import verify_refit
        else:
            from round327 import verify_refit
        verify_refit(run,manifest,endpoint)
        assert checkpoint['config']==json.loads((run/'config.json').read_text(encoding='utf-8'))
    model=SingleMarkerStainer(); model.load_state_dict(checkpoint['model'],strict=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    model=model.cuda().eval()
    marker_tag='cd68' if round_id==306 else 'mixed'
    output=Path(f'submissions/semifinal_round{round_id}_{marker_tag}_q95.zip')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    counts={}; expected=[]; started=time.perf_counter()
    # Verify that output encoding is exactly the encoding scored during Val.
    sample=torch.linspace(0,1,256*256).reshape(1,1,256,256)
    with Image.open(io.BytesIO(jpeg_bytes(sample))) as im:
        assert np.array_equal(np.asarray(im.convert('L')),encoded_prediction(sample))
    with zipfile.ZipFile(output.with_suffix('.writing.zip'),'w',compression=zipfile.ZIP_STORED) as archive:
        with torch.inference_mode():
            for organ in ORGANS:
                folder=Path(manifest[organ]['folder']).parent/'test'/'DAPI'
                names=sorted(p.name for p in folder.glob('*.jpg'))
                assert len(names)=={'colon':1346,'liver':560,'stomach':280}[organ]
                marker=markers[organ]; index=torch.tensor([MARKERS.index(marker)],device='cuda')
                for name in names:
                    image=read_image(folder/name)
                    prediction=model(tensor(image,'cuda'),index)
                    assert prediction.shape==(1,1,*image.shape) and torch.isfinite(prediction).all()
                    payload=jpeg_bytes(prediction)
                    with Image.open(io.BytesIO(payload)) as decoded:
                        decoded.load()
                        assert decoded.mode=='RGB' and decoded.size==(256,256)
                    member=f'results/{organ}/test/{marker}/{name}'
                    archive.writestr(member,payload); expected.append(member)
                counts[organ]=len(names)
                print(json.dumps({'organ':organ,'images':len(names),'seconds':time.perf_counter()-started}),flush=True)
    output.with_suffix('.writing.zip').replace(output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert archive.namelist()==expected and len(expected)==len(set(expected))==2186
        assert sum(i.file_size for i in archive.infolist())<8192*1024**2
    assert output.stat().st_size<2048*1024**2
    result={'checkpoint':str(source),'checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'epoch':checkpoint['epoch'],'markers':markers,'validation_score':chosen['weighted_score'],
            'selection':(f'fixed full-data epoch{endpoint} from Round{326 if round_id==327 else 323 if round_id==324 else (320 if round_id==321 else (318 if round_id==319 else (313 if round_id==314 else (308 if round_id==309 else (305 if round_id==306 else 303)))))} Val; no full-data validation'
                         if full_train else 'best grouped Val checkpoint within this run; no cross-run performance gate'),
            'images':counts,'package':str(output),'bytes':output.stat().st_size,
            'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'jpeg_quality':95,
            'tta':False,'context':False,'postprocessing':False,'seconds':time.perf_counter()-started}
    manifest_path=Path(f'reports/round{round_id}_submission.json')
    temporary=manifest_path.with_suffix('.json.writing')
    write_json(temporary,result)
    temporary.replace(manifest_path)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round',dest='round_id',type=int,choices=(302,303,304,306,307,309,314,319,321,324,327),default=302)
    main(parser.parse_args().round_id)


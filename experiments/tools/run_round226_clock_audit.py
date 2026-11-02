"""Validation-only common-clock reassessment, not a hidden-score epoch search."""
import json
import math
import subprocess
import sys
from pathlib import Path
from run_round204_cd45_cv import DATASET
from run_round210_relational import digest


def select_clock(rows,omit=None):
    folds=[f for f in rows if f!=omit]
    values=[]
    for e in range(14):
        count=sum(rows[f][e]['validation']['count'] for f in folds)
        values.append(sum(rows[f][e]['validation']['count']*rows[f][e]['validation']['markers']['CD45RO']['competition_score_mean_image_psnr'] for f in folds)/count)
    return {'epoch':max(range(14),key=lambda i:values[i])+1,'scores':values}


def main():
    protocol=json.loads(Path('configs/semifinal_cv5_baseline.json').read_text())
    folders={
      'base':{f['name']:(f"round207_{f['name']}" if f['name'] in ('fold0','fold3') else f"round216_{f['name']}") for f in protocol['folds']},
      'signal':{f['name']:(f"round223_{f['name']}" if f['name'] in ('fold0','fold3') else f"round224_{f['name']}") for f in protocol['folds']}}
    decisions={}
    for recipe,runs in folders.items():
        rows={fold:[json.loads(l) for l in Path(f'runs/{prefix}_stage2/metrics.jsonl').read_text().splitlines()] for fold,prefix in runs.items()}
        for trajectory in rows.values():
            assert [r['epoch'] for r in trajectory]==list(range(1,15))
            assert all(math.isfinite(r['selection_score']) and r['validation']['ssim_mode']=='gaussian11' for r in trajectory)
        decisions[recipe]={'common':select_clock(rows),'leave_one_fold_out':{f:select_clock(rows,f)['epoch'] for f in rows},
                           'attempted_steps_per_epoch':{f:rs[-1]['optimizer_steps'] for f,rs in rows.items()}}
    output=Path('reports/round226_clock_audit.json')
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(decisions,indent=2)+'\n')
    print('Validation-derived clocks',json.dumps({k:{'common':v['common']['epoch'],'leave_one_fold_out':v['leave_one_fold_out']} for k,v in decisions.items()}),flush=True)
    for recipe,runs in folders.items():
        epoch=decisions[recipe]['common']['epoch']
        sources=[]
        for fold in protocol['folds']:
            ck=Path(f"runs/{runs[fold['name']]}_stage2/epochs/epoch_{epoch:03d}.pt")
            sources.append({**fold,'images':140*len(fold['val_rois']),'checkpoint':str(ck.resolve()),'checkpoint_sha256':digest(ck)})
        directory=Path(f'runs/round226_{recipe}_sources');directory.mkdir(exist_ok=False)
        (directory/'manifest.json').write_text(json.dumps({'method':'fold-excluded-single-recipe-d4-teacher',
            'output_source_offset_applied':False,'images':2380,'folds':sources},indent=2)+'\n')
        subprocess.run([sys.executable,'tools/audit_semifinal_markers.py',DATASET,str(directory),
                        f'reports/round226_{recipe}_d4.json','--mode','models','--device','cuda:0'],check=True)
    base=json.loads(Path('reports/round226_base_d4.json').read_text())['markers']['CD45RO']
    signal=json.loads(Path('reports/round226_signal_d4.json').read_text())['markers']['CD45RO']
    gains={scope:r['gaussian11_score']-base[scope]['gaussian11_score'] for scope,r in signal.items()}
    Path('reports/round226_cd45ro_gains.json').write_text(json.dumps(gains,indent=2)+'\n')
    print('D4 changes at separately validation-selected common clocks',json.dumps(gains),flush=True)


if __name__=='__main__':main()

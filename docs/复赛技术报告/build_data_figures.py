"""Build editable method and qualitative figures from verified, real Liver images.

Teacher-cache inputs live outside Git in runs/report_figure_sources. They are the three
fixed intensity-quantile PNG targets and manifest from runs/round245_oof_targets.
No prediction is generated, selected by score, or spatially enhanced here.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree.ElementTree import SubElement

import numpy as np
from PIL import Image

from build_drawio import Diagram, ENC, DEC, COND, OUT

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dataset/复赛数据集(包括训练集和测试集输入)'
CACHE = ROOT / 'runs/report_figure_sources'
DELIVERY = ROOT / 'deliverables/AIC-2026-51865112-通信速通生物-复赛-团队成员'
NAMES = ['ROI010_07_09.jpg', 'ROI093_02_01.jpg', 'ROI081_08_06.jpg']
TEST_NAME = 'ROI025_09_02.jpg'


def load(path):
    return np.asarray(Image.open(path).convert('L'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(array):
    # One common, fixed linear window for every biological image in both figures.
    return Image.fromarray(np.rint(np.clip(array.astype(float) / 128, 0, 1) * 255).astype('uint8'))


def picture(d, key, array, x, y, size):
    buf = io.BytesIO()
    display(array).save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode()
    c = SubElement(d.root, 'mxCell', id=key, value='', vertex='1', parent='1',
                   style='shape=image;imageAspect=0;aspect=fixed;html=1;strokeColor=none;'
                   'image=data:image/png,' + encoded + ';')
    SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(size), height=str(size), **{'as': 'geometry'})
    d.boxes[key] = (x, y, size, size)


def scale_bar(d, key, x, y, length, label):
    d.node(key, '', x, y, length, 4, '#FFFFFF', style='strokeColor=none;spacing=0;')
    d.text(key+'-label', label, x-8, y-26, length+16, 22, 17)
    c = d.root.find(f"mxCell[@id='{key}-label']")
    c.set('style', c.get('style')+'fontColor=#FFFFFF;')


manifest = json.loads((CACHE/'manifest.json').read_text())
expected = json.loads((ROOT/'reports/round245_artifacts.json').read_text())
for fold in manifest['folds']:
    assert fold['sha256'] == next(f['sha256'] for f in expected['teachers'] if f['name'] == fold['name'])
ranked = sorted((float(load(p).mean()), p.name) for p in (DATA/'train/CD45RO').iterdir())
assert [ranked[round(q*(len(ranked)-1))][1] for q in (.25, .5, .75)] == NAMES
test_ranked = sorted((float(load(p).mean()), p.name) for p in (DATA/'test/DAPI').iterdir())
assert test_ranked[len(test_ranked)//2][1] == TEST_NAME

d = Diagram('Training and deployment', 1280, 950)
for key, y, height in [('pretraining-panel', -15, 280),
                       ('distillation-panel', 285, 330),
                       ('deployment-panel', 645, 285)]:
    d.node(key, '', -15, y, 1280, height, 'none',
           style='rounded=1;absoluteArcSize=1;arcSize=28;dashed=1;'
                 'dashPattern=5 4;strokeColor=#657786;strokeWidth=3.5;')
d.text('a','a  DAPI domain pretraining',15,5,1180,40,30,True,'left')
train = load(DATA/'train/DAPI'/NAMES[1])
picture(d,'train-dapi',train,25,75,130)
d.text('train-l','Training DAPI',10,211,160,28,22)
mask = np.ones(256, dtype=bool)
mask[np.random.default_rng(2026).permutation(256)[:128]] = False
masked = train * np.repeat(np.repeat(mask.reshape(16,16),16,axis=0),16,axis=1)
picture(d,'masked',masked,245,75,130)
d.text('mask-l','50% masked',225,211,170,28,22)
d.edge('mask-edge','train-dapi','masked','e','w')
d.node('fcmae','FCMAE<br>Masked reconstruction',475,90,250,95,ENC,26)
d.edge('pretrain-edge','masked','fcmae','e','w')
d.node('init','Encoder initialization<br>Fold-specific / full-data',845,90,345,95,ENC,25)
d.edge('init-edge','fcmae','init','e','w')

d.text('b','b  ROI-excluded knowledge distillation',15,300,1180,42,30,True,'left')
d.node('teachers','Five fold teachers<br>Supervised on training ROIs',25,387,250,100,ENC,25)
d.text('teacher-init','Fold-specific encoder',25,349,250,30,21)
picture(d,'teacher-target',load(CACHE/NAMES[1]),340,375,125)
d.text('target-l','Held-out target',305,507,195,28,22)
d.edge('target-edge','teachers','teacher-target','e','w')
d.node('stage1','Student: stage 1<br>True labels only',550,387,235,100,DEC,25)
d.text('student-init','Full-data encoder',550,349,235,30,21)
d.node('stage2','Student: stage 2<br>Labels + teacher target',945,387,250,100,DEC,25)
d.edge('student-edge','stage1','stage2','e','w')
d.text('warmstart','Initialize',795,398,140,30,21)
d.edge('distill-edge','teacher-target','stage2','e','s',[(500,437.5),(500,581),(1070,581)],dashed=True)
d.text('loss','0.8 label loss + 0.2 distillation loss',565,535,530,35,24)

d.text('c','c  Single-student deployment',15,669,1180,40,30,True,'left')
picture(d,'test-dapi',load(DATA/'test/DAPI'/TEST_NAME),25,741,130)
d.text('test-l','Test DAPI',10,875,160,28,22)
d.node('student','2 translations x D4<br>One fixed student',250,754,260,100,DEC,24)
d.node('tta','Inverse transforms<br>Equal-weight mean',635,754,300,100,'#F2F5F7',24)
with ZipFile(DELIVERY/'results.zip') as archive:
    prediction_bytes = archive.read('AIC-2026-51865112-赛题结果文档/results/test/cd45ro/' + TEST_NAME)
prediction = np.asarray(Image.open(io.BytesIO(prediction_bytes)).convert('L'))
picture(d,'test-output',prediction,1060,741,130)
d.text('output-l','CD45RO prediction',1010,875,225,28,22)
d.edge('test-input','test-dapi','student','e','w')
d.edge('single-forward','student','tta','e','w')
d.edge('tta-output','tta','test-output','e','w')
d.save('workflow.drawio')

q = Diagram('Held-out qualitative examples', 1370, 990)
cols = [('DAPI input',15),('CD45RO reference',285),('Teacher prediction',555),('Reference crop',825),('Prediction crop',1095)]
for i,(label,x) in enumerate(cols):
    q.text(f'column-{i}',label,x,5,250,36,25,True)
records=[]
for r,(name,quantile) in enumerate(zip(NAMES,(.25,.5,.75))):
    y=90+r*310
    q.text(f'row-{r}',f'{chr(97+r)}  {Path(name).stem}  |  intensity quantile {int(quantile*100)}%',15,y-42,1260,32,23,True,'left')
    arrays=[load(DATA/'train/DAPI'/name),load(DATA/'train/CD45RO'/name),load(CACHE/name)]
    arrays += [arrays[1][96:160,96:160],arrays[2][96:160,96:160]]
    for c,((label,x),array) in enumerate(zip(cols,arrays)):
        picture(q,f'image-{r}-{c}',array,x,y,245)
        if c in (1,2):
            q.node(f'crop-box-{r}-{c}','',x+245*96/256,y+245*96/256,245*64/256,245*64/256,
                   'none',style='strokeColor=#61D3D4;strokeWidth=2;spacing=0;')
        if c==0:
            scale_bar(q,f'scale-{r}',x+165,y+226,245*64/256,'64 px')
    roi=name.split('_')[0]
    fold=next(f for f in manifest['folds'] if roi in f['val_rois'])
    records.append({'filename':name,'target_intensity_quantile':quantile,'roi':roi,
                    'excluded_by_teacher':fold['name'],'teacher_sha256':fold['sha256'],
                    'dapi_sha256':sha(DATA/'train/DAPI'/name),
                    'reference_sha256':sha(DATA/'train/CD45RO'/name),
                    'cached_prediction_sha256':sha(CACHE/name)})
q.save('qualitative.drawio')
audit={'display':{'window':[0,128],'mapping':'linear, clipped to [0,255], identical for all images',
                  'crop_xyxy':[96,96,160,160],'units':'pixels; physical pixel size not assumed'},
       'selection':'All 2380 training tiles ranked by mean true CD45RO intensity; nearest 25/50/75 percentiles, without prediction inspection.',
       'qualitative_prediction':'PNG teacher cache used by final student; eight-view D4, halo128; not final-student held-out validation or JPEG-scored images.',
       'training_examples':records,
       'deployment_example':{'filename':TEST_NAME,'selection':'Median of all 560 test DAPI mean intensities',
                             'dapi_sha256':sha(DATA/'test/DAPI'/TEST_NAME),
                             'prediction_sha256':hashlib.sha256(prediction_bytes).hexdigest(),'model_sha256':expected['checkpoint_sha256']},
       'mask_illustration':{'seed':2026,'grid':[16,16],'masked_fraction':.5,'scope':'Illustrative mask, not a recorded training minibatch'}}
(OUT/'image_provenance.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Created workflow and qualitative figures from verified data.')

"""Build native, editable draw.io diagrams; export with draw.io Desktop."""
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, indent, ElementTree

OUT = Path(__file__).resolve().parent / 'figures'
OUT.mkdir(exist_ok=True)
ENC, DEC, COND = '#DCECF0', '#F6E6D2', '#E8E2F1'


def math(s):
    return r'\(' + s + r'\)'


class Diagram:
    def __init__(self, name, width, height):
        self.file = Element('mxfile', host='Electron', agent='draw.io Desktop', version='31.7.0')
        page = SubElement(self.file, 'diagram', name=name, id=name)
        model = SubElement(page, 'mxGraphModel', dx=str(width), dy=str(height), grid='1',
                           gridSize='10', guides='1', tooltips='1', connect='1', arrows='1',
                           fold='1', page='0', pageScale='1', pageWidth=str(width),
                           pageHeight=str(height), math='1', shadow='0', background='#FFFFFF')
        self.root = SubElement(model, 'root')
        SubElement(self.root, 'mxCell', id='0')
        SubElement(self.root, 'mxCell', id='1', parent='0')
        self.boxes = {}

    def node(self, key, label, x, y, w, h, fill='#FFFFFF', size=24, style=''):
        base = ('rounded=0;whiteSpace=wrap;html=1;align=center;verticalAlign=middle;'
                'fontFamily=Arial;fontColor=#243442;strokeColor=#7F929C;strokeWidth=1.3;'
                'spacing=4;labelBackgroundColor=none;labelBorderColor=none;')
        c = SubElement(self.root, 'mxCell', id=key, value=label,
                       style=base+f'fillColor={fill};fontSize={size};'+style, vertex='1', parent='1')
        SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as': 'geometry'})
        self.boxes[key] = (x, y, w, h)

    def text(self, key, label, x, y, w, h=35, size=23, bold=False, align='center'):
        self.node(key, label, x, y, w, h, 'none', size,
                  f'strokeColor=none;align={align};fontStyle={1 if bold else 0};')

    def group(self, key, label, x, y, w, h):
        self.node(key, '', x, y, w, h, 'none', style='rounded=1;absoluteArcSize=1;arcSize=20;dashed=1;dashPattern=6 4;strokeColor=#A0A0A0;')
        if label:
            self.text(key+'-title', label, x+10, y-40, w, 32, 24, True, 'left')

    def edge(self, key, source, target, start='s', end='n', points=(), gray=False, dashed=False, arrow=True):
        anchors = {'n': (.5, 0), 's': (.5, 1), 'e': (1, .5), 'w': (0, .5)}
        a, b = anchors[start], anchors[end]
        style = ('edgeStyle=none;rounded=0;html=1;endArrow=block;endFill=1;endSize=9;'
                 'strokeWidth=2;labelBackgroundColor=none;labelBorderColor=none;'
                 f'strokeColor={"#858585" if gray else "#303030"};'
                 f'exitX={a[0]};exitY={a[1]};exitPerimeter=1;entryX={b[0]};entryY={b[1]};entryPerimeter=1;')
        c = SubElement(self.root, 'mxCell', id=key, value='', style=style+('dashed=1;' if dashed else '')+('endArrow=none;' if not arrow else ''),
                       edge='1', parent='1', source=source, target=target)
        g = SubElement(c, 'mxGeometry', relative='1', **{'as': 'geometry'})
        if points:
            arr = SubElement(g, 'Array', **{'as': 'points'})
            for x, y in points:
                SubElement(arr, 'mxPoint', x=str(x), y=str(y))

    def save(self, name):
        indent(self.file)
        ElementTree(self.file).write(OUT / name, encoding='utf-8', xml_declaration=True)


d = Diagram('Architecture', 1220, 810)
d.text('title-a','a  Single-view network',15,5,760,45,30,True,'left')
d.text('title-b','b  Spatial conditioning',825,5,395,45,30,True,'left')
d.group('enc-group','',45,245,220,475)
d.group('dec-group','',495,245,220,355)
d.text('enc-title','Encoder',30,202,112,35,22,True,'left')
d.text('dec-title','Decoder',638,202,125,35,22,True,'left')
d.node('x','DAPI context '+math('x')+'<br>'+math(r'1\times512\times512'),65,110,180,75,'#F2F5F7',22)
d.node('head','Output '+math(r'\hat Y')+'<>'+math(r'4\times256\times256'),515,110,180,75,'#F2F5F7',22)
# Use explicit HTML breaks, retaining editable formulas.
d.root.find("mxCell[@id='head']").set('value','Output '+math(r'\hat Y')+'<br>'+math(r'4\times256\times256'))
for j,y,dim,blocks in [(1,260,r'96\times256\times256',3),(2,380,r'192\times128\times128',3),(3,500,r'384\times64\times64',9),(4,620,r'768\times32\times32',3)]:
 d.node(f'e{j}',math(f'e_{j}')+'  |  '+str(blocks)+' blocks<br>'+math(dim),65,y,180,80,ENC,20)
for j,y,dim in [(1,260,r'16\times256\times256'),(2,380,r'192\times128\times128'),(3,500,r'384\times64\times64')]:
 d.node(f'd{j}',math(f'd_{j}')+'<br>'+math(dim),515,y,180,80,DEC,20)
d.node('film','FiLM<br>'+math(r'd_4=\widetilde e_4'),305,620,180,80,COND,24)
d.edge('stem','x','e1');d.text('stem-l','Stem '+math(r'\to e_0'),240,205,160,30,20,align='left')
for j in [2,3,4]:d.edge(f'down{j}',f'e{j-1}',f'e{j}')
d.edge('enc-film','e4','film','e','w')
d.edge('film-dec','film','d3','e','s',[(605,660)])
for j in [2,1]:d.edge(f'up{j}',f'd{j+1}',f'd{j}','n','s')
for j,y in [(1,260),(2,380),(3,500)]:
 d.edge(f'skip{j}',f'e{j}',f'd{j}','e','w',gray=True)
 d.text(f'skip-l{j}','Skip '+math(f'e_{j}')+' / concatenate',275,y-7,220,32,20)
d.edge('head-edge','d1','head','n','s')
d.text('head-note','PixelShuffle / sigmoid / crop',365,84,460,25,19)
d.node('roi','DAPI ROI '+math('I')+' + tile coordinates',840,110,360,75,'#F2F5F7',23)
d.node('stats','Statistics<br>'+math(r's\in\mathbb R^{101}'),840,250,165,80,COND,22)
d.node('thumb','Thumbnail<br>'+math(r'T:2\times80\times80'),1035,250,165,80,COND,19)
d.node('mlp','MLP<br>'+math(r'a_s\in\mathbb R^{1536}'),840,395,165,80,COND,22)
d.node('cnn','CNN / D4 mean<br>'+math(r'a_T\in\mathbb R^{1536}'),1035,395,165,80,COND,21)
d.edge('roi-stats','roi','stats','s','n',[(1020,215),(922.5,215)])
d.edge('roi-thumb','roi','thumb','s','n',[(1020,215),(1117.5,215)])
d.edge('stats-mlp','stats','mlp');d.edge('thumb-cnn','thumb','cnn')
d.node('affine',math(r'[\gamma_F;\beta_F]=a_s+a_T')+'<br>'+math(r'\gamma_F,\beta_F\in\mathbb R^{768}'),865,565,315,90,COND,24)
d.edge('as-merge','mlp','affine','s','n',[(922.5,525),(1022.5,525)])
d.edge('at-merge','cnn','affine','s','n',[(1117.5,525),(1022.5,525)])
d.edge('condition-film','affine','film','s','s',[(1022.5,790),(395,790)],dashed=True)
d.save('architecture.drawio')

b=Diagram('Building blocks',1640,720)
for key,title,x in [('a','(a) ConvNeXt V2 block',10),('b','(b) Decode stage '+math('j'),555),('c','(c) Bottleneck FiLM',1090)]:
    b.text(key+'-title',title,x,5,535,45,30,True,'left')
b.text('u',math(r'u\in\mathbb R^{C\times H\times W}'),195,50,210,40)
b.node('input-junction','',297,107,6,6,'#303030',style='ellipse;strokeColor=#303030;spacing=0;')
for key,label,y,h,fill in [
 ('dw',math(r'v=\operatorname{DWConv}_{7,1}(u)'),125,55,ENC),
 ('ln',math(r'\operatorname{LN}(v)'),215,55,'#FFFFFF'),
 ('mlp',math(r'a=\phi(W_1\operatorname{LN}(v)+b_1)')+'<br>'+math(r'C\to4C')+'; GELU',305,75,ENC),
 ('grn',math(r'\operatorname{GRN}(a)'),415,55,ENC),
 ('proj',math(r'W_2\operatorname{GRN}(a)+b_2')+'<br>'+math(r'4C\to C'),505,75,ENC)]:
    b.node(key,label,140,y,320,h,fill)
b.node('sum','+',280,625,40,40,style='ellipse;aspect=fixed;')
b.text('bu',math(r'\mathcal B(u)'),205,695,190,35)
b.edge('input-stem','u','input-junction',arrow=False)
for a,c in [('input-junction','dw'),('dw','ln'),('ln','mlp'),('mlp','grn'),('grn','proj'),('proj','sum'),('sum','bu')]:b.edge(a+'-'+c,a,c)
b.edge('residual','input-junction','sum','w','w',[(80,110),(80,645)])
b.text('residual-label',math('u'),125,610,70,30)
b.text('din',math('d_{j+1}'),670,60,300,40)
for key,label,y,h,fill in [
 ('ps',math(r'\operatorname{PS}_2(d_{j+1})'),125,55,DEC),
 ('concat',math(r'v_j=[\operatorname{PS}_2(d_{j+1});e_j]')+'<br>Concatenate',225,75,DEC),
 ('dln',math(r'\operatorname{LN}(v_j)'),345,55,'#FFFFFF'),
 ('project',math(r'P_j\operatorname{LN}(v_j)+b_{D,j}')+'<br>'+math(r'1\times1')+' convolution',440,75,DEC),
 ('blocks',math(r'\mathcal B_{D,j}^{[2]}')+'<br>Two ConvNeXt V2 blocks',560,75,DEC)]:
    b.node(key,label,650,y,340,h,fill)
b.text('dj',math('d_j'),730,690,180,35)
for a,c in [('din','ps'),('ps','concat'),('concat','dln'),('dln','project'),('project','blocks'),('blocks','dj')]:b.edge(a+'-'+c,a,c)
b.text('ej',math('e_j'),540,211,90,35)
b.edge('skip','ej','concat','s','w',[(585,262.5)])
b.node('sm','Statistics MLP<br>'+math(r's\to a_s'),1085,115,235,80,COND)
b.node('tm','Thumbnail encoder<br>'+math(r'T\to\bar z\to a_T'),1400,115,245,80,COND)
b.node('affines',math(r'[\gamma_F;\beta_F]=a_s+a_T'),1155,290,440,65,COND)
b.edge('s-merge','sm','affines','s','n',[(1202.5,240),(1375,240)])
b.edge('t-merge','tm','affines','s','n',[(1522.5,240),(1375,240)])
b.node('film-op',math(r'(1+\gamma_F)\odot e_4+\beta_F')+'<br>Channel-wise affine modulation',1155,490,440,85,COND)
b.edge('broadcast','affines','film-op')
b.text('broadcast-label','Broadcast over '+math('H,W'),1395,405,250,35,22,align='left')
b.text('e4',math('e_4'),1070,415,70,35)
b.edge('feature','e4','film-op','s','w',[(1105,532.5)])
b.text('d4',math(r'\widetilde e_4=d_4'),1210,625,330,45)
b.edge('conditioned','film-op','d4')
b.text('legend',math(r'\odot')+': element-wise multiplication',1090,697,560,35,22)
b.save('blocks.drawio')
print('Created two editable draw.io files in',OUT)

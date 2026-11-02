"""Resolve named PDF links to direct destinations for reader compatibility."""
import json
from collections import Counter
from pathlib import Path

import fitz
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, ArrayObject, FloatObject, NumberObject

ROOT = Path(__file__).resolve().parent
path = ROOT / 'report.pdf'
writer = PdfWriter(clone_from=PdfReader(path))
names = writer.named_destinations
converted = []
colors_by_page = []


def resolve(obj):
    action = obj.get('/A')
    container, key = (obj, '/Dest')
    if action is not None and action.get_object().get('/S') == '/GoTo':
        container, key = action.get_object(), '/D'
    dest = container.get(key)
    if isinstance(dest, str):
        name = str(dest)
        if name not in names:
            name = name.lstrip('/')
        if name not in names:
            raise ValueError(f'Unresolved PDF destination: {dest}')
        container[NameObject(key)] = names[name].dest_array
        converted.append(name)


for page in writer.pages:
    colors = []
    for ref in page.get('/Annots', []):
        annotation = ref.get_object()
        if annotation.get('/Subtype') == '/Link':
            resolve(annotation)
            annotation[NameObject('/Border')] = ArrayObject([NumberObject(0), NumberObject(0), FloatObject(.6)])
            color = tuple(float(value) for value in annotation.get('/C', (1, 0, 0)))
            colors.append(color)
            annotation[NameObject('/H')] = NameObject('/I')
    colors_by_page.append(colors)


def outline(node):
    while node:
        item = node.get_object()
        resolve(item)
        outline(item.get('/First'))
        node = item.get('/Next')


outlines = writer.root_object.get('/Outlines')
if outlines is not None:
    outline(outlines.get_object().get('/First'))
temporary = path.with_name('report.links.pdf')
writer.write(temporary)
with fitz.open(temporary) as doc:
    links = [link for page in doc for link in page.get_links()]
    assert links and all(link['kind'] in (fitz.LINK_GOTO, fitz.LINK_URI) for link in links)
    internal = [link for link in links if link['kind'] == fitz.LINK_GOTO]
    assert internal and all(0 <= link['page'] < len(doc) for link in internal)
    result = {'internal_links': len(internal), 'external_links': len(links)-len(internal),
              'resolved_destinations': sorted(set(converted)), 'border_width': .6,
              'border_colors': {str(color): count for color, count in Counter(color for colors in colors_by_page for color in colors).items()},
              'all_internal_targets_valid': True}
    # Some embedded viewers omit link annotation borders. Draw the same thin,
    # unfilled border in page content so the clickable area remains visible.
    for page, colors in zip(doc, colors_by_page):
        page_links = page.get_links()
        assert len(page_links) == len(colors)
        for link, color in zip(page_links, colors):
            page.draw_rect(link['from'], color=color, width=.5, overlay=True)
    doc.saveIncr()
temporary.replace(path)
(ROOT / 'report_link_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))

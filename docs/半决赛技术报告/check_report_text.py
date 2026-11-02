"""Check report text, references and page bounds without rendering any images."""
import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'report.tex').read_text(encoding='utf-8')
log = (ROOT / 'report.log').read_text(encoding='utf-8')
assert not re.search(r'Overfull|Missing character|undefined|^!', log, re.M)
equations = re.findall(r'\\begin\{equation\}.*?\\end\{equation\}', source, re.S)
assert len(equations) == 19
assert len(re.findall(r'\\end\{equation\}\s*\\sym\{', source)) == len(equations)
labels = re.findall(r'\\label\{([^}]+)\}', source)
refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', source)
assert len(labels) == len(set(labels)) and set(refs) <= set(labels)
bib = set(re.findall(r'\\bibitem\{([^}]+)\}', source))
cites = {key for group in re.findall(r'\\cite\{([^}]+)\}', source) for key in group.split(',')}
assert cites == bib and len(bib) == 11
assert not re.search(r'Round\d+|2026年|不是|并非|不应|不要|无需|不能', source)
assert not re.search(r'仍待|待补充|后续可|需要独立指标|twocolumn|hidelinks', source)
doc = fitz.open(ROOT / 'report.pdf')
pages = []
for number, page in enumerate(doc, 1):
    text = page.get_text()
    assert text.strip() and '\ufffd' not in text
    outside = []
    for word in page.get_text('words'):
        if not page.rect.contains(fitz.Rect(word[:4])):
            outside.append(word[4])
    assert not outside, (number, outside)
    pages.append({'page': number, 'characters': len(text), 'out_of_page_words': len(outside)})
full_text = '\n'.join(page.get_text() for page in doc)
for token in ['通信速通生物', '31,228,917', '74.0677']:
    assert token in full_text, token
assert len(re.findall(r'[\u4e00-\u9fff]', full_text)) >= 2000
result = {'pages': len(doc), 'equations_with_following_definitions': len(equations),
          'references': len(bib), 'cjk_characters': len(re.findall(r'[\u4e00-\u9fff]', full_text)),
          'page_checks': pages, 'image_inspection': False,
          'scope': 'Text, compilation diagnostics and word bounding boxes only; no visual layout verification.'}
(ROOT / 'report_text_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))

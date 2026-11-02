from __future__ import annotations
from data import write_json

def control_record(path, value):
    temporary = path.with_suffix(path.suffix + '.writing')
    write_json(temporary, value)
    temporary.replace(path)

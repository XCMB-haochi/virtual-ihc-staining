"""Extract the user-supplied archive inside the project dataset directory."""
from pathlib import Path
import zipfile

root=Path('dataset').resolve()
archives=list(root.glob('*.zip'))
assert len(archives)==1, archives
with zipfile.ZipFile(archives[0]) as archive:
    for member in archive.infolist():
        target=(root/member.filename).resolve()
        if not target.is_relative_to(root) or member.filename.startswith(('/', '\\')):
            raise ValueError(member.filename)
        if target.exists() and not member.is_dir():
            raise FileExistsError(target)
    archive.extractall(root)
print('Extracted',len(archive.infolist()),'entries',flush=True)

from pathlib import Path


def mkdirs(path):
    if isinstance(path, str):
        p = Path(path)
    else:
        p = path
    if not p.exists():
        p.mkdir(parents=True)
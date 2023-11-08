from pathlib import Path


__all__ = [
    "mkdirs"
]


def mkdirs(path):
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True)
import json
from pathlib import Path

def ensure_dir(path: Path) -> None:
    """Create *path* recursively if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
"""Small persisted settings under app_home() (settings.json). Nothing here touches the DB.
A missing or unreadable file reads as {} so a corrupt settings file never blocks the app."""
from __future__ import annotations
import json
from pathlib import Path
from .config import settings_path

def load() -> dict:
    p = settings_path()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def save(d: dict) -> None:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2))

def get_export_base() -> Path | None:
    """Where exports go instead of export_root(), or None for the default."""
    v = load().get("export_base")
    return Path(v) if isinstance(v, str) and v else None

def set_export_base(p: Path | None) -> None:
    d = load()
    if p is None:
        d.pop("export_base", None)
    else:
        d["export_base"] = str(p)
    save(d)

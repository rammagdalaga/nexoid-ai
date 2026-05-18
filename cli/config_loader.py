import json
from pathlib import Path

DEFAULTS = {
    "theme": "black",
    "api_base_url": "http://localhost:8000",
    "api_key": "",
    "streaming": True,
    "model": "atlas",
    "temperature": 0.7,
    "mode": "chat",
}


def mask_key(key: str) -> str:
    if not key:
        return "<missing>"
    return key[:4] + "...****"


def load_settings(path: str = ".nexoid/settings.json") -> dict:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(DEFAULTS, indent=2))
        cfg = dict(DEFAULTS)
    else:
        cfg = json.loads(p.read_text())
        for k, v in DEFAULTS.items():
            cfg.setdefault(k, v)
    if not cfg.get("api_key"):
        raise ValueError("Missing API key in .nexoid/settings.json")
    return cfg

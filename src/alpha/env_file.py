from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def load_env_file(path: str | Path, override: bool = False) -> Dict[str, str]:
    env_path = Path(path)
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _unquote(value.strip())
        if not key:
            continue
        loaded[key] = value
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return loaded


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value

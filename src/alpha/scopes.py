from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


Scope = Dict[str, Any]


SCOPE_PRESETS: Dict[str, Scope] = {
    "us": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
    "usa": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
    "us-d0": {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
    "usa-d0": {"region": "USA", "universe": "TOP3000", "delay": 0, "neutralization": "INDUSTRY"},
    "us-d1": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
    "usa-d1": {"region": "USA", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
    "glb": {"region": "GLB", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
    "global": {"region": "GLB", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
    "glb-d1": {"region": "GLB", "universe": "TOP3000", "delay": 1, "neutralization": "INDUSTRY"},
    "eur": {"region": "EUR", "universe": "TOP2500", "delay": 1, "neutralization": "INDUSTRY"},
    "eur-d0": {"region": "EUR", "universe": "TOP2500", "delay": 0, "neutralization": "INDUSTRY"},
    "eur-d1": {"region": "EUR", "universe": "TOP2500", "delay": 1, "neutralization": "INDUSTRY"},
    "chn": {"region": "CHN", "universe": "TOP2000U", "delay": 1, "neutralization": "INDUSTRY"},
    "chn-d0": {"region": "CHN", "universe": "TOP2000U", "delay": 0, "neutralization": "INDUSTRY"},
    "chn-d1": {"region": "CHN", "universe": "TOP2000U", "delay": 1, "neutralization": "INDUSTRY"},
    "ind": {"region": "IND", "universe": "TOP500", "delay": 1, "neutralization": "INDUSTRY"},
    "ind-d1": {"region": "IND", "universe": "TOP500", "delay": 1, "neutralization": "INDUSTRY"},
    "asi": {"region": "ASI", "universe": "MINVOL1M", "delay": 1, "neutralization": "INDUSTRY"},
    "asi-d1": {"region": "ASI", "universe": "MINVOL1M", "delay": 1, "neutralization": "INDUSTRY"},
    "kor": {"region": "KOR", "universe": "TOP600", "delay": 1, "neutralization": "INDUSTRY"},
    "kor-d1": {"region": "KOR", "universe": "TOP600", "delay": 1, "neutralization": "INDUSTRY"},
    "hkg": {"region": "HKG", "universe": "TOP800", "delay": 1, "neutralization": "INDUSTRY"},
    "hkg-d1": {"region": "HKG", "universe": "TOP800", "delay": 1, "neutralization": "INDUSTRY"},
    "mea": {"region": "MEA", "universe": "TOP300", "delay": 1, "neutralization": "MARKET"},
    "mea-d1": {"region": "MEA", "universe": "TOP300", "delay": 1, "neutralization": "MARKET"},
}

PLATFORM_SCOPE_OPTIONS: List[Dict[str, Any]] = [
    {
        "region": "USA",
        "delay": 1,
        "universes": ["TOP3000", "TOP2000", "TOP1000", "TOP500", "TOP200", "ILLIQUID_MINVOL1M", "TOPSP500"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "STATISTICAL",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "USA",
        "delay": 0,
        "universes": ["TOP3000", "TOP2000", "TOP1000", "TOP500", "TOP200", "ILLIQUID_MINVOL1M", "TOPSP500"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "STATISTICAL",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "GLB",
        "delay": 1,
        "universes": ["TOP3000", "MINVOL1M", "MINVOL10M", "TOPDIV3000"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "STATISTICAL",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "COUNTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "EUR",
        "delay": 1,
        "universes": ["TOP2500", "TOP1200", "TOP800", "TOP400", "ILLIQUID_MINVOL1M", "TOPCS1600"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "STATISTICAL",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "COUNTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "EUR",
        "delay": 0,
        "universes": ["TOP2500", "TOP1200", "TOP800", "TOP400", "ILLIQUID_MINVOL1M", "TOPCS1600"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "STATISTICAL",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "COUNTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "ASI",
        "delay": 1,
        "universes": ["MINVOL1M", "MINVOL10M", "ILLIQUID_MINVOL1M", "TOP500"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "STATISTICAL",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "COUNTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "CHN",
        "delay": 0,
        "universes": ["TOP2000U"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "CHN",
        "delay": 1,
        "universes": ["TOP2000U"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "KOR",
        "delay": 1,
        "universes": ["TOP600"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "HKG",
        "delay": 1,
        "universes": ["TOP800", "TOP500"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "IND",
        "delay": 1,
        "universes": ["TOP500"],
        "neutralizations": [
            "NONE",
            "REVERSION_AND_MOMENTUM",
            "CROWDING",
            "FAST",
            "SLOW",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "SLOW_AND_FAST",
        ],
    },
    {
        "region": "MEA",
        "delay": 1,
        "universes": ["TOP300"],
        "neutralizations": [
            "NONE",
            "MARKET",
            "SECTOR",
            "INDUSTRY",
            "SUBINDUSTRY",
            "COUNTRY",
        ],
    },
]


_UPPERCASE_KEYS = {"region", "universe", "neutralization"}


def apply_scope(base: Scope, preset: str | None = None, overrides: Scope | None = None) -> Scope:
    context = dict(base)
    if preset:
        try:
            context.update(SCOPE_PRESETS[preset])
        except KeyError as exc:
            raise ValueError(f"unknown preset: {preset}") from exc

    for key, value in (overrides or {}).items():
        if value is not None:
            context[key] = value

    for key in _UPPERCASE_KEYS:
        value = context.get(key)
        if isinstance(value, str):
            context[key] = value.strip().upper()

    return context


def preset_rows() -> Iterable[Tuple[str, Scope]]:
    for name in sorted(SCOPE_PRESETS):
        yield name, SCOPE_PRESETS[name]


def platform_scope_rows() -> Iterable[Dict[str, Any]]:
    for row in PLATFORM_SCOPE_OPTIONS:
        yield {
            "region": row["region"],
            "delay": row["delay"],
            "universes": list(row["universes"]),
            "neutralizations": list(row["neutralizations"]),
        }

"""Helpers for loading the repository's presets in tests.

The engine itself is path-only: the CLI and `load_session` take explicit
paths. Tests exercise the presets shipped in this repository, so the
repo-relative lookup lives here, not in the package.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from farfield.session import Session, load_session

PRESET_DIR = Path(__file__).resolve().parent.parent / "presets"


def preset_path(name: str) -> Path:
    return PRESET_DIR / f"{name}.yaml"


def load_preset(name: str, total_duration_s: float | None = None) -> Session:
    return load_session(preset_path(name), total_duration_s=total_duration_s)


def preset_names() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.yaml"))


_TIER_ORDER = {"measured-tape": 0, "measured-mss": 1, "patent": 2, "original": 3}


def preset_metadata() -> list[dict]:
    entries = []
    for name in preset_names():
        data = yaml.safe_load(preset_path(name).read_text())
        entries.append(
            {
                "name": name,
                "title": str(data.get("title", name)),
                "fidelity": str(data.get("fidelity", "original")),
                "notes": str(data.get("notes", "")).strip(),
            }
        )
    entries.sort(key=lambda e: (_TIER_ORDER.get(e["fidelity"], 9), e["name"]))
    return entries

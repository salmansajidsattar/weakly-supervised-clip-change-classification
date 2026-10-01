"""YAML config loading + light validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        if name in self.raw:
            v = self.raw[name]
            return Config(v) if isinstance(v, dict) else v
        raise AttributeError(name)

    def get(self, name: str, default: Any = None) -> Any:
        return self.raw.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return self.raw


def load_config(path: str | Path) -> Config:
    """Load a YAML config and return as a dot-accessible Config object."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw)

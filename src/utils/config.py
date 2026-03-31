"""
src/utils/config.py
───────────────────
Loads and exposes the central config.yaml.
Usage:
    from src.utils.config import cfg
    data_path = cfg.paths.data.stock
"""

import yaml
import os
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_config(config_path: str = None) -> SimpleNamespace:
    """
    Load config.yaml and return as dot-accessible namespace.

    Automatically finds config.yaml by walking up from the current file
    until it finds the project root (where config.yaml lives).
    """
    if config_path is None:
        # Walk up from src/utils/ to find project root
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break
        if config_path is None:
            raise FileNotFoundError(
                "config.yaml not found. Make sure you run from the project root."
            )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return _dict_to_namespace(raw)


# ── Singleton: import `cfg` anywhere ──────────────────────────
cfg = load_config()


if __name__ == "__main__":
    print("Config loaded successfully!")
    print(f"  Project : {cfg.project.name}")
    print(f"  Stock data: {cfg.paths.data.stock}")
    print(f"  Horizons: {cfg.timemixer.horizons}")
    print(f"  Primary vol estimator: {cfg.data.volatility.primary_estimator}")

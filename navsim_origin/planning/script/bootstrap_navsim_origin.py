import importlib
import os
import sys
from pathlib import Path


def force_navsim_origin_alias() -> None:
    """
    Route `import navsim` to `navsim_origin` package in this repo.

    This keeps existing absolute imports (`navsim.*`) working while forcing
    runtime to use navsim_origin modules.
    """
    if os.environ.get("DRIVOR_FORCE_NAVSIM_ORIGIN", "1") != "1":
        return

    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    navsim_origin_pkg = importlib.import_module("navsim_origin")
    sys.modules["navsim"] = navsim_origin_pkg

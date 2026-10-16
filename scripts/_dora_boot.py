"""Load a perception_models module from this src tree (ignore install site-packages)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_main(module_filename: str):
    root = Path(__file__).resolve().parent.parent  # .../perception_models
    pkg_dir = root / "perception_models"
    # Prefer this tree for all perception_models.* imports the module triggers.
    root_s = str(root)
    if root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)

    # Drop a stale installed perception_models if already imported.
    for key in list(sys.modules):
        if key == "perception_models" or key.startswith("perception_models."):
            del sys.modules[key]

    path = pkg_dir / module_filename
    name = f"_hound_src_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.main

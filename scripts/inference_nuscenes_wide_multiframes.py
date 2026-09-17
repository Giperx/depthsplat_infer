#!/usr/bin/env python3
"""nuScenes multi-frame wide-view inference (thin per-dataset entry point).

Pins ``--dataset nuscenes`` on the shared ``wide_inference_multiframes_core``; ``--dataset`` can
still be passed to override the preset explicitly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

DEFAULT_DATASET = "nuscenes"
_CORE_NAME = "wide_inference_multiframes_core"
SCRIPT_DIR = Path(__file__).resolve().parent


def _load_core():
    """Load the shared core by path (works without torch; no package import)."""
    if _CORE_NAME in sys.modules:
        return sys.modules[_CORE_NAME]
    core_path = SCRIPT_DIR / f"{_CORE_NAME}.py"
    spec = importlib.util.spec_from_file_location(_CORE_NAME, core_path)
    if spec is None or spec.loader is None:  # pragma: no cover - path dependent
        raise ImportError(f"Could not load core from {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CORE_NAME] = module
    spec.loader.exec_module(module)
    return module


core = _load_core()


def main(argv=None):
    """Run multi-frame wide-view inference pinned to ``DEFAULT_DATASET``."""
    return core.main(argv, default_dataset=DEFAULT_DATASET)


if __name__ == "__main__":
    sys.exit(main())

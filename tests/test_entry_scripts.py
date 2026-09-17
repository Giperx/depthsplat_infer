"""Tests for the thin per-dataset entry-point wrappers.

Run from the repository root with::

    python -m unittest tests.test_entry_scripts

Each wrapper pins a dataset on a shared core (``wide_inference_core``,
``wide_inference_multiframes_core`` or ``benchmark_wide_core``) and delegates to
``core.main(default_dataset=...)``.  These tests load each wrapper by path and
check its metadata, then exercise ``--help`` through a subprocess (which must
work without torch/CUDA).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

# filename -> (pinned dataset, family)
SINGLE = "single"
MULTI = "multi"
BENCH = "benchmark"

WRAPPERS = {
    "inference_nuscenes_wide.py": ("nuscenes", SINGLE),
    "inference_lyft1920_wide.py": ("lyft1920", SINGLE),
    "inference_lyft1224_wide.py": ("lyft1224", SINGLE),
    "inference_ddad_wide.py": ("ddad", SINGLE),
    "inference_nuscenes_wide_multiframes.py": ("nuscenes", MULTI),
    "inference_lyft1920_wide_multiframes.py": ("lyft1920", MULTI),
    "inference_lyft1224_wide_multiframes.py": ("lyft1224", MULTI),
    "inference_ddad_wide_multiframes.py": ("ddad", MULTI),
    "benchmark_nuscenes_wide.py": ("nuscenes", BENCH),
    "benchmark_lyft1920_wide.py": ("lyft1920", BENCH),
    "benchmark_lyft1224_wide.py": ("lyft1224", BENCH),
    "benchmark_ddad_wide.py": ("ddad", BENCH),
}

CORE_FOR_FAMILY = {
    SINGLE: "wide_inference_core",
    MULTI: "wide_inference_multiframes_core",
    BENCH: "benchmark_wide_core",
}


def _load_wrapper(path: Path):
    """Load a wrapper by path, registering it before execution."""
    name = path.stem
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - path dependent
        raise RuntimeError(f"Could not load wrapper from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class WrapperMetadataTest(unittest.TestCase):
    """Every wrapper exposes its pinned dataset and a callable main."""

    def test_all_twelve_wrappers_present(self):
        self.assertEqual(len(WRAPPERS), 12)
        for filename in WRAPPERS:
            with self.subTest(wrapper=filename):
                self.assertTrue((SCRIPTS / filename).is_file(), filename)

    def test_default_dataset_matches_filename(self):
        for filename, (dataset, _) in WRAPPERS.items():
            with self.subTest(wrapper=filename):
                module = _load_wrapper(SCRIPTS / filename)
                self.assertTrue(hasattr(module, "DEFAULT_DATASET"))
                self.assertEqual(module.DEFAULT_DATASET, dataset)
                self.assertTrue(callable(module.main))
                # The pinned dataset name appears in the wrapper filename.
                self.assertIn(dataset, filename)

    def test_wrapper_delegates_to_matching_core(self):
        for filename, (_, family) in WRAPPERS.items():
            with self.subTest(wrapper=filename):
                module = _load_wrapper(SCRIPTS / filename)
                self.assertEqual(module._CORE_NAME, CORE_FOR_FAMILY[family])
                self.assertEqual(module.core.__name__, CORE_FOR_FAMILY[family])


@unittest.skipUnless(hasattr(subprocess, "run"), "subprocess required")
class WrapperHelpTest(unittest.TestCase):
    """``--help`` works for every wrapper and reports its pinned dataset."""

    def test_help_for_all_wrappers(self):
        for filename, (dataset, _) in WRAPPERS.items():
            with self.subTest(wrapper=filename):
                result = subprocess.run(
                    [sys.executable, str(SCRIPTS / filename), "--help"],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"{filename} --help failed:\n{result.stderr}",
                )
                output = result.stdout + result.stderr
                # ArgumentDefaultsHelpFormatter prints the pinned --dataset default.
                self.assertIn(dataset, output)


if __name__ == "__main__":
    unittest.main()

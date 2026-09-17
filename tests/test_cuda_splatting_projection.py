"""Pure (CPU) tests for the normalized-intrinsics projection in cuda_splatting.

``cuda_splatting`` imports the CUDA Gaussian rasterizer at module import time, so
a minimal stub is installed before importing it.  This lets the projection
helpers -- the only part exercised here -- run without a GPU build::

    python -m unittest tests.test_cuda_splatting_projection

Run these with an environment that has ``torch``, ``einops`` and ``jaxtyping``
(for example the project's ``depthsplat`` environment).
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The pure wide-view helpers now live in the shared core; the old
# ``inference_nuscenes_wide.py`` path is a thin per-dataset wrapper.
SCRIPT_PATH = REPO_ROOT / "scripts" / "wide_inference_core.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# cuda_splatting imports the rasterizer extension at module import time; stub it
# so the pure projection math can be imported on CPU-only machines.
if "diff_gaussian_rasterization" not in sys.modules:
    _rasterizer_stub = types.ModuleType("diff_gaussian_rasterization")

    class _GaussianRasterizationSettings:  # pragma: no cover - stub
        pass

    class _GaussianRasterizer:  # pragma: no cover - stub
        pass

    _rasterizer_stub.GaussianRasterizationSettings = _GaussianRasterizationSettings
    _rasterizer_stub.GaussianRasterizer = _GaussianRasterizer
    sys.modules["diff_gaussian_rasterization"] = _rasterizer_stub

try:
    import torch

    from src.geometry.projection import get_fov
    from src.model.decoder import cuda_splatting

    _HAVE_DEPS = True
except ImportError:  # pragma: no cover - environment dependent
    _HAVE_DEPS = False


def _load_wide_module():
    spec = importlib.util.spec_from_file_location(
        "inference_nuscenes_wide_projection", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(_HAVE_DEPS, "torch/einops/jaxtyping are required")
class CenteredCompatibilityTest(unittest.TestCase):
    """A centred principal point must reproduce the legacy symmetric frustum."""

    def test_centered_matches_symmetric_projection(self):
        h, w = 448, 768
        fx_px = 400.0
        intrinsics = torch.tensor(
            [
                [
                    [fx_px / w, 0.0, 0.5],
                    [0.0, fx_px / h, 0.5],
                    [0.0, 0.0, 1.0],
                ]
            ]
        )
        near = torch.tensor([0.5])
        far = torch.tensor([200.0])

        matrix, tan_x, tan_y = cuda_splatting.get_projection_matrix_from_intrinsics(
            near, far, intrinsics, (h, w)
        )
        fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
        legacy = cuda_splatting.get_projection_matrix(near, far, fov_x, fov_y)

        torch.testing.assert_close(matrix, legacy, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(tan_x, (0.5 * fov_x).tan(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(tan_y, (0.5 * fov_y).tan(), rtol=1e-5, atol=1e-6)


@unittest.skipUnless(_HAVE_DEPS, "torch/einops/jaxtyping are required")
class AsymmetricIntrinsicsTest(unittest.TestCase):
    """Deliberately off-centre cx/cy must map to the requested pixel coordinates."""

    H, W = 480, 640
    FX, FY = 500.0, 510.0
    CX, CY = 300.0, 260.0

    def _project(self, x, y, z, near=0.5, far=200.0):
        intrinsics = torch.tensor(
            [
                [
                    [self.FX / self.W, 0.0, self.CX / self.W],
                    [0.0, self.FY / self.H, self.CY / self.H],
                    [0.0, 0.0, 1.0],
                ]
            ]
        )
        matrix, tan_x, tan_y = cuda_splatting.get_projection_matrix_from_intrinsics(
            torch.tensor([near]), torch.tensor([far]), intrinsics, (self.H, self.W)
        )
        clip = matrix[0] @ torch.tensor([x, y, z, 1.0])
        ndc = clip[:2] / clip[3]
        return ndc, tan_x[0], tan_y[0]

    def test_ndc_honours_principal_point(self):
        for x, y, z in [(0.1, -0.05, 2.0), (-0.3, 0.2, 1.5), (0.0, 0.0, 3.0)]:
            ndc, _, _ = self._project(x, y, z)
            expected_x = 2.0 * (self.FX * x / z + self.CX) / self.W - 1.0
            expected_y = 2.0 * (self.FY * y / z + self.CY) / self.H - 1.0
            self.assertAlmostEqual(ndc[0].item(), expected_x, places=5)
            self.assertAlmostEqual(ndc[1].item(), expected_y, places=5)

    def test_tan_fov_matches_pixel_focal_length(self):
        _, tan_x, tan_y = self._project(0.0, 0.0, 1.0)
        self.assertAlmostEqual(tan_x.item(), 0.5 / (self.FX / self.W), places=6)
        self.assertAlmostEqual(tan_y.item(), 0.5 / (self.FY / self.H), places=6)

    def test_rasterizer_pixel_coordinate(self):
        # The rasterizer's ndc2Pix: pix = ((ndc + 1) * size - 1) / 2.
        x, y, z = 0.2, 0.1, 2.0
        ndc, _, _ = self._project(x, y, z)
        u = ((ndc[0].item() + 1.0) * self.W - 1.0) * 0.5
        v = ((ndc[1].item() + 1.0) * self.H - 1.0) * 0.5
        self.assertAlmostEqual(u, self.FX * x / z + self.CX - 0.5, places=4)
        self.assertAlmostEqual(v, self.FY * y / z + self.CY - 0.5, places=4)


@unittest.skipUnless(_HAVE_DEPS, "torch/einops/jaxtyping are required")
class MakeWideProjectionTest(unittest.TestCase):
    """The wide-render intrinsics must be projected with the preserved cy."""

    CAM5 = (402.792, 402.792, 398.792, 239.752)  # fx, fy, cx, cy

    def test_wide_vertical_principal_point_is_honoured(self):
        wide = _load_wide_module()
        h, w = 448, 768
        render_px = wide.PixelIntrinsics(*self.CAM5)
        wide_px, (out_h, out_w) = wide.make_wide_intrinsics(render_px, (h, w), 2.0)

        # cx is recentred by design, cy is preserved.
        self.assertAlmostEqual(wide_px.cx, out_w / 2.0, places=6)
        self.assertAlmostEqual(wide_px.cy, render_px.cy, places=6)

        normalized = wide.pixel_to_normalized_intrinsics(wide_px, out_h, out_w)
        matrix, _, _ = cuda_splatting.get_projection_matrix_from_intrinsics(
            torch.tensor([0.5]),
            torch.tensor([200.0]),
            torch.tensor(normalized[None], dtype=torch.float32),
            (out_h, out_w),
        )

        x, y, z = 0.0, 0.05, 2.0
        clip = matrix[0] @ torch.tensor([x, y, z, 1.0])
        ndc_y = (clip[1] / clip[3]).item()
        expected = 2.0 * (wide_px.fy * y / z + wide_px.cy) / out_h - 1.0
        self.assertAlmostEqual(ndc_y, expected, places=5)

        # A centred-cy assumption would be measurably wrong.
        centred = 2.0 * (wide_px.fy * y / z + out_h / 2.0) / out_h - 1.0
        self.assertGreater(abs(ndc_y - centred), 0.01)


if __name__ == "__main__":
    unittest.main()

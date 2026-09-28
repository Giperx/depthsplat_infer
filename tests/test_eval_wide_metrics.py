"""Tests for the wide-image metrics script (no CUDA, no LPIPS).

Run from the repository root with::

    python -m unittest tests.test_eval_wide_metrics
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "eval_wide_metrics.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_wide_metrics", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


metrics = _load_module()


class PanelSlicesTest(unittest.TestCase):
    def test_width_1554_has_panel_width_518(self):
        left, center, right = metrics.panel_slices(1554)
        self.assertEqual(left, slice(0, 518))
        self.assertEqual(center, slice(518, 1036))
        self.assertEqual(right, slice(1036, 1554))

    def test_span_is_three_panels(self):
        left, center, right = metrics.panel_slices(1554)
        self.assertEqual(left.stop - left.start, 518)
        self.assertEqual(center.stop - center.start, 518)
        self.assertEqual(right.stop - right.start, 518)


class SparseSsimTest(unittest.TestCase):
    def test_identical_images_score_one(self):
        rng = np.random.default_rng(0)
        image = rng.random((8, 8, 3))
        mask = np.ones((8, 8), dtype=bool)
        self.assertAlmostEqual(metrics.sparse_ssim(image, image, mask), 1.0, places=6)

    def test_empty_mask_scores_zero(self):
        image = np.ones((4, 4, 3))
        self.assertEqual(metrics.sparse_ssim(image, image, np.zeros((4, 4), bool)), 0.0)


class MetricsNpTest(unittest.TestCase):
    def test_constant_error(self):
        render = np.zeros((4, 4, 3))
        gt = np.full((4, 4, 3), 0.5)
        mask = np.ones((4, 4), dtype=bool)
        result = metrics.compute_metrics_np(render, gt, mask)
        self.assertAlmostEqual(result["mae"], 0.5)
        self.assertAlmostEqual(result["rmse"], 0.5)
        # mse = 0.25 -> PSNR = 10*log10(1/0.25) = 10*log10(4).
        self.assertAlmostEqual(result["psnr"], 10.0 * np.log10(4.0), places=6)
        self.assertEqual(result["n_pixels"], 16)

    def test_empty_mask_returns_none(self):
        render = np.zeros((4, 4, 3))
        self.assertIsNone(
            metrics.compute_metrics_np(render, render, np.zeros((4, 4), bool))
        )

    def test_perfect_match_is_infinite_psnr(self):
        render = np.full((2, 2, 3), 0.25)
        result = metrics.compute_metrics_np(render, render, np.ones((2, 2), bool))
        self.assertEqual(result["psnr"], float("inf"))


class RegionMaskTest(unittest.TestCase):
    def test_left_right_use_gt_only_center_intersects_render(self):
        # Panels for width 9 are [0:3], [3:6], [6:9].
        gt_mask = np.ones((4, 9), dtype=bool)
        render_valid = np.zeros((4, 9), dtype=bool)
        render_valid[:, 0:3] = True  # only the left panel is rendered
        masks = metrics.build_region_masks(gt_mask, render_valid, None)
        # L/R are GT only: a partial render mask must not shrink them.
        self.assertTrue(masks["Left"].all())
        self.assertTrue(masks["Right"].all())
        # Center is intersected with the render-valid mask -> empty here.
        self.assertFalse(masks["Center"].any())

    def test_center_masked_intersects_car_keep(self):
        gt_mask = np.ones((4, 9), dtype=bool)
        render_valid = np.ones((4, 9), dtype=bool)
        car_keep = np.zeros((4, 3), dtype=bool)
        car_keep[:, 1:] = True
        masks = metrics.build_region_masks(gt_mask, render_valid, car_keep)
        self.assertTrue(masks["Center"].all())
        self.assertFalse(masks["Center_masked"][:, 0].any())
        self.assertTrue(masks["Center_masked"][:, 1:].all())


class FramePairingTest(unittest.TestCase):
    def test_pairs_wide_render_and_ignores_fixedfov(self):
        with tempfile.TemporaryDirectory() as tmp:
            render_rgb = Path(tmp) / "render" / "rgb"
            gt_rgb = Path(tmp) / "gt" / "rgb"
            render_rgb.mkdir(parents=True)
            gt_rgb.mkdir(parents=True)
            (render_rgb / "006_5_wide.jpg").write_bytes(b"")
            (render_rgb / "006_fixedfov_wide.jpg").write_bytes(b"")
            (render_rgb / "007_5_wide.jpg").write_bytes(b"")
            (gt_rgb / "006_5_sparse_wide.png").write_bytes(b"")
            # Frame 007 has no GT: it must not be paired.

            pairs = metrics.pair_scene_frames(render_rgb, gt_rgb)
            self.assertEqual(len(pairs), 1)
            frame, render_path, gt_path = pairs[0]
            self.assertEqual(frame, "006")
            self.assertTrue(render_path.name == "006_5_wide.jpg")
            self.assertEqual(gt_path.name, "006_5_sparse_wide.png")

    def test_pairs_do_not_require_zero_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            render_rgb = Path(tmp) / "render" / "rgb"
            gt_rgb = Path(tmp) / "gt" / "rgb"
            render_rgb.mkdir(parents=True)
            gt_rgb.mkdir(parents=True)
            (render_rgb / "042_5_wide.jpg").write_bytes(b"")
            (gt_rgb / "042_5_sparse_wide.png").write_bytes(b"")
            pairs = metrics.pair_scene_frames(render_rgb, gt_rgb)
            self.assertEqual([p[0] for p in pairs], ["042"])


class ResizeDecisionTest(unittest.TestCase):
    def test_smaller_render_is_resized_to_gt_size(self):
        render = np.zeros((2, 3, 3), dtype=np.float64)
        resized, changed = metrics.resize_if_needed(render, (4, 8), is_mask=False)
        self.assertTrue(changed)
        self.assertEqual(resized.shape[:2], (4, 8))
        self.assertEqual(resized.shape[2], 3)

    def test_gt_size_array_is_unchanged(self):
        gt = np.zeros((4, 8, 3), dtype=np.float64)
        same, changed = metrics.resize_if_needed(gt, (4, 8), is_mask=False)
        self.assertFalse(changed)
        self.assertEqual(same.shape[:2], (4, 8))

    def test_mask_resize_is_nearest_bool(self):
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        resized, changed = metrics.resize_if_needed(mask, (4, 4), is_mask=True)
        self.assertTrue(changed)
        self.assertEqual(resized.dtype, bool)
        self.assertEqual(resized.shape, (4, 4))


class GtRootTest(unittest.TestCase):
    def test_missing_gt_root_names_path_and_flag(self):
        with self.assertRaises(SystemExit) as ctx:
            metrics.resolve_gt_root("/no/such/sparse/gt", "nuscenes")
        message = str(ctx.exception)
        self.assertIn("/no/such/sparse/gt", message)
        self.assertIn("--gt-root", message)

    def test_nuscenes_default_path_is_documented(self):
        cfg = metrics.DATASETS["nuscenes"]
        self.assertEqual(
            cfg.gt_root, "datasets/nuscenes/sparseWideFOVImages3_1344x256"
        )
        self.assertEqual(cfg.expected_hw, (256, 1344))

    def test_all_dataset_gt_roots_present_in_table(self):
        self.assertEqual(
            set(metrics.DATASETS), {"nuscenes", "lyft1920", "lyft1224", "ddad"}
        )
        self.assertEqual(
            metrics.DATASETS["ddad"].gt_root,
            "datasets/ddad/sparseWideFOVImages3_1344x256",
        )
        self.assertEqual(
            metrics.DATASETS["lyft1920"].gt_root,
            "datasets/lyft/1920_sparseWideFOVImages3_1344x256",
        )
        self.assertEqual(
            metrics.DATASETS["lyft1224"].gt_root,
            "datasets/lyft/1224_sparseWideFOVImages3_1344x256",
        )
        for cfg in metrics.DATASETS.values():
            self.assertEqual(cfg.expected_hw, (256, 1344))


class EqualWeightMeanTest(unittest.TestCase):
    def test_center_masked_mean_uses_center_masked_not_center(self):
        def sample(mae):
            return {"mae": mae, "rmse": mae, "psnr": 10.0, "ssim": 0.5, "n_pixels": 1}

        all_metrics = {
            "Left": [sample(0.1)],
            "Center": [sample(0.4)],
            "Center_masked": [sample(0.2)],
            "Right": [sample(0.3)],
        }
        report = metrics.format_report(
            "lyft1920",
            Path("renders"),
            Path("gt"),
            Path("list.txt"),
            all_metrics,
            {},
            {},
            1,
            0.0,
        )
        self.assertIn("Mean L/Center/R (equal weight):", report)
        self.assertIn("Mean L/Center_masked/R (equal weight):", report)
        # (0.1+0.4+0.3)/3 = 0.2666... -> 68.00 on the 0-255 scale
        # (0.1+0.2+0.3)/3 = 0.2 -> 51.00
        center_block = report.split("Mean L/Center/R (equal weight):", 1)[1]
        center_mae = center_block.split("Mean L/Center_masked/R", 1)[0]
        masked_block = report.split("Mean L/Center_masked/R (equal weight):", 1)[1]
        self.assertIn("MAE  : 68.00", center_mae)
        self.assertIn("MAE  : 51.00", masked_block)


class DenseSsimTest(unittest.TestCase):
    def test_identical_center_scores_one(self):
        rng = np.random.default_rng(1)
        image = rng.random((16, 16, 3))
        ssim_map = metrics.dense_ssim_map(image, image)
        self.assertAlmostEqual(float(ssim_map.mean()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

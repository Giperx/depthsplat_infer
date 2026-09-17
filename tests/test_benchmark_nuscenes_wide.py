"""Focused tests for the nuScenes wide-view speed benchmark helpers.

Run from the repository root with::

    python -m unittest tests.test_benchmark_nuscenes_wide

These tests exercise the pure helpers (statistics summarization, sample
selection, mode/model/num-frames/local-mv-match validation and parser defaults)
and deliberately avoid CUDA and torch so they run in any environment that can
import the inference scripts.  Loading the benchmark module also loads the two
inference modules by path, which is torch-free at import time.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "benchmark_nuscenes_wide.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_nuscenes_wide", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so dataclasses can resolve the module in 3.10.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_module()


class ParserDefaultsTest(unittest.TestCase):
    """CLI defaults must mirror the inference scripts."""

    def setUp(self):
        self.args = bench.build_arg_parser().parse_args([])
        # Dataset-derived fields default to None and are filled from the
        # selected --dataset preset (nuscenes by default).
        bench.resolve_dataset_defaults(self.args)

    def test_mode_and_benchmark_defaults(self):
        self.assertEqual(self.args.mode, "multi")
        self.assertEqual(bench.DEFAULT_MODE, "multi")
        self.assertEqual(self.args.warmup, 10)
        self.assertEqual(self.args.measure, 50)
        self.assertIsNone(self.args.json)

    def test_model_and_input_defaults(self):
        self.assertEqual(self.args.model, "256x448")
        self.assertEqual(bench.DEFAULT_PRESET, "256x448")
        self.assertIsNone(self.args.input_size)
        self.assertIsNone(self.args.height)
        self.assertIsNone(self.args.width)
        self.assertIsNone(self.args.checkpoint)
        self.assertIsNone(self.args.gaussian_scale_max)

    def test_multi_frame_defaults(self):
        self.assertEqual(self.args.num_frames, 3)
        self.assertEqual(bench.DEFAULT_NUM_FRAMES, 3)
        self.assertIsNone(self.args.local_mv_match)

    def test_camera_and_render_defaults(self):
        self.assertEqual(self.args.cameras, "5,4,3")
        self.assertEqual(self.args.render_camera, 5)
        self.assertEqual(self.args.width_factor, 3.0)
        self.assertEqual(self.args.near, 0.5)
        self.assertEqual(self.args.far, 200.0)

    def test_scene_selection_defaults(self):
        self.assertIsNone(self.args.scene)
        self.assertIsNone(self.args.frame)
        self.assertEqual(
            self.args.scene_list,
            "datasets/nuscenes/processed_10Hz/trainval2/nuScenes_Val2.txt",
        )

    def test_car_mask_defaults_enabled(self):
        self.assertEqual(
            self.args.car_mask_root,
            "datasets/nuscenes/processed_10Hz/nuscenes_mask",
        )
        self.assertFalse(self.args.disable_car_mask)
        self.assertFalse(self.args.mask_render_view)

    def test_runtime_defaults(self):
        self.assertIsNone(self.args.device)
        self.assertFalse(self.args.amp)

    def test_dinov2_default_is_exposed(self):
        # Environment dependent value; the attribute must exist.
        self.assertTrue(hasattr(self.args, "dinov2_source"))

    def test_parser_accepts_overrides(self):
        args = bench.build_arg_parser().parse_args(
            [
                "--mode",
                "single",
                "--model",
                "448x768",
                "--input-size",
                "448x768",
                "--scene",
                "037",
                "--frame",
                "0",
                "--cameras",
                "5,4,3",
                "--render-camera",
                "4",
                "--width-factor",
                "3",
                "--near",
                "1",
                "--far",
                "100",
                "--warmup",
                "3",
                "--measure",
                "7",
                "--height",
                "256",
                "--width",
                "512",
                "--gaussian-scale-max",
                "1.5",
                "--disable-car-mask",
                "--amp",
                "--json",
                "out.json",
            ]
        )
        self.assertEqual(args.mode, "single")
        self.assertEqual(args.model, "448x768")
        self.assertEqual(args.input_size, (448, 768))
        self.assertEqual(args.scene, "037")
        self.assertEqual(args.frame, "0")
        self.assertEqual(args.render_camera, 4)
        self.assertEqual(args.width_factor, 3.0)
        self.assertEqual(args.near, 1.0)
        self.assertEqual(args.far, 100.0)
        self.assertEqual(args.warmup, 3)
        self.assertEqual(args.measure, 7)
        self.assertEqual(args.height, 256)
        self.assertEqual(args.width, 512)
        self.assertEqual(args.gaussian_scale_max, 1.5)
        self.assertTrue(args.disable_car_mask)
        self.assertTrue(args.amp)
        self.assertEqual(args.json, "out.json")

    def test_parser_rejects_unknown_mode_and_model(self):
        with self.assertRaises(SystemExit):
            bench.build_arg_parser().parse_args(["--mode", "triple"])
        with self.assertRaises(SystemExit):
            bench.build_arg_parser().parse_args(["--model", "1024x2048"])

    def test_dataset_default_is_nuscenes(self):
        args = bench.build_arg_parser().parse_args([])
        self.assertEqual(args.dataset, "nuscenes")
        self.assertEqual(bench.DEFAULT_DATASET, "nuscenes")
        bench.resolve_dataset_defaults(args)
        self.assertEqual(args.data_root, str(bench.DEFAULT_DATA_ROOT))
        self.assertEqual(
            args.scene_list, str(bench.DEFAULT_SCENE_LIST_PATH)
        )

    def test_dataset_overrides_defaults(self):
        args = bench.build_arg_parser().parse_args(["--dataset", "lyft1920"])
        bench.resolve_dataset_defaults(args)
        self.assertEqual(args.data_root, "datasets/lyft/lyft_val1920_3cams")
        self.assertEqual(
            args.scene_list,
            "datasets/lyft/lyft_val1920_3cams/lyft_val1920.txt",
        )
        self.assertEqual(
            args.car_mask_root,
            "datasets/lyft/lyft_val1920_3cams/ego_car_masks",
        )
        self.assertEqual(args.cameras, "5,4,3")
        self.assertEqual(args.render_camera, 5)

    def test_ddad_dataset_defaults(self):
        args = bench.build_arg_parser().parse_args(["--dataset", "ddad"])
        bench.resolve_dataset_defaults(args)
        self.assertEqual(args.data_root, "datasets/ddad/valid")
        self.assertEqual(args.car_mask_root, "datasets/ddad/valid")

    def test_parser_rejects_unknown_dataset(self):
        with self.assertRaises(SystemExit):
            bench.build_arg_parser().parse_args(["--dataset", "kitti"])

    def test_default_width_factor_is_three(self):
        self.assertEqual(self.args.width_factor, 3.0)
        self.assertEqual(bench.DEFAULT_WIDTH_FACTOR, 3.0)

    def test_dataset_derived_input_resolution(self):
        model = bench.wide.MODEL_PRESETS["256x448"]
        for dataset, expected in (("ddad", (256, 448)), ("lyft1224", (384, 448))):
            with self.subTest(dataset=dataset):
                args = bench.build_arg_parser().parse_args(["--dataset", dataset])
                preset = bench.resolve_dataset_defaults(args)
                self.assertEqual(
                    bench.wide.resolve_dataset_input_hw(args, model, preset, 64),
                    expected,
                )

    def test_resolution_is_a_model_alias(self):
        args = bench.build_arg_parser().parse_args(["--resolution", "448x768"])
        self.assertEqual(args.model, "448x768")


class ModeModelValidationTest(unittest.TestCase):
    def test_valid_combinations(self):
        for mode in ("single", "multi"):
            for model in ("256x448", "448x768"):
                self.assertEqual(
                    bench.validate_mode_and_model(mode, model), (mode, model)
                )

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            bench.validate_mode_and_model("triple", "256x448")

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError):
            bench.validate_mode_and_model("single", "1024x2048")


class NumFramesTest(unittest.TestCase):
    def test_single_mode_is_one_frame(self):
        self.assertEqual(bench.resolve_num_frames("single", 99), 1)

    def test_multi_mode_uses_requested(self):
        self.assertEqual(bench.resolve_num_frames("multi", 3), 3)
        self.assertEqual(bench.resolve_num_frames("multi", 1), 1)

    def test_multi_mode_rejects_zero_or_negative(self):
        with self.assertRaises(ValueError):
            bench.resolve_num_frames("multi", 0)
        with self.assertRaises(ValueError):
            bench.resolve_num_frames("multi", -2)


class LocalMvMatchTest(unittest.TestCase):
    def test_single_mode_default_is_none(self):
        args = SimpleNamespace(local_mv_match=None)
        self.assertIsNone(
            bench.resolve_local_mv_match("single", args, 1, (5, 4, 3))
        )

    def test_single_mode_rejects_override(self):
        args = SimpleNamespace(local_mv_match=2)
        with self.assertRaises(SystemExit):
            bench.resolve_local_mv_match("single", args, 1, (5, 4, 3))

    def test_multi_mode_defaults_to_v_minus_one(self):
        args = SimpleNamespace(local_mv_match=None)
        # V = 3 frames * 3 cameras = 9 -> 8.
        self.assertEqual(
            bench.resolve_local_mv_match("multi", args, 3, (5, 4, 3)), 8
        )

    def test_multi_mode_explicit_value(self):
        args = SimpleNamespace(local_mv_match=2)
        self.assertEqual(
            bench.resolve_local_mv_match("multi", args, 3, (5, 4, 3)), 2
        )

    def test_multi_mode_rejects_zero_for_many_views(self):
        args = SimpleNamespace(local_mv_match=0)
        with self.assertRaises(SystemExit):
            bench.resolve_local_mv_match("multi", args, 3, (5, 4, 3))

    def test_multi_mode_allows_zero_for_at_most_three_views(self):
        args = SimpleNamespace(local_mv_match=0)
        self.assertEqual(
            bench.resolve_local_mv_match("multi", args, 1, (5, 4, 3)), 0
        )


class SummarizeTimesTest(unittest.TestCase):
    def test_basic_stats(self):
        stats = bench.summarize_times([1.0, 2.0, 3.0])
        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean_ms"], 2.0)
        self.assertAlmostEqual(stats["median_ms"], 2.0)
        self.assertAlmostEqual(stats["min_ms"], 1.0)
        self.assertAlmostEqual(stats["max_ms"], 3.0)
        self.assertAlmostEqual(stats["stdev_ms"], 1.0)
        self.assertAlmostEqual(stats["fps"], 500.0)

    def test_single_sample_has_zero_stdev(self):
        stats = bench.summarize_times([10.0])
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["stdev_ms"], 0.0)
        self.assertAlmostEqual(stats["fps"], 100.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            bench.summarize_times([])

    def test_accepts_ints(self):
        stats = bench.summarize_times([2, 4])
        self.assertAlmostEqual(stats["mean_ms"], 3.0)
        self.assertAlmostEqual(stats["fps"], 1000.0 / 3.0)


class SceneSelectionTest(unittest.TestCase):
    def test_default_is_first_scene(self):
        self.assertEqual(bench.select_scene(["037", "038", "039"]), "037")

    def test_requested_scene_is_returned(self):
        self.assertEqual(bench.select_scene(["037", "038"], "038"), "038")

    def test_requested_scene_must_be_in_list(self):
        with self.assertRaises(ValueError):
            bench.select_scene(["037", "038"], "999")

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            bench.select_scene([])
        with self.assertRaises(ValueError):
            bench.select_scene(["", "   "])


class SingleFrameSelectionTest(unittest.TestCase):
    FRAMES = ["000", "001", "002", "003"]

    def test_default_is_first_frame(self):
        self.assertEqual(bench.select_single_frame(self.FRAMES), "000")

    def test_normalizes_requested_frame(self):
        self.assertEqual(bench.select_single_frame(self.FRAMES, "1"), "001")
        self.assertEqual(bench.select_single_frame(self.FRAMES, "001"), "001")

    def test_unknown_frame_returns_none(self):
        self.assertEqual(bench.select_single_frame(self.FRAMES, "009"), None)

    def test_no_frames_returns_none(self):
        self.assertEqual(bench.select_single_frame([]), None)


class WindowSelectionTest(unittest.TestCase):
    FRAMES = ["000", "001", "002", "003"]

    def test_default_is_first_complete_window(self):
        self.assertEqual(
            bench.select_window(self.FRAMES, 3), ("000", "001", "002")
        )

    def test_requested_newest_frame_selects_window(self):
        self.assertEqual(
            bench.select_window(self.FRAMES, 3, "3"), ("001", "002", "003")
        )
        self.assertEqual(
            bench.select_window(self.FRAMES, 3, "003"), ("001", "002", "003")
        )

    def test_frame_without_enough_predecessors_returns_none(self):
        self.assertEqual(bench.select_window(self.FRAMES, 3, "1"), None)
        self.assertEqual(bench.select_window(self.FRAMES, 3, "000"), None)

    def test_too_few_frames_returns_none(self):
        self.assertEqual(bench.select_window(["000", "001"], 3), None)
        self.assertEqual(bench.select_window([], 3), None)

    def test_single_frame_window(self):
        self.assertEqual(bench.select_window(self.FRAMES, 1), ("000",))


if __name__ == "__main__":
    unittest.main()

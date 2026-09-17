"""Focused tests for the multi-frame nuScenes wide-view inference helpers.

Run from the repository root with::

    python -m unittest tests.test_inference_nuscenes_wide_multiframes

The tests only exercise the pure helpers (causal window enumeration, frame-major
flattening / newest-render index, per-frame extrinsic paths, parser defaults and
``local_mv_match`` resolution) plus a temp-scene loader check, so they do not
require CUDA or the Gaussian rasterizer.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "inference_nuscenes_wide_multiframes.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "inference_nuscenes_wide_multiframes", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so dataclasses can resolve the module in 3.10.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mf = _load_module()


class WindowEnumerationTest(unittest.TestCase):
    """Windows are causal and contiguous: output is the newest frame."""

    def test_causal_windows_for_four_frames(self):
        windows = mf.enumerate_windows(["000", "001", "002", "003"], 3)
        self.assertEqual(
            windows, [("000", "001", "002"), ("001", "002", "003")]
        )
        # The rendered/output frame is always the last element.
        self.assertEqual([w[-1] for w in windows], ["002", "003"])

    def test_first_window_ends_at_num_frames_minus_one(self):
        windows = mf.enumerate_windows(["000", "001", "002"], 3)
        self.assertEqual(windows, [("000", "001", "002")])

    def test_single_frame_windows(self):
        windows = mf.enumerate_windows(["000", "001"], 1)
        self.assertEqual(windows, [("000",), ("001",)])

    def test_two_frame_windows(self):
        windows = mf.enumerate_windows(["000", "001", "002"], 2)
        self.assertEqual(windows, [("000", "001"), ("001", "002")])

    def test_too_few_frames_yields_no_incomplete_window(self):
        self.assertEqual(mf.enumerate_windows(["000", "001"], 3), [])
        self.assertEqual(mf.enumerate_windows([], 3), [])

    def test_rejects_zero_or_negative(self):
        with self.assertRaises(ValueError):
            mf.enumerate_windows(["000"], 0)
        with self.assertRaises(ValueError):
            mf.enumerate_windows(["000"], -1)


class SelectWindowsTest(unittest.TestCase):
    """``--frame`` selects the window whose newest frame matches."""

    FRAMES = ["000", "001", "002", "003"]

    def test_default_returns_all_complete_windows(self):
        self.assertEqual(
            mf.select_windows(self.FRAMES, 3),
            [("000", "001", "002"), ("001", "002", "003")],
        )

    def test_frame_selects_output_newest_frame(self):
        self.assertEqual(
            mf.select_windows(self.FRAMES, 3, frame="3"), [("001", "002", "003")]
        )
        self.assertEqual(
            mf.select_windows(self.FRAMES, 3, frame="003"),
            [("001", "002", "003")],
        )

    def test_frame_without_enough_predecessors_has_no_window(self):
        self.assertEqual(mf.select_windows(self.FRAMES, 3, frame="1"), [])
        self.assertEqual(mf.select_windows(self.FRAMES, 3, frame="000"), [])

    def test_max_frames_limited_list_yields_single_window(self):
        # --max-frames is applied before windowing: limiting to 3 frames means
        # only the [000,001,002] window exists.
        limited = self.FRAMES[:3]
        self.assertEqual(
            mf.select_windows(limited, 3), [("000", "001", "002")]
        )


class FlatteningTest(unittest.TestCase):
    """Views are flattened frame-major (oldest frame first)."""

    def test_frame_major_order(self):
        views = mf.flatten_window_views(("000", "001"), (5, 4, 3))
        self.assertEqual(
            views,
            [
                ("000", 5),
                ("000", 4),
                ("000", 3),
                ("001", 5),
                ("001", 4),
                ("001", 3),
            ],
        )

    def test_newest_render_index_matches_flattening(self):
        cameras = (5, 4, 3)
        window = ("000", "001", "002")
        views = mf.flatten_window_views(window, cameras)
        for render_camera in cameras:
            expected = views.index((window[-1], render_camera))
            self.assertEqual(
                mf.newest_render_index(len(window), cameras, render_camera),
                expected,
            )
        # For T=3, V=3: the newest frame's views start at index 6.
        self.assertEqual(mf.newest_render_index(3, cameras, 5), 6)
        self.assertEqual(mf.newest_render_index(3, cameras, 4), 7)
        self.assertEqual(mf.newest_render_index(3, cameras, 3), 8)

    def test_single_frame_render_index(self):
        self.assertEqual(mf.newest_render_index(1, (5, 4, 3), 3), 2)

    def test_unknown_render_camera_rejected(self):
        with self.assertRaises(ValueError):
            mf.newest_render_index(3, (5, 4, 3), 9)


class LocalMvMatchTest(unittest.TestCase):
    """By default every flattened view participates in matching."""

    class _Args:
        def __init__(self, value=None):
            self.local_mv_match = value

    def test_default_is_v_minus_one(self):
        # 3 frames x 3 cameras -> 9 views -> 8 additional matches.
        self.assertEqual(
            mf.resolve_local_mv_match(self._Args(None), 3, (5, 4, 3)), 8
        )
        self.assertEqual(
            mf.resolve_local_mv_match(self._Args(None), 1, (5, 4, 3)), 2
        )

    def test_explicit_override_wins(self):
        self.assertEqual(
            mf.resolve_local_mv_match(self._Args(2), 3, (5, 4, 3)), 2
        )
        self.assertEqual(
            mf.resolve_local_mv_match(self._Args(1), 3, (5, 4, 3)), 1
        )

    def test_zero_rejected_for_more_than_three_views(self):
        # V = 3 frames x 3 cameras = 9 > 3: keeping only each view itself leaves
        # an empty cross-view cost volume that can produce NaNs.
        for num_frames, cameras in ((3, (5, 4, 3)), (2, (5, 4, 3)), (4, (5, 4, 3))):
            with self.subTest(num_frames=num_frames, cameras=cameras):
                with self.assertRaises(SystemExit) as ctx:
                    mf.resolve_local_mv_match(
                        self._Args(0), num_frames, cameras
                    )
                message = str(ctx.exception)
                self.assertIn("--local-mv-match 0", message)
                self.assertIn("empty cross-view cost volume", message)

    def test_zero_allowed_for_at_most_three_views(self):
        # V <= 3: the encoder matches all views and ignores local_mv_match.
        self.assertEqual(
            mf.resolve_local_mv_match(self._Args(0), 1, (5, 4, 3)), 0
        )
        self.assertEqual(
            mf.resolve_local_mv_match(self._Args(0), 1, (5, 4)), 0
        )
        self.assertEqual(
            mf.resolve_local_mv_match(self._Args(0), 1, (5,)), 0
        )

    def test_default_never_hits_invalid_zero(self):
        # Default is V - 1, so > 3 views always resolve to >= 1.
        self.assertEqual(
            mf.resolve_local_mv_match(self._Args(None), 3, (5, 4, 3)), 8
        )
        self.assertEqual(
            mf.resolve_local_mv_match(self._Args(None), 2, (5, 4, 3)), 5
        )

    def test_negative_rejected(self):
        with self.assertRaises(SystemExit):
            mf.resolve_local_mv_match(self._Args(-1), 3, (5, 4, 3))


class PerFrameExtrinsicsPathTest(unittest.TestCase):
    """Multi-frame extrinsics are always the per-frame global matrices."""

    def test_path_uses_frame_and_camera(self):
        self.assertEqual(
            mf.per_frame_extrinsics_path(Path("/scene"), "002", 5),
            Path("/scene/extrinsics/002_5.txt"),
        )
        self.assertEqual(
            mf.per_frame_extrinsics_path(Path("/scene"), "010", 3),
            Path("/scene/extrinsics/010_3.txt"),
        )

    def test_path_is_never_cam2ego(self):
        path = mf.per_frame_extrinsics_path(Path("/scene"), "006", 3)
        self.assertEqual(path.parent.name, "extrinsics")
        self.assertNotIn("cam2ego", str(path))


class ParserDefaultsTest(unittest.TestCase):
    """The CLI defaults describe the full multi-frame run."""

    def setUp(self):
        self.args = mf.build_arg_parser().parse_args([])

    def test_model_and_input_defaults(self):
        self.assertEqual(self.args.model, "256x448")
        self.assertEqual(mf.wide.DEFAULT_PRESET, "256x448")
        self.assertIsNone(self.args.input_size)
        self.assertIsNone(self.args.height)
        self.assertIsNone(self.args.width)
        # The effective default input follows the model preset.
        preset = mf.wide.MODEL_PRESETS[mf.wide.DEFAULT_PRESET]
        self.assertEqual(mf.wide.resolve_input_hw(self.args, preset), (256, 448))

    def test_explicit_large_preset_still_supported(self):
        args = mf.build_arg_parser().parse_args(
            ["--model", "448x768", "--input-size", "448x768"]
        )
        self.assertEqual(args.model, "448x768")
        self.assertEqual(args.input_size, (448, 768))
        preset = mf.wide.MODEL_PRESETS["448x768"]
        self.assertEqual(mf.wide.resolve_input_hw(args, preset), (448, 768))

    def test_num_frames_default(self):
        self.assertEqual(self.args.num_frames, 3)
        self.assertEqual(mf.DEFAULT_NUM_FRAMES, 3)

    def test_camera_defaults(self):
        self.assertEqual(self.args.cameras, "5,4,3")
        self.assertEqual(self.args.render_camera, 5)
        self.assertEqual(self.args.width_factor, 2.0)

    def test_scene_list_default(self):
        self.assertEqual(
            self.args.scene_list,
            "datasets/nuscenes/processed_10Hz/trainval2/nuScenes_Val2.txt",
        )
        self.assertIsNone(self.args.scene)
        self.assertIsNone(self.args.frame)
        self.assertEqual(self.args.max_frames, -1)

    def test_local_mv_match_default_is_none(self):
        self.assertIsNone(self.args.local_mv_match)
        self.assertIsNone(mf.DEFAULT_LOCAL_MV_MATCH)

    def test_output_defaults(self):
        self.assertEqual(
            self.args.output_dir, "outputs/nuscenes_wide_multiframes"
        )
        self.assertFalse(self.args.save_inputs)

    def test_parser_accepts_overrides(self):
        args = mf.build_arg_parser().parse_args(
            [
                "--num-frames",
                "5",
                "--cameras",
                "6,5,4",
                "--render-camera",
                "4",
                "--local-mv-match",
                "12",
                "--max-frames",
                "10",
            ]
        )
        self.assertEqual(args.num_frames, 5)
        self.assertEqual(args.cameras, "6,5,4")
        self.assertEqual(args.render_camera, 4)
        self.assertEqual(args.local_mv_match, 12)
        self.assertEqual(args.max_frames, 10)


class ExtrinsicsSourceTest(unittest.TestCase):
    """The multi-frame CLI must not expose a cam2ego A/B switch."""

    def test_no_extrinsics_source_flag(self):
        args = mf.build_arg_parser().parse_args([])
        self.assertFalse(hasattr(args, "extrinsics_source"))


class ComposeLocalMvMatchTest(unittest.TestCase):
    """The extended single-frame composer forwards local_mv_match."""

    def setUp(self):
        try:
            import hydra  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("hydra is required to compose the repository config")

    def test_override_is_composed(self):
        args = mf.build_arg_parser().parse_args([])
        preset = mf.wide.MODEL_PRESETS[mf.wide.DEFAULT_PRESET]
        cfg_dict = mf.wide.compose_config_dict(
            args, preset, None, local_mv_match=8
        )
        self.assertEqual(int(cfg_dict.model.encoder.local_mv_match), 8)

    def test_default_keeps_training_value(self):
        args = mf.build_arg_parser().parse_args([])
        preset = mf.wide.MODEL_PRESETS[mf.wide.DEFAULT_PRESET]
        cfg_dict = mf.wide.compose_config_dict(args, preset, None)
        self.assertEqual(int(cfg_dict.model.encoder.local_mv_match), 2)


class LoadWindowInputsTest(unittest.TestCase):
    """Each frame in a window uses its own per-frame global extrinsics."""

    def setUp(self):
        try:
            import PIL  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("PIL is required to write temporary frame images")

    CAMERAS = (5, 4, 3)
    FRAMES = ("000", "001", "002")

    @staticmethod
    def _matrix_text(value):
        rows = [
            [1.0, 0.0, 0.0, value],
            [0.0, 1.0, 0.0, value + 1.0],
            [0.0, 0.0, 1.0, value + 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return "\n".join(" ".join(str(v) for v in row) for row in rows)

    def _translation_value(self, frame_index, cam):
        # Unique per (frame, cam) so frame-major ordering is checkable.
        return 100.0 * frame_index + cam

    def _make_scene(self, tmp, image_size=(8, 8)):
        from PIL import Image

        scene_dir = Path(tmp) / "037"
        for sub in ("images", "intrinsics", "extrinsics", "cam2ego_extrinsics"):
            (scene_dir / sub).mkdir(parents=True)
        for frame_index, frame in enumerate(self.FRAMES):
            for cam in self.CAMERAS:
                Image.new("RGB", image_size, (frame_index, cam, 0)).save(
                    scene_dir / "images" / f"{frame}_{cam}.jpg"
                )
                (scene_dir / "intrinsics" / f"{cam}.txt").write_text(
                    "1 1 4 4 0 0 0 0 1"
                )
                (scene_dir / "extrinsics" / f"{frame}_{cam}.txt").write_text(
                    self._matrix_text(self._translation_value(frame_index, cam))
                )
                # Deliberately misleading static rig; it must never be read.
                (scene_dir / "cam2ego_extrinsics" / f"{cam}.txt").write_text(
                    self._matrix_text(9000.0 + cam)
                )
        return scene_dir

    def _load(self, scene_dir, window=None, render_camera=5):
        window = window or self.FRAMES
        return mf.load_window_inputs(
            scene_dir,
            "037",
            window,
            self.CAMERAS,
            render_camera,
            (8, 8),
            (8, 8),
        )

    def test_shapes_and_frame_major_extrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(tmp)
            inputs = self._load(scene_dir)
            self.assertEqual(inputs.frames, self.FRAMES)
            self.assertEqual(inputs.newest_frame, "002")
            self.assertEqual(inputs.images.shape, (9, 8, 8, 3))
            self.assertEqual(inputs.intrinsics.shape, (9, 3, 3))
            self.assertEqual(inputs.extrinsics.shape, (9, 4, 4))

            expected = [
                self._translation_value(frame_index, cam)
                for frame_index, frame in enumerate(self.FRAMES)
                for cam in self.CAMERAS
            ]
            np.testing.assert_allclose(
                inputs.extrinsics[:, 0, 3], expected, rtol=0, atol=1e-9
            )
            # The static cam2ego rig (9000+) must not leak in.
            self.assertTrue((inputs.extrinsics[:, 0, 3] < 1000).all())

    def test_render_index_is_newest_render_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(tmp)
            for render_camera in self.CAMERAS:
                inputs = self._load(scene_dir, render_camera=render_camera)
                self.assertEqual(
                    inputs.render_index,
                    mf.newest_render_index(3, self.CAMERAS, render_camera),
                )
                # The target view's extrinsics is the newest frame's render cam.
                expected = self._translation_value(2, render_camera)
                self.assertAlmostEqual(
                    float(inputs.extrinsics[inputs.render_index, 0, 3]),
                    expected,
                    places=6,
                )

    def test_sub_window_uses_matching_extrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(tmp)
            inputs = self._load(scene_dir, window=("001", "002"))
            self.assertEqual(inputs.render_index, 3)  # (2-1)*3 + 0
            # Newest frame is on-disk frame "002" (frame index 2, cam 5).
            np.testing.assert_allclose(
                inputs.extrinsics[inputs.render_index, 0, 3],
                self._translation_value(2, 5),
            )

    def test_missing_per_frame_extrinsics_fails_with_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(tmp)
            (scene_dir / "extrinsics" / "001_3.txt").unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                self._load(scene_dir)
            message = str(ctx.exception)
            self.assertIn("extrinsics/001_3.txt", message)
            self.assertIn("{frame}_{cam}.txt", message)
            self.assertIn("cam2ego", message)

    def test_frame_shape_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(tmp)
            # Replace one frame's image with a different source shape: every
            # frame must share the resize plan.
            from PIL import Image

            Image.new("RGB", (16, 16), (0, 0, 0)).save(
                scene_dir / "images" / "001_5.jpg"
            )
            with self.assertRaises(ValueError):
                self._load(scene_dir)


if __name__ == "__main__":
    unittest.main()

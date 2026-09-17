"""Focused tests for the multi-frame nuScenes wide-view inference helpers.

Run from the repository root with::

    python -m unittest tests.test_inference_nuscenes_wide_multiframes

The tests only exercise the pure helpers (causal window enumeration, frame-major
flattening / newest-render index, per-frame extrinsic paths, parser defaults and
``local_mv_match`` resolution) plus a temp-scene loader check, so they do not
require CUDA or the Gaussian rasterizer.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_car_mask_defaults_enabled(self):
        # Masking is on by default: car-mask-root has its default, the disable
        # flag is off and the render view stays preserved.
        self.assertEqual(
            self.args.car_mask_root,
            "datasets/nuscenes/processed_10Hz/nuscenes_mask",
        )
        self.assertFalse(self.args.disable_car_mask)
        self.assertFalse(self.args.mask_render_view)

    def test_disable_car_mask_flag(self):
        args = mf.build_arg_parser().parse_args(["--disable-car-mask"])
        self.assertTrue(args.disable_car_mask)
        self.assertFalse(args.mask_render_view)

    def test_mask_render_view_flag(self):
        args = mf.build_arg_parser().parse_args(["--mask-render-view"])
        self.assertTrue(args.mask_render_view)
        self.assertFalse(args.disable_car_mask)

    def test_car_mask_root_override(self):
        args = mf.build_arg_parser().parse_args(
            ["--car-mask-root", "/some/other/masks"]
        )
        self.assertEqual(args.car_mask_root, "/some/other/masks")
        self.assertFalse(args.disable_car_mask)
        self.assertFalse(args.mask_render_view)

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


class CameraMaskMappingTest(unittest.TestCase):
    """The nuScenes camera id -> mask file mapping is exact."""

    EXPECTED = {
        0: "CAM_FRONT_mask.png",
        1: "CAM_FRONT_LEFT_mask.png",
        2: "CAM_FRONT_RIGHT_mask.png",
        3: "CAM_BACK_LEFT_mask.png",
        4: "CAM_BACK_RIGHT_mask.png",
        5: "CAM_BACK_mask.png",
    }

    def test_camera_to_file_mapping(self):
        self.assertEqual(mf.CAMERA_MASK_FILES, self.EXPECTED)

    def test_default_root(self):
        self.assertEqual(
            mf.DEFAULT_CAR_MASK_ROOT,
            Path("datasets/nuscenes/processed_10Hz/nuscenes_mask"),
        )

    def test_paths_use_camera_names(self):
        root = Path("/masks")
        for cam, name in self.EXPECTED.items():
            with self.subTest(cam=cam):
                self.assertEqual(mf.camera_mask_path(root, cam), root / name)

    def test_unmapped_camera_is_a_hard_error(self):
        with self.assertRaises(KeyError):
            mf.camera_mask_path(Path("/masks"), 6)


class CarMaskResizeTest(unittest.TestCase):
    """Masks follow the exact image resize/crop plan, never a plain resize."""

    def setUp(self):
        try:
            from PIL import Image  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("PIL is required to write temporary mask images")

    @staticmethod
    def _checkerboard(h, w):
        y, x = np.indices((h, w))
        return np.where((x + y) % 2 == 0, 255, 0).astype(np.uint8)

    def test_nearest_resize_then_centre_crop(self):
        from PIL import Image

        plan = mf.wide.plan_resize_and_crop((4, 8), (8, 4))
        source = self._checkerboard(4, 8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "CAM_BACK_mask.png"
            Image.fromarray(source, "L").save(path)

            got = mf.load_resized_keep_mask(path, plan)

            # Reference: NEAREST resize to the scaled size, then the centre crop.
            image = Image.open(path).convert("L")
            image = image.resize((plan.scaled_w, plan.scaled_h), Image.NEAREST)
            image = image.crop(
                (plan.col, plan.row, plan.col + plan.out_w, plan.row + plan.out_h)
            )
            expected = np.asarray(image) >= mf.CAR_MASK_KEEP_THRESHOLD
            np.testing.assert_array_equal(got, expected)
            self.assertEqual(got.shape, (plan.out_h, plan.out_w))

            # A plain resize straight to the destination gives a different
            # alignment; the helper must not do that.
            direct = np.asarray(
                Image.open(path).convert("L").resize(
                    (plan.out_w, plan.out_h), Image.NEAREST
                )
            ) >= mf.CAR_MASK_KEEP_THRESHOLD
            self.assertFalse(np.array_equal(got, direct))

    def test_polarity_black_removed_white_kept(self):
        from PIL import Image

        plan = mf.wide.plan_resize_and_crop((2, 2), (2, 2))
        source = np.array([[0, 127], [128, 255]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.png"
            Image.fromarray(source, "L").save(path)
            got = mf.load_resized_keep_mask(path, plan)
            np.testing.assert_array_equal(got, [[False, False], [True, True]])

    def test_source_dimension_mismatch_fails_loudly(self):
        from PIL import Image

        plan = mf.wide.plan_resize_and_crop((4, 8), (8, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.png"
            Image.fromarray(self._checkerboard(2, 2), "L").save(path)
            with self.assertRaises(ValueError) as ctx:
                mf.load_resized_keep_mask(path, plan)
            message = str(ctx.exception)
            self.assertIn("4x8", message)
            self.assertIn("resolution", message)


class CarMaskPolicyTest(unittest.TestCase):
    """Default masks every view except the current render view."""

    CAMERAS = (5, 4, 3)
    NUM_FRAMES = 3
    DST_HW = (2, 2)
    # Newest frame (t3) camera 5 -> flattened index 6.
    RENDER_INDEX = mf.newest_render_index(3, (5, 4, 3), 5)

    def _masks(self):
        # Distinct, non-trivial masks so per-view camera mapping is checkable.
        return {
            5: np.array([[True, True], [True, False]], dtype=bool),
            4: np.array([[False, True], [True, True]], dtype=bool),
            3: np.array([[True, False], [False, True]], dtype=bool),
        }

    def test_frame_major_camera_mapping(self):
        # view v uses cameras[v % len(cameras)].
        expected = [5, 4, 3, 5, 4, 3, 5, 4, 3]
        self.assertEqual(
            [self.CAMERAS[v % len(self.CAMERAS)] for v in range(9)], expected
        )
        self.assertEqual(self.RENDER_INDEX, 6)

    def test_default_masks_every_view_except_render_view(self):
        masks = self._masks()
        keep = mf.build_car_keep_mask(
            masks, self.CAMERAS, self.NUM_FRAMES, self.DST_HW, self.RENDER_INDEX
        )
        self.assertEqual(keep.shape, (9, 2, 2))
        self.assertTrue(keep.dtype == bool)
        for view in range(9):
            cam = self.CAMERAS[view % len(self.CAMERAS)]
            if view == self.RENDER_INDEX:
                self.assertTrue(keep[view].all())
            else:
                np.testing.assert_array_equal(keep[view], masks[cam])
        # The render view is the only all-ones view.
        all_ones = [v for v in range(9) if keep[v].all()]
        self.assertEqual(all_ones, [self.RENDER_INDEX])

    def test_default_masks_newest_side_cameras_but_not_render(self):
        masks = self._masks()
        keep = mf.build_car_keep_mask(
            masks, self.CAMERAS, self.NUM_FRAMES, self.DST_HW, self.RENDER_INDEX
        )
        # Current frame t3: cam5 render preserved; cams 4 and 3 masked.
        self.assertTrue(keep[6].all())
        np.testing.assert_array_equal(keep[7], masks[4])
        np.testing.assert_array_equal(keep[8], masks[3])
        # Historical frames t1/t2: every camera masked.
        for view in (0, 1, 2, 3, 4, 5):
            self.assertFalse(keep[view].all())

    def test_diagnostic_masks_all_views_including_render(self):
        masks = self._masks()
        keep = mf.build_car_keep_mask(
            masks,
            self.CAMERAS,
            self.NUM_FRAMES,
            self.DST_HW,
            self.RENDER_INDEX,
            mask_render_view=True,
        )
        for view in range(9):
            cam = self.CAMERAS[view % len(self.CAMERAS)]
            np.testing.assert_array_equal(keep[view], masks[cam])
        self.assertFalse(keep[self.RENDER_INDEX].all())

    def test_diagnostic_removes_exactly_the_render_view_pixels_more(self):
        masks = self._masks()
        default = mf.build_car_keep_mask(
            masks, self.CAMERAS, self.NUM_FRAMES, self.DST_HW, self.RENDER_INDEX
        )
        diagnostic = mf.build_car_keep_mask(
            masks,
            self.CAMERAS,
            self.NUM_FRAMES,
            self.DST_HW,
            self.RENDER_INDEX,
            mask_render_view=True,
        )
        extra_removed = int((~diagnostic).sum()) - int((~default).sum())
        self.assertEqual(extra_removed, int((~masks[5]).sum()))
        # Default kept pixels are a superset of the diagnostic's kept pixels.
        self.assertTrue(np.array_equal(default | diagnostic, default))

    def test_single_frame_default_masks_side_cameras_only(self):
        render_index = mf.newest_render_index(1, self.CAMERAS, 5)  # 0
        masks = self._masks()
        keep = mf.build_car_keep_mask(
            masks, self.CAMERAS, 1, self.DST_HW, render_index
        )
        self.assertEqual(keep.shape, (3, 2, 2))
        self.assertTrue(keep[0].all())
        np.testing.assert_array_equal(keep[1], masks[4])
        np.testing.assert_array_equal(keep[2], masks[3])

    def test_single_frame_diagnostic_masks_render_too(self):
        render_index = mf.newest_render_index(1, self.CAMERAS, 5)
        masks = self._masks()
        keep = mf.build_car_keep_mask(
            masks, self.CAMERAS, 1, self.DST_HW, render_index, mask_render_view=True
        )
        for view in range(3):
            np.testing.assert_array_equal(keep[view], masks[self.CAMERAS[view]])

    def test_render_index_must_be_newest_frame(self):
        with self.assertRaises(ValueError) as ctx:
            mf.build_car_keep_mask(
                self._masks(), self.CAMERAS, self.NUM_FRAMES, self.DST_HW, 0
            )
        self.assertIn("newest frame", str(ctx.exception))

    def test_render_index_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            mf.build_car_keep_mask(
                self._masks(), self.CAMERAS, self.NUM_FRAMES, self.DST_HW, 9
            )

    def test_camera_mask_shape_mismatch_fails(self):
        masks = self._masks()
        masks[4] = np.ones((3, 3), dtype=bool)
        with self.assertRaises(ValueError):
            mf.build_car_keep_mask(
                masks, self.CAMERAS, self.NUM_FRAMES, self.DST_HW, self.RENDER_INDEX
            )

    def test_missing_camera_mask_fails(self):
        masks = self._masks()
        del masks[4]
        with self.assertRaises(KeyError):
            mf.build_car_keep_mask(
                masks, self.CAMERAS, self.NUM_FRAMES, self.DST_HW, self.RENDER_INDEX
            )


class CarMaskPolicyResolutionTest(unittest.TestCase):
    """``--disable-car-mask`` has the highest precedence."""

    def test_default_is_all_except_render_view(self):
        self.assertEqual(
            mf.resolve_car_mask_policy(False, False),
            mf.CAR_MASK_POLICY_ALL_EXCEPT_RENDER,
        )

    def test_mask_render_view_selects_all_views(self):
        self.assertEqual(
            mf.resolve_car_mask_policy(False, True), mf.CAR_MASK_POLICY_ALL
        )

    def test_disable_wins_over_everything(self):
        self.assertIsNone(mf.resolve_car_mask_policy(True, False))
        self.assertIsNone(mf.resolve_car_mask_policy(True, True))


class GaussianPruningTest(unittest.TestCase):
    """Multiplicity inference and synchronized Gaussian pruning."""

    @dataclasses.dataclass
    class _Gaussians:
        means: object
        covariances: object
        harmonics: object
        opacities: object

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("torch is required to build Gaussian test tensors")

    def _make(self, v, h, w, k):
        import torch

        base = v * h * w
        g = base * k
        idx = torch.arange(g, dtype=torch.float32)
        return self._Gaussians(
            means=idx.view(1, g, 1).repeat(1, 1, 3),
            covariances=idx.view(1, g, 1, 1).repeat(1, 1, 3, 3),
            harmonics=idx.view(1, g, 1, 1).repeat(1, 1, 3, 2),
            opacities=idx.view(1, g),
        )

    def test_expand_keep_mask_over_multiplicity(self):
        keep = np.array(
            [[[True, False], [True, True]], [[False, False], [True, False]]]
        )
        flat = mf.expand_keep_mask_to_gaussians(keep, 2)
        expected = np.repeat(keep.reshape(-1), 2)
        np.testing.assert_array_equal(flat, expected)
        self.assertEqual(flat.shape, (2 * 4 * 2,))

    def test_multiplicity_is_inferred_and_all_fields_pruned(self):
        # V=2, H=2, W=2, K=3 -> G = 24 > V*H*W = 8.
        v, h, w, k = 2, 2, 2, 3
        gaussians = self._make(v, h, w, k)
        keep_mask = np.array(
            [[[True, False], [True, True]], [[False, False], [True, False]]]
        )
        pruned = mf.filter_gaussians_by_camera_mask(
            gaussians, keep_mask, v, h, w
        )
        flat = np.repeat(keep_mask.reshape(-1), k)
        expected_index = np.nonzero(flat)[0]
        self.assertEqual(pruned.means.shape[1], expected_index.size)
        # Every field is sliced identically along dim=1.
        np.testing.assert_array_equal(
            pruned.means[0, :, 0].numpy(), expected_index.astype(np.float32)
        )
        np.testing.assert_array_equal(
            pruned.covariances[0, :, 0, 0].numpy(),
            expected_index.astype(np.float32),
        )
        np.testing.assert_array_equal(
            pruned.harmonics[0, :, 0, 0].numpy(),
            expected_index.astype(np.float32),
        )
        np.testing.assert_array_equal(
            pruned.opacities[0].numpy(), expected_index.astype(np.float32)
        )
        self.assertIsInstance(pruned, type(gaussians))

    def test_indivisible_gaussian_count_fails(self):
        # G = 3 but V*H*W = 4 does not divide it.
        gaussians = self._make(v=1, h=2, w=2, k=1)
        bad = self._Gaussians(
            means=gaussians.means[:, :3],
            covariances=gaussians.covariances[:, :3],
            harmonics=gaussians.harmonics[:, :3],
            opacities=gaussians.opacities[:, :3],
        )
        with self.assertRaises(ValueError) as ctx:
            mf.filter_gaussians_by_camera_mask(
                bad, np.ones((1, 2, 2), dtype=bool), 1, 2, 2
            )
        self.assertIn("does not divide", str(ctx.exception))

    def test_all_removed_fails_loudly(self):
        v, h, w, k = 1, 2, 2, 1
        gaussians = self._make(v, h, w, k)
        with self.assertRaises(ValueError) as ctx:
            mf.filter_gaussians_by_camera_mask(
                gaussians, np.zeros((v, h, w), dtype=bool), v, h, w
            )
        self.assertIn("every Gaussian", str(ctx.exception))

    def test_mask_shape_mismatch_fails(self):
        gaussians = self._make(1, 2, 2, 1)
        with self.assertRaises(ValueError):
            mf.filter_gaussians_by_camera_mask(
                gaussians, np.ones((2, 2, 2), dtype=bool), 1, 2, 2
            )


class MissingMaskTest(unittest.TestCase):
    """A missing required mask never silently becomes all-ones."""

    def test_missing_mask_raises_with_path(self):
        root = Path("/nonexistent/mask/root")
        with self.assertRaises(FileNotFoundError) as ctx:
            mf.load_camera_keep_masks(
                root, (5, 4, 3), mf.wide.plan_resize_and_crop((2, 2), (2, 2))
            )
        self.assertIn("CAM_BACK_mask.png", str(ctx.exception))

    def test_partial_root_reports_missing_camera(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("L", (2, 2), 255).save(root / "CAM_BACK_mask.png")
            Image.new("L", (2, 2), 255).save(root / "CAM_BACK_RIGHT_mask.png")
            with self.assertRaises(FileNotFoundError) as ctx:
                mf.load_camera_keep_masks(
                    root,
                    (5, 4, 3),
                    mf.wide.plan_resize_and_crop((2, 2), (2, 2)),
                )
            self.assertIn("CAM_BACK_LEFT_mask.png", str(ctx.exception))

    def test_all_present_loads_every_camera(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cam, name in mf.CAMERA_MASK_FILES.items():
                Image.new("L", (2, 2), 255).save(root / name)
            masks = mf.load_camera_keep_masks(
                root, (5, 4, 3), mf.wide.plan_resize_and_crop((2, 2), (2, 2))
            )
            self.assertEqual(set(masks), {5, 4, 3})


class RenderFilterCallbackTest(unittest.TestCase):
    """``_render_wide_impl`` calls the filter between encoder and decoder."""

    @dataclasses.dataclass
    class _Gaussians:
        means: object
        covariances: object
        harmonics: object
        opacities: object

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("torch is required to exercise the render callback")

    def _inputs(self):
        cameras = 3
        h = w = 8
        return mf.wide.FrameInputs(
            scene="s",
            frame="000",
            images=np.zeros((cameras, h, w, 3), dtype=np.float32),
            intrinsics=np.tile(np.eye(3, dtype=np.float32), (cameras, 1, 1)),
            extrinsics=np.tile(np.eye(4, dtype=np.float32), (cameras, 1, 1)),
            resize_plan=mf.wide.plan_resize_and_crop((h, w), (h, w)),
            render_intrinsics_px=mf.wide.PixelIntrinsics(4, 4, 4, 4),
        )

    def _fake_gaussians(self, tag):
        import torch

        obj = self._Gaussians(
            means=torch.zeros(1, 3, 3),
            covariances=torch.zeros(1, 3, 3, 3),
            harmonics=torch.zeros(1, 3, 3, 1),
            opacities=torch.zeros(1, 3),
        )
        obj.tag = tag
        return obj

    def test_filter_runs_between_encoder_and_decoder(self):
        import torch

        raw = self._fake_gaussians("encoded")
        filtered = self._fake_gaussians("filtered")

        class _Encoder:
            def __call__(self, context, step, flag):
                return {"gaussians": raw}

        seen = {}

        class _Decoder:
            def __call__(self, gaussians, *args, **kwargs):
                seen["gaussians"] = gaussians
                return SimpleNamespace(color=torch.zeros(1, 1, 3, 8, 8))

        calls = []

        def callback(gaussians, num_views, height, width):
            calls.append((gaussians, num_views, height, width))
            return filtered

        mf.wide._render_wide_impl(
            self._inputs(),
            _Encoder(),
            _Decoder(),
            0,
            2.0,
            0.5,
            200.0,
            torch.device("cpu"),
            False,
            gaussian_filter=callback,
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], raw)
        self.assertEqual(calls[0][1:], (3, 8, 8))
        self.assertIs(seen["gaussians"], filtered)

    def test_no_filter_leaves_encoder_output_untouched(self):
        import torch

        raw = self._fake_gaussians("encoded")

        class _Encoder:
            def __call__(self, context, step, flag):
                return {"gaussians": raw}

        seen = {}

        class _Decoder:
            def __call__(self, gaussians, *args, **kwargs):
                seen["gaussians"] = gaussians
                return SimpleNamespace(color=torch.zeros(1, 1, 3, 8, 8))

        mf.wide._render_wide_impl(
            self._inputs(),
            _Encoder(),
            _Decoder(),
            0,
            2.0,
            0.5,
            200.0,
            torch.device("cpu"),
            False,
        )
        self.assertIs(seen["gaussians"], raw)


if __name__ == "__main__":
    unittest.main()

"""Focused tests for the nuScenes wide-view inference helpers.

Run from the repository root with::

    python -m unittest tests.test_inference_nuscenes_wide

The tests only exercise the pure helpers (resize/crop, intrinsics adjustment,
pixel-to-normalized K conversion, wide-K construction and data enumeration) so
they do not require CUDA or the Gaussian rasterizer.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "inference_nuscenes_wide.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "inference_nuscenes_wide", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so dataclasses can resolve the module in 3.10.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


wide = _load_module()

try:  # torch is only needed for the strict-loading tests.
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


class ModelPresetGaussianScaleTest(unittest.TestCase):
    """The gaussian scale max is part of the checkpoint architecture."""

    def test_preset_scale_max_values(self):
        self.assertEqual(wide.MODEL_PRESETS["448x768"].gaussian_scale_max, 0.1)
        self.assertEqual(wide.MODEL_PRESETS["256x448"].gaussian_scale_max, 3.0)

    def test_every_preset_carries_scale_max(self):
        for preset in wide.MODEL_PRESETS.values():
            self.assertIsInstance(preset.gaussian_scale_max, float)


@unittest.skipIf(torch is None, "torch is required for these checks")
class StrictEncoderLoadingTest(unittest.TestCase):
    """``load_encoder_state_dict`` must fail loudly on any key mismatch."""

    def _make_encoder(self):
        import torch.nn as nn

        class TinyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

        return TinyEncoder()

    def _state(self, encoder):
        return {key: value.clone() for key, value in encoder.state_dict().items()}

    def test_matching_state_loads(self):
        encoder = self._make_encoder()
        wide.load_encoder_state_dict(encoder, self._state(encoder))

    def test_missing_key_fails_loudly(self):
        encoder = self._make_encoder()
        state = self._state(encoder)
        del state["linear.weight"]
        with self.assertRaises(SystemExit):
            wide.load_encoder_state_dict(encoder, state)

    def test_unexpected_key_fails_loudly(self):
        encoder = self._make_encoder()
        state = self._state(encoder)
        state["extra.weight"] = torch.zeros(1, 1)
        with self.assertRaises(SystemExit):
            wide.load_encoder_state_dict(encoder, state)


class ResizeCropPlanTest(unittest.TestCase):
    def test_plan_for_448x768_from_1600x900(self):
        plan = wide.plan_resize_and_crop((900, 1600), (448, 768))
        # scale = max(448/900, 768/1600) = 0.49777...
        self.assertEqual((plan.scaled_h, plan.scaled_w), (448, 796))
        self.assertEqual((plan.row, plan.col), (0, 14))
        self.assertEqual((plan.out_h, plan.out_w), (448, 768))

    def test_plan_keeps_aspect_and_crops_long_side(self):
        plan = wide.plan_resize_and_crop((720, 1280), (448, 768))
        # The shorter-aspect dimension fills the target, the other overflows.
        self.assertGreaterEqual(plan.scaled_h, 448)
        self.assertGreaterEqual(plan.scaled_w, 768)
        self.assertTrue(plan.scaled_h == 448 or plan.scaled_w == 768)

    def test_plan_identity(self):
        plan = wide.plan_resize_and_crop((448, 768), (448, 768))
        self.assertEqual((plan.scaled_h, plan.scaled_w), (448, 768))
        self.assertEqual((plan.row, plan.col), (0, 0))

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            wide.plan_resize_and_crop((0, 100), (10, 10))


class IntrinsicsTest(unittest.TestCase):
    CAM5 = wide.PixelIntrinsics(809.22099, 809.22099, 829.21960, 481.77842)

    def test_resize_and_crop_adjusts_pixel_k(self):
        plan = wide.plan_resize_and_crop((900, 1600), (448, 768))
        resized = wide.resize_and_crop_intrinsics(self.CAM5, plan)
        self.assertAlmostEqual(resized.fx, self.CAM5.fx * 796 / 1600, places=6)
        self.assertAlmostEqual(resized.fy, self.CAM5.fy * 448 / 900, places=6)
        self.assertAlmostEqual(resized.cx, self.CAM5.cx * 796 / 1600 - 14, places=6)
        self.assertAlmostEqual(resized.cy, self.CAM5.cy * 448 / 900, places=6)

    def test_normalized_k_matches_definition(self):
        plan = wide.plan_resize_and_crop((900, 1600), (448, 768))
        resized = wide.resize_and_crop_intrinsics(self.CAM5, plan)
        normalized = wide.pixel_to_normalized_intrinsics(resized, 448, 768)
        expected = np.array(
            [
                [resized.fx / 768, 0.0, resized.cx / 768],
                [0.0, resized.fy / 448, resized.cy / 448],
                [0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-9)

    def test_normalized_focal_is_invariant_to_resize(self):
        # Pure resizing (no crop) must not change the normalized intrinsics.
        plan = wide.plan_resize_and_crop((900, 1600), (448, 768))
        scale_x = plan.scaled_w / plan.src_w
        scale_y = plan.scaled_h / plan.src_h
        resized_only = wide.PixelIntrinsics(
            self.CAM5.fx * scale_x,
            self.CAM5.fy * scale_y,
            self.CAM5.cx * scale_x,
            self.CAM5.cy * scale_y,
        )
        original = wide.pixel_to_normalized_intrinsics(self.CAM5, 900, 1600)
        resized = wide.pixel_to_normalized_intrinsics(
            resized_only, plan.scaled_h, plan.scaled_w
        )
        np.testing.assert_allclose(original, resized, rtol=1e-9, atol=1e-12)

    def test_crop_only_shifts_and_rescales_normalized_x(self):
        plan = wide.plan_resize_and_crop((900, 1600), (448, 768))
        resized = wide.resize_and_crop_intrinsics(self.CAM5, plan)
        adjusted = wide.pixel_to_normalized_intrinsics(resized, 448, 768)
        # The crop removes `col` pixels; the normalized focal length grows by
        # scaled_w / out_w relative to the resize-only value.
        expected_fx = (self.CAM5.fx * plan.scaled_w / plan.src_w) / plan.out_w
        self.assertAlmostEqual(adjusted[0, 0], expected_fx, places=6)

    def test_pixel_to_normalized_rejects_bad_size(self):
        with self.assertRaises(ValueError):
            wide.pixel_to_normalized_intrinsics(self.CAM5, 0, 100)


class WideIntrinsicsTest(unittest.TestCase):
    CAM5 = wide.PixelIntrinsics(402.792, 402.792, 398.792, 239.752)

    def test_wide_pixel_k(self):
        wide_px, (out_h, out_w) = wide.make_wide_intrinsics(self.CAM5, (448, 768), 2.0)
        self.assertEqual((out_h, out_w), (448, 1536))
        self.assertAlmostEqual(wide_px.fx, self.CAM5.fx, places=6)
        self.assertAlmostEqual(wide_px.fy, self.CAM5.fy, places=6)
        self.assertAlmostEqual(wide_px.cx, 1536 / 2.0, places=6)
        self.assertAlmostEqual(wide_px.cy, self.CAM5.cy, places=6)

    def test_wide_normalized_focal_halves_and_center(self):
        wide_px, (out_h, out_w) = wide.make_wide_intrinsics(self.CAM5, (448, 768), 2.0)
        base_norm = wide.pixel_to_normalized_intrinsics(self.CAM5, 448, 768)
        wide_norm = wide.pixel_to_normalized_intrinsics(wide_px, out_h, out_w)
        # Keeping the pixel focal length while doubling the width halves the
        # normalized focal length.
        self.assertAlmostEqual(wide_norm[0, 0], base_norm[0, 0] / 2.0, places=6)
        self.assertAlmostEqual(wide_norm[0, 2], 0.5, places=6)
        # Height (and its focal / principal point) is unchanged.
        self.assertAlmostEqual(wide_norm[1, 1], base_norm[1, 1], places=6)
        self.assertAlmostEqual(wide_norm[1, 2], base_norm[1, 2], places=6)

    def test_wide_non_integer_factor(self):
        wide_px, (out_h, out_w) = wide.make_wide_intrinsics(self.CAM5, (448, 768), 3.0)
        self.assertEqual((out_h, out_w), (448, 2304))
        self.assertAlmostEqual(wide_px.cx, 1152.0, places=6)

    def test_wide_rejects_bad_factor(self):
        with self.assertRaises(ValueError):
            wide.make_wide_intrinsics(self.CAM5, (448, 768), 0.0)


class PatchRoundingTest(unittest.TestCase):
    def test_round_down(self):
        self.assertEqual(wide.round_hw_to_multiple((450, 770), 64), (448, 768))
        self.assertEqual(wide.round_hw_to_multiple((448, 768), 64), (448, 768))

    def test_minimum_one_patch(self):
        self.assertEqual(wide.round_hw_to_multiple((10, 20), 64), (64, 64))

    def test_bad_multiple(self):
        with self.assertRaises(ValueError):
            wide.round_hw_to_multiple((64, 64), 0)


class EnumerationTest(unittest.TestCase):
    def test_read_scene_list_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.txt"
            path.write_text("037\n074\n\n  \n")
            self.assertEqual(wide.read_scene_list(path), ["037", "074"])

    def test_enumerate_frames_filters_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "037"
            images = scene_dir / "images"
            images.mkdir(parents=True)
            for frame in ("000", "001", "002"):
                for cam in (5, 4, 3):
                    (images / f"{frame}_{cam}.jpg").write_bytes(b"")
            # Frame 003 is missing camera 4 and must be dropped.
            (images / "003_5.jpg").write_bytes(b"")
            (images / "003_3.jpg").write_bytes(b"")

            frames = wide.enumerate_frames(scene_dir, (5, 4, 3), max_frames=None)
            self.assertEqual(frames, ["000", "001", "002"])

    def test_enumerate_frames_max_and_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "037"
            images = scene_dir / "images"
            images.mkdir(parents=True)
            for frame in ("000", "001", "002"):
                for cam in (5, 4, 3):
                    (images / f"{frame}_{cam}.jpg").write_bytes(b"")

            self.assertEqual(
                wide.enumerate_frames(scene_dir, (5, 4, 3), max_frames=1), ["000"]
            )
            self.assertEqual(
                wide.enumerate_frames(scene_dir, (5, 4, 3), frame="1"), ["001"]
            )
            self.assertEqual(
                wide.enumerate_frames(scene_dir, (5, 4, 3), frame="000"), ["000"]
            )

    def test_enumerate_frames_negative_max_returns_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "037"
            images = scene_dir / "images"
            images.mkdir(parents=True)
            for frame in ("000", "001", "002"):
                for cam in (5, 4, 3):
                    (images / f"{frame}_{cam}.jpg").write_bytes(b"")

            frames = wide.enumerate_frames(scene_dir, (5, 4, 3), max_frames=-1)
            self.assertEqual(frames, ["000", "001", "002"])
            self.assertEqual(
                wide.enumerate_frames(scene_dir, (5, 4, 3), max_frames=None),
                ["000", "001", "002"],
            )

    def test_enumerate_frames_missing_dir(self):
        self.assertEqual(
            wide.enumerate_frames(Path("/nonexistent/scene"), (5, 4, 3)), []
        )


class IntrinsicsParsingTest(unittest.TestCase):
    def test_parse_9_values(self):
        text = "\n".join(str(float(i)) for i in range(9))
        k = wide.parse_pixel_intrinsics(text)
        self.assertEqual((k.fx, k.fy, k.cx, k.cy), (0.0, 1.0, 2.0, 3.0))

    def test_parse_comma_separated(self):
        k = wide.parse_pixel_intrinsics("1,2,3,4,5,6,7,8,9")
        self.assertEqual((k.fx, k.fy, k.cx, k.cy), (1.0, 2.0, 3.0, 4.0))

    def test_parse_too_few_values(self):
        with self.assertRaises(ValueError):
            wide.parse_pixel_intrinsics("1 2 3")


class Dinov2SourceTest(unittest.TestCase):
    """Offline DINOv2 source resolution/validation (no network fallback)."""

    def _make_source(self, tmp, with_hubconf=True):
        source = Path(tmp) / "facebookresearch_dinov2_main"
        source.mkdir(parents=True)
        if with_hubconf:
            (source / wide.HUB_MARKER).write_text("dependencies = ['torch']\n")
        return source

    def test_valid_source_with_hubconf(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._make_source(tmp)
            self.assertTrue(wide.is_valid_dinov2_source(source))
            self.assertEqual(wide.validate_dinov2_source(source), source)
            self.assertEqual(wide.resolve_dinov2_source(str(source)), source)

    def test_source_without_hubconf_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self._make_source(tmp, with_hubconf=False)
            self.assertFalse(wide.is_valid_dinov2_source(source))
            with self.assertRaises(ValueError):
                wide.validate_dinov2_source(source)
            with self.assertRaises(SystemExit):
                wide.resolve_dinov2_source(str(source))

    def test_missing_directory_rejected(self):
        missing = Path("/nonexistent/dinov2/source")
        self.assertFalse(wide.is_valid_dinov2_source(missing))
        with self.assertRaises(ValueError):
            wide.validate_dinov2_source(missing)

    def test_none_source_fails_loudly(self):
        with self.assertRaises(SystemExit) as ctx:
            wide.resolve_dinov2_source(None)
        message = str(ctx.exception)
        self.assertIn("--dinov2-source", message)
        self.assertIn(wide.HUB_MARKER, message)

    def test_default_prefers_env_var(self):
        env = {wide.DINOV2_ENV_VAR: "/tmp/custom_dinov2"}
        self.assertEqual(
            wide.default_dinov2_source(env), Path("/tmp/custom_dinov2")
        )

    def test_default_none_when_cache_missing(self):
        # An empty environment with a non-existent cache returns None.
        # ``default_dinov2_source`` resolves its cache constant from the helper
        # module, so patch it there.
        helper = wide._dinov2_source
        original = helper.DEFAULT_DINOV2_HUB_CACHE
        helper.DEFAULT_DINOV2_HUB_CACHE = Path("/nonexistent/dinov2/cache")
        try:
            self.assertIsNone(wide.default_dinov2_source(env={}))
        finally:
            helper.DEFAULT_DINOV2_HUB_CACHE = original


class ComposeConfigDinov2Test(unittest.TestCase):
    """The inference config overrides force local, non-pretrained DINOv2."""

    def setUp(self):
        try:
            import hydra  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("hydra is required to compose the repository config")

    def test_compose_sets_local_source_and_disables_pretrained(self):
        args = wide.build_arg_parser().parse_args([])
        preset = wide.MODEL_PRESETS[wide.DEFAULT_PRESET]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "facebookresearch_dinov2_main"
            source.mkdir(parents=True)
            (source / wide.HUB_MARKER).write_text("dependencies = ['torch']\n")
            cfg_dict = wide.compose_config_dict(args, preset, source)
        self.assertEqual(str(cfg_dict.model.encoder.dinov2_source), str(source))
        self.assertFalse(bool(cfg_dict.model.encoder.dinov2_pretrained))

    def test_compose_without_source_keeps_training_defaults(self):
        args = wide.build_arg_parser().parse_args([])
        preset = wide.MODEL_PRESETS[wide.DEFAULT_PRESET]
        cfg_dict = wide.compose_config_dict(args, preset, None)
        self.assertIsNone(cfg_dict.model.encoder.dinov2_source)
        self.assertTrue(bool(cfg_dict.model.encoder.dinov2_pretrained))


class CliDinov2DefaultTest(unittest.TestCase):
    def test_parser_exposes_dinov2_source_option(self):
        parser = wide.build_arg_parser()
        args = parser.parse_args([])
        self.assertTrue(hasattr(args, "dinov2_source"))

    def test_explicit_source_is_parsed(self):
        parser = wide.build_arg_parser()
        args = parser.parse_args(["--dinov2-source", "/tmp/some/source"])
        self.assertEqual(args.dinov2_source, "/tmp/some/source")


class ParserDefaultsTest(unittest.TestCase):
    """The CLI defaults must describe the full, no-input-saving run."""

    def setUp(self):
        self.args = wide.build_arg_parser().parse_args([])

    def test_model_preset_default(self):
        self.assertEqual(self.args.model, "256x448")
        self.assertEqual(wide.DEFAULT_PRESET, "256x448")
        # Input size defaults to the model preset (no explicit override)...
        self.assertIsNone(self.args.input_size)
        self.assertIsNone(self.args.height)
        self.assertIsNone(self.args.width)
        # ...so the effective default input is the 256x448 preset size.
        preset = wide.MODEL_PRESETS[wide.DEFAULT_PRESET]
        self.assertEqual(wide.resolve_input_hw(self.args, preset), (256, 448))

    def test_explicit_large_preset_still_supported(self):
        args = wide.build_arg_parser().parse_args(
            ["--model", "448x768", "--input-size", "448x768"]
        )
        self.assertEqual(args.model, "448x768")
        self.assertEqual(args.input_size, (448, 768))
        preset = wide.MODEL_PRESETS["448x768"]
        self.assertEqual(wide.resolve_input_hw(args, preset), (448, 768))

    def test_resolution_is_legacy_alias_for_model(self):
        parser = wide.build_arg_parser()
        legacy = parser.parse_args(["--resolution", "256x448"])
        modern = parser.parse_args(["--model", "256x448"])
        self.assertEqual(legacy.model, "256x448")
        self.assertEqual(legacy.model, modern.model)

    def test_scene_list_default(self):
        self.assertEqual(
            self.args.scene_list,
            "datasets/nuscenes/processed_10Hz/trainval2/nuScenes_Val2.txt",
        )
        self.assertEqual(
            self.args.scene_list, str(wide.DEFAULT_SCENE_LIST_PATH)
        )
        # No scene/frame restriction by default.
        self.assertIsNone(self.args.scene)
        self.assertIsNone(self.args.frame)

    def test_max_frames_defaults_to_all(self):
        self.assertEqual(self.args.max_frames, -1)
        self.assertEqual(wide.DEFAULT_MAX_FRAMES, -1)

    def test_save_inputs_off_by_default(self):
        self.assertFalse(self.args.save_inputs)

    def test_camera_defaults(self):
        self.assertEqual(self.args.cameras, "5,4,3")
        self.assertEqual(self.args.render_camera, 5)
        self.assertEqual(self.args.width_factor, 2.0)

    def test_output_dir_default(self):
        self.assertEqual(self.args.output_dir, "outputs/nuscenes_wide")


class ParseInputSizeTest(unittest.TestCase):
    """``--input-size`` accepts HxW (and H,W / H W) and rejects bad values."""

    def test_parses_x_separator(self):
        self.assertEqual(wide.parse_hw("448x768"), (448, 768))
        self.assertEqual(wide.parse_hw("256x448"), (256, 448))

    def test_parses_other_separators_and_case(self):
        self.assertEqual(wide.parse_hw("448,768"), (448, 768))
        self.assertEqual(wide.parse_hw("448 768"), (448, 768))
        self.assertEqual(wide.parse_hw("448X768"), (448, 768))

    def test_rejects_malformed(self):
        import argparse

        for bad in ("448", "448x", "a x b", "448x768x1", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                wide.parse_hw(bad)

    def test_rejects_non_positive(self):
        import argparse

        for bad in ("0x768", "448x0", "-1x768"):
            with self.assertRaises(argparse.ArgumentTypeError):
                wide.parse_hw(bad)

    def test_cli_parses_input_size(self):
        args = wide.build_arg_parser().parse_args(
            ["--model", "256x448", "--input-size", "256x448"]
        )
        self.assertEqual(args.model, "256x448")
        self.assertEqual(args.input_size, (256, 448))

    def test_cli_rejects_bad_input_size(self):
        with self.assertRaises(SystemExit):
            wide.build_arg_parser().parse_args(["--input-size", "nope"])

    def test_cli_rejects_unknown_model(self):
        with self.assertRaises(SystemExit):
            wide.build_arg_parser().parse_args(["--model", "512x960"])


class ResolveInputHwTest(unittest.TestCase):
    """Input size is independent from the model preset but defaults to it."""

    def setUp(self):
        self.preset = wide.MODEL_PRESETS["256x448"]

    def _args(self, **kwargs):
        base = dict(
            model="256x448", input_size=None, height=None, width=None
        )
        base.update(kwargs)
        return type("Args", (), base)

    def test_defaults_to_preset_size(self):
        self.assertEqual(wide.resolve_input_hw(self._args(), self.preset), (256, 448))

    def test_input_size_overrides_both(self):
        args = self._args(input_size=(448, 768))
        self.assertEqual(wide.resolve_input_hw(args, self.preset), (448, 768))

    def test_height_width_override_individual_dimensions(self):
        args = self._args(input_size=(448, 768), height=320)
        self.assertEqual(wide.resolve_input_hw(args, self.preset), (320, 768))
        args = self._args(input_size=(448, 768), width=1024)
        self.assertEqual(wide.resolve_input_hw(args, self.preset), (448, 1024))
        args = self._args(height=320, width=1024)
        self.assertEqual(wide.resolve_input_hw(args, self.preset), (320, 1024))


class ValidateCheckpointPresetTest(unittest.TestCase):
    """A known other-preset checkpoint must be rejected with a clear error."""

    def test_other_preset_checkpoint_rejected(self):
        wrong = wide.resolve_local(wide.MODEL_PRESETS["448x768"].checkpoint)
        with self.assertRaises(SystemExit) as ctx:
            wide.validate_checkpoint_preset(wrong, "256x448")
        message = str(ctx.exception)
        self.assertIn("--model 448x768", message)
        self.assertIn("--checkpoint", message)

    def test_matching_preset_checkpoint_allowed(self):
        right = wide.resolve_local(wide.MODEL_PRESETS["448x768"].checkpoint)
        wide.validate_checkpoint_preset(right, "448x768")

    def test_arbitrary_custom_checkpoint_allowed(self):
        # Unknown filenames are deferred to the strict encoder load.
        wide.validate_checkpoint_preset(
            Path("/tmp/my_custom_checkpoint.pth"), "256x448"
        )


class ExtrinsicsSourceTest(unittest.TestCase):
    """Default extrinsics source is the static cam2ego rig; per_frame is opt-in."""

    def test_parser_default_is_cam2ego(self):
        args = wide.build_arg_parser().parse_args([])
        self.assertEqual(args.extrinsics_source, "cam2ego")
        self.assertEqual(wide.DEFAULT_EXTRINSICS_SOURCE, "cam2ego")
        self.assertEqual(wide.EXTRINSICS_SOURCE_CHOICES, ("cam2ego", "per_frame"))

    def test_parser_accepts_per_frame(self):
        args = wide.build_arg_parser().parse_args(
            ["--extrinsics-source", "per_frame"]
        )
        self.assertEqual(args.extrinsics_source, "per_frame")

    def test_parser_rejects_unknown_source(self):
        with self.assertRaises(SystemExit):
            wide.build_arg_parser().parse_args(
                ["--extrinsics-source", "global"]
            )

    def test_resolve_path_cam2ego_default(self):
        # No frame component: the static rig is per camera, not per frame.
        self.assertEqual(
            wide.resolve_extrinsics_path(Path("/scene"), "006", 3),
            Path("/scene/cam2ego_extrinsics/3.txt"),
        )
        self.assertEqual(
            wide.resolve_extrinsics_path(Path("/scene"), "006", 3, "cam2ego"),
            Path("/scene/cam2ego_extrinsics/3.txt"),
        )

    def test_resolve_path_per_frame(self):
        self.assertEqual(
            wide.resolve_extrinsics_path(Path("/scene"), "006", 3, "per_frame"),
            Path("/scene/extrinsics/006_3.txt"),
        )

    def test_resolve_path_rejects_unknown(self):
        with self.assertRaises(ValueError):
            wide.resolve_extrinsics_path(Path("/scene"), "006", 3, "global")


class CarMaskParserDefaultsTest(unittest.TestCase):
    """Single-frame masking is on by default with the shared policy flags."""

    def setUp(self):
        self.args = wide.build_arg_parser().parse_args([])

    def test_defaults(self):
        self.assertEqual(
            self.args.car_mask_root,
            "datasets/nuscenes/processed_10Hz/nuscenes_mask",
        )
        self.assertEqual(
            self.args.car_mask_root, str(wide.DEFAULT_CAR_MASK_ROOT)
        )
        self.assertFalse(self.args.mask_render_view)
        self.assertFalse(self.args.disable_car_mask)

    def test_mask_render_view_flag(self):
        args = wide.build_arg_parser().parse_args(["--mask-render-view"])
        self.assertTrue(args.mask_render_view)
        self.assertFalse(args.disable_car_mask)

    def test_disable_flag(self):
        args = wide.build_arg_parser().parse_args(["--disable-car-mask"])
        self.assertTrue(args.disable_car_mask)
        self.assertFalse(args.mask_render_view)

    def test_car_mask_root_override(self):
        args = wide.build_arg_parser().parse_args(
            ["--car-mask-root", "/some/other/masks"]
        )
        self.assertEqual(args.car_mask_root, "/some/other/masks")

    def test_extrinsics_source_default_unchanged(self):
        # The mask work must not change the cam2ego default.
        self.assertEqual(self.args.extrinsics_source, "cam2ego")


class SharedMaskConstantsTest(unittest.TestCase):
    """The single-frame script is the single source of the mask constants."""

    def test_camera_mapping(self):
        self.assertEqual(
            wide.CAMERA_MASK_FILES,
            {
                0: "CAM_FRONT_mask.png",
                1: "CAM_FRONT_LEFT_mask.png",
                2: "CAM_FRONT_RIGHT_mask.png",
                3: "CAM_BACK_LEFT_mask.png",
                4: "CAM_BACK_RIGHT_mask.png",
                5: "CAM_BACK_mask.png",
            },
        )

    def test_policy_names(self):
        self.assertEqual(
            wide.CAR_MASK_POLICY_ALL_EXCEPT_RENDER, "all_except_render_view"
        )
        self.assertEqual(wide.CAR_MASK_POLICY_ALL, "all_views")

    def test_disable_precedence(self):
        self.assertIsNone(wide.resolve_car_mask_policy(True, False))
        self.assertIsNone(wide.resolve_car_mask_policy(True, True))
        self.assertEqual(
            wide.resolve_car_mask_policy(False, False),
            wide.CAR_MASK_POLICY_ALL_EXCEPT_RENDER,
        )
        self.assertEqual(
            wide.resolve_car_mask_policy(False, True), wide.CAR_MASK_POLICY_ALL
        )


class SingleFrameCarMaskPolicyTest(unittest.TestCase):
    """Default masks cams 4/3; render cam 5 is preserved."""

    CAMERAS = (5, 4, 3)
    DST_HW = (2, 2)
    # Single frame -> render index is cameras.index(render_camera) = 0 for cam5.
    RENDER_INDEX = 0

    def _masks(self):
        return {
            5: np.array([[True, True], [True, False]], dtype=bool),
            4: np.array([[False, True], [True, True]], dtype=bool),
            3: np.array([[True, False], [False, True]], dtype=bool),
        }

    def test_default_masks_cam4_and_cam3_keeps_cam5(self):
        masks = self._masks()
        keep = wide.build_car_keep_mask(
            masks, self.CAMERAS, 1, self.DST_HW, self.RENDER_INDEX
        )
        self.assertEqual(keep.shape, (3, 2, 2))
        self.assertTrue(keep[0].all())  # render cam 5 preserved
        np.testing.assert_array_equal(keep[1], masks[4])
        np.testing.assert_array_equal(keep[2], masks[3])
        self.assertEqual([v for v in range(3) if keep[v].all()], [0])

    def test_mask_render_view_masks_all_cameras(self):
        masks = self._masks()
        keep = wide.build_car_keep_mask(
            masks, self.CAMERAS, 1, self.DST_HW, self.RENDER_INDEX,
            mask_render_view=True,
        )
        for view, cam in enumerate(self.CAMERAS):
            np.testing.assert_array_equal(keep[view], masks[cam])
        self.assertFalse(keep[0].all())

    def test_default_removes_only_side_camera_pixels(self):
        masks = self._masks()
        keep = wide.build_car_keep_mask(
            masks, self.CAMERAS, 1, self.DST_HW, self.RENDER_INDEX
        )
        removed = int((~keep).sum())
        expected = int((~masks[4]).sum()) + int((~masks[3]).sum())
        self.assertEqual(removed, expected)

    def test_render_camera_index_matches_list_index(self):
        # For a different render camera the preserved view follows the list.
        masks = self._masks()
        keep = wide.build_car_keep_mask(
            masks, self.CAMERAS, 1, self.DST_HW, list(self.CAMERAS).index(4)
        )
        np.testing.assert_array_equal(keep[0], masks[5])
        self.assertTrue(keep[1].all())
        np.testing.assert_array_equal(keep[2], masks[3])


class SingleFrameMissingMaskTest(unittest.TestCase):
    """A missing required mask is a hard error (no all-ones fallback)."""

    def test_missing_camera_mask_raises_with_path(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            wide.load_camera_keep_masks(
                Path("/nonexistent/mask/root"),
                (5, 4, 3),
                wide.plan_resize_and_crop((2, 2), (2, 2)),
            )
        self.assertIn("CAM_BACK_mask.png", str(ctx.exception))

    def test_partial_root_reports_missing_camera(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("L", (2, 2), 255).save(root / "CAM_BACK_mask.png")
            Image.new("L", (2, 2), 255).save(root / "CAM_BACK_RIGHT_mask.png")
            with self.assertRaises(FileNotFoundError) as ctx:
                wide.load_camera_keep_masks(
                    root, (5, 4, 3), wide.plan_resize_and_crop((2, 2), (2, 2))
                )
            self.assertIn("CAM_BACK_LEFT_mask.png", str(ctx.exception))


class SingleFrameGaussianPruningIntegrationTest(unittest.TestCase):
    """A V=3 keep mask prunes all four Gaussian fields in a synchronized way."""

    @dataclasses.dataclass
    class _Gaussians:
        means: object
        covariances: object
        harmonics: object
        opacities: object

    def setUp(self):
        if torch is None:  # pragma: no cover - environment dependent
            self.skipTest("torch is required to build Gaussian test tensors")

    def _gaussians(self, v, h, w):
        idx = torch.arange(v * h * w, dtype=torch.float32)
        g = idx.numel()
        return self._Gaussians(
            means=idx.view(1, g, 1).repeat(1, 1, 3),
            covariances=idx.view(1, g, 1, 1).repeat(1, 1, 3, 3),
            harmonics=idx.view(1, g, 1, 1).repeat(1, 1, 3, 2),
            opacities=idx.view(1, g),
        )

    def test_default_v3_prunes_cam4_and_cam3_pixels(self):
        v, h, w = 3, 2, 2
        gaussians = self._gaussians(v, h, w)
        masks = {
            5: np.array([[True, True], [True, False]], dtype=bool),
            4: np.array([[False, True], [True, True]], dtype=bool),
            3: np.array([[True, False], [False, True]], dtype=bool),
        }
        keep_mask = wide.build_car_keep_mask(
            masks, (5, 4, 3), 1, (h, w), 0
        )
        keep_flat = wide.expand_keep_mask_to_gaussians(keep_mask, 1)
        self.assertEqual(keep_flat.shape, (v * h * w,))

        pruned = wide.filter_gaussians_by_camera_mask(
            gaussians, keep_mask, v, h, w
        )
        expected_index = np.nonzero(keep_flat)[0]
        self.assertEqual(pruned.means.shape[1], expected_index.size)
        for tensor in (
            pruned.means[0, :, 0],
            pruned.covariances[0, :, 0, 0],
            pruned.harmonics[0, :, 0, 0],
            pruned.opacities[0],
        ):
            np.testing.assert_array_equal(
                tensor.numpy(), expected_index.astype(np.float32)
            )
        self.assertIsInstance(pruned, type(gaussians))

    def test_make_gaussian_filter_closure_uses_v3_mask(self):
        v, h, w = 3, 2, 2
        masks = {
            5: np.ones((h, w), dtype=bool),
            4: np.zeros((h, w), dtype=bool),
            3: np.zeros((h, w), dtype=bool),
        }
        keep_mask = wide.build_car_keep_mask(masks, (5, 4, 3), 1, (h, w), 0)
        callback = wide.make_gaussian_filter(keep_mask)
        gaussians = self._gaussians(v, h, w)
        pruned = callback(gaussians, v, h, w)
        # Only cam5 (view 0) survives: h*w Gaussians remain.
        self.assertEqual(pruned.means.shape[1], h * w)
        self.assertFalse(keep_mask[1].any())
        self.assertFalse(keep_mask[2].any())
        self.assertTrue(keep_mask[0].all())


class ExtrinsicsLoaderTest(unittest.TestCase):
    """The loader reads exactly the file the selected source names."""

    def setUp(self):
        try:
            import PIL  # noqa: F401
        except ImportError:  # pragma: no cover - environment dependent
            self.skipTest("PIL is required to write temporary frame images")

    @staticmethod
    def _matrix_text(value):
        rows = [
            [1.0, 0.0, 0.0, value],
            [0.0, 1.0, 0.0, value + 1.0],
            [0.0, 0.0, 1.0, value + 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        return "\n".join(" ".join(str(v) for v in row) for row in rows)

    def _make_scene(self, tmp, cam2ego_value, per_frame_value):
        from PIL import Image

        scene_dir = Path(tmp) / "037"
        for sub in ("images", "intrinsics", "cam2ego_extrinsics", "extrinsics"):
            (scene_dir / sub).mkdir(parents=True)
        for cam in (5, 4, 3):
            Image.new("RGB", (8, 8), (0, 0, 0)).save(
                scene_dir / "images" / f"000_{cam}.jpg"
            )
            (scene_dir / "intrinsics" / f"{cam}.txt").write_text(
                "1 1 4 4 0 0 0 0 1"
            )
            (scene_dir / "cam2ego_extrinsics" / f"{cam}.txt").write_text(
                self._matrix_text(cam2ego_value)
            )
            (scene_dir / "extrinsics" / f"000_{cam}.txt").write_text(
                self._matrix_text(per_frame_value)
            )
        return scene_dir

    def _load(self, scene_dir, source=None):
        kwargs = {} if source is None else {"extrinsics_source": source}
        return wide.load_frame_inputs(
            scene_dir, "037", "000", (5, 4, 3), 5, (8, 8), (8, 8), **kwargs
        )

    def test_default_uses_cam2ego_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(tmp, 1.0, 100.0)
            inputs = self._load(scene_dir)
            np.testing.assert_allclose(
                inputs.extrinsics[:, :3, 3], np.tile([1.0, 2.0, 3.0], (3, 1))
            )

    def test_per_frame_uses_per_frame_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(tmp, 1.0, 100.0)
            inputs = self._load(scene_dir, "per_frame")
            np.testing.assert_allclose(
                inputs.extrinsics[:, :3, 3],
                np.tile([100.0, 101.0, 102.0], (3, 1)),
            )

    def test_default_missing_cam2ego_fails_with_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(tmp, 1.0, 100.0)
            for cam in (5, 4, 3):
                (scene_dir / "cam2ego_extrinsics" / f"{cam}.txt").unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                self._load(scene_dir)
            message = str(ctx.exception)
            self.assertIn("cam2ego", message)
            self.assertIn("--extrinsics-source per_frame", message)

    def test_per_frame_missing_fails_with_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = self._make_scene(tmp, 1.0, 100.0)
            for cam in (5, 4, 3):
                (scene_dir / "extrinsics" / f"000_{cam}.txt").unlink()
            with self.assertRaises(FileNotFoundError) as ctx:
                self._load(scene_dir, "per_frame")
            message = str(ctx.exception)
            self.assertIn("per-frame", message)
            self.assertIn("--extrinsics-source cam2ego", message)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Standalone multi-frame nuScenes wide-view inference for DepthSplat.

This is the temporal-context counterpart of ``inference_nuscenes_wide.py``.  It
reuses that script's pure helpers (resize/crop, intrinsics, wide-K construction,
enumeration, model construction) but feeds the DepthSplat encoder several
*consecutive* frames at once so it can use temporal context, and renders only the
newest frame, analogous to
``dggt_infer/inference_nuscenes_multiframes.py``.

Causal windows
--------------
For a scene whose valid frames are ``[000, 001, 002, 003]`` and the default
``--num-frames 3`` the processed windows are::

    [000, 001, 002]  -> output 002
    [001, 002, 003]  -> output 003

The window always ends on (and renders) its newest frame, so output frame ids
start at ``num_frames - 1`` (i.e. ``002`` for three frames).  Incomplete windows
at the start of a scene are never rendered: a scene with fewer than
``--num-frames`` valid frames produces no output.  Use ``--frame`` to select a
window by its **newest** (output) frame; ``--max-frames`` limits the source frame
enumeration per scene *before* windowing (so ``--max-frames 3`` yields only the
``[000, 001, 002]`` window).

Extrinsics
----------
Unlike the single-frame script, multi-frame inference deliberately uses the
already-computed per-frame **global** camera-to-world matrices from
``extrinsics/{frame}_{cam}.txt`` directly.  The static ``cam2ego`` rig is *not*
used and the per-frame ego pose is *not* recomputed: each view in a window keeps
its own global C2W matrix so the frames stay in a single, consistent world
frame.  Missing per-frame extrinsics are a hard error (there is no fallback).

Input layout (per scene folder)::

    images/{frame}_{cam}.jpg              # e.g. 000_5.jpg
    intrinsics/{cam}.txt                  # 9 values, first four fx fy cx cy
    extrinsics/{frame}_{cam}.txt          # per-frame 4x4 OpenCV global C2W

Per window the views are flattened **frame-major**::

    [oldest cam5, oldest cam4, oldest cam3, ..., newest cam5, newest cam4,
     newest cam3]

giving a ``[1, num_frames * len(cameras), 3, H, W]`` image tensor with normalized
K per view and the per-frame global C2W per view.  All frames of a window share
one source shape and one resize/crop plan (a shape mismatch fails loudly).

Encoder multi-view matching
---------------------------
DepthSplat's multi-view transformer only selects ``local_mv_match + 1`` nearest
views per reference view (by camera-centre distance) when there are more than
three views.  With the trained default ``local_mv_match=2`` a nine-view window
would mostly match each view against the *temporal same-camera* views (their
centres are close) and drop the lateral cameras entirely.  This script therefore
defaults ``local_mv_match`` to ``num_frames * len(cameras) - 1`` (e.g. ``8``
for ``V = 9``) so every flattened view participates, and exposes
``--local-mv-match`` to override it.  Because the encoder only consults this
setting when ``V > 3`` and then keeps ``local_mv_match + 1`` views, an explicit
``--local-mv-match 0`` is rejected for ``V > 3``: it would keep only each view
itself, leaving an empty cross-view cost volume that can produce NaNs.  For
``V <= 3`` (including single-frame windows) the encoder matches all views, so
``0`` is accepted.

Output layout::

    <output>/<scene>/rgb/{newest_frame}_{render_cam}_wide.jpg   # quality 95

Only the newest frame's render camera is rendered (the current single-frame
naming is kept, e.g. ``002_5_wide.jpg``); no mask/alpha image is invented.  Use a
separate ``--output-dir`` from the single-frame run to avoid overwriting those
files.

See ``README.md`` (section "nuScenes Multi-Frame Wide-View Inference") for usage.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Repository / single-frame helper module loading
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

# Allow `import src.*` when this script is executed directly (the repository
# uses implicit namespace packages, so the repo root must be importable).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_single_frame_module():
    """Load ``inference_nuscenes_wide`` by path without requiring torch.

    The single-frame script is a sibling module, not an installed package, so
    load it directly by path (registering it before execution so dataclasses can
    resolve the module on Python 3.10).  Importing it also makes the repo root
    importable for the ``src.*`` model stack later; nothing torch-related is
    imported here, so ``--help`` still works without torch.
    """
    name = "inference_nuscenes_wide"
    if name in sys.modules:
        return sys.modules[name]
    helper_path = SCRIPT_DIR / "inference_nuscenes_wide.py"
    spec = importlib.util.spec_from_file_location(name, helper_path)
    if spec is None or spec.loader is None:  # pragma: no cover - path dependent
        raise ImportError(f"Could not load single-frame helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wide = _load_single_frame_module()

# Re-exported helper types (plain aliases so static analysis and annotations can
# refer to them without the dynamic ``wide.`` prefix).
PixelIntrinsics = wide.PixelIntrinsics
ResizeCropPlan = wide.ResizeCropPlan

# ---------------------------------------------------------------------------
# Defaults (mirroring the single-frame script where sensible)
# ---------------------------------------------------------------------------

DEFAULT_NUM_FRAMES = 3
DEFAULT_CAMERAS = wide.DEFAULT_CAMERAS  # (5, 4, 3)
DEFAULT_RENDER_CAMERA = wide.DEFAULT_RENDER_CAMERA  # 5
DEFAULT_MAX_FRAMES = wide.DEFAULT_MAX_FRAMES  # -1 == all valid frames
DEFAULT_DATA_ROOT = wide.DEFAULT_DATA_ROOT
DEFAULT_SCENE_LIST_PATH = wide.DEFAULT_SCENE_LIST_PATH
DEFAULT_SCENE_LIST_NAME = wide.DEFAULT_SCENE_LIST_NAME
DEFAULT_WIDTH_FACTOR = wide.DEFAULT_WIDTH_FACTOR
# A distinct default output root so a multi-frame run cannot silently overwrite
# the single-frame run's same-named outputs.
DEFAULT_OUTPUT_DIR = Path("outputs/nuscenes_wide_multiframes")

# The local_mv_match override for a window with V views: V - 1 means "match
# against every other view" (the encoder keeps local_mv_match + 1 neighbours).
# An explicit 0 is only valid for V <= 3 (the encoder ignores the setting there);
# for V > 3 it would leave an empty cross-view cost volume (see
# resolve_local_mv_match).
DEFAULT_LOCAL_MV_MATCH = None  # None -> num_frames * len(cameras) - 1


# ---------------------------------------------------------------------------
# Pure helpers (no torch at import time)
# ---------------------------------------------------------------------------


def enumerate_windows(
    frames: Sequence[str], num_frames: int
) -> list[tuple[str, ...]]:
    """Enumerate causal contiguous windows over already-valid frames.

    For ``frames = [000, 001, 002, 003]`` and ``num_frames = 3`` this returns
    ``[(000, 001, 002), (001, 002, 003)]``: each window is ordered oldest ->
    newest and its **last** element is the frame that gets rendered.  Frames are
    assumed to be in ascending order (as produced by
    ``inference_nuscenes_wide.enumerate_frames``); an incomplete trailing window
    is naturally never produced, and a scene with fewer than ``num_frames`` valid
    frames yields an empty list.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}.")
    frames = list(frames)
    if len(frames) < num_frames:
        return []
    return [
        tuple(frames[start : start + num_frames])
        for start in range(len(frames) - num_frames + 1)
    ]


def flatten_window_views(
    window: Sequence[str], cameras: Sequence[int]
) -> list[tuple[str, int]]:
    """Flatten a window frame-major into ``[(frame, cam), ...]``.

    Order is ``[oldest cam0, oldest cam1, ..., newest cam0, ...]`` which is the
    order of the ``[1, T*V, ...]`` tensors handed to the encoder.
    """
    return [(frame, int(cam)) for frame in window for cam in cameras]


def newest_render_index(
    num_frames: int, cameras: Sequence[int], render_camera: int
) -> int:
    """Index of the newest frame's render camera in the flattened view list."""
    camera_list = [int(cam) for cam in cameras]
    if render_camera not in camera_list:
        raise ValueError(
            f"render_camera {render_camera} is not among cameras {tuple(camera_list)}."
        )
    return (int(num_frames) - 1) * len(camera_list) + camera_list.index(render_camera)


def per_frame_extrinsics_path(scene_dir: Path, frame: str, cam: int) -> Path:
    """Path of one view's per-frame global OpenCV camera-to-world matrix."""
    return Path(scene_dir) / "extrinsics" / f"{frame}_{cam}.txt"


def select_windows(
    frames: Sequence[str], num_frames: int, frame: Optional[str] = None
) -> list[tuple[str, ...]]:
    """Window ``frames``; with ``frame`` keep the window whose newest is that id.

    ``frames`` is the already-enumerated, ascending list of valid frames for a
    scene (``--max-frames`` applied upstream).  ``frame`` is normalized to the
    on-disk padded form against that list.  A requested frame with fewer than
    ``num_frames - 1`` valid predecessors has no complete window and yields an
    empty list (never a partially-filled window).
    """
    windows = enumerate_windows(frames, num_frames)
    if frame is None:
        return windows
    wanted = wide.normalize_frame_id(frame, available=frames)
    return [window for window in windows if window[-1] == wanted]


def resolve_local_mv_match(
    args, num_frames: int, cameras: Sequence[int]
) -> int:
    """Resolve the encoder ``local_mv_match`` (an explicit flag, else ``V - 1``).

    ``V = num_frames * len(cameras)`` is the flattened view count.  For ``V > 3``
    the encoder selects the ``local_mv_match + 1`` nearest views per reference
    view, so ``local_mv_match=0`` keeps only each view itself: the cross-view cost
    volume is empty and the depth decoder can produce NaNs.  An explicit ``0`` is
    therefore rejected for ``V > 3``.  It stays valid for ``V <= 3`` (including
    single-frame windows), where the encoder matches all views and ignores this
    setting entirely.
    """
    requested = getattr(args, "local_mv_match", None)
    num_views = int(num_frames) * len(list(cameras))
    if requested is None:
        value = num_views - 1
    else:
        value = int(requested)
    if value < 0:
        raise SystemExit(f"--local-mv-match must be >= 0, got {value}.")
    if value == 0 and num_views > 3:
        raise SystemExit(
            "--local-mv-match 0 is invalid for more than 3 views "
            f"(V = {num_views} = num_frames * len(cameras)): the encoder would "
            "keep only each view itself, leaving an empty cross-view cost volume "
            "that can produce NaNs. Use a value >= 1 (the default V - 1 matches "
            "every other view)."
        )
    return value


# ---------------------------------------------------------------------------
# Window loading
# ---------------------------------------------------------------------------


@dataclass
class MultiFrameInputs:
    """All flattened views of one causal window.

    Attribute names intentionally match the single-frame ``FrameInputs`` so the
    single-frame ``_render_wide_impl`` can render these inputs unchanged.
    """

    scene: str
    frames: tuple[str, ...]  # oldest -> newest
    images: np.ndarray  # [T*V, H, W, 3] float32
    intrinsics: np.ndarray  # [T*V, 3, 3] normalized
    extrinsics: np.ndarray  # [T*V, 4, 4] per-frame global OpenCV C2W
    resize_plan: ResizeCropPlan
    render_intrinsics_px: PixelIntrinsics  # newest render cam's resized pixel K
    render_index: int  # position of (newest frame, render camera) in the flattening

    @property
    def newest_frame(self) -> str:
        return self.frames[-1]


def load_window_inputs(
    scene_dir: Path,
    scene: str,
    window: Sequence[str],
    cameras: Sequence[int],
    render_camera: int,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> MultiFrameInputs:
    """Load one window's images/intrinsics/extrinsics, frame-major flattened.

    Every frame uses its own per-frame global camera-to-world matrix
    ``extrinsics/{frame}_{cam}.txt`` (the single-frame ``cam2ego`` rig is not
    consulted).  All frames share the resize/crop plan built from ``src_hw``;
    ``load_resized_rgb`` raises if a frame's source shape differs.
    """
    plan = wide.plan_resize_and_crop(src_hw, dst_hw)
    newest_frame = window[-1]
    render_index = newest_render_index(len(window), cameras, render_camera)

    images: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    render_intrinsics_px: Optional[PixelIntrinsics] = None

    for frame in window:
        for cam in cameras:
            image_path = Path(scene_dir) / "images" / f"{frame}_{cam}.jpg"
            intrinsics_path = Path(scene_dir) / "intrinsics" / f"{cam}.txt"
            extrinsics_path = per_frame_extrinsics_path(scene_dir, frame, cam)

            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image: {image_path}")
            if not intrinsics_path.is_file():
                raise FileNotFoundError(f"Missing intrinsics: {intrinsics_path}")
            if not extrinsics_path.is_file():
                raise FileNotFoundError(
                    "Missing per-frame global extrinsics: "
                    f"{extrinsics_path}. Multi-frame inference uses the "
                    "per-frame camera-to-world matrices "
                    "extrinsics/{frame}_{cam}.txt directly for every frame "
                    "(cam2ego is intentionally not used)."
                )

            pixel_k = wide.read_pixel_intrinsics(intrinsics_path)
            resized_k = wide.resize_and_crop_intrinsics(pixel_k, plan)
            normalized_k = wide.pixel_to_normalized_intrinsics(
                resized_k, dst_hw[0], dst_hw[1]
            )

            images.append(wide.load_resized_rgb(image_path, plan))
            intrinsics.append(normalized_k)
            extrinsics.append(wide.read_matrix(extrinsics_path, size=4))

            if frame == newest_frame and int(cam) == int(render_camera):
                render_intrinsics_px = resized_k

    if render_intrinsics_px is None:  # pragma: no cover - cameras already checked
        raise ValueError(
            f"Render camera {render_camera} is not among cameras {tuple(cameras)}."
        )

    return MultiFrameInputs(
        scene=scene,
        frames=tuple(window),
        images=np.stack(images, axis=0).astype(np.float32),
        intrinsics=np.stack(intrinsics, axis=0).astype(np.float32),
        extrinsics=np.stack(extrinsics, axis=0).astype(np.float32),
        resize_plan=plan,
        render_intrinsics_px=render_intrinsics_px,
        render_index=render_index,
    )


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference_nuscenes_wide_multiframes.py",
        description=(
            "nuScenes multi-frame wide-view rendering with DepthSplat. Feeds "
            "num-frames consecutive frames x cameras 5,4,3 to the encoder and "
            "renders a widened image from the newest frame's camera 5. By "
            "default every scene in the scene list and every complete "
            "num-frames window is processed. Extrinsics are the per-frame "
            "global camera-to-world matrices extrinsics/{frame}_{cam}.txt."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Directory containing the scene folders (relative to repo root).",
    )
    data.add_argument(
        "--scene-list",
        default=str(DEFAULT_SCENE_LIST_PATH),
        help=(
            "Scene id list (one scene id per line). Defaults to "
            "<data-root>/" + DEFAULT_SCENE_LIST_NAME + ", i.e. "
            f"{DEFAULT_SCENE_LIST_PATH}. Every scene in the list is processed."
        ),
    )
    data.add_argument(
        "--scene",
        default=None,
        help="Process a single scene id instead of the scene list.",
    )
    data.add_argument(
        "--frame",
        default=None,
        help=(
            "Process the window whose NEWEST (output) frame is this id (e.g. 2 "
            "or 002) for every selected scene. A frame without num_frames-1 "
            "valid predecessors is skipped."
        ),
    )
    data.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=(
            "Maximum valid source frames per scene, applied BEFORE windowing; "
            "-1 (default) enumerates every valid frame. For example "
            "--num-frames 3 --max-frames 3 yields only the [000,001,002] window."
        ),
    )
    data.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=(
            "Number of consecutive frames per inference window (causal: the "
            "newest frame is rendered). Must be >= 1."
        ),
    )
    data.add_argument(
        "--cameras",
        default=",".join(str(c) for c in DEFAULT_CAMERAS),
        help="Comma separated context camera ids (order preserved).",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model",
        "--resolution",
        dest="model",
        choices=sorted(wide.MODEL_PRESETS),
        default=wide.DEFAULT_PRESET,
        help=(
            "Model architecture preset; selects vitb/num_scales/upsample_factor/"
            "gaussian_scale_max and the matching default checkpoint. Defaults to "
            "the smaller local preset; use --input-size to resize independently. "
            "--resolution is a legacy alias for --model."
        ),
    )
    model.add_argument(
        "--input-size",
        type=wide.parse_hw,
        metavar="HxW",
        default=None,
        help=(
            "Input resize size as HxW (e.g. 448x768; also H,W or H W). Defaults "
            "to the --model preset size; rounded down to a multiple of the "
            "effective patch size. --height/--width override it per dimension."
        ),
    )
    model.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path (defaults to the one matching --model). It must be "
            "built for the --model architecture or strict loading fails loudly."
        ),
    )
    model.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "Override the input height (highest priority; floored to the "
            "effective patch size)."
        ),
    )
    model.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Override the input width (highest priority; floored to the "
            "effective patch size)."
        ),
    )
    model.add_argument(
        "--gaussian-scale-max",
        type=float,
        default=None,
        help="Optional override of gaussian_adapter.gaussian_scale_max.",
    )
    _default_dinov2_source = wide.default_dinov2_source()
    model.add_argument(
        "--dinov2-source",
        default=(
            str(_default_dinov2_source) if _default_dinov2_source is not None else None
        ),
        help=(
            "Local DINOv2 torch.hub source directory (must contain hubconf.py). "
            f"Defaults to ${wide.DINOV2_ENV_VAR}, else "
            f"{wide.DEFAULT_DINOV2_HUB_CACHE} when it exists. Offline inference "
            "never downloads DINOv2; a missing/invalid source is an error."
        ),
    )
    model.add_argument(
        "--local-mv-match",
        type=int,
        default=DEFAULT_LOCAL_MV_MATCH,
        help=(
            "Override model.encoder.local_mv_match, the number of additional "
            "nearest views matched per reference view (must be >= 0; 0 is only "
            "valid when V = num_frames*len(cameras) <= 3, since for V > 3 it "
            "would keep only each view itself and leave an empty cross-view cost "
            "volume). Defaults to num_frames*len(cameras)-1 so every flattened "
            "view participates (the trained nearest-3 default, 2, would otherwise "
            "mostly select temporal same-camera views and drop the lateral "
            "cameras)."
        ),
    )

    render = parser.add_argument_group("rendering")
    render.add_argument(
        "--render-camera",
        type=int,
        default=DEFAULT_RENDER_CAMERA,
        help="Camera id rendered in the wide view (from the newest frame).",
    )
    render.add_argument(
        "--width-factor",
        type=float,
        default=DEFAULT_WIDTH_FACTOR,
        help="Output width multiplier relative to the model input width.",
    )
    render.add_argument("--near", type=float, default=0.5, help="Near depth bound.")
    render.add_argument("--far", type=float, default=200.0, help="Far depth bound.")

    out = parser.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Output root directory. Use a distinct directory from the "
            "single-frame run: file names are shared "
            "(<scene>/rgb/{newest_frame}_{render_cam}_wide.jpg)."
        ),
    )
    out.add_argument(
        "--save-inputs",
        action="store_true",
        help=(
            "Also save every window's resized context images under "
            "<output>/<scene>/inputs/ (disabled by default)."
        ),
    )
    out.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and load windows only; do not build the model or render.",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument(
        "--device",
        default=None,
        help="Torch device (default: cuda when available, else cpu).",
    )
    runtime.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA float16 autocast (experimental; rasterizer may not support it).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    import torch  # imported here so --help works without torch

    torch.set_float32_matmul_precision("high")

    data_root = wide.resolve_local(args.data_root)
    # Keep the parser's canonical scene-list default relative to --data-root so
    # overriding only --data-root still reads that root's nuScenes_Val2.txt; an
    # explicit --scene-list wins.
    if args.scene_list == str(wide.DEFAULT_SCENE_LIST_PATH):
        scene_list = data_root / DEFAULT_SCENE_LIST_NAME
    else:
        scene_list = wide.resolve_local(args.scene_list)
    output_dir = wide.resolve_local(args.output_dir)
    cameras = wide.parse_cameras(args.cameras)

    num_frames = int(args.num_frames)
    if num_frames < 1:
        raise SystemExit(f"--num-frames must be >= 1, got {num_frames}.")
    if args.render_camera not in cameras:
        raise SystemExit(
            f"--render-camera {args.render_camera} must be one of --cameras {cameras}."
        )

    local_mv_match = resolve_local_mv_match(args, num_frames, cameras)
    print(
        f"[info] Multi-frame windows of {num_frames} frame(s) x cameras "
        f"{cameras}; local_mv_match={local_mv_match}. Using the per-frame "
        "global camera-to-world matrices extrinsics/{frame}_{cam}.txt directly.",
        file=sys.stderr,
    )

    if args.scene is not None:
        scenes = [args.scene]
    else:
        if not scene_list.is_file():
            raise SystemExit(f"Scene list not found: {scene_list}")
        scenes = wide.read_scene_list(scene_list)
    if not scenes:
        raise SystemExit(f"No scenes selected (scene list: {scene_list}).")

    preset = wide.MODEL_PRESETS[args.model]
    requested_hw = wide.resolve_input_hw(args, preset)

    # DINOv2 must resolve to a valid local source for real inference: this path
    # deliberately has no network fallback. ``--dry-run`` never builds a model,
    # so it may proceed without one.
    try:
        dinov2_source = wide.resolve_dinov2_source(args.dinov2_source)
    except SystemExit:
        if not args.dry_run:
            raise
        dinov2_source = None

    cfg_dict = wide.compose_config_dict(
        args, preset, dinov2_source, local_mv_match=local_mv_match
    )
    patch_size = wide.effective_patch_size(cfg_dict)
    dst_hw = wide.round_hw_to_multiple(requested_hw, patch_size)
    if dst_hw != requested_hw:
        print(
            f"[info] Input size {requested_hw[0]}x{requested_hw[1]} rounded to "
            f"{dst_hw[0]}x{dst_hw[1]} (multiple of effective patch {patch_size}).",
            file=sys.stderr,
        )

    checkpoint_path = (
        wide.resolve_local(args.checkpoint)
        if args.checkpoint is not None
        else wide.resolve_local(preset.checkpoint)
    )
    if args.checkpoint is not None:
        wide.validate_checkpoint_preset(checkpoint_path, args.model)
        print(
            f"[info] Using custom --checkpoint {checkpoint_path} with --model "
            f"'{args.model}' architecture; a mismatch fails at strict load.",
            file=sys.stderr,
        )

    # Enumerate and validate jobs before touching the GPU/model.  --max-frames
    # limits the source frame enumeration; windowing (and --frame) then operate
    # on that already-limited list.
    jobs: list[tuple[str, tuple[str, ...]]] = []
    for scene in scenes:
        scene_dir = data_root / scene
        if not scene_dir.is_dir():
            message = f"Scene directory not found: {scene_dir}"
            if args.scene is not None:
                raise SystemExit(message)
            print(f"[warn] {message}, skipping.", file=sys.stderr)
            continue
        frames = wide.enumerate_frames(
            scene_dir, cameras, max_frames=args.max_frames, frame=None
        )
        if not frames:
            print(
                f"[warn] No frames found for scene {scene} with cameras {cameras}.",
                file=sys.stderr,
            )
            continue
        windows = select_windows(frames, num_frames, frame=args.frame)
        if not windows:
            if args.frame is not None:
                print(
                    f"[warn] Scene {scene} has no complete {num_frames}-frame "
                    f"window ending at frame {args.frame}; skipping.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[warn] Scene {scene} has {len(frames)} valid frame(s), "
                    f"fewer than --num-frames {num_frames}; skipping.",
                    file=sys.stderr,
                )
            continue
        for window in windows:
            jobs.append((scene, window))

    if not jobs:
        raise SystemExit("No valid (scene, window) jobs were found.")

    print(
        f"Selected {len(jobs)} window(s) from {len(scenes)} scene(s); "
        f"model '{args.model}', input {dst_hw[0]}x{dst_hw[1]}, "
        f"effective patch {patch_size}, local_mv_match={local_mv_match}."
    )

    # Determine source resolution from the first valid image and validate.
    first_scene, first_window = jobs[0]
    first_image = (
        data_root / first_scene / "images" / f"{first_window[0]}_{cameras[0]}.jpg"
    )
    if not first_image.is_file():
        raise SystemExit(f"Missing first image: {first_image}")
    src_hw = wide.image_size(first_image)
    print(f"Source image resolution: {src_hw[0]}x{src_hw[1]}")

    if args.dry_run:
        with tqdm(
            jobs, total=len(jobs), unit="window", desc="Validating windows (dry-run)"
        ) as progress:
            for index, (scene, window) in enumerate(progress):
                inputs = load_window_inputs(
                    data_root / scene,
                    scene,
                    window,
                    cameras,
                    args.render_camera,
                    src_hw,
                    dst_hw,
                )
                wide_px, out_hw = wide.make_wide_intrinsics(
                    inputs.render_intrinsics_px,
                    (dst_hw[0], dst_hw[1]),
                    args.width_factor,
                )
                progress.set_postfix(
                    scene=scene,
                    out=inputs.newest_frame,
                    wide=f"{out_hw[0]}x{out_hw[1]}",
                )
                # One concrete diagnostic line so a failure can be traced without
                # flooding the terminal.
                if index == 0:
                    progress.write(
                        f"[dry-run] first window scene={scene} "
                        f"frames={inputs.frames} render_index={inputs.render_index} "
                        f"images={inputs.images.shape} "
                        f"extrinsics={inputs.extrinsics.shape} "
                        f"K_norm={inputs.intrinsics.shape} "
                        f"wide={out_hw[0]}x{out_hw[1]} "
                        f"wide_K_px=({wide_px.fx:.2f},{wide_px.fy:.2f},"
                        f"{wide_px.cx:.2f},{wide_px.cy:.2f})"
                    )
        return 0

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise SystemExit(
            "CUDA is required for the Gaussian splatting decoder but was not "
            "available. Pass --device cuda on a machine with a working CUDA "
            "build of diff-gaussian-rasterization."
        )

    _, encoder, decoder = wide.build_model(cfg_dict, checkpoint_path, device)

    render_index = newest_render_index(num_frames, cameras, args.render_camera)
    progress = tqdm(
        jobs, total=len(jobs), unit="window", desc="Rendering wide views"
    )
    try:
        for scene, window in progress:
            newest_frame = window[-1]
            progress.set_postfix(scene=scene, out=newest_frame)
            scene_dir = data_root / scene
            inputs = load_window_inputs(
                scene_dir,
                scene,
                window,
                cameras,
                args.render_camera,
                src_hw,
                dst_hw,
            )
            # ``_render_wide_impl`` is reused unchanged: it only needs the
            # ``images``/``intrinsics``/``extrinsics``/``render_intrinsics_px``
            # attributes, which MultiFrameInputs provides for V = T*len(cameras).
            color, (out_h, out_w) = wide._render_wide_impl(
                inputs,
                encoder,
                decoder,
                render_index,
                args.width_factor,
                args.near,
                args.far,
                device,
                args.amp,
            )
            progress.set_postfix(
                scene=scene, out=newest_frame, res=f"{out_h}x{out_w}"
            )
            save_path = (
                output_dir
                / scene
                / "rgb"
                / f"{newest_frame}_{args.render_camera}_wide.jpg"
            )
            wide.save_rgb(color, save_path, quality=95)

            if args.save_inputs:
                for (frame, cam), image in zip(
                    flatten_window_views(window, cameras), inputs.images
                ):
                    input_path = output_dir / scene / "inputs" / f"{frame}_{cam}.jpg"
                    wide.save_rgb(
                        torch.from_numpy(image).permute(2, 0, 1),
                        input_path,
                        quality=95,
                    )
    finally:
        progress.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

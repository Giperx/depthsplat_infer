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

Ego-car masking (enabled by default, ``--disable-car-mask`` to turn off)
-----------------------------------------------------------------------
Each selected camera has a static ego-car mask under ``--car-mask-root``
(default ``datasets/nuscenes/processed_10Hz/nuscenes_mask``), mapped by camera
id 0..5 to ``CAM_FRONT``/``CAM_FRONT_LEFT``/``CAM_FRONT_RIGHT``/
``CAM_BACK_LEFT``/``CAM_BACK_RIGHT``/``CAM_BACK``.  A mask pixel is removed when
it is black (``<128``) and kept when white (``>=128``); the mask is transformed
with exactly the images' resize/crop plan (NEAREST resize to the scaled size,
then the identical centre crop).  A missing required mask is a hard error (no
all-ones fallback).

Default policy (``all_except_render_view``): each camera's mask is applied to
**every view except the single current/newest render view**.  For a 3-frame
``[5, 4, 3]`` window with render camera 5 (``t3_cam5``) that masks historical
``t1/t2`` cams 5/4/3 and the current ``t3`` cams 4/3, leaving only ``t3_cam5``
fully preserved.  Diagnostic policy (``--mask-render-view``, ``all_views``)
additionally applies the masks to the render view itself, so the preset ego-car
region is removed there too (useful to check for a missing rear ego-car hood).
``--disable-car-mask`` has the highest precedence and disables masking
completely.

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

# Per-camera nuScenes ego-car masks.  Camera ids 0..5 are, in order, CAM_FRONT,
# CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK_RIGHT, CAM_BACK; the
# default cameras 5,4,3 therefore map to CAM_BACK, CAM_BACK_RIGHT, CAM_BACK_LEFT.
DEFAULT_CAR_MASK_ROOT = Path("datasets/nuscenes/processed_10Hz/nuscenes_mask")
CAMERA_MASK_FILES: dict[int, str] = {
    0: "CAM_FRONT_mask.png",
    1: "CAM_FRONT_LEFT_mask.png",
    2: "CAM_FRONT_RIGHT_mask.png",
    3: "CAM_BACK_LEFT_mask.png",
    4: "CAM_BACK_RIGHT_mask.png",
    5: "CAM_BACK_mask.png",
}

# Polarity threshold: black (< 128) is the ego car to remove, white (>= 128)
# is kept.
CAR_MASK_KEEP_THRESHOLD = 128

# Masking policy names (reported in info/dry-run diagnostics).
CAR_MASK_POLICY_ALL_EXCEPT_RENDER = "all_except_render_view"
CAR_MASK_POLICY_ALL = "all_views"


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
# Ego-car mask helpers (no torch at import time)
# ---------------------------------------------------------------------------


def camera_mask_path(mask_root, cam: int) -> Path:
    """Path of ``cam``'s ego-car mask under ``mask_root``.

    Raises ``KeyError`` for a camera without a nuScenes mask mapping (only
    0..5 exist); there is deliberately no fallback mask.
    """
    cam = int(cam)
    if cam not in CAMERA_MASK_FILES:
        raise KeyError(
            f"No ego-car mask is defined for camera {cam}; known nuScenes "
            f"cameras are {sorted(CAMERA_MASK_FILES)}."
        )
    return Path(mask_root) / CAMERA_MASK_FILES[cam]


def load_resized_keep_mask(path, plan: ResizeCropPlan) -> np.ndarray:
    """Load one ego-car mask through exactly the image resize/crop plan.

    The mask is loaded as PIL ``L`` and transformed *identically* to the RGB
    context images in :func:`inference_nuscenes_wide.load_resized_rgb`: a
    NEAREST resize to ``(plan.scaled_w, plan.scaled_h)`` followed by the same
    centre crop ``(plan.col, plan.row)``.  It is never plain-resized straight to
    the destination, so mask pixels stay aligned with the resized/cropped image.
    Returns a ``[plan.out_h, plan.out_w]`` bool array; ``True`` = keep.
    """
    from PIL import Image

    path = Path(path)
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != (plan.src_w, plan.src_h):
            raise ValueError(
                f"Ego-car mask {path} has size {mask.size[1]}x{mask.size[0]} but "
                f"the source images/resize plan are {plan.src_h}x{plan.src_w}; "
                "the mask must match the source image resolution."
            )
        if mask.size != (plan.scaled_w, plan.scaled_h):
            mask = mask.resize((plan.scaled_w, plan.scaled_h), Image.NEAREST)
        mask = mask.crop(
            (plan.col, plan.row, plan.col + plan.out_w, plan.row + plan.out_h)
        )
        array = np.asarray(mask)
    return array >= CAR_MASK_KEEP_THRESHOLD


def load_camera_keep_masks(
    mask_root, cameras: Sequence[int], plan: ResizeCropPlan
) -> dict[int, np.ndarray]:
    """Load and resize every selected camera's keep mask exactly once.

    Fails loudly (``FileNotFoundError``/``ValueError``) when a required mask is
    missing or has the wrong source resolution; masking never silently falls
    back to all-ones.
    """
    masks: dict[int, np.ndarray] = {}
    for cam in cameras:
        cam = int(cam)
        path = camera_mask_path(mask_root, cam)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing ego-car mask for camera {cam}: {path}. Masking is "
                "enabled by default; provide the mask or pass --disable-car-mask "
                "for a controlled comparison that keeps every Gaussian."
            )
        masks[cam] = load_resized_keep_mask(path, plan)
    return masks


def build_car_keep_mask(
    camera_masks: dict[int, np.ndarray],
    cameras: Sequence[int],
    num_frames: int,
    dst_hw: tuple[int, int],
    render_index: int,
    mask_render_view: bool = False,
) -> np.ndarray:
    """Build the ``[num_frames * len(cameras), H, W]`` per-view keep mask.

    Flattening is frame-major (same as the encoder input), so view
    ``v = frame_index * len(cameras) + camera_index`` uses
    ``cameras[camera_index]``.

    Default policy (``mask_render_view=False``): each view uses its own
    camera's keep mask for **every view except** the single current/newest
    render view at ``render_index`` (``t3_cam<render_camera>``), which is kept
    whole.  For a 3-frame ``[5, 4, 3]`` window that masks historical
    ``t1/t2`` cams 5/4/3 and the current ``t3`` cams 4/3, leaving only
    ``t3_cam5`` fully preserved.

    Diagnostic policy (``mask_render_view=True``): the camera mask is applied
    to **all** views, including the render view, so the preset ego-car region is
    removed there too.

    ``render_index`` must address the newest frame of the window; result is
    bool with ``True`` = keep.
    """
    num_frames = int(num_frames)
    cameras = [int(cam) for cam in cameras]
    out_h, out_w = int(dst_hw[0]), int(dst_hw[1])
    num_views = num_frames * len(cameras)
    render_index = int(render_index)
    if not cameras:
        raise ValueError("At least one camera is required to build a mask.")
    if not 0 <= render_index < num_views:
        raise ValueError(
            f"render_index {render_index} is out of range for {num_views} views."
        )
    if render_index // len(cameras) != num_frames - 1:
        raise ValueError(
            f"render_index {render_index} does not address the newest frame "
            f"(frame index {num_frames - 1} of {num_frames}); the render view "
            "must be the current/newest window view."
        )

    keep = np.ones((num_views, out_h, out_w), dtype=bool)
    for view_index in range(num_views):
        if view_index == render_index and not mask_render_view:
            continue
        cam = cameras[view_index % len(cameras)]
        mask = camera_masks.get(cam)
        if mask is None:
            raise KeyError(f"No ego-car mask was loaded for camera {cam}.")
        if mask.shape != (out_h, out_w):
            raise ValueError(
                f"Ego-car mask for camera {cam} has shape {mask.shape} but "
                f"the model input is {out_h}x{out_w}."
            )
        keep[view_index] = mask
    return keep


def resolve_car_mask_policy(
    disable_car_mask: bool, mask_render_view: bool
) -> Optional[str]:
    """Resolve the masking policy name (``--disable-car-mask`` wins).

    Returns ``None`` when masking is disabled, else
    ``CAR_MASK_POLICY_ALL_EXCEPT_RENDER`` (default) or ``CAR_MASK_POLICY_ALL``
    when ``--mask-render-view`` is requested.
    """
    if disable_car_mask:
        return None
    return (
        CAR_MASK_POLICY_ALL if mask_render_view else CAR_MASK_POLICY_ALL_EXCEPT_RENDER
    )


def expand_keep_mask_to_gaussians(
    keep_mask: np.ndarray, multiplicity: int
) -> np.ndarray:
    """Expand a ``[V, H, W]`` pixel keep mask to the flat Gaussian ordering.

    Encoder Gaussians are ordered ``(v, h*w, surface, spp)`` (see
    ``encoder_depthsplat``'s ``b (v r srf spp)`` rearrange), so each pixel's
    keep flag is repeated ``K = num_surfaces * gaussians_per_pixel`` times.
    Returns a flat bool array of length ``V*H*W*K``.
    """
    multiplicity = int(multiplicity)
    if multiplicity < 1:
        raise ValueError(f"multiplicity must be >= 1, got {multiplicity}.")
    mask = np.asarray(keep_mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError(f"keep_mask must be [V, H, W], got shape {mask.shape}.")
    return np.repeat(mask.reshape(-1), multiplicity)


def prune_gaussians_by_keep_flat(gaussians, keep_flat: np.ndarray):
    """Prune a Gaussians object along dim=1 with a flat bool keep mask.

    All four encoder fields (``means``, ``covariances``, ``harmonics``,
    ``opacities``) are indexed identically, and the same Gaussians type is
    reconstructed.  Works for any batch size (the mask is shared across the
    batch).  Raises if no Gaussian survives.
    """
    import torch

    # The keep index must live on the Gaussians' device (they are on CUDA during
    # real inference) for index_select.
    keep = torch.as_tensor(
        np.asarray(keep_flat, dtype=bool), device=gaussians.means.device
    )
    if keep.ndim != 1:
        raise ValueError(f"keep_flat must be 1-D, got shape {keep.shape}.")
    num_gaussians = int(gaussians.means.shape[1])
    if keep.shape[0] != num_gaussians:
        raise ValueError(
            f"keep mask has {keep.shape[0]} entries but the encoder produced "
            f"{num_gaussians} Gaussians (dim=1)."
        )
    index = keep.nonzero(as_tuple=False).reshape(-1)
    if index.numel() == 0:
        raise ValueError(
            "Ego-car masking would remove every Gaussian; refusing to render an "
            "empty scene. Check the mask polarity/root."
        )
    return type(gaussians)(
        means=gaussians.means.index_select(1, index),
        covariances=gaussians.covariances.index_select(1, index),
        harmonics=gaussians.harmonics.index_select(1, index),
        opacities=gaussians.opacities.index_select(1, index),
    )


def filter_gaussians_by_camera_mask(
    gaussians, keep_mask: np.ndarray, num_views: int, height: int, width: int
):
    """Apply a ``[V, H, W]`` keep mask to encoder Gaussians.

    Infers the per-pixel multiplicity ``K = G // (V*H*W)`` from the encoder
    output, validates that ``V*H*W`` divides ``G`` exactly, expands the keep
    mask over ``K`` and prunes all Gaussian fields together.
    """
    num_views, height, width = int(num_views), int(height), int(width)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.shape != (num_views, height, width):
        raise ValueError(
            f"keep_mask shape {keep_mask.shape} does not match "
            f"(V, H, W) = ({num_views}, {height}, {width})."
        )
    expected = num_views * height * width
    num_gaussians = int(gaussians.means.shape[1])
    if expected <= 0 or num_gaussians % expected != 0:
        raise ValueError(
            f"Encoder produced {num_gaussians} Gaussians but V*H*W = "
            f"{num_views}*{height}*{width} = {expected} does not divide it "
            "exactly; the encoder resolution and the mask resolution disagree."
        )
    multiplicity = num_gaussians // expected
    keep_flat = expand_keep_mask_to_gaussians(keep_mask, multiplicity)
    return prune_gaussians_by_keep_flat(gaussians, keep_flat)


def make_gaussian_filter(keep_mask: np.ndarray):
    """Closure for ``_render_wide_impl``'s ``gaussian_filter`` callback.

    The per-view mask is built once and reused for every window; only the
    multiplicity is inferred per call from the encoder's actual output.
    """
    mask = np.asarray(keep_mask, dtype=bool)

    def gaussian_filter(gaussians, num_views, height, width):
        return filter_gaussians_by_camera_mask(
            gaussians, mask, num_views, height, width
        )

    return gaussian_filter


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
    data.add_argument(
        "--car-mask-root",
        default=str(DEFAULT_CAR_MASK_ROOT),
        help=(
            "Directory with the per-camera nuScenes ego-car masks "
            "(CAM_FRONT_mask.png, CAM_FRONT_LEFT_mask.png, CAM_FRONT_RIGHT_mask.png, "
            "CAM_BACK_LEFT_mask.png, CAM_BACK_RIGHT_mask.png, CAM_BACK_mask.png for "
            "cameras 0..5). Black (<128) pixels are removed and white (>=128) kept, "
            "transformed with exactly the same resize/crop plan as the images. "
            "By default each camera's mask is applied to every view EXCEPT the "
            "current/newest render view: historical t1/t2 cams 5/4/3 are masked and "
            "the current t3 cams 4/3 are masked, while only the current render view "
            "(t3 cam5) is fully preserved. A missing mask for any selected camera is "
            "a hard error."
        ),
    )
    data.add_argument(
        "--mask-render-view",
        action="store_true",
        help=(
            "Diagnostic: also apply each camera's mask to the current/newest render "
            "view (t<num_frames-1> cam<render-camera>), so the preset ego-car "
            "region is removed there too. Default off: the render view is fully "
            "preserved."
        ),
    )
    data.add_argument(
        "--disable-car-mask",
        action="store_true",
        help=(
            "Disable ego-car masking entirely (keep every Gaussian from every view) "
            "for a controlled comparison. Highest precedence: it overrides "
            "--mask-render-view and the default policy."
        ),
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

    # Ego-car masking is enabled by default.  The masks are static, T/camera
    # order is fixed and the resize/crop plan is constant, so every mask (and
    # the per-view keep mask) is loaded/built exactly once here, after
    # src_hw/dst_hw are known and before any window is touched.  The render view
    # is the flattened index of the newest frame's render camera.
    render_index = newest_render_index(num_frames, cameras, args.render_camera)
    mask_policy = resolve_car_mask_policy(
        args.disable_car_mask, args.mask_render_view
    )
    gaussian_filter = None
    keep_mask = None
    if mask_policy is None:
        print(
            "[info] Ego-car masking DISABLED (--disable-car-mask, highest "
            "precedence): every Gaussian from every view is kept.",
            file=sys.stderr,
        )
    else:
        mask_root = wide.resolve_local(args.car_mask_root)
        plan = wide.plan_resize_and_crop(src_hw, dst_hw)
        camera_keep_masks = load_camera_keep_masks(mask_root, cameras, plan)
        keep_mask = build_car_keep_mask(
            camera_keep_masks,
            cameras,
            num_frames,
            dst_hw,
            render_index,
            mask_render_view=args.mask_render_view,
        )
        gaussian_filter = make_gaussian_filter(keep_mask)
        retained = int(keep_mask.sum())
        total = int(keep_mask.size)
        if mask_policy == CAR_MASK_POLICY_ALL:
            policy_detail = (
                "every view (including the current render view; diagnostic "
                "--mask-render-view)"
            )
        else:
            policy_detail = (
                "every view except the current/newest render view (newest "
                f"frame cam{args.render_camera}) at render_index={render_index}, "
                "which is fully preserved"
            )
        print(
            f"[info] Ego-car masking enabled from {mask_root} "
            f"(policy={mask_policy}): {policy_detail}; kept {retained}/{total} "
            f"resized mask pixels ({total - retained} removed) over "
            f"V={keep_mask.shape[0]} views.",
            file=sys.stderr,
        )

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
                    if keep_mask is None:
                        mask_info = " car_mask=disabled car_mask_policy=disabled"
                    else:
                        mask_info = (
                            f" car_mask_policy={mask_policy}"
                            f" car_mask_removed={int((~keep_mask).sum())}"
                            f"/{int(keep_mask.size)}"
                            f" render_view_preserved="
                            f"{str(mask_policy == CAR_MASK_POLICY_ALL_EXCEPT_RENDER).lower()}"
                        )
                    progress.write(
                        f"[dry-run] first window scene={scene} "
                        f"frames={inputs.frames} render_index={inputs.render_index} "
                        f"images={inputs.images.shape} "
                        f"extrinsics={inputs.extrinsics.shape} "
                        f"K_norm={inputs.intrinsics.shape} "
                        f"wide={out_hw[0]}x{out_hw[1]} "
                        f"wide_K_px=({wide_px.fx:.2f},{wide_px.fy:.2f},"
                        f"{wide_px.cx:.2f},{wide_px.cy:.2f}){mask_info}"
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
            # The optional callback prunes ego-car Gaussians from historical
            # views between the encoder and the decoder (``None`` when masking
            # is disabled, leaving single-frame behaviour untouched).
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
                gaussian_filter=gaussian_filter,
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

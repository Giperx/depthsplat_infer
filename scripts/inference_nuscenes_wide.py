#!/usr/bin/env python3
"""Standalone nuScenes wide-view inference for DepthSplat.

This script does **not** go through ``ModelWrapper.test_step``.  By default it
processes every scene in the scene list and every valid frame per scene (use
``--max-frames`` / ``--scene`` / ``--frame`` to narrow this).  For each selected
frame it builds the DepthSplat encoder/decoder directly from the repository
configuration, reconstructs Gaussians from cameras ``[5, 4, 3]`` and renders a
horizontally widened image from camera ``5``.

Rendering convention
--------------------
The context images are resized (aspect preserving) and centre-cropped to the
model input resolution, and each camera's pixel intrinsics are adjusted through
exactly the same resize/crop.  The wide render keeps the resized camera-5 focal
length (``fx``, ``fy``) and ``cy`` and only moves the principal point to the
centre of the wider canvas (``cx = wide_W / 2``).  This widens the horizontal
field of view while preserving the per-pixel angular scale and the output
height.  The resulting pixel K is converted to the normalized K expected by
DepthSplat (first row ``/ width``, second row ``/ height``).  The decoder's
projection builds an asymmetric frustum from that normalized K, so an
off-centre ``cx``/``cy`` is honoured instead of being approximated by a
symmetric field of view.

Input layout (per scene folder)::

    images/{frame}_{cam}.jpg              # e.g. 000_5.jpg
    intrinsics/{cam}.txt                  # 9 values, first four fx fy cx cy
    cam2ego_extrinsics/{cam}.txt          # static rig 4x4 OpenCV camera-to-ego
    extrinsics/{frame}_{cam}.txt          # per-frame 4x4 OpenCV global C2W

Extrinsics source (``--extrinsics-source``)
-------------------------------------------
``cam2ego`` (default) reads the static per-camera rig transform
``cam2ego_extrinsics/{cam}.txt`` and uses it directly as OpenCV
camera-to-world: the per-frame ego frame *is* the world here, so the rig is
frame-independent and every independently processed frame reconstructs in the
same ego/world frame.  ``per_frame`` instead reads
``extrinsics/{frame}_{cam}.txt``, a per-frame *global* camera-to-world matrix
(ego pose baked in).  That global source is not always exactly
``ego_pose @ cam2ego`` in this processed data, so it can yield a frame-dependent
rig and a slightly different reconstruction; it is offered only for explicit A/B
comparison and is never used as a silent fallback.

Output layout::

    <output>/<scene>/rgb/{frame}_5_wide.jpg   # quality 95

See ``README.md`` (section "nuScenes Wide-View Inference") for usage details.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (no workstation-specific paths)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# Allow `import src.*` when this script is executed directly (the repository
# uses implicit namespace packages, so the repo root must be importable).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_ROOT = Path("datasets/nuscenes/processed_10Hz/trainval2")
DEFAULT_SCENE_LIST_NAME = "nuScenes_Val2.txt"
DEFAULT_SCENE_LIST_PATH = DEFAULT_DATA_ROOT / DEFAULT_SCENE_LIST_NAME
DEFAULT_MAX_FRAMES = -1  # -1 == process every valid frame of every selected scene
DEFAULT_OUTPUT_DIR = Path("outputs/nuscenes_wide")

DEFAULT_CAMERAS = (5, 4, 3)
DEFAULT_RENDER_CAMERA = 5
DEFAULT_WIDTH_FACTOR = 2.0

# How OpenCV camera-to-world extrinsics are obtained.  ``cam2ego`` reads the
# static per-camera rig transform ``cam2ego_extrinsics/{cam}.txt`` (ego is the
# per-frame world), so reconstruction is frame-independent.  ``per_frame`` reads
# the per-frame global ``extrinsics/{frame}_{cam}.txt`` and exists only for
# explicit A/B comparison of this processed data.
DEFAULT_EXTRINSICS_SOURCE = "cam2ego"
EXTRINSICS_SOURCE_CHOICES = ("cam2ego", "per_frame")

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

# The dl3dv experiment applies a patch shim of ``shim_patch_size`` at a
# ``downscale_factor`` stride, so images must be a multiple of the *effective*
# patch size ``shim_patch_size * downscale_factor`` (= 16 * 4 = 64 here).  This
# is intentionally separate from DINOv2's 14-pixel ViT patch: the encoder
# floors the image to a multiple of 14 internally, but the explicit crop that
# protects the cost volume / UNet alignment is the effective patch size.
FALLBACK_EFFECTIVE_PATCH_SIZE = 64


# ---------------------------------------------------------------------------
# DINOv2 local source helpers (torch-free)
# ---------------------------------------------------------------------------
#
# The canonical helpers live in
# ``src/model/encoder/unimatch/dinov2_source.py``.  A normal import of that
# module would execute ``src/model/encoder/__init__.py``, which imports torch;
# this script (and ``--help``) must work without torch, so load the standalone
# module directly by path and re-export its helpers.


def _load_dinov2_source_helpers():
    import importlib.util

    helper_path = (
        REPO_ROOT
        / "src"
        / "model"
        / "encoder"
        / "unimatch"
        / "dinov2_source.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_depthsplat_dinov2_source", helper_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DINOv2 source helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_dinov2_source = _load_dinov2_source_helpers()

HUB_MARKER = _dinov2_source.HUB_MARKER
DINOV2_ENV_VAR = _dinov2_source.DINOV2_ENV_VAR
DEFAULT_DINOV2_HUB_CACHE = _dinov2_source.DEFAULT_DINOV2_HUB_CACHE
is_valid_dinov2_source = _dinov2_source.is_valid_dinov2_source
validate_dinov2_source = _dinov2_source.validate_dinov2_source
default_dinov2_source = _dinov2_source.default_dinov2_source


def resolve_dinov2_source(source) -> Path:
    """Validate the ``--dinov2-source`` value, failing loudly when unavailable.

    Offline inference deliberately has no network fallback: ``source`` must be a
    directory containing ``hubconf.py``.
    """
    if source is None:
        raise SystemExit(
            "No local DINOv2 source available and offline inference never "
            "downloads it from the network. Pass --dinov2-source "
            "/path/to/facebookresearch_dinov2_main (the directory must contain "
            f"{HUB_MARKER}), or set {DINOV2_ENV_VAR}. The standard torch.hub "
            f"cache location is {DEFAULT_DINOV2_HUB_CACHE}."
        )
    try:
        return validate_dinov2_source(source)
    except ValueError as error:
        raise SystemExit(
            f"Invalid --dinov2-source: {error} The source must be a directory "
            f"containing {HUB_MARKER}."
        ) from error


@dataclass(frozen=True)
class ModelPreset:
    """Architecture preset compatible with one of the locally shipped checkpoints."""

    name: str
    height: int
    width: int
    checkpoint: str
    gaussian_scale_max: float
    monodepth_vit_type: str = "vitb"
    num_scales: int = 2
    upsample_factor: int = 4
    lowest_feature_resolution: int = 8


# Both shipped "base" (117M) checkpoints use the same ``vitb`` architecture.
# ``gaussian_scale_max`` is part of the trained architecture, not a free
# rendering knob: the 448x768 model (re10k/dl3dv, used at 0.1 in the README) and
# the 256x448 dl3dv model (trained with the 3.0 default) each need their own
# value, so it is carried on the preset rather than left to the config default.
MODEL_PRESETS: dict[str, ModelPreset] = {
    "448x768": ModelPreset(
        name="448x768",
        height=448,
        width=768,
        checkpoint=(
            "pretrained/depthsplat-gs-base-re10kdl3dv-448x768-randview2-6-f8ddd845.pth"
        ),
        gaussian_scale_max=0.1,
    ),
    "256x448": ModelPreset(
        name="256x448",
        height=256,
        width=448,
        checkpoint=(
            "pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth"
        ),
        gaussian_scale_max=3.0,
    ),
}
# Default to the smaller local model so the default run needs less memory; the
# larger 448x768 preset stays available via ``--model 448x768`` (optionally with
# ``--input-size 448x768``).
DEFAULT_PRESET = "256x448"


# ---------------------------------------------------------------------------
# Pure helpers (no torch / no PIL at import time)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelIntrinsics:
    """Pinhole intrinsics in pixels for one specific image size."""

    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ResizeCropPlan:
    """Explicit aspect-preserving resize followed by a centre crop."""

    src_h: int
    src_w: int
    scaled_h: int
    scaled_w: int
    out_h: int
    out_w: int
    row: int
    col: int


def _floats_from_text(text: str) -> list[float]:
    return [float(tok) for tok in text.replace(",", " ").split()]


def parse_pixel_intrinsics(text: str) -> PixelIntrinsics:
    """Parse 9-value intrinsics text, keeping the first four fx/fy/cx/cy."""
    values = _floats_from_text(text)
    if len(values) < 4:
        raise ValueError(
            f"Expected at least 4 intrinsic values, found {len(values)}."
        )
    return PixelIntrinsics(values[0], values[1], values[2], values[3])


def read_pixel_intrinsics(path: Path) -> PixelIntrinsics:
    return parse_pixel_intrinsics(Path(path).read_text())


def parse_matrix(text: str, size: int = 4) -> np.ndarray:
    values = _floats_from_text(text)
    if len(values) != size * size:
        raise ValueError(f"Expected {size * size} matrix values, found {len(values)}.")
    return np.asarray(values, dtype=np.float64).reshape(size, size)


def read_matrix(path: Path, size: int = 4) -> np.ndarray:
    return parse_matrix(Path(path).read_text(), size=size)


def read_scene_list(path: Path) -> list[str]:
    """One scene id per non-empty line."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def normalize_frame_id(frame: str, available: Optional[Iterable[str]] = None) -> str:
    """Normalize a user supplied frame id to the zero-padded on-disk form."""
    text = str(frame).strip()
    if available is not None and text in set(available):
        return text
    if text.isdigit():
        return f"{int(text):03d}"
    return text


def round_hw_to_multiple(
    hw: tuple[int, int], multiple: int
) -> tuple[int, int]:
    """Floor each spatial dimension to a multiple of ``multiple`` (min one step)."""
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    h, w = int(hw[0]), int(hw[1])
    return (max(multiple, h // multiple * multiple), max(multiple, w // multiple * multiple))


def plan_resize_and_crop(
    src_hw: tuple[int, int], dst_hw: tuple[int, int]
) -> ResizeCropPlan:
    """Plan an aspect-preserving resize followed by a centre crop.

    ``scale = max(dst_h / src_h, dst_w / src_w)`` so the shorter dimension is
    filled exactly and the longer dimension is centre-cropped.
    """
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    out_h, out_w = int(dst_hw[0]), int(dst_hw[1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid source shape {src_hw}.")
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Invalid target shape {dst_hw}.")

    scale = max(out_h / src_h, out_w / src_w)
    scaled_h = int(round(src_h * scale))
    scaled_w = int(round(src_w * scale))
    # Guarantee we never crop below the requested size because of rounding.
    scaled_h = max(scaled_h, out_h)
    scaled_w = max(scaled_w, out_w)
    row = (scaled_h - out_h) // 2
    col = (scaled_w - out_w) // 2
    return ResizeCropPlan(src_h, src_w, scaled_h, scaled_w, out_h, out_w, row, col)


def resize_and_crop_intrinsics(
    intrinsics: PixelIntrinsics, plan: ResizeCropPlan
) -> PixelIntrinsics:
    """Apply the resize/crop plan to pixel intrinsics.

    The image is resized from ``(src_h, src_w)`` to ``(scaled_h, scaled_w)`` and
    then centre-cropped at ``(row, col)``; ``fx``/``fy`` are scaled, and
    ``cx``/``cy`` are scaled and shifted by the crop origin.
    """
    scale_x = plan.scaled_w / plan.src_w
    scale_y = plan.scaled_h / plan.src_h
    return PixelIntrinsics(
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=intrinsics.cx * scale_x - plan.col,
        cy=intrinsics.cy * scale_y - plan.row,
    )


def pixel_to_normalized_intrinsics(
    intrinsics: PixelIntrinsics, height: int, width: int
) -> np.ndarray:
    """Convert pixel K to DepthSplat's normalized K.

    Normalized K is ``[[fx/W, 0, cx/W], [0, fy/H, cy/H], [0, 0, 1]]``.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    return np.array(
        [
            [intrinsics.fx / width, 0.0, intrinsics.cx / width],
            [0.0, intrinsics.fy / height, intrinsics.cy / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_wide_intrinsics(
    context_intrinsics: PixelIntrinsics,
    context_hw: tuple[int, int],
    width_factor: float,
) -> tuple[PixelIntrinsics, tuple[int, int]]:
    """Build the wide-view pixel K from the render camera's resized pixel K.

    ``fx``/``fy`` (pixel focal length) and ``cy`` are preserved; ``cx`` is moved
    to the centre of the wider canvas.  Returns the wide intrinsics and the
    output ``(height, width)``.
    """
    if width_factor <= 0:
        raise ValueError("width_factor must be positive")
    context_h, context_w = int(context_hw[0]), int(context_hw[1])
    wide_w = int(round(context_w * float(width_factor)))
    wide = PixelIntrinsics(
        fx=context_intrinsics.fx,
        fy=context_intrinsics.fy,
        cx=wide_w / 2.0,
        cy=context_intrinsics.cy,
    )
    return wide, (context_h, wide_w)


def enumerate_frames(
    scene_dir: Path,
    cameras: Sequence[int],
    max_frames: Optional[int] = None,
    frame: Optional[str] = None,
) -> list[str]:
    """Enumerate frames that have an image for every requested camera.

    ``max_frames`` is ``None`` or negative for "all frames".  ``frame`` selects
    a single frame id (normalized to the on-disk padded form).
    """
    images_dir = Path(scene_dir) / "images"
    if not images_dir.is_dir():
        return []

    suffixes = {f"_{cam}.jpg" for cam in cameras}
    all_frames: set[str] = set()
    for entry in images_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        for suffix in suffixes:
            if name.endswith(suffix):
                all_frames.add(name[: -len(suffix)])
                break

    # A frame is usable only when *every* requested camera image is present.
    frames = [
        frame_id
        for frame_id in sorted(all_frames)
        if all((images_dir / f"{frame_id}_{cam}.jpg").is_file() for cam in cameras)
    ]

    if frame is not None:
        wanted = normalize_frame_id(frame, frames)
        return [wanted] if wanted in frames else []

    if max_frames is not None and max_frames >= 0:
        frames = frames[:max_frames]
    return frames


# ---------------------------------------------------------------------------
# Image IO (PIL imported lazily)
# ---------------------------------------------------------------------------


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return (height, width)


def load_resized_rgb(path: Path, plan: ResizeCropPlan) -> np.ndarray:
    """Load an image, apply the resize/crop plan, return float32 HxWx3 in [0,1]."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (plan.src_w, plan.src_h):
            raise ValueError(
                f"Image {path} has size {image.size[1]}x{image.size[0]} but the "
                f"resize plan was built for {plan.src_h}x{plan.src_w}."
            )
        if image.size != (plan.scaled_w, plan.scaled_h):
            image = image.resize((plan.scaled_w, plan.scaled_h), Image.LANCZOS)
        image = image.crop(
            (plan.col, plan.row, plan.col + plan.out_w, plan.row + plan.out_h)
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def save_rgb(tensor_chw, path: Path, quality: int = 95) -> None:
    """Save a [3,H,W] float tensor in [0,1] as a JPEG."""
    from PIL import Image

    array = (
        tensor_chw.detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    array = (array * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, quality=quality)


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
    context images in :func:`load_resized_rgb`: a NEAREST resize to
    ``(plan.scaled_w, plan.scaled_h)`` followed by the same centre crop
    ``(plan.col, plan.row)``.  It is never plain-resized straight to the
    destination, so mask pixels stay aligned with the resized/cropped image.
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
    ``t3_cam5`` fully preserved.  With ``num_frames=1`` it masks only the
    non-render context cameras (e.g. cams 4 and 3) and preserves the render
    camera.

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
# argparse
# ---------------------------------------------------------------------------


def parse_hw(text: str) -> tuple[int, int]:
    """Parse an ``HxW`` input size (also accepts ``H,W`` / ``H W``).

    Used for ``--input-size``.  Raises ``argparse.ArgumentTypeError`` so a bad
    value is reported by argparse as a normal usage error rather than a
    traceback.
    """
    tokens = (
        str(text).strip().lower().replace("x", " ").replace(",", " ").split()
    )
    if len(tokens) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected an input size like 448x768, got {text!r}."
        )
    try:
        height, width = int(tokens[0]), int(tokens[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Input size {text!r} must be two integers like 448x768."
        ) from error
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(
            f"Input size must be positive, got {height}x{width}."
        )
    return (height, width)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference_nuscenes_wide.py",
        description=(
            "nuScenes wide-view rendering with DepthSplat. Reconstructs "
            "Gaussians from cameras 5,4,3 and renders a widened image from "
            "camera 5 (default 2x width, same height). By default every scene "
            "in the scene list and every valid frame per scene is processed."
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
        help="Process a single frame id (e.g. 0 or 000) for every selected scene.",
    )
    data.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help=(
            "Maximum valid frames per scene; -1 (default) processes every valid "
            "frame of every selected scene."
        ),
    )
    data.add_argument(
        "--cameras",
        default=",".join(str(c) for c in DEFAULT_CAMERAS),
        help="Comma separated context camera ids (order preserved).",
    )
    data.add_argument(
        "--extrinsics-source",
        choices=EXTRINSICS_SOURCE_CHOICES,
        default=DEFAULT_EXTRINSICS_SOURCE,
        help=(
            "OpenCV camera-to-world extrinsics to use. 'cam2ego' (default) "
            "reads the static per-camera rig transform "
            "cam2ego_extrinsics/{cam}.txt directly (the per-frame ego frame is "
            "the world), giving frame-independent reconstruction. 'per_frame' "
            "reads the per-frame global extrinsics/{frame}_{cam}.txt (ego pose "
            "baked in), which can be inconsistent in this processed data; it is "
            "for explicit A/B comparison only."
        ),
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
            "By default each context camera's mask is applied to its own view, "
            "except the render camera which is fully preserved: for cameras 5,4,3 "
            "with render camera 5, cameras 4 and 3 are masked. A missing mask for "
            "any selected camera is a hard error."
        ),
    )
    data.add_argument(
        "--mask-render-view",
        action="store_true",
        help=(
            "Diagnostic: also apply each camera's mask to the render view "
            "(e.g. camera 5), so the preset ego-car region is removed there too. "
            "Default off: the render view is fully preserved."
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
        choices=sorted(MODEL_PRESETS),
        default=DEFAULT_PRESET,
        help=(
            "Model architecture preset; selects vitb/num_scales/upsample_factor/"
            "gaussian_scale_max and the matching default checkpoint. Defaults to "
            "the smaller local preset; use --input-size to resize independently. "
            "--resolution is a legacy alias for --model."
        ),
    )
    model.add_argument(
        "--input-size",
        type=parse_hw,
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
    _default_dinov2_source = default_dinov2_source()
    model.add_argument(
        "--dinov2-source",
        default=(
            str(_default_dinov2_source)
            if _default_dinov2_source is not None
            else None
        ),
        help=(
            "Local DINOv2 torch.hub source directory (must contain hubconf.py). "
            f"Defaults to ${DINOV2_ENV_VAR}, else "
            f"{DEFAULT_DINOV2_HUB_CACHE} when it exists. Offline inference never "
            "downloads DINOv2; a missing/invalid source is an error."
        ),
    )

    render = parser.add_argument_group("rendering")
    render.add_argument(
        "--render-camera",
        type=int,
        default=DEFAULT_RENDER_CAMERA,
        help="Camera id rendered in the wide view.",
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
        help="Output root directory.",
    )
    out.add_argument(
        "--save-inputs",
        action="store_true",
        help=(
            "Also save the resized context images under <output>/<scene>/inputs/ "
            "(disabled by default)."
        ),
    )
    out.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and load data only; do not build the model or render.",
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


def parse_cameras(text: str) -> tuple[int, ...]:
    cameras = tuple(int(tok) for tok in text.replace(",", " ").split())
    if not cameras:
        raise ValueError("At least one camera id is required.")
    if len(set(cameras)) != len(cameras):
        raise ValueError(f"Duplicate camera ids in {cameras}.")
    return cameras


def resolve_local(path: str, base: Path = REPO_ROOT) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


def resolve_input_hw(args, preset: ModelPreset) -> tuple[int, int]:
    """Resolve the requested input resize size for a parsed ``args``.

    Precedence (lowest to highest): the ``--model`` preset size, then
    ``--input-size`` for both dimensions, then ``--height`` / ``--width`` for
    their individual dimension.  The result is the *requested* size; callers
    still floor it to the effective patch size.
    """
    height, width = int(preset.height), int(preset.width)
    if args.input_size is not None:
        height, width = int(args.input_size[0]), int(args.input_size[1])
    if args.height is not None:
        height = int(args.height)
    if args.width is not None:
        width = int(args.width)
    return (height, width)


def validate_checkpoint_preset(checkpoint_path: Path, preset_name: str) -> None:
    """Reject a ``--checkpoint`` that is another preset's known checkpoint.

    A custom checkpoint is allowed (its architecture is confirmed by the strict
    load in :func:`load_encoder_state_dict`), but shipping a preset's file under
    the wrong ``--model`` is a definite mistake that can be reported before
    building the model.  Arbitrary filenames are left to the strict load.
    """
    resolved = Path(checkpoint_path).resolve()
    for name, preset in MODEL_PRESETS.items():
        if name == preset_name:
            continue
        if resolved == resolve_local(preset.checkpoint).resolve():
            raise SystemExit(
                f"--checkpoint {checkpoint_path} is the '{name}' preset "
                f"checkpoint but --model is '{preset_name}'. Use "
                f"--model {name} (optionally with --input-size) instead, or "
                f"pass a checkpoint built for the '{preset_name}' architecture."
            )


# ---------------------------------------------------------------------------
# Config / model construction (heavy imports are lazy)
# ---------------------------------------------------------------------------


def compose_config_dict(
    args, preset: ModelPreset, dinov2_source=None, local_mv_match=None
):
    """Compose the repository Hydra config for the dl3dv experiment.

    ``dinov2_source`` (a validated local directory) is composed into the encoder
    config with ``dinov2_pretrained=false``: the full checkpoint carries the
    trained DINOv2 weights under ``encoder.depth_predictor.pretrained.*``, so the
    architecture is built with random weights and then strict-loaded, never
    downloading DINOv2 from the network.

    ``local_mv_match`` is an optional override of
    ``model.encoder.local_mv_match`` (the number of *additional* nearest views
    the multi-view transformer matches against).  ``None`` keeps the training
    default, so single-frame callers are unaffected; the multi-frame script
    passes ``V - 1`` so that every flattened view participates.
    """
    from hydra import compose, initialize_config_dir

    config_dir = REPO_ROOT / "config"
    gaussian_scale_max = (
        args.gaussian_scale_max
        if args.gaussian_scale_max is not None
        else preset.gaussian_scale_max
    )
    overrides = [
        "+experiment=dl3dv",
        f"model.encoder.monodepth_vit_type={preset.monodepth_vit_type}",
        f"model.encoder.num_scales={preset.num_scales}",
        f"model.encoder.upsample_factor={preset.upsample_factor}",
        f"model.encoder.lowest_feature_resolution={preset.lowest_feature_resolution}",
        "model.encoder.gaussian_adapter.gaussian_scale_max="
        f"{gaussian_scale_max}",
        f"dataset.image_shape=[{preset.height},{preset.width}]",
    ]
    if dinov2_source is not None:
        overrides.append(f"model.encoder.dinov2_source={dinov2_source}")
        overrides.append("model.encoder.dinov2_pretrained=false")
    if local_mv_match is not None:
        overrides.append(f"model.encoder.local_mv_match={int(local_mv_match)}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    return cfg_dict


def effective_patch_size(cfg_dict) -> int:
    encoder_cfg = cfg_dict.model.encoder
    return int(encoder_cfg.shim_patch_size) * int(encoder_cfg.downscale_factor)


def load_encoder_state_dict(encoder, encoder_state: dict) -> None:
    """Load encoder weights strictly, failing loudly on any key mismatch.

    The released checkpoints are full-model ``state_dict``s; the caller passes
    only the ``encoder.*`` entries with that prefix stripped.  A missing or
    unexpected key means ``--model`` / ``--checkpoint`` do not describe the same
    architecture, so there is deliberately no ``strict=False`` fallback: it would
    silently leave parts of the encoder randomly initialized (or report a
    "load" that never happened).  Any mismatch is surfaced as a clear error.
    """
    try:
        encoder.load_state_dict(encoder_state, strict=True)
    except RuntimeError as error:  # pragma: no cover - checkpoint dependent
        raise SystemExit(
            "Failed to load encoder weights strictly: --checkpoint does not "
            "match the architecture selected by --model. A custom checkpoint "
            "must be built for the chosen --model preset (vitb / num_scales / "
            "upsample_factor / gaussian_scale_max).\n"
            f"Original error: {error}"
        ) from error


def build_model(cfg_dict, checkpoint_path: Path, device):
    """Construct EncoderDepthSplat + DecoderSplattingCUDA and load the checkpoint."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(f"PyTorch is required but could not be imported: {exc}") from exc

    try:
        import diff_gaussian_rasterization  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The CUDA Gaussian rasterizer is required but could not be imported:\n"
            "  pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified\n"
            f"Original error: {exc}"
        ) from exc

    try:
        from src.config import load_typed_root_config
        from src.model.encoder import get_encoder
        from src.model.decoder import get_decoder
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Failed to import the DepthSplat model stack. Make sure the "
            "dependencies in requirements.txt are installed and the repository "
            f"root is importable.\nOriginal error: {exc}"
        ) from exc

    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    if device.type != "cuda":
        raise SystemExit(
            "DecoderSplattingCUDA requires CUDA. Use --device cuda (or check "
            "`torch.cuda.is_available()`)."
        )

    cfg = load_typed_root_config(cfg_dict)
    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)

    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise SystemExit(
            f"Checkpoint {checkpoint_path} does not contain any 'encoder.*' weights."
        )

    load_encoder_state_dict(encoder, encoder_state)

    encoder.eval().to(device)
    decoder.eval().to(device)
    for param in encoder.parameters():
        param.requires_grad_(False)
    for param in decoder.parameters():
        param.requires_grad_(False)
    return cfg, encoder, decoder


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------


@dataclass
class FrameInputs:
    scene: str
    frame: str
    images: np.ndarray  # [V, H, W, 3] float32
    intrinsics: np.ndarray  # [V, 3, 3] normalized
    extrinsics: np.ndarray  # [V, 4, 4] OpenCV C2W
    resize_plan: ResizeCropPlan
    render_intrinsics_px: PixelIntrinsics  # render camera's resized pixel K


def resolve_extrinsics_path(
    scene_dir: Path,
    frame: str,
    cam: int,
    source: str = DEFAULT_EXTRINSICS_SOURCE,
) -> Path:
    """Return the extrinsics file for one camera under the selected source.

    ``cam2ego`` (default) is the static per-camera rig transform
    ``cam2ego_extrinsics/{cam}.txt``, used directly as OpenCV camera-to-world.
    ``per_frame`` is the per-frame global camera-to-world matrix
    ``extrinsics/{frame}_{cam}.txt`` (ego pose baked in), offered only for
    explicit A/B comparison; it is never selected implicitly.
    """
    if source == "cam2ego":
        return Path(scene_dir) / "cam2ego_extrinsics" / f"{cam}.txt"
    if source == "per_frame":
        return Path(scene_dir) / "extrinsics" / f"{frame}_{cam}.txt"
    raise ValueError(
        f"Unknown extrinsics source {source!r}; expected one of "
        f"{EXTRINSICS_SOURCE_CHOICES}."
    )


def load_frame_inputs(
    scene_dir: Path,
    scene: str,
    frame: str,
    cameras: Sequence[int],
    render_camera: int,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
    extrinsics_source: str = DEFAULT_EXTRINSICS_SOURCE,
) -> FrameInputs:
    """Load one frame's context images/intrinsics/extrinsics.

    ``extrinsics_source`` selects the camera-to-world source (see
    :func:`resolve_extrinsics_path`): ``"cam2ego"`` (default) reads the static
    rig ``cam2ego_extrinsics/{cam}.txt``, ``"per_frame"`` reads the per-frame
    global ``extrinsics/{frame}_{cam}.txt``.  The chosen source is never
    silently substituted by the other one.
    """
    plan = plan_resize_and_crop(src_hw, dst_hw)
    images: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    render_intrinsics_px: Optional[PixelIntrinsics] = None
    render_index: Optional[int] = None

    for index, cam in enumerate(cameras):
        image_path = scene_dir / "images" / f"{frame}_{cam}.jpg"
        intrinsics_path = scene_dir / "intrinsics" / f"{cam}.txt"
        extrinsics_path = resolve_extrinsics_path(
            scene_dir, frame, cam, extrinsics_source
        )

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not intrinsics_path.is_file():
            raise FileNotFoundError(f"Missing intrinsics: {intrinsics_path}")
        if not extrinsics_path.is_file():
            if extrinsics_source == "cam2ego":
                raise FileNotFoundError(
                    f"Missing cam2ego extrinsics: {extrinsics_path}. The "
                    "default --extrinsics-source cam2ego expects the static rig "
                    "transform cam2ego_extrinsics/<cam>.txt; pass "
                    "--extrinsics-source per_frame to read the per-frame "
                    "extrinsics/<frame>_<cam>.txt instead."
                )
            raise FileNotFoundError(
                f"Missing per-frame extrinsics: {extrinsics_path}. "
                "--extrinsics-source per_frame expects "
                "extrinsics/<frame>_<cam>.txt; the default --extrinsics-source "
                "cam2ego reads the static rig cam2ego_extrinsics/<cam>.txt "
                "instead."
            )

        pixel_k = read_pixel_intrinsics(intrinsics_path)
        resized_k = resize_and_crop_intrinsics(pixel_k, plan)
        normalized_k = pixel_to_normalized_intrinsics(resized_k, dst_hw[0], dst_hw[1])

        images.append(load_resized_rgb(image_path, plan))
        intrinsics.append(normalized_k)
        extrinsics.append(read_matrix(extrinsics_path, size=4))

        if cam == render_camera:
            render_intrinsics_px = resized_k
            render_index = index

    if render_intrinsics_px is None or render_index is None:
        raise ValueError(
            f"Render camera {render_camera} is not among the context cameras {tuple(cameras)}."
        )

    return FrameInputs(
        scene=scene,
        frame=frame,
        images=np.stack(images, axis=0).astype(np.float32),
        intrinsics=np.stack(intrinsics, axis=0).astype(np.float32),
        extrinsics=np.stack(extrinsics, axis=0).astype(np.float32),
        resize_plan=plan,
        render_intrinsics_px=render_intrinsics_px,
    )


def _render_wide_impl(
    frame_inputs: FrameInputs,
    encoder,
    decoder,
    render_index: int,
    width_factor: float,
    near: float,
    far: float,
    device,
    amp: bool,
    gaussian_filter=None,
):
    """Encode views, optionally filter the Gaussians, then render.

    ``gaussian_filter`` is an optional callback invoked with
    ``(gaussians, num_views, height, width)`` right after the encoder returns
    and before the decoder runs; it must return a Gaussians object (e.g. a
    pruned copy).  It defaults to ``None`` so the single-frame callers are
    completely unchanged.  The multi-frame script uses it to drop ego-car
    Gaussians from historical views.
    """
    import torch

    images = torch.from_numpy(frame_inputs.images).permute(0, 3, 1, 2).contiguous()
    intrinsics = torch.from_numpy(frame_inputs.intrinsics)
    extrinsics = torch.from_numpy(frame_inputs.extrinsics)

    v = images.shape[0]
    context = {
        "image": images[None].to(device),
        "intrinsics": intrinsics[None].to(device),
        "extrinsics": extrinsics[None].to(device),
        "near": torch.full((1, v), float(near), device=device),
        "far": torch.full((1, v), float(far), device=device),
    }

    context_h, context_w = frame_inputs.images.shape[1], frame_inputs.images.shape[2]
    wide_px, (out_h, out_w) = make_wide_intrinsics(
        frame_inputs.render_intrinsics_px, (context_h, context_w), width_factor
    )
    wide_normalized = pixel_to_normalized_intrinsics(wide_px, out_h, out_w)

    target_extrinsics = torch.from_numpy(
        frame_inputs.extrinsics[render_index : render_index + 1]
    ).to(device)
    target_intrinsics = torch.from_numpy(
        wide_normalized[None].astype(np.float32)
    ).to(device)
    target_near = torch.full((1, 1), float(near), device=device)
    target_far = torch.full((1, 1), float(far), device=device)

    autocast = torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=bool(amp)
    )
    with torch.no_grad(), autocast:
        encoded = encoder(context, 0, True)
        gaussians = encoded["gaussians"] if isinstance(encoded, dict) else encoded
        if gaussian_filter is not None:
            gaussians = gaussian_filter(gaussians, v, context_h, context_w)
        output = decoder(
            gaussians,
            target_extrinsics[None],
            target_intrinsics[None],
            target_near,
            target_far,
            (out_h, out_w),
            depth_mode=None,
        )
    return output.color[0, 0], (out_h, out_w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    import torch  # imported here so --help works without torch

    torch.set_float32_matmul_precision("high")

    data_root = resolve_local(args.data_root)
    # The parser's scene-list default is the canonical trainval2 list.  Keep it
    # relative to --data-root so overriding only --data-root still picks up that
    # root's <data-root>/nuScenes_Val2.txt; an explicit --scene-list wins.
    if args.scene_list == str(DEFAULT_SCENE_LIST_PATH):
        scene_list = data_root / DEFAULT_SCENE_LIST_NAME
    else:
        scene_list = resolve_local(args.scene_list)
    output_dir = resolve_local(args.output_dir)
    cameras = parse_cameras(args.cameras)

    if args.render_camera not in cameras:
        raise SystemExit(
            f"--render-camera {args.render_camera} must be one of --cameras {cameras}."
        )
    if args.frame is not None and args.max_frames >= 0 and args.max_frames != 1:
        print(
            "[warn] --frame is set; --max-frames is ignored for that selection.",
            file=sys.stderr,
        )

    # Extrinsics provenance is part of the result: report it explicitly and warn
    # when the A/B-only per-frame global source is requested.
    if args.extrinsics_source == "per_frame":
        print(
            "[warn] --extrinsics-source per_frame uses the per-frame global "
            "cam-to-world extrinsics/<frame>_<cam>.txt (ego pose baked in); "
            "this can be inconsistent in the processed data. The default "
            "cam2ego static rig is recommended for frame-independent "
            "reconstruction.",
            file=sys.stderr,
        )
    else:
        print(
            "[info] Using --extrinsics-source cam2ego: static per-camera rig "
            "transforms cam2ego_extrinsics/<cam>.txt as OpenCV camera-to-world.",
            file=sys.stderr,
        )

    if args.scene is not None:
        scenes = [args.scene]
    else:
        if not scene_list.is_file():
            raise SystemExit(f"Scene list not found: {scene_list}")
        scenes = read_scene_list(scene_list)
    if not scenes:
        raise SystemExit(f"No scenes selected (scene list: {scene_list}).")

    preset = MODEL_PRESETS[args.model]
    requested_hw = resolve_input_hw(args, preset)

    # DINOv2 must resolve to a valid local source for real inference: this path
    # deliberately has no network fallback. ``--dry-run`` never builds a model,
    # so it may proceed without one.
    try:
        dinov2_source = resolve_dinov2_source(args.dinov2_source)
    except SystemExit:
        if not args.dry_run:
            raise
        dinov2_source = None

    # Compose config early (cheap, no torch) so the effective patch size is explicit.
    # ``--model`` alone controls the architecture (vitb/num_scales/upsample_factor/
    # gaussian_scale_max); ``--input-size`` only sets the resize target.
    cfg_dict = compose_config_dict(args, preset, dinov2_source)
    patch_size = effective_patch_size(cfg_dict)
    if patch_size != FALLBACK_EFFECTIVE_PATCH_SIZE:
        print(
            f"[info] Using effective patch size {patch_size} from the composed config.",
            file=sys.stderr,
        )
    dst_hw = round_hw_to_multiple(requested_hw, patch_size)
    if dst_hw != requested_hw:
        print(
            f"[info] Input size {requested_hw[0]}x{requested_hw[1]} rounded to "
            f"{dst_hw[0]}x{dst_hw[1]} (multiple of effective patch {patch_size}).",
            file=sys.stderr,
        )

    checkpoint_path = (
        resolve_local(args.checkpoint)
        if args.checkpoint is not None
        else resolve_local(preset.checkpoint)
    )
    if args.checkpoint is not None:
        validate_checkpoint_preset(checkpoint_path, args.model)
        print(
            f"[info] Using custom --checkpoint {checkpoint_path} with --model "
            f"'{args.model}' architecture; a mismatch fails at strict load.",
            file=sys.stderr,
        )

    # Enumerate and validate jobs before touching the GPU/model.
    jobs: list[tuple[str, str]] = []
    for scene in scenes:
        scene_dir = data_root / scene
        if not scene_dir.is_dir():
            message = f"Scene directory not found: {scene_dir}"
            if args.scene is not None:
                raise SystemExit(message)
            print(f"[warn] {message}, skipping.", file=sys.stderr)
            continue
        frames = enumerate_frames(
            scene_dir, cameras, max_frames=args.max_frames, frame=args.frame
        )
        if not frames:
            print(
                f"[warn] No frames found for scene {scene} with cameras {cameras}.",
                file=sys.stderr,
            )
            continue
        for frame in frames:
            jobs.append((scene, frame))

    if not jobs:
        raise SystemExit("No valid (scene, frame) jobs were found.")

    print(
        f"Selected {len(jobs)} frame(s) from {len(scenes)} scene(s); "
        f"model '{args.model}', input {dst_hw[0]}x{dst_hw[1]}, "
        f"effective patch {patch_size}."
    )

    # Determine source resolution from the first valid image and validate.
    first_scene, first_frame = jobs[0]
    first_image = data_root / first_scene / "images" / f"{first_frame}_{cameras[0]}.jpg"
    if not first_image.is_file():
        raise SystemExit(f"Missing first image: {first_image}")
    src_hw = image_size(first_image)
    print(f"Source image resolution: {src_hw[0]}x{src_hw[1]}")

    # Ego-car masking is enabled by default (same policy as multi-frame).  The
    # masks are static and the resize/crop plan is constant, so every mask and
    # the single-frame per-view keep mask are loaded/built exactly once here,
    # after src_hw/dst_hw are known and before any frame is touched.  For
    # cameras 5,4,3 with render camera 5 the default masks cams 4 and 3 and
    # fully preserves the render cam 5.
    render_index = list(cameras).index(args.render_camera)
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
        mask_root = resolve_local(args.car_mask_root)
        plan = plan_resize_and_crop(src_hw, dst_hw)
        camera_keep_masks = load_camera_keep_masks(mask_root, cameras, plan)
        # Single frame: one view per camera, render view at cameras.index(...).
        keep_mask = build_car_keep_mask(
            camera_keep_masks,
            cameras,
            1,
            dst_hw,
            render_index,
            mask_render_view=args.mask_render_view,
        )
        gaussian_filter = make_gaussian_filter(keep_mask)
        retained = int(keep_mask.sum())
        total = int(keep_mask.size)
        if mask_policy == CAR_MASK_POLICY_ALL:
            policy_detail = (
                "every view (including the render view; diagnostic "
                "--mask-render-view)"
            )
        else:
            policy_detail = (
                "every view except the render view (camera "
                f"{args.render_camera}) at render_index={render_index}, which is "
                "fully preserved"
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
            jobs, total=len(jobs), unit="frame", desc="Validating frames (dry-run)"
        ) as progress:
            for index, (scene, frame) in enumerate(progress):
                inputs = load_frame_inputs(
                    data_root / scene,
                    scene,
                    frame,
                    cameras,
                    args.render_camera,
                    src_hw,
                    dst_hw,
                    extrinsics_source=args.extrinsics_source,
                )
                wide_px, out_hw = make_wide_intrinsics(
                    inputs.render_intrinsics_px,
                    (dst_hw[0], dst_hw[1]),
                    args.width_factor,
                )
                progress.set_postfix(
                    scene=scene, frame=frame, wide=f"{out_hw[0]}x{out_hw[1]}"
                )
                # Keep one concrete diagnostic line instead of one per frame so a
                # failure can still be traced without flooding the terminal.
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
                        f"[dry-run] first frame scene={scene} frame={frame} "
                        f"extrinsics_source={args.extrinsics_source} "
                        f"images={inputs.images.shape} "
                        f"extrinsics={inputs.extrinsics.shape} "
                        f"K_norm={inputs.intrinsics.shape} "
                        f"wide={out_hw[0]}x{out_hw[1]} "
                        f"wide_K_px=({wide_px.fx:.2f},{wide_px.fy:.2f},"
                        f"{wide_px.cx:.2f},{wide_px.cy:.2f}){mask_info}"
                    )
        return 0

    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise SystemExit(
            "CUDA is required for the Gaussian splatting decoder but was not "
            "available. Pass --device cuda on a machine with a working CUDA "
            "build of diff-gaussian-rasterization."
        )

    _, encoder, decoder = build_model(cfg_dict, checkpoint_path, device)

    progress = tqdm(
        jobs,
        total=len(jobs),
        unit="frame",
        desc="Rendering wide views",
    )
    try:
        for scene, frame in progress:
            progress.set_postfix(scene=scene, frame=frame)
            scene_dir = data_root / scene
            inputs = load_frame_inputs(
                scene_dir,
                scene,
                frame,
                cameras,
                args.render_camera,
                src_hw,
                dst_hw,
                extrinsics_source=args.extrinsics_source,
            )
            color, (out_h, out_w) = _render_wide_impl(
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
                scene=scene, frame=frame, out=f"{out_h}x{out_w}"
            )
            save_path = (
                output_dir / scene / "rgb" / f"{frame}_{args.render_camera}_wide.jpg"
            )
            save_rgb(color, save_path, quality=95)

            if args.save_inputs:
                for cam, image in zip(cameras, inputs.images):
                    input_path = output_dir / scene / "inputs" / f"{frame}_{cam}.jpg"
                    save_rgb(
                        torch.from_numpy(image).permute(2, 0, 1), input_path, quality=95
                    )
    finally:
        # Close cleanly on errors so the traceback stays readable.
        progress.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

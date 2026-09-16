#!/usr/bin/env python3
"""Standalone single-frame nuScenes wide-view inference for DepthSplat.

This script does **not** go through ``ModelWrapper.test_step``.  It loads one
nuScenes frame, builds the DepthSplat encoder/decoder directly from the
repository configuration, reconstructs Gaussians from cameras ``[5, 4, 3]`` and
renders a horizontally widened image from camera ``5``.

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

    images/{frame}_{cam}.jpg          # e.g. 000_5.jpg
    intrinsics/{cam}.txt              # 9 values, first four fx fy cx cy
    extrinsics/{frame}_{cam}.txt      # 4x4 OpenCV camera-to-world

Only per-frame ``extrinsics/{frame}_{cam}.txt`` are used.  ``cam2ego_extrinsics``
is never silently substituted when the per-frame file is missing.

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
DEFAULT_OUTPUT_DIR = Path("outputs/nuscenes_wide")

DEFAULT_CAMERAS = (5, 4, 3)
DEFAULT_RENDER_CAMERA = 5
DEFAULT_WIDTH_FACTOR = 2.0

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
DEFAULT_PRESET = "448x768"


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
# argparse
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference_nuscenes_wide.py",
        description=(
            "Single-frame nuScenes wide-view rendering with DepthSplat. "
            "Reconstructs Gaussians from cameras 5,4,3 and renders a widened "
            "image from camera 5 (default 2x width, same height)."
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
        default=None,
        help=(
            "Scene id list. Defaults to <data-root>/" + DEFAULT_SCENE_LIST_NAME + "."
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
        default=1,
        help="Maximum frames per scene; -1 processes all frames.",
    )
    data.add_argument(
        "--cameras",
        default=",".join(str(c) for c in DEFAULT_CAMERAS),
        help="Comma separated context camera ids (order preserved).",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--resolution",
        choices=sorted(MODEL_PRESETS),
        default=DEFAULT_PRESET,
        help="Model input resolution preset; selects the matching default checkpoint.",
    )
    model.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (defaults to the one matching --resolution).",
    )
    model.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override the preset input height (floored to the effective patch size).",
    )
    model.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override the preset input width (floored to the effective patch size).",
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
        help="Also save the resized context images under <output>/<scene>/inputs/.",
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


# ---------------------------------------------------------------------------
# Config / model construction (heavy imports are lazy)
# ---------------------------------------------------------------------------


def compose_config_dict(args, preset: ModelPreset, dinov2_source=None):
    """Compose the repository Hydra config for the dl3dv experiment.

    ``dinov2_source`` (a validated local directory) is composed into the encoder
    config with ``dinov2_pretrained=false``: the full checkpoint carries the
    trained DINOv2 weights under ``encoder.depth_predictor.pretrained.*``, so the
    architecture is built with random weights and then strict-loaded, never
    downloading DINOv2 from the network.
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
    unexpected key means ``--resolution`` / ``--checkpoint`` do not describe the
    same architecture, so there is deliberately no ``strict=False`` fallback: it
    would silently leave parts of the encoder randomly initialized (or report a
    "load" that never happened).  Any mismatch is surfaced as a clear error.
    """
    try:
        encoder.load_state_dict(encoder_state, strict=True)
    except RuntimeError as error:  # pragma: no cover - checkpoint dependent
        raise SystemExit(
            "Failed to load encoder weights strictly: the checkpoint does not "
            "match the constructed encoder architecture. Ensure --resolution "
            "and --checkpoint refer to the same model.\n"
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


def load_frame_inputs(
    scene_dir: Path,
    scene: str,
    frame: str,
    cameras: Sequence[int],
    render_camera: int,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> FrameInputs:
    plan = plan_resize_and_crop(src_hw, dst_hw)
    images: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    render_intrinsics_px: Optional[PixelIntrinsics] = None
    render_index: Optional[int] = None

    for index, cam in enumerate(cameras):
        image_path = scene_dir / "images" / f"{frame}_{cam}.jpg"
        intrinsics_path = scene_dir / "intrinsics" / f"{cam}.txt"
        extrinsics_path = scene_dir / "extrinsics" / f"{frame}_{cam}.txt"

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")
        if not intrinsics_path.is_file():
            raise FileNotFoundError(f"Missing intrinsics: {intrinsics_path}")
        if not extrinsics_path.is_file():
            raise FileNotFoundError(
                "Missing per-frame extrinsics: "
                f"{extrinsics_path} (cam2ego_extrinsics is not used as a silent fallback)"
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
):
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
    scene_list = (
        resolve_local(args.scene_list)
        if args.scene_list is not None
        else data_root / DEFAULT_SCENE_LIST_NAME
    )
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

    if args.scene is not None:
        scenes = [args.scene]
    else:
        if not scene_list.is_file():
            raise SystemExit(f"Scene list not found: {scene_list}")
        scenes = read_scene_list(scene_list)
    if not scenes:
        raise SystemExit(f"No scenes selected (scene list: {scene_list}).")

    preset = MODEL_PRESETS[args.resolution]
    preset_height = args.height if args.height is not None else preset.height
    preset_width = args.width if args.width is not None else preset.width

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
    cfg_dict = compose_config_dict(args, preset, dinov2_source)
    patch_size = effective_patch_size(cfg_dict)
    if patch_size != FALLBACK_EFFECTIVE_PATCH_SIZE:
        print(
            f"[info] Using effective patch size {patch_size} from the composed config.",
            file=sys.stderr,
        )
    dst_hw = round_hw_to_multiple((int(preset_height), int(preset_width)), patch_size)
    if dst_hw != (int(preset_height), int(preset_width)):
        print(
            f"[info] Input resolution {preset_height}x{preset_width} rounded to "
            f"{dst_hw[0]}x{dst_hw[1]} (multiple of effective patch {patch_size}).",
            file=sys.stderr,
        )

    checkpoint_path = (
        resolve_local(args.checkpoint)
        if args.checkpoint is not None
        else resolve_local(preset.checkpoint)
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
        f"input {dst_hw[0]}x{dst_hw[1]}, effective patch {patch_size}."
    )

    # Determine source resolution from the first valid image and validate.
    first_scene, first_frame = jobs[0]
    first_image = data_root / first_scene / "images" / f"{first_frame}_{cameras[0]}.jpg"
    if not first_image.is_file():
        raise SystemExit(f"Missing first image: {first_image}")
    src_hw = image_size(first_image)
    print(f"Source image resolution: {src_hw[0]}x{src_hw[1]}")

    if args.dry_run:
        for scene, frame in jobs:
            inputs = load_frame_inputs(
                data_root / scene,
                scene,
                frame,
                cameras,
                args.render_camera,
                src_hw,
                dst_hw,
            )
            wide_px, out_hw = make_wide_intrinsics(
                inputs.render_intrinsics_px, (dst_hw[0], dst_hw[1]), args.width_factor
            )
            print(
                f"[dry-run] scene={scene} frame={frame} "
                f"images={inputs.images.shape} extrinsics={inputs.extrinsics.shape} "
                f"K_norm={inputs.intrinsics.shape} wide={out_hw[0]}x{out_hw[1]} "
                f"wide_K_px=({wide_px.fx:.2f},{wide_px.fy:.2f},{wide_px.cx:.2f},{wide_px.cy:.2f})"
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

    render_index = list(cameras).index(args.render_camera)
    for scene, frame in jobs:
        scene_dir = data_root / scene
        inputs = load_frame_inputs(
            scene_dir, scene, frame, cameras, args.render_camera, src_hw, dst_hw
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
        )
        save_path = output_dir / scene / "rgb" / f"{frame}_{args.render_camera}_wide.jpg"
        save_rgb(color, save_path, quality=95)
        print(f"Saved {save_path} ({out_w}x{out_h})")

        if args.save_inputs:
            for cam, image in zip(cameras, inputs.images):
                input_path = output_dir / scene / "inputs" / f"{frame}_{cam}.jpg"
                save_rgb(
                    torch.from_numpy(image).permute(2, 0, 1), input_path, quality=95
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())

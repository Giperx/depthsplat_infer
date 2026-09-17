#!/usr/bin/env python3
"""Standalone speed benchmark for the nuScenes wide-view DepthSplat paths.

This script times the *real* inference pipeline of the two existing entry
points without saving any images:

* ``--mode single`` reuses ``inference_nuscenes_wide.py``
  (:func:`inference_nuscenes_wide.load_frame_inputs` +
  :func:`inference_nuscenes_wide._render_wide_impl`) with the static
  ``cam2ego`` extrinsics, exactly like the single-frame script.
* ``--mode multi`` reuses ``inference_nuscenes_wide_multiframes.py``
  (:func:`...load_window_inputs`) and calls the very same
  ``wide._render_wide_impl`` with the per-frame global extrinsics, exactly like
  the multi-frame script.

The model is built once (``wide.build_model``), the inputs are loaded once
(*outside* the timing loop; the load time is reported separately) and the
ego-car Gaussian filter is built with the same policy resolution as the
inference scripts.  Warmup iterations are run first, then ``--measure``
iterations are timed with ``torch.cuda.Event``; the mean / median / min / max /
stdev latency and the FPS (``1000 / mean``) are reported.

``--help`` and the pure helpers work without torch (torch, CUDA and DINOv2 are
only imported/resolved inside :func:`main` for a real run).

Usage::

    # four combinations (see README "Speed benchmark")
    python scripts/benchmark_nuscenes_wide.py --mode single --model 256x448
    python scripts/benchmark_nuscenes_wide.py --mode single --model 448x768
    python scripts/benchmark_nuscenes_wide.py --mode multi  --model 256x448
    python scripts/benchmark_nuscenes_wide.py --mode multi  --model 448x768

    # write the stats to JSON for scripted comparisons
    python scripts/benchmark_nuscenes_wide.py --json outputs/bench_256_multi.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    """Load a sibling script by path, registering it before execution.

    Registering before ``exec_module`` lets the module's ``dataclass``
    decorators resolve it on Python 3.10.  Existing registrations are reused so
    ``inference_nuscenes_wide_multiframes``'s own loader shares this exact
    ``inference_nuscenes_wide`` module (and its helpers) rather than importing a
    second copy.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - path dependent
        raise ImportError(f"Could not load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the single-frame helpers first; the multi-frame module reuses this exact
# object (it checks ``sys.modules`` before loading its own copy).
wide = _load_module(
    "inference_nuscenes_wide", SCRIPT_DIR / "inference_nuscenes_wide.py"
)
mf = _load_module(
    "inference_nuscenes_wide_multiframes",
    SCRIPT_DIR / "inference_nuscenes_wide_multiframes.py",
)

# ---------------------------------------------------------------------------
# Defaults (mirroring the inference scripts)
# ---------------------------------------------------------------------------

MODE_CHOICES = ("single", "multi")
DEFAULT_MODE = "multi"
DEFAULT_WARMUP = 10
DEFAULT_MEASURE = 50
DEFAULT_NUM_FRAMES = mf.DEFAULT_NUM_FRAMES  # 3

DEFAULT_DATA_ROOT = wide.DEFAULT_DATA_ROOT
DEFAULT_SCENE_LIST_PATH = wide.DEFAULT_SCENE_LIST_PATH
DEFAULT_SCENE_LIST_NAME = wide.DEFAULT_SCENE_LIST_NAME
DEFAULT_CAMERAS = wide.DEFAULT_CAMERAS  # (5, 4, 3)
DEFAULT_RENDER_CAMERA = wide.DEFAULT_RENDER_CAMERA  # 5
DEFAULT_WIDTH_FACTOR = wide.DEFAULT_WIDTH_FACTOR  # 2.0
DEFAULT_CAR_MASK_ROOT = wide.DEFAULT_CAR_MASK_ROOT
DEFAULT_PRESET = wide.DEFAULT_PRESET  # "256x448"

MODE_SINGLE = "single"
MODE_MULTI = "multi"


# ---------------------------------------------------------------------------
# Pure helpers (no torch / no CUDA)
# ---------------------------------------------------------------------------


def summarize_times(times_ms: Sequence[float]) -> dict:
    """Summarize latency samples (milliseconds) into mean/median/min/max/stdev/FPS.

    ``fps`` is ``1000 / mean``.  A single sample has ``stdev_ms == 0.0``.
    Raises ``ValueError`` for an empty sequence.
    """
    values = [float(value) for value in times_ms]
    if not values:
        raise ValueError("At least one timing sample is required to summarize.")
    mean_ms = statistics.mean(values)
    return {
        "count": len(values),
        "mean_ms": mean_ms,
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "fps": (1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
    }


def select_scene(scenes: Sequence[str], requested: Optional[str] = None) -> str:
    """Resolve the benchmark scene: ``requested`` if given, else the first in the list.

    Raises ``ValueError`` for an empty list or a requested scene that is not in
    it, so a typo cannot silently benchmark a different scene.
    """
    cleaned = [str(scene).strip() for scene in scenes if str(scene).strip()]
    if not cleaned:
        raise ValueError("The scene list is empty; cannot select a scene.")
    if requested is None:
        return cleaned[0]
    wanted = str(requested).strip()
    if wanted not in cleaned:
        raise ValueError(
            f"Scene {wanted!r} is not in the scene list. Available scenes: "
            f"{cleaned[:5]}{'...' if len(cleaned) > 5 else ''}."
        )
    return wanted


def select_single_frame(
    frames: Sequence[str], requested: Optional[str] = None
) -> Optional[str]:
    """Resolve the single-mode frame: ``requested`` if valid, else the first.

    Returns ``None`` when there are no valid frames or the requested frame has
    no usable image (never falls back to a different frame).
    """
    cleaned = [str(frame) for frame in frames]
    if not cleaned:
        return None
    if requested is None:
        return cleaned[0]
    wanted = wide.normalize_frame_id(requested, available=cleaned)
    return wanted if wanted in cleaned else None


def select_window(
    frames: Sequence[str],
    num_frames: int,
    requested: Optional[str] = None,
) -> Optional[tuple]:
    """Resolve the multi-mode window: ``requested`` newest frame, else the first.

    Wraps :func:`inference_nuscenes_wide_multiframes.select_windows` (which
    normalizes ``requested`` and never returns an incomplete window).  Returns
    ``None`` when no complete window exists.
    """
    windows = mf.select_windows(
        [str(frame) for frame in frames], num_frames, frame=requested
    )
    return windows[0] if windows else None


def validate_mode_and_model(mode: str, model: str) -> tuple:
    """Validate ``--mode`` / ``--model`` and return them.

    A pure guard used by tests and by :func:`main`; ``argparse`` already
    restricts the choices, but this keeps the failure explicit and testable.
    """
    if mode not in MODE_CHOICES:
        raise ValueError(
            f"Unknown mode {mode!r}; expected one of {MODE_CHOICES}."
        )
    if model not in wide.MODEL_PRESETS:
        raise ValueError(
            f"Unknown model preset {model!r}; expected one of "
            f"{sorted(wide.MODEL_PRESETS)}."
        )
    return mode, model


def resolve_num_frames(mode: str, num_frames: int) -> int:
    """Effective flattened window length: ``num_frames`` in multi, ``1`` in single."""
    if mode == MODE_SINGLE:
        return 1
    value = int(num_frames)
    if value < 1:
        raise ValueError(f"--num-frames must be >= 1, got {value}.")
    return value


def resolve_local_mv_match(mode: str, args, num_frames: int, cameras) -> Optional[int]:
    """Resolve ``local_mv_match`` for the selected mode.

    Single mode has no multi-view transformer override, so an explicit
    ``--local-mv-match`` is rejected; multi mode defers to
    :func:`inference_nuscenes_wide_multiframes.resolve_local_mv_match` (default
    ``V - 1``, ``0`` rejected for ``V > 3``).
    """
    requested = getattr(args, "local_mv_match", None)
    if mode == MODE_SINGLE:
        if requested is not None:
            raise SystemExit(
                "--local-mv-match only applies to --mode multi; the single-frame "
                "path does not compose an encoder local_mv_match override."
            )
        return None
    return mf.resolve_local_mv_match(args, num_frames, cameras)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_nuscenes_wide.py",
        description=(
            "Speed benchmark for the nuScenes wide-view DepthSplat inference "
            "paths. Times the real encoder+filter+decoder call with "
            "torch.cuda.Event after warmup; no images are saved. Defaults mirror "
            "the inference scripts (masking on, cameras 5,4,3, render camera 5, "
            "model 256x448)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    bench = parser.add_argument_group("benchmark")
    bench.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default=DEFAULT_MODE,
        help=(
            "'single' times the single-frame path (static cam2ego extrinsics); "
            "'multi' times the multi-frame window path (per-frame global "
            "extrinsics)."
        ),
    )
    bench.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup iterations run before timing (excluded from the stats).",
    )
    bench.add_argument(
        "--measure",
        type=int,
        default=DEFAULT_MEASURE,
        help="Timed iterations; must be >= 1.",
    )
    bench.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Optional path to write the timing stats (and metadata) as JSON.",
    )

    data = parser.add_argument_group("sample selection")
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
            "<data-root>/" + DEFAULT_SCENE_LIST_NAME + "."
        ),
    )
    data.add_argument(
        "--scene",
        default=None,
        help="Scene id to benchmark (default: the first scene in the scene list).",
    )
    data.add_argument(
        "--frame",
        default=None,
        help=(
            "Frame to benchmark. Single mode selects that frame; multi mode "
            "selects the window whose NEWEST (output) frame is this id. Default: "
            "the first valid frame / first complete window."
        ),
    )
    data.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="Number of consecutive frames per window (multi mode only).",
    )
    data.add_argument(
        "--cameras",
        default=",".join(str(cam) for cam in DEFAULT_CAMERAS),
        help="Comma separated context camera ids (order preserved).",
    )
    data.add_argument(
        "--render-camera",
        type=int,
        default=DEFAULT_RENDER_CAMERA,
        help="Camera id rendered in the wide view.",
    )
    data.add_argument(
        "--width-factor",
        type=float,
        default=DEFAULT_WIDTH_FACTOR,
        help="Output width multiplier relative to the model input width.",
    )
    data.add_argument("--near", type=float, default=0.5, help="Near depth bound.")
    data.add_argument("--far", type=float, default=200.0, help="Far depth bound.")
    data.add_argument(
        "--local-mv-match",
        type=int,
        default=None,
        help=(
            "Multi mode only: override model.encoder.local_mv_match (an "
            "explicit 0 is rejected for V > 3). Defaults to "
            "num_frames*len(cameras)-1 so every flattened view participates."
        ),
    )

    masks = parser.add_argument_group("ego-car masking")
    masks.add_argument(
        "--car-mask-root",
        default=str(DEFAULT_CAR_MASK_ROOT),
        help=(
            "Directory with the per-camera nuScenes ego-car masks. Black "
            "(<128) pixels are removed and white (>=128) kept, transformed with "
            "exactly the same resize/crop plan as the images. A missing mask for "
            "any selected camera is a hard error."
        ),
    )
    masks.add_argument(
        "--mask-render-view",
        action="store_true",
        help=(
            "Diagnostic: also apply each camera's mask to the render view. "
            "Default off: the render view is fully preserved."
        ),
    )
    masks.add_argument(
        "--disable-car-mask",
        action="store_true",
        help=(
            "Disable ego-car masking entirely (keep every Gaussian). Highest "
            "precedence: overrides --mask-render-view and the default policy."
        ),
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model",
        "--resolution",
        dest="model",
        choices=sorted(wide.MODEL_PRESETS),
        default=DEFAULT_PRESET,
        help=(
            "Model architecture preset; selects the matching default checkpoint. "
            "--resolution is a legacy alias."
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
        "--height",
        type=int,
        default=None,
        help="Override the input height (highest priority).",
    )
    model.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override the input width (highest priority).",
    )
    model.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (defaults to the one matching --model).",
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
            str(_default_dinov2_source)
            if _default_dinov2_source is not None
            else None
        ),
        help=(
            "Local DINOv2 torch.hub source directory (must contain hubconf.py). "
            f"Defaults to ${wide.DINOV2_ENV_VAR}, else "
            f"{wide.DEFAULT_DINOV2_HUB_CACHE} when it exists. Offline inference "
            "never downloads DINOv2; a missing/invalid source is an error."
        ),
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
        help="Enable CUDA float16 autocast (experimental; matches the inference scripts).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _resolve_sample(args, data_root: Path):
    """Pick the (scene, frames, sample) to benchmark using the pure helpers."""
    if args.scene is not None:
        scene = str(args.scene)
    else:
        scene_list = (
            data_root / DEFAULT_SCENE_LIST_NAME
            if args.scene_list == str(DEFAULT_SCENE_LIST_PATH)
            else wide.resolve_local(args.scene_list)
        )
        if not scene_list.is_file():
            raise SystemExit(f"Scene list not found: {scene_list}")
        scene = select_scene(wide.read_scene_list(scene_list))

    scene_dir = data_root / scene
    if not scene_dir.is_dir():
        raise SystemExit(f"Scene directory not found: {scene_dir}")

    cameras = wide.parse_cameras(args.cameras)
    num_frames = resolve_num_frames(args.mode, args.num_frames)

    frames = wide.enumerate_frames(scene_dir, cameras, max_frames=None, frame=None)
    if not frames:
        raise SystemExit(
            f"No valid frames found for scene {scene} with cameras {cameras}."
        )

    if args.mode == MODE_SINGLE:
        sample = select_single_frame(frames, args.frame)
        if sample is None:
            raise SystemExit(
                f"Frame {args.frame!r} has no usable images for scene {scene} "
                f"with cameras {cameras}."
            )
    else:
        sample = select_window(frames, num_frames, args.frame)
        if sample is None:
            raise SystemExit(
                f"Scene {scene} has no complete {num_frames}-frame window "
                f"{'ending at frame ' + repr(args.frame) if args.frame is not None else ''}"
                f" with cameras {cameras}."
            )
    return scene, scene_dir, cameras, num_frames, sample


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Pure validation first so bad arguments fail without importing torch.
    mode, model = validate_mode_and_model(args.mode, args.model)
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be >= 0, got {args.warmup}.")
    if args.measure < 1:
        raise SystemExit(f"--measure must be >= 1, got {args.measure}.")

    cameras = wide.parse_cameras(args.cameras)
    if args.render_camera not in cameras:
        raise SystemExit(
            f"--render-camera {args.render_camera} must be one of --cameras {cameras}."
        )
    num_frames = resolve_num_frames(mode, args.num_frames)
    local_mv_match = resolve_local_mv_match(mode, args, num_frames, cameras)

    data_root = wide.resolve_local(args.data_root)
    scene, scene_dir, cameras, num_frames, sample = _resolve_sample(args, data_root)

    import torch  # imported here so --help works without torch

    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for the DepthSplat Gaussian-splatting benchmark "
            "(torch.cuda.is_available() is False). Run this on a machine with a "
            "working CUDA build of diff-gaussian-rasterization, or pass "
            "--device cuda once CUDA is visible."
        )

    preset = wide.MODEL_PRESETS[model]
    requested_hw = wide.resolve_input_hw(args, preset)

    # Offline DINOv2: resolve to a valid local source (there is deliberately no
    # network fallback).
    dinov2_source = wide.resolve_dinov2_source(args.dinov2_source)

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
        wide.validate_checkpoint_preset(checkpoint_path, model)

    # Source resolution comes from the first loaded image (as in inference).
    first_frame = sample if mode == MODE_SINGLE else sample[0]
    first_image = scene_dir / "images" / f"{first_frame}_{cameras[0]}.jpg"
    if not first_image.is_file():
        raise SystemExit(f"Missing first image: {first_image}")
    src_hw = wide.image_size(first_image)

    # Render index / ego-car mask policy resolution (identical to inference).
    if mode == MODE_SINGLE:
        render_index = list(cameras).index(args.render_camera)
    else:
        render_index = mf.newest_render_index(num_frames, cameras, args.render_camera)

    mask_policy = wide.resolve_car_mask_policy(
        args.disable_car_mask, args.mask_render_view
    )
    gaussian_filter = None
    keep_mask = None
    if mask_policy is None:
        print(
            "[info] Ego-car masking DISABLED (--disable-car-mask): every "
            "Gaussian is kept.",
            file=sys.stderr,
        )
    else:
        mask_root = wide.resolve_local(args.car_mask_root)
        plan = wide.plan_resize_and_crop(src_hw, dst_hw)
        camera_keep_masks = wide.load_camera_keep_masks(mask_root, cameras, plan)
        keep_mask = wide.build_car_keep_mask(
            camera_keep_masks,
            cameras,
            num_frames,
            dst_hw,
            render_index,
            mask_render_view=args.mask_render_view,
        )
        gaussian_filter = wide.make_gaussian_filter(keep_mask)
        retained = int(keep_mask.sum())
        total = int(keep_mask.size)
        print(
            f"[info] Ego-car masking enabled from {mask_root} "
            f"(policy={mask_policy}, render_index={render_index}): kept "
            f"{retained}/{total} resized mask pixels over V={keep_mask.shape[0]} "
            "views.",
            file=sys.stderr,
        )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type != "cuda":
        raise SystemExit(
            "CUDA is required for the Gaussian-splatting benchmark but "
            f"--device resolved to {device.type!r}. Pass --device cuda on a "
            "machine with a working CUDA build."
        )

    # ---- Model build (timed separately, outside the latency loop) ----
    build_start = time.perf_counter()
    _, encoder, decoder = wide.build_model(cfg_dict, checkpoint_path, device)
    model_build_ms = (time.perf_counter() - build_start) * 1000.0

    # ---- Input loading (timed separately, outside the latency loop) ----
    load_start = time.perf_counter()
    if mode == MODE_SINGLE:
        inputs = wide.load_frame_inputs(
            scene_dir,
            scene,
            sample,
            cameras,
            args.render_camera,
            src_hw,
            dst_hw,
            extrinsics_source=wide.DEFAULT_EXTRINSICS_SOURCE,
        )
    else:
        inputs = mf.load_window_inputs(
            scene_dir,
            scene,
            sample,
            cameras,
            args.render_camera,
            src_hw,
            dst_hw,
        )
    data_load_ms = (time.perf_counter() - load_start) * 1000.0

    input_shape = tuple(int(dim) for dim in inputs.images.shape)
    v = input_shape[0]

    def run_once():
        """One real encode (+filter) + decode call; returns the rendered RGB."""
        color, out_hw = wide._render_wide_impl(
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
        return color, out_hw

    # ---- Warmup (excluded from the stats) ----
    for iteration in range(args.warmup):
        try:
            run_once()
        except Exception as error:  # noqa: BLE001 - re-raised with context
            raise SystemExit(
                f"Warmup iteration {iteration + 1}/{args.warmup} failed: {error!r}. "
                "No timing samples were collected."
            ) from error
    torch.cuda.synchronize()

    # ---- Measure (torch.cuda.Event around each call) ----
    times_ms: list[float] = []
    output_shape = None
    for iteration in range(args.measure):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        try:
            color, out_hw = run_once()
        except Exception as error:  # noqa: BLE001 - re-raised with context
            raise SystemExit(
                f"Measurement iteration {iteration + 1}/{args.measure} failed "
                f"after {len(times_ms)} successful sample(s): {error!r}."
            ) from error
        end_event.record()
        torch.cuda.synchronize()
        times_ms.append(float(start_event.elapsed_time(end_event)))
        output_shape = tuple(int(dim) for dim in color.shape)

    stats = summarize_times(times_ms)

    sample_label = sample if mode == MODE_SINGLE else "->".join(sample)
    resolved_input = f"{dst_hw[0]}x{dst_hw[1]}"
    mask_label = mask_policy if mask_policy is not None else "disabled"

    # ---- Report ----
    print(f"\n{'=' * 72}")
    print(" DepthSplat nuScenes wide-view speed benchmark")
    print(f"{'=' * 72}")
    print(f"  Mode:            {mode}")
    print(f"  Scene / sample:  {scene} / {sample_label}")
    if mode == MODE_MULTI:
        print(f"  Window:          {tuple(sample)} (newest rendered)")
    print(f"  Model preset:    {model}")
    print(f"  Input size:      {resolved_input} (requested {requested_hw[0]}x{requested_hw[1]})")
    print(f"  Input shape:     {input_shape}  (V, C, H, W)")
    print(f"  V (views):       {v}")
    print(f"  Output shape:    {output_shape}  (C, H, W)")
    print(f"  local_mv_match:  {local_mv_match}")
    print(f"  Mask policy:     {mask_label}")
    print(f"  AMP:             {bool(args.amp)}")
    print(f"  Warmup iters:    {args.warmup} (excluded)")
    print(f"  Measure iters:   {args.measure}")
    print(f"{'─' * 72}")
    print(f"  Latency (ms):    mean={stats['mean_ms']:8.3f}  median={stats['median_ms']:8.3f}  "
          f"min={stats['min_ms']:8.3f}  max={stats['max_ms']:8.3f}  stdev={stats['stdev_ms']:6.3f}")
    print(f"  Throughput:      {stats['fps']:8.2f} FPS (1000 / mean)")
    print(f"{'─' * 72}")
    print(f"  Data load:       {data_load_ms:8.2f} ms (one-time, excluded)")
    print(f"  Model build:     {model_build_ms:8.2f} ms (one-time, excluded)")
    print(f"{'=' * 72}")

    if args.json is not None:
        payload = {
            "mode": mode,
            "model": model,
            "scene": scene,
            "sample": list(sample) if mode == MODE_MULTI else sample,
            "num_frames": num_frames if mode == MODE_MULTI else 1,
            "cameras": list(cameras),
            "render_camera": args.render_camera,
            "width_factor": args.width_factor,
            "near": args.near,
            "far": args.far,
            "local_mv_match": local_mv_match,
            "mask_policy": mask_label,
            "amp": bool(args.amp),
            "device": str(device),
            "input_size": resolved_input,
            "input_shape": list(input_shape),
            "v": v,
            "output_shape": list(output_shape) if output_shape is not None else None,
            "warmup": args.warmup,
            "measure": args.measure,
            "data_load_ms": data_load_ms,
            "model_build_ms": model_build_ms,
            **stats,
        }
        json_path = Path(args.json)
        if json_path.parent != Path(""):
            json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote JSON stats to {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

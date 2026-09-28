#!/usr/bin/env python3
"""Photometric metrics for *this* project's direct 3x-wide renders.

This evaluates the single/multi-frame wide renders produced by
``inference_*_wide*.py`` (files named ``{frame}_5_wide.jpg``) against the sparse
multipl wide GT used by the DWSplat ``*_multiplane_v2.py`` metric scripts.  It
is a self-contained re-implementation: **nothing is imported from the DWSplat
repository**.

Protocol (adapted from the DWSplat v2 sparse-wide protocol)
-----------------------------------------------------------
* The render is resized (bilinear) to the GT image size when they differ; the
  GT is never resized.  Panel width is ``gt_width // 3`` and Left / Center /
  Right are three contiguous panels.
* GT mask is grayscale ``> 127``.  A render mask (``{frame}_5_wide.png``) is
  optional; when absent the render is treated as fully valid.
* Left/Right use the **GT mask only** (the v2 rule: no render-mask
  intersection).  Center is ``GT mask & render-valid`` and ``Center_masked`` is
  Center ``&`` the camera-5 ego-car keep mask.
* MAE/RMSE/PSNR on masked pixels in ``[0, 1]``.  Left/Right SSIM is the DWSplat
  sparse per-pixel SSIM (no window) on the masked pixels in 0-255 scale.
  Center SSIM is an 11x11 Gaussian (sigma 1.5) windowed SSIM map averaged over
  the center mask.  Center LPIPS (alex, spatial) is averaged over the Center and
  Center_masked masks.

Run from the repository root, e.g.::

    python scripts/eval_wide_metrics.py --dataset lyft1920 \
        --render-root outputs/lyft1920_wide --max-scenes 1 --max-frames 2

Relative paths resolve against the repository root unless absolute.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

RENDER_CAMERA = 5
PANEL_COUNT = 3
# DWSplat v2 mask thresholds.
GT_MASK_THRESHOLD = 127
RENDER_MASK_THRESHOLD = 127
CAR_MASK_KEEP_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Dataset table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WideDataset:
    """Sparse-GT / scene-list / car-mask locations for one dataset split.

    ``car_mask`` is either a dataset-level mask file or (``car_mask_per_scene``)
    a root under which ``<root>/<scene>/ego_car_masks/5.<ext>`` lives.
    """

    name: str
    gt_root: str
    scene_list: str
    car_mask: str
    car_mask_per_scene: bool
    car_mask_ext: str
    expected_hw: tuple[int, int]


DATASETS: dict[str, WideDataset] = {
    "nuscenes": WideDataset(
        name="nuscenes",
        gt_root="datasets/nuscenes/sparseMultiplaneImages3_1554x294",
        scene_list="datasets/nuscenes/processed_10Hz/trainval2/nuScenes_Val2.txt",
        car_mask="datasets/nuscenes/processed_10Hz/nuscenes_mask/CAM_BACK_mask.png",
        car_mask_per_scene=False,
        car_mask_ext="png",
        expected_hw=(294, 1554),
    ),
    "lyft1920": WideDataset(
        name="lyft1920",
        gt_root="datasets/lyft/1920_sparseMultiplaneWideFOVImages3",
        scene_list="datasets/lyft/lyft_val1920_3cams/lyft_val1920.txt",
        car_mask="datasets/lyft/lyft_val1920_3cams/ego_car_masks/5.jpg",
        car_mask_per_scene=False,
        car_mask_ext="jpg",
        expected_hw=(294, 1554),
    ),
    "lyft1224": WideDataset(
        name="lyft1224",
        gt_root="datasets/lyft/1224_sparseMultiplaneWideFOVImages3",
        scene_list="datasets/lyft/lyft_val1224_3cams/lyft_val1224.txt",
        car_mask="datasets/lyft/lyft_val1224_3cams/ego_car_masks/5.jpg",
        car_mask_per_scene=False,
        car_mask_ext="jpg",
        expected_hw=(434, 1554),
    ),
    "ddad": WideDataset(
        name="ddad",
        gt_root="datasets/ddad/sparseMultiplaneImages3_1554x322",
        scene_list="datasets/ddad/valid/valid.txt",
        car_mask="datasets/ddad/valid",
        car_mask_per_scene=True,
        car_mask_ext="jpg",
        expected_hw=(322, 1554),
    ),
}

DEFAULT_DATASET = "nuscenes"

# Renders are the *direct* wide outputs, ``{frame}_5_wide.jpg``.  The regex
# deliberately does not match ``{frame}_fixedfov_wide.jpg``.
_RENDER_RE = re.compile(r"^(\d+)_5_wide\.(?:jpg|jpeg|png)$", re.IGNORECASE)
_GT_RE = re.compile(r"^(\d+)_5_multiplane_wide\.png$", re.IGNORECASE)

REGION_NAMES = ("Left", "Center", "Center_masked", "Right")


# ---------------------------------------------------------------------------
# Path / scene helpers
# ---------------------------------------------------------------------------


def resolve_local(path, base: Path = REPO_ROOT) -> Path:
    """Resolve ``path`` against ``base`` unless it is already absolute."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p)


def default_gt_root(dataset: str) -> Path:
    return REPO_ROOT / DATASETS[dataset].gt_root


def resolve_gt_root(gt_root, dataset: str, base: Path = REPO_ROOT) -> Path:
    """Resolve and validate the sparse GT root, failing loudly when missing.

    A missing directory (including the nuScenes sparse GT, which is not in this
    checkout) names the expected path and ``--gt-root`` in the error.
    """
    path = default_gt_root(dataset) if gt_root is None else resolve_local(gt_root, base)
    if not path.is_dir():
        raise SystemExit(
            f"Sparse GT root not found: {path}\n"
            f"Pass --gt-root /path/to/{DATASETS[dataset].gt_root} (this checkout "
            f"expects the {dataset} sparse multiplane GT there)."
        )
    return path


def read_scene_list(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def pair_scene_frames(render_rgb_dir: Path, gt_rgb_dir: Path) -> list[tuple[str, Path, Path]]:
    """Pair render ``{frame}_5_wide.*`` with GT ``{frame}_5_multiplane_wide.png``.

    Frames are matched by their leading integer id (so frames need not start at
    000) and the canonical frame string is zero-padded to 3 digits.  Other
    render suffixes (e.g. ``_fixedfov_wide``) are ignored.
    """
    if not render_rgb_dir.is_dir() or not gt_rgb_dir.is_dir():
        return []
    render_by_id: dict[int, Path] = {}
    for name in os.listdir(render_rgb_dir):
        match = _RENDER_RE.match(name)
        if match:
            render_by_id[int(match.group(1))] = render_rgb_dir / name
    pairs: list[tuple[str, Path, Path]] = []
    for name in os.listdir(gt_rgb_dir):
        match = _GT_RE.match(name)
        if not match:
            continue
        frame_id = int(match.group(1))
        render_path = render_by_id.get(frame_id)
        if render_path is not None:
            pairs.append((f"{frame_id:03d}", render_path, gt_rgb_dir / name))
    pairs.sort(key=lambda item: int(item[0]))
    return pairs


def panel_slices(width: int) -> tuple[slice, slice, slice]:
    """Three contiguous panels; panel width is ``width // 3``."""
    panel_w = int(width) // PANEL_COUNT
    return (
        slice(0, panel_w),
        slice(panel_w, 2 * panel_w),
        slice(2 * panel_w, 3 * panel_w),
    )


def car_mask_path_for_scene(
    dataset_cfg: WideDataset, scene: str, override=None, base: Path = REPO_ROOT
) -> Path:
    """Resolve one scene's camera-5 ego-car keep mask.

    ``--car-mask`` overrides a single file for every dataset.  For a per-scene
    dataset (ddad) a directory override is interpreted as a mask root with the
    same ``<dir>/<scene>/ego_car_masks/5.<ext>`` layout (falling back to
    ``<dir>/<scene>/5.<ext>``); for a dataset-level mask a directory override
    means ``<dir>/5.<ext>``.
    """
    ext = dataset_cfg.car_mask_ext
    if override is not None:
        path = resolve_local(override, base)
        if dataset_cfg.car_mask_per_scene:
            if path.is_dir():
                candidate = path / scene / "ego_car_masks" / f"{RENDER_CAMERA}.{ext}"
                if candidate.is_file():
                    return candidate
                return path / scene / f"{RENDER_CAMERA}.{ext}"
            return path
        if path.is_dir():
            return path / f"{RENDER_CAMERA}.{ext}"
        return path
    root = resolve_local(dataset_cfg.car_mask, base)
    if dataset_cfg.car_mask_per_scene:
        return root / scene / "ego_car_masks" / f"{RENDER_CAMERA}.{ext}"
    return root


# ---------------------------------------------------------------------------
# Image / mask IO and resizing
# ---------------------------------------------------------------------------


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as float64 ``HxWx3`` in ``[0, 1]``."""
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return array


def resize_if_needed(array: np.ndarray, target_hw, is_mask: bool = False):
    """Resize ``array`` to ``target_hw`` (H, W); return ``(array, changed)``.

    Renders use bilinear; masks use nearest.  Arrays already at the target size
    are returned unchanged.
    """
    from PIL import Image

    target_hw = (int(target_hw[0]), int(target_hw[1]))
    if tuple(array.shape[:2]) == target_hw:
        return array, False
    if is_mask:
        img = Image.fromarray((np.asarray(array, dtype=bool).astype(np.uint8) * 255))
        img = img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
        return np.asarray(img) > 127, True
    img = Image.fromarray(
        (np.asarray(array, dtype=np.float64) * 255.0).round().astype(np.uint8)
    )
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0, True


def load_binary_mask(path: Path, target_hw, threshold: int = GT_MASK_THRESHOLD) -> np.ndarray:
    """Load a grayscale mask, nearest-resize to ``target_hw`` and threshold."""
    from PIL import Image

    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != (int(target_hw[1]), int(target_hw[0])):
            mask = mask.resize((int(target_hw[1]), int(target_hw[0])), Image.NEAREST)
        array = np.asarray(mask)
    return array > threshold


def load_car_keep_mask(path: Path, target_hw) -> np.ndarray:
    """Load the camera-5 ego-car mask resized to the *center panel* size.

    Bilinear resize then ``> 0.5`` (white/bright = keep), exactly as DWSplat v2.
    """
    from PIL import Image

    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != (int(target_hw[1]), int(target_hw[0])):
            mask = mask.resize((int(target_hw[1]), int(target_hw[0])), Image.BILINEAR)
        array = np.asarray(mask, dtype=np.float64) / 255.0
    return array > CAR_MASK_KEEP_THRESHOLD


def build_region_masks(
    gt_mask: np.ndarray, render_valid: np.ndarray, car_keep: Optional[np.ndarray] = None
) -> dict[str, np.ndarray]:
    """Build the Left/Center/Center_masked/Right masks (DWSplat v2 rules).

    Left/Right are the GT mask only.  Center is ``GT & render_valid``;
    ``Center_masked`` is Center ``& car_keep`` (camera-5 ego-car keep mask).
    """
    left, center, right = panel_slices(gt_mask.shape[1])
    masks = {
        "Left": gt_mask[:, left],
        "Center": gt_mask[:, center] & render_valid[:, center],
        "Right": gt_mask[:, right],
    }
    if car_keep is not None:
        masks["Center_masked"] = masks["Center"] & car_keep
    else:
        masks["Center_masked"] = masks["Center"]
    return masks


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics_np(render: np.ndarray, gt: np.ndarray, mask: np.ndarray):
    """MAE/RMSE/PSNR on masked pixels; ``None`` for an empty mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return None
    diff = np.abs(render - gt)
    diff2 = (render - gt) ** 2
    mae = float(diff[mask].mean())
    mse = float(diff2[mask].mean())
    rmse = float(np.sqrt(mse))
    psnr = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf")
    return {"mae": mae, "rmse": rmse, "psnr": psnr, "n_pixels": int(mask.sum())}


def sparse_ssim(render: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """DWSplat sparse per-pixel SSIM (no window), 0-255 scale, C1 only."""
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return 0.0
    p = render[mask] * 255.0
    g = gt[mask] * 255.0
    c1 = (0.01 * 255.0) ** 2
    luminance = (2 * p * g + c1) / (p ** 2 + g ** 2 + c1)
    return float(luminance.mean())


def _gaussian_kernel_1d(win_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    coords = np.arange(win_size, dtype=np.float64) - win_size // 2
    kernel = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _separable_conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Zero-padded separable convolution, ``same`` output (Gaussian symmetric)."""
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape
    win = kernel.shape[0]
    pad = win // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="constant")
    vertical = np.zeros((height, width), dtype=np.float64)
    for index, weight in enumerate(kernel):
        vertical += weight * padded[index : index + height, pad : pad + width]
    padded_v = np.pad(vertical, ((0, 0), (pad, pad)), mode="constant")
    out = np.zeros((height, width), dtype=np.float64)
    for index, weight in enumerate(kernel):
        out += weight * padded_v[:, index : index + width]
    return out


def dense_ssim_map(
    render: np.ndarray, gt: np.ndarray, win_size: int = 11, sigma: float = 1.5
) -> np.ndarray:
    """11x11 Gaussian-windowed SSIM map (mean over channels), CPU numpy.

    Mirrors the DWSplat ``dense_ssim_gpu_map``: ``data_range=1.0``,
    ``C1=(0.01)**2``, ``C2=(0.03)**2``, zero padding.
    """
    render = np.asarray(render, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    kernel = _gaussian_kernel_1d(win_size, sigma)
    channels = render.shape[2]

    def conv(x):
        out = np.empty_like(x, dtype=np.float64)
        for channel in range(channels):
            out[..., channel] = _separable_conv2d(x[..., channel], kernel)
        return out

    mu_x = conv(render)
    mu_y = conv(gt)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x_sq = conv(render ** 2) - mu_x_sq
    sigma_y_sq = conv(gt ** 2) - mu_y_sq
    sigma_xy = conv(render * gt) - mu_xy
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim.mean(axis=2)


def masked_mean(values: np.ndarray, mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return None
    return float(np.asarray(values)[mask].mean())


# ---------------------------------------------------------------------------
# Per-frame evaluation
# ---------------------------------------------------------------------------


def evaluate_pair(
    render_rgb_path: Path,
    gt_rgb_path: Path,
    gt_mask_dir: Path,
    render_mask_dir: Path,
    car_keep: np.ndarray,
    lpips_fn=None,
    device=None,
):
    """Evaluate one (render, GT) pair.

    Returns ``(results, lpips_frame)`` or ``(None, None)`` when the GT mask (or
    a required file) is missing.  ``results`` maps region name to a metric dict;
    ``lpips_frame`` maps ``"masked"``/``"unmasked"`` to the per-frame LPIPS.
    """
    gt = load_rgb(gt_rgb_path)
    gt_hw = gt.shape[:2]
    gt_mask_path = gt_mask_dir / gt_rgb_path.name
    if not gt_mask_path.is_file():
        print(f"  [warn] missing GT mask, skipping: {gt_mask_path}", file=sys.stderr)
        return None, None
    gt_mask = load_binary_mask(gt_mask_path, gt_hw)

    render = load_rgb(render_rgb_path)
    render, _ = resize_if_needed(render, gt_hw, is_mask=False)

    render_mask_path = render_mask_dir / f"{render_rgb_path.stem}.png"
    if render_mask_path.is_file():
        render_valid = load_binary_mask(
            render_mask_path, gt_hw, threshold=RENDER_MASK_THRESHOLD
        )
    else:
        render_valid = np.ones(gt_hw, dtype=bool)

    masks = build_region_masks(gt_mask, render_valid, car_keep)
    left_sl, center_sl, right_sl = panel_slices(gt.shape[1])

    results: dict[str, dict] = {}
    # Left / Right: GT mask only, sparse SSIM.
    for name, region_sl in (("Left", left_sl), ("Right", right_sl)):
        region_mask = masks[name]
        res = compute_metrics_np(render[:, region_sl], gt[:, region_sl], region_mask)
        if res is not None:
            res["ssim"] = sparse_ssim(render[:, region_sl], gt[:, region_sl], region_mask)
            results[name] = res

    # Center: GT & render_valid; dense SSIM map averaged over the mask.
    center_render = render[:, center_sl]
    center_gt = gt[:, center_sl]
    center_mask = masks["Center"]
    masked_mask = masks["Center_masked"]
    res_center = compute_metrics_np(center_render, center_gt, center_mask)
    res_masked = compute_metrics_np(center_render, center_gt, masked_mask)
    if res_center is not None or res_masked is not None:
        ssim_map = dense_ssim_map(center_render, center_gt)
        if res_center is not None:
            res_center["ssim"] = masked_mean(ssim_map, center_mask) or 0.0
            results["Center"] = res_center
        if res_masked is not None:
            res_masked["ssim"] = masked_mean(ssim_map, masked_mask) or 0.0
            results["Center_masked"] = res_masked

    lpips_frame = None
    if lpips_fn is not None:
        import torch

        render_t = (
            torch.from_numpy(center_render.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        gt_t = (
            torch.from_numpy(center_gt.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            lpips_map = lpips_fn(render_t * 2 - 1, gt_t * 2 - 1)
        lpips_map = lpips_map.squeeze().cpu().numpy()
        lpips_frame = {
            "masked": masked_mean(lpips_map, masked_mask),
            "unmasked": masked_mean(lpips_map, center_mask),
        }
    return results, lpips_frame


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _region_frames_mean(all_metrics, center_key=None):
    keys = ["Left", "Center", "Right"] if center_key is None else ["Left", center_key, "Right"]
    avgs = {}
    for name in keys:
        metrics = all_metrics.get(name, [])
        if metrics:
            avgs[name] = {
                "mae": float(np.mean([x["mae"] for x in metrics])),
                "rmse": float(np.mean([x["rmse"] for x in metrics])),
                "psnr": float(np.mean([x["psnr"] for x in metrics])),
                "ssim": float(np.mean([x["ssim"] for x in metrics])),
            }
    return avgs, keys


def _write_metric_lines(lines: list[str], mean: dict) -> None:
    lines.append(f"    {'MAE':5s}: {mean['mae'] * 255:.2f}")
    lines.append(f"    {'RMSE':5s}: {mean['rmse'] * 255:.2f}")
    lines.append(f"    {'PSNR':5s}: {mean['psnr']:.2f} dB")
    lines.append(f"    {'SSIM':5s}: {mean['ssim']:.4f}")


def format_report(
    dataset: str,
    render_root: Path,
    gt_root: Path,
    scene_list_path: Path,
    all_metrics: dict,
    scene_results: dict,
    lpips_results: dict,
    total_pairs: int,
    elapsed: float,
) -> str:
    lines: list[str] = []
    lines.append("=== Wide-Image Metrics (direct 3x render vs sparse multiplane GT) ===")
    lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Dataset: {dataset}")
    lines.append(f"Render root: {render_root}")
    lines.append(f"Sparse GT root: {gt_root}")
    lines.append(f"Scene list: {scene_list_path}")
    lines.append(f"Paired frames: {total_pairs}")
    lines.append(f"Elapsed: {elapsed:.1f}s")
    lines.append("")

    lines.append("=" * 80)
    lines.append("Summary Metrics")
    lines.append("=" * 80)

    for region in REGION_NAMES:
        metrics = all_metrics.get(region, [])
        if not metrics:
            continue
        lines.append(f"\n  {region}:")
        lines.append(f"    {'MAE':5s}: {np.mean([x['mae'] for x in metrics]) * 255:.2f}")
        lines.append(f"    {'RMSE':5s}: {np.mean([x['rmse'] for x in metrics]) * 255:.2f}")
        lines.append(f"    {'PSNR':5s}: {np.mean([x['psnr'] for x in metrics]):.2f} dB")
        lines.append(f"    {'SSIM':5s}: {np.mean([x['ssim'] for x in metrics]):.4f}")
        lines.append(f"    Frames: {len(metrics)}")

    base_metrics = []
    for region in ("Left", "Center", "Right"):
        base_metrics.extend(all_metrics.get(region, []))
    if base_metrics:
        total_px = sum(x["n_pixels"] for x in base_metrics)
        lines.append("\n  Overall (pixel-weighted L+Center+R):")
        for metric, scale in (("mae", 255.0), ("rmse", 255.0), ("psnr", 1.0), ("ssim", 1.0)):
            value = sum(x[metric] * x["n_pixels"] for x in base_metrics) / total_px
            if metric in ("mae", "rmse"):
                lines.append(f"    {metric.upper():5s}: {value * scale:.2f}")
            elif metric == "psnr":
                lines.append(f"    {metric.upper():5s}: {value:.2f} dB")
            else:
                lines.append(f"    {metric.upper():5s}: {value:.4f}")

    for center_key, label in (("Center", "L/Center/R"), ("Center_masked", "L/Center_masked/R")):
        avgs, keys = _region_frames_mean(all_metrics, center_key)
        if len(avgs) == len(keys):
            mean = {
                metric: float(np.mean([avgs[name][metric] for name in keys]))
                for metric in ("mae", "rmse", "psnr", "ssim")
            }
            lines.append(f"\n  Mean {label} (equal weight):")
            _write_metric_lines(lines, mean)

    extrap, extrap_keys = _region_frames_mean(all_metrics, None)
    if all(name in extrap for name in ("Left", "Right")):
        mean = {
            metric: float(np.mean([extrap[name][metric] for name in ("Left", "Right")]))
            for metric in ("mae", "rmse", "psnr", "ssim")
        }
        lines.append("\n  Extrapolation (mean L/R):")
        _write_metric_lines(lines, mean)

    if lpips_results.get("masked"):
        values = [x for x in lpips_results["masked"] if x is not None]
        if values:
            lines.append(
                f"\n  Center LPIPS (masked): {np.mean(values):.4f} ({len(values)} frames)"
            )
    if lpips_results.get("unmasked"):
        values = [x for x in lpips_results["unmasked"] if x is not None]
        if values:
            lines.append(
                f"  Center LPIPS (unmasked): {np.mean(values):.4f} ({len(values)} frames)"
            )

    lines.append("")
    lines.append("=" * 80)
    lines.append("Per-Scene Results")
    lines.append("=" * 80)
    for scene in sorted(scene_results):
        summary = scene_results[scene]
        lines.append(f"\n  Scene {scene} ({summary['evaluated']} evaluated / "
                     f"{summary['paired']} paired):")
        for region in REGION_NAMES:
            if region in summary["regions"]:
                avg = summary["regions"][region]
                lines.append(
                    f"    {region:12s}: MAE={avg['mae'] * 255:.2f}, "
                    f"RMSE={avg['rmse'] * 255:.2f}, PSNR={avg['psnr']:.2f}, "
                    f"SSIM={avg['ssim']:.4f} ({avg['n_frames']} frames)"
                )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name if sys.argv and sys.argv[0] else None,
        description=(
            "Evaluate this project's direct 3x-wide renders ({frame}_5_wide.jpg) "
            "against the sparse multiplane GT, following the DWSplat v2 protocol."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default=DEFAULT_DATASET,
        help="Dataset preset (no WideDrive preset; the render is the direct wide output).",
    )
    parser.add_argument(
        "--render-root",
        required=True,
        help="Root of the direct wide renders: <root>/<scene>/rgb/{frame}_5_wide.jpg.",
    )
    parser.add_argument(
        "--gt-root",
        default=None,
        help=(
            "Sparse multiplane GT root: <root>/<scene>/{rgb,mask}/"
            "{frame}_5_multiplane_wide.png. Default: the --dataset preset path."
        ),
    )
    parser.add_argument(
        "--scene-list",
        default=None,
        help="Scene id list (one per line). Default: the --dataset preset list.",
    )
    parser.add_argument("--max-scenes", type=int, default=None,
                        help="Maximum number of scenes to evaluate.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames per scene to evaluate.")
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Skip the center LPIPS computation (no lpips/torch needed).",
    )
    parser.add_argument(
        "--car-mask",
        default=None,
        help=(
            "Override the camera-5 ego-car mask. A file is used for every scene; "
            "for ddad a directory is treated as a per-scene mask root."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Report path. Default: "
            "<render-root>/wide_metrics_<dataset>_<timestamp>.txt."
        ),
    )
    return parser


def main(argv=None) -> int:
    import time

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    start = time.time()

    dataset_cfg = DATASETS[args.dataset]
    render_root = resolve_local(args.render_root)
    if not render_root.is_dir():
        raise SystemExit(f"Render root not found: {render_root}")
    gt_root = resolve_gt_root(args.gt_root, args.dataset)

    if args.scene_list is not None:
        scene_list_path = resolve_local(args.scene_list)
    else:
        scene_list_path = resolve_local(dataset_cfg.scene_list)
    if not scene_list_path.is_file():
        raise SystemExit(f"Scene list not found: {scene_list_path}")
    scenes = read_scene_list(scene_list_path)
    if args.max_scenes is not None:
        scenes = scenes[: max(0, args.max_scenes)]

    lpips_fn = None
    device = None
    if not args.no_lpips:
        try:
            import torch
            import lpips as lpips_lib
        except ImportError as error:  # pragma: no cover - environment dependent
            raise SystemExit(
                "LPIPS was requested but torch/lpips are unavailable "
                f"({error}). Install them or pass --no-lpips."
            ) from error
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_fn = lpips_lib.LPIPS(net="alex", spatial=True).to(device)

    all_metrics = {name: [] for name in REGION_NAMES}
    lpips_results = {"masked": [], "unmasked": []}
    scene_results: dict[str, dict] = {}
    total_pairs = 0
    total_evaluated = 0
    car_cache: dict[tuple[str, int, int], np.ndarray] = {}

    for scene in scenes:
        render_scene_dir = render_root / scene
        gt_scene_dir = gt_root / scene
        pairs = pair_scene_frames(render_scene_dir / "rgb", gt_scene_dir / "rgb")
        if args.max_frames is not None:
            pairs = pairs[: max(0, args.max_frames)]
        if not pairs:
            print(f"[warn] scene {scene}: no render/GT pairs, skipping", file=sys.stderr)
            scene_results[scene] = {
                "regions": {},
                "paired": 0,
                "evaluated": 0,
            }
            continue

        scene_pairs = len(pairs)
        total_pairs += scene_pairs
        scene_metrics = {name: [] for name in REGION_NAMES}
        scene_evaluated = 0
        gt_mask_dir = gt_scene_dir / "mask"
        render_mask_dir = render_scene_dir / "mask"

        for frame, render_rgb_path, gt_rgb_path in pairs:
            # Resolve the per-scene car mask lazily at the GT center-panel size.
            gt_hw = None
            try:
                from PIL import Image

                with Image.open(gt_rgb_path) as image:
                    gt_hw = (image.height, image.width)
            except OSError:
                gt_hw = None
            if gt_hw is None:
                print(f"  [warn] scene {scene} frame {frame}: unreadable GT, skipping",
                      file=sys.stderr)
                continue
            cache_key = (scene, gt_hw[0], gt_hw[1])
            if cache_key not in car_cache:
                car_mask_path = car_mask_path_for_scene(
                    dataset_cfg, scene, override=args.car_mask
                )
                if not car_mask_path.is_file():
                    raise FileNotFoundError(
                        f"Camera-5 ego-car mask not found for scene {scene}: "
                        f"{car_mask_path}. Pass --car-mask to override it."
                    )
                car_cache[cache_key] = load_car_keep_mask(
                    car_mask_path, (gt_hw[0], gt_hw[1] // PANEL_COUNT)
                )
            car_keep = car_cache[cache_key]

            try:
                results, lpips_frame = evaluate_pair(
                    render_rgb_path,
                    gt_rgb_path,
                    gt_mask_dir,
                    render_mask_dir,
                    car_keep,
                    lpips_fn=lpips_fn,
                    device=device,
                )
            except (OSError, FileNotFoundError) as error:
                print(f"  [warn] scene {scene} frame {frame}: {error}", file=sys.stderr)
                continue
            if results is None:
                continue

            scene_evaluated += 1
            total_evaluated += 1
            for region in REGION_NAMES:
                if region in results:
                    scene_metrics[region].append(results[region])
                    all_metrics[region].append(results[region])
            if lpips_frame is not None:
                lpips_results["masked"].append(lpips_frame["masked"])
                lpips_results["unmasked"].append(lpips_frame["unmasked"])

        regions_summary = {}
        for region in REGION_NAMES:
            metrics = scene_metrics[region]
            if metrics:
                regions_summary[region] = {
                    "n_frames": len(metrics),
                    "mae": float(np.mean([x["mae"] for x in metrics])),
                    "rmse": float(np.mean([x["rmse"] for x in metrics])),
                    "psnr": float(np.mean([x["psnr"] for x in metrics])),
                    "ssim": float(np.mean([x["ssim"] for x in metrics])),
                }
        scene_results[scene] = {
            "regions": regions_summary,
            "paired": scene_pairs,
            "evaluated": scene_evaluated,
        }

    elapsed = time.time() - start
    if total_pairs == 0:
        print(
            f"[error] no render/GT pairs found under {render_root} for dataset "
            f"{args.dataset}; nothing to evaluate.",
            file=sys.stderr,
        )
        return 1

    report = format_report(
        args.dataset,
        render_root,
        gt_root,
        scene_list_path,
        all_metrics,
        scene_results,
        lpips_results,
        total_pairs,
        elapsed,
    )

    if args.output is not None:
        output_path = resolve_local(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = render_root / f"wide_metrics_{args.dataset}_{timestamp}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(report)
    print(f"Report saved to {output_path}")
    print(f"Evaluated {total_evaluated} frame(s) from {len(scene_results)} scene(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

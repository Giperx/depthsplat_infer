"""Pure helpers for locating and validating a local DINOv2 ``torch.hub`` source.

This module deliberately does **not** import torch.  It is used by both the
model code (``MultiViewUniMatch``) and the standalone nuScenes inference script,
and is unit-tested directly, so it must stay importable without the heavy
torch / CUDA dependencies.

A "source" here is the directory normally populated by
``torch.hub.load("facebookresearch/dinov2", ...)``.  Loading it with
``source="local"`` requires the directory to contain a ``hubconf.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Union

#: File that marks a directory as a valid ``torch.hub`` source.
HUB_MARKER = "hubconf.py"

#: Environment variable overriding the automatically detected local source.
DINOV2_ENV_VAR = "DINOV2_SOURCE"

#: Default location populated by a previous ``torch.hub.load`` of DINOv2.
DEFAULT_DINOV2_HUB_CACHE = (
    Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main"
)

PathLike = Union[str, "os.PathLike[str]"]


def is_valid_dinov2_source(source: PathLike) -> bool:
    """Return ``True`` when ``source`` is a directory containing ``hubconf.py``."""
    path = Path(source).expanduser()
    return path.is_dir() and (path / HUB_MARKER).is_file()


def validate_dinov2_source(source: PathLike) -> Path:
    """Validate a local DINOv2 source and return its resolved path.

    Raises:
        ValueError: if the path is not a directory or has no ``hubconf.py``.
    """
    path = Path(source).expanduser()
    if not path.is_dir():
        raise ValueError(
            f"Local DINOv2 source directory not found: {path}. Pass a directory "
            f"that contains {HUB_MARKER} (for example the torch.hub cache "
            f"directory {DEFAULT_DINOV2_HUB_CACHE})."
        )
    if not (path / HUB_MARKER).is_file():
        raise ValueError(
            f"Local DINOv2 source {path} does not contain {HUB_MARKER}; it is not "
            "a valid torch.hub source."
        )
    return path


def default_dinov2_source(
    env: Optional[Mapping[str, str]] = None,
) -> Optional[Path]:
    """Return the default local DINOv2 source, or ``None`` when unavailable.

    ``$DINOV2_SOURCE`` takes precedence and is returned as-is (it is validated
    later).  Otherwise the standard torch hub cache directory is returned only
    when it already exists and looks valid.
    """
    environ = os.environ if env is None else env
    candidate = environ.get(DINOV2_ENV_VAR)
    if candidate:
        return Path(candidate).expanduser()
    if is_valid_dinov2_source(DEFAULT_DINOV2_HUB_CACHE):
        return DEFAULT_DINOV2_HUB_CACHE
    return None

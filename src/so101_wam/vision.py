"""Deterministic wrist-RGB preprocessing shared by training and inference."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


COMPACT_IMAGE_MAX_SIDE = 64


class VisionPreprocessError(ValueError):
    """Raised when a wrist image cannot enter the compact model."""


def compact_rgb_image(
    image: NDArray[np.uint8],
    *,
    max_side: int = COMPACT_IMAGE_MAX_SIDE,
) -> NDArray[np.uint8]:
    """Downsample RGB with deterministic nearest-neighbor indexing.

    Small images are copied unchanged. Larger frames preserve aspect ratio and
    are reduced before tensor stacking, avoiding a large float32 expansion of
    a full 3--12 second 640x480 prompt.
    """

    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
        raise VisionPreprocessError(
            f"wrist image must be uint8[H,W,3], got {value.dtype}{value.shape}"
        )
    if not isinstance(max_side, int) or isinstance(max_side, bool) or max_side < 8:
        raise VisionPreprocessError("max_side must be an integer >= 8")

    height, width = int(value.shape[0]), int(value.shape[1])
    if height < 1 or width < 1:
        raise VisionPreprocessError("wrist image height and width must be positive")
    if max(height, width) <= max_side:
        return np.array(value, copy=True, order="C")

    scale = max_side / float(max(height, width))
    output_height = max(8, int(round(height * scale)))
    output_width = max(8, int(round(width * scale)))
    row_indices = np.linspace(0, height - 1, output_height).round().astype(np.int64)
    column_indices = np.linspace(0, width - 1, output_width).round().astype(np.int64)
    resized = np.take(np.take(value, row_indices, axis=0), column_indices, axis=1)
    return np.ascontiguousarray(resized, dtype=np.uint8)


__all__ = ["COMPACT_IMAGE_MAX_SIDE", "VisionPreprocessError", "compact_rgb_image"]

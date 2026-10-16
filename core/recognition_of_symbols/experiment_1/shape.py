"""Pixel-level shape comparison between a candidate stroke and reference braces.

Each mask is cropped to its bounding box and resampled to a fixed grid, so a
two-line and a seven-line brace become comparable. Two normalised shapes are
compared with the symmetric chamfer distance: the mean distance from each ink
pixel of one shape to the nearest ink pixel of the other, in grid pixels.
Low is similar; 0 is identical.
"""
from __future__ import annotations

import cv2
import numpy as np

GRID = (96, 24)  # rows, columns


def normalise(mask: np.ndarray, grid: tuple[int, int] = GRID) -> np.ndarray:
    """Crop `mask` to its bounding box and resample it to `grid`."""
    ys, xs = np.where(mask)
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.uint8) * 255
    out = cv2.resize(crop, (grid[1], grid[0]), interpolation=cv2.INTER_AREA) >= 64
    return out if out.any() else cv2.resize(crop, (grid[1], grid[0]), interpolation=cv2.INTER_NEAREST) > 0


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric chamfer distance between two normalised shapes."""
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
    return float((db[a].mean() + da[b].mean()) / 2)


def best_match(shape: np.ndarray, references: list[np.ndarray]) -> float:
    """Distance from `shape` to the closest reference."""
    return min(chamfer(shape, r) for r in references)
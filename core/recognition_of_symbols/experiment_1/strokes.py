"""Pixel-level stroke primitives used by the mark detectors.

`path_length` is a directional path opening: for every ink pixel it gives the
length of the longest ink path through that pixel that keeps moving along one
axis while drifting at most one pixel sideways per step. Unlike a straight
line-kernel opening it follows curved strokes, and unlike connected components
it is not fooled when a mark touches a letter.
"""
from __future__ import annotations

import cv2
import numpy as np


def _one_way(ink: np.ndarray) -> np.ndarray:
    """Longest downward-cone path ending at each pixel, scanning top to bottom."""
    h, w = ink.shape
    out = np.zeros((h, w), np.int32)
    prev = np.zeros(w, np.int32)
    for y in range(h):
        best = prev.copy()
        best[1:] = np.maximum(best[1:], prev[:-1])
        best[:-1] = np.maximum(best[:-1], prev[1:])
        prev = np.where(ink[y], best + 1, 0)
        out[y] = prev
    return out


def path_length(ink: np.ndarray, axis: str = "vertical") -> np.ndarray:
    """Per-pixel length of the longest near-`axis` ink path through the pixel.

    axis="vertical"   -> paths within 45 degrees of vertical
    axis="horizontal" -> paths within 45 degrees of horizontal
    """
    m = ink if axis == "vertical" else ink.T
    down = _one_way(m)
    up = _one_way(m[::-1])[::-1]
    total = np.where(m, down + up - 1, 0)
    return total if axis == "vertical" else total.T


def line_spacing(ink: np.ndarray, lo: int = 30, hi: int = 160) -> int:
    """Dominant text line pitch in pixels, from the row-profile autocorrelation."""
    prof = ink.sum(axis=1).astype(np.float64)
    prof -= prof.mean()
    ac = np.correlate(prof, prof, "full")[len(prof) - 1 :]
    return int(np.argmax(ac[lo:hi]) + lo)


def stroke_width(ink: np.ndarray) -> float:
    """Median pen stroke width, from the distance transform on the skeleton ridge."""
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = dist[(dist > 0) & (dist == cv2.dilate(dist, np.ones((3, 3), np.uint8)))]
    return float(2 * np.median(ridge)) if ridge.size else 3.0
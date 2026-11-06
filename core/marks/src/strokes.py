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


def line_spacing(ink: np.ndarray, lo: int = 8, hi_frac: float = 0.12, sub_ratio: float = 0.25) -> int:
    """Text line pitch in pixels, from the row-profile autocorrelation.

    The autocorrelation of the ink-per-row profile peaks at the line pitch
    and at its multiples. In a register the entries repeat too (two or three
    lines each), so the strongest peak may be the entry pitch rather than the
    line pitch. The strongest peak is therefore taken first, and replaced by
    the peak near a half or a third of it when that one is clearly present
    (at least `sub_ratio` of its height). The search runs from `lo` pixels to
    `hi_frac` of the page height, so it works at any image resolution.
    """
    from scipy.signal import find_peaks

    prof = ink.sum(axis=1).astype(np.float64)
    prof -= prof.mean()
    ac = np.correlate(prof, prof, "full")[len(prof) - 1 :]
    if ac[0] <= 0:
        return 40  # an empty page: any value will do
    ac /= ac[0]
    hi = max(lo + 3, int(hi_frac * len(prof)))
    peaks, _ = find_peaks(ac[:hi], prominence=0.02)
    peaks = peaks[peaks >= lo]
    if not len(peaks):
        return int(np.argmax(ac[lo:hi]) + lo)
    best = int(peaks[np.argmax(ac[peaks])])
    for k in (3, 2):  # prefer the smallest clear fundamental
        near = peaks[np.abs(peaks - best / k) <= 0.15 * best / k]
        if len(near):
            cand = int(near[np.argmax(ac[near])])
            if ac[cand] > 0 and ac[cand] >= sub_ratio * ac[best]:
                return cand
    return best


def stroke_width(ink: np.ndarray) -> float:
    """Median pen stroke width, from the distance transform on the skeleton ridge."""
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 3)
    ridge = dist[(dist > 0) & (dist == cv2.dilate(dist, np.ones((3, 3), np.uint8)))]
    return float(2 * np.median(ridge)) if ridge.size else 3.0
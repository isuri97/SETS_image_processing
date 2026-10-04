"""
Handwriting normalisation for word crops (label-free).

Removes writer/pen variation that is not about word identity, so that the
same word written by different scribes or pens looks more alike:

1. Binarisation   : Otsu on the tight crop (ink = 1).
2. Slant removal  : shear angle chosen to maximise the sum of squared column
                    ink counts (vertical strokes concentrate ink in few
                    columns); the crop is sheared back to upright.
3. Zone alignment : the core (x-height) band is located from the horizontal
                    ink profile and mapped to fixed rows of the output, so
                    ascender, core and descender zones line up across words.
                    Scale is set by the x-height, not the box height, so a
                    tall flourish no longer shrinks the whole word.
4. Stroke width   : the ink is thinned to a 1-px skeleton and re-thickened to
                    a fixed width, removing heavy-vs-light pen differences.

`normalise_word` returns a float image (ink = 1) of fixed height and a width
proportional to the word, for DTW; `to_canvas` resizes it to the fixed HOG
canvas.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class NormParams:
    height: int = 64                  # output height (px)
    zones: tuple = (24, 16, 24)       # ascender / core / descender rows
    slant_range: int = 45             # search shear angles in +/- degrees
    slant_step: int = 3
    core_frac: float = 0.35           # row is "core" if ink >= frac x peak
    stroke_px: int = 3                # re-thickened stroke width
    pad: int = 2


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def binarise_crop(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return ink.astype(np.uint8)


def shear(ink: np.ndarray, deg: float) -> np.ndarray:
    """Horizontal shear x' = x + tan(deg) * (y - (h-1)): the bottom row stays
    fixed and upper rows move left (deg > 0) or right (deg < 0). The canvas
    is widened so nothing is cut off."""
    h, w = ink.shape
    k = np.tan(np.radians(deg))
    extra = int(np.ceil(abs(k) * (h - 1))) + 1
    off = k * (h - 1) if k > 0 else 0.0
    M = np.float32([[1, k, off - k * (h - 1)], [0, 1, 0]])
    return cv2.warpAffine(ink, M, (w + extra, h), flags=cv2.INTER_NEAREST)


def deslant(ink: np.ndarray, p: NormParams) -> np.ndarray:
    best, best_score = ink, -1.0
    for deg in range(-p.slant_range, p.slant_range + 1, p.slant_step):
        s = shear(ink, deg)
        score = float((s.sum(axis=0).astype(np.float64) ** 2).sum())
        if score > best_score:
            best, best_score = s, score
    return best


def core_band(ink: np.ndarray, p: NormParams) -> tuple[int, int]:
    """Rows [top, bottom) of the x-height band: the contiguous run of rows
    around the ink peak whose ink count stays above core_frac x peak."""
    prof = cv2.GaussianBlur(ink.sum(axis=1).astype(np.float32).reshape(-1, 1),
                            (1, 5), 0).ravel()
    peak = int(prof.argmax())
    thr = p.core_frac * prof[peak]
    top, bottom = peak, peak + 1
    while top > 0 and prof[top - 1] >= thr:
        top -= 1
    while bottom < len(prof) and prof[bottom] >= thr:
        bottom += 1
    return top, max(bottom, top + 2)


def align_zones(ink: np.ndarray, p: NormParams) -> np.ndarray:
    """Scale so the core band has zones[1] rows, and place it at rows
    zones[0]..zones[0]+zones[1]; ascenders/descenders are clipped to their
    zones (they carry less identity information than the core)."""
    top, bottom = core_band(ink, p)
    scale = p.zones[1] / (bottom - top)
    h, w = ink.shape
    new_w = max(4, int(round(w * scale)))
    new_h = max(4, int(round(h * scale)))
    resized = cv2.resize(ink.astype(np.float32), (new_w, new_h),
                         interpolation=cv2.INTER_AREA)
    out = np.zeros((p.height, new_w), np.float32)
    shift = p.zones[0] - int(round(top * scale))      # core top -> row zones[0]
    src0, dst0 = max(0, -shift), max(0, shift)
    n = min(new_h - src0, p.height - dst0)
    if n > 0:
        out[dst0:dst0 + n] = resized[src0:src0 + n]
    return (out > 0.3).astype(np.uint8)


def fix_stroke_width(ink: np.ndarray, p: NormParams) -> np.ndarray:
    skel = skeletonize(ink.astype(bool)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.stroke_px, p.stroke_px))
    return cv2.dilate(skel, k)


def trim_columns(ink: np.ndarray, pad: int) -> np.ndarray:
    cols = np.flatnonzero(ink.any(axis=0))
    if len(cols) == 0:
        return ink
    return np.pad(ink[:, cols[0]:cols[-1] + 1], ((0, 0), (pad, pad)))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def normalise_word(gray_crop: np.ndarray, p: NormParams = NormParams()) -> np.ndarray:
    """Tight grayscale word crop -> normalised float ink image
    (height p.height, width proportional to the word)."""
    ink = binarise_crop(gray_crop)
    if ink.sum() == 0:
        return np.zeros((p.height, 8), np.float32)
    ink = deslant(ink, p)
    ink = align_zones(ink, p)
    ink = fix_stroke_width(ink, p)
    return trim_columns(ink, p.pad).astype(np.float32)


def to_canvas(word: np.ndarray, width: int = 144) -> np.ndarray:
    """Stretch horizontally to the fixed HOG canvas width. Height (zones) is
    already normalised, so only width is resampled."""
    img = cv2.resize(word, (width, word.shape[0]), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(img, (3, 3), 0.8)
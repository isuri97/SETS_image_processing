"""Page preprocessing: page crop, illumination flattening, ink binarisation.

All later stages work on the boolean ink mask returned by `binarise`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter


def paper_mask(bgr: np.ndarray, erode: int = 25, sat_max: int = 36) -> np.ndarray:
    """Boolean mask of the sheet: largest non-desk region with holes filled.

    The desk is either strongly coloured (wood) or dark; paper is neither.
    Thresholds are deliberately loose so shadows and colour casts on the
    paper stay inside the mask: on the two test pages shaded paper reaches a
    saturation of about 28, while the desk is above 45 or much darker.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    clean = cv2.GaussianBlur(cv2.morphologyEx(bgr, cv2.MORPH_CLOSE, k), (0, 0), 4)
    hsv = cv2.cvtColor(clean, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    v_paper = np.percentile(v[s < 25], 50) if (s < 25).any() else 200.0
    raw = ((s < sat_max) & (v > 0.45 * v_paper)).astype(np.uint8)
    # opening wider than any sliver of pale desk that hugs the sheet edge
    k_open = max(15, int(0.05 * bgr.shape[1])) | 1
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw)
    if n < 2:
        return np.ones(bgr.shape[:2], bool)
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(bgr.shape[:2], np.uint8)
    cv2.drawContours(filled, cnts, -1, 1, cv2.FILLED)  # outline with holes filled
    filled = _straighten_edges(filled.astype(bool))
    filled = cv2.erode(filled.astype(np.uint8), np.ones((erode, erode), np.uint8))
    return filled.astype(bool)


def _straighten_edges(mask: np.ndarray, window: float = 0.25, slack: int = 10) -> np.ndarray:
    """Clip bulges of the sheet mask that stick out past its typical edge.

    A sheet edge is close to straight, so the outermost mask pixel of each row
    (or column) should follow a running median of its neighbours. Patches of
    pale desk that survive the colour test show up as local bulges and are cut.
    """
    out = mask.copy()
    for axis in (0, 1):  # 0: left/right edge per row, 1: top/bottom edge per column
        m = out if axis == 0 else out.T
        has = m.any(axis=1)
        if has.sum() < 10:
            continue
        idx = np.arange(m.shape[1])
        lo = np.where(has, np.where(m, idx, m.shape[1]).min(axis=1), 0)
        hi = np.where(has, np.where(m, idx, -1).max(axis=1), 0)
        size = max(3, int(window * has.sum())) | 1
        lo_med = median_filter(lo[has], size=size, mode="nearest")
        hi_med = median_filter(hi[has], size=size, mode="nearest")
        keep = np.zeros_like(m)
        rows = np.where(has)[0]
        inside = (idx[None, :] >= (lo_med - slack)[:, None]) & (idx[None, :] <= (hi_med + slack)[:, None])
        keep[rows] = inside
        m &= keep  # in-place on `out` (or its transpose view)
    return out


def cut_fold(ink: np.ndarray, min_frac: float = 0.3, edge_frac: float = 0.15) -> np.ndarray:
    """Blank out a gutter fold shadow and everything beyond it (facing page)."""
    h, w = ink.shape
    out = ink.copy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, cw, ch = stats[i, :4]
        if ch < min_frac * h:
            continue
        if x + cw < edge_frac * w:  # fold near left edge
            out[:, : x + cw // 2] = False
            out[lab == i] = False
        elif x > (1 - edge_frac) * w:  # fold near right edge
            out[:, x + cw // 2 :] = False
            out[lab == i] = False
        else:
            out[lab == i] = False
    return out


def drop_border_touching(ink: np.ndarray, margin: int = 2, band: int = 40) -> np.ndarray:
    """Remove ink components that hug the crop border (desk and edge shadows).

    A component is dropped if it touches the border and at least half of it
    lies within `band` pixels of it. A rule or a word that merely runs up to
    the sheet edge is kept.
    """
    h, w = ink.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    x, y = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    x1, y1 = x + stats[:, cv2.CC_STAT_WIDTH], y + stats[:, cv2.CC_STAT_HEIGHT]
    touches = (x < margin) | (y < margin) | (x1 > w - margin) | (y1 > h - margin)
    edge = np.ones((h, w), bool)
    edge[band : h - band, band : w - band] = False
    near_edge = np.bincount(lab[edge], minlength=n)
    drop = touches & (near_edge >= 0.5 * stats[:, cv2.CC_STAT_AREA])
    drop[0] = True
    return ~drop[lab]


def crop_page(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Crop to the sheet. Returns (bgr_crop, sheet_mask_crop, (x0, y0))."""
    m = paper_mask(bgr)
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return bgr[y0:y1, x0:x1], m[y0:y1, x0:x1], (x0, y0)


def flatten(gray: np.ndarray, sigma: float = 40) -> np.ndarray:
    """Divide by a blurred background estimate to remove lighting gradients."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), sigma)
    norm = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1)
    return np.clip(norm * 255, 0, 255).astype(np.uint8)


def binarise(bgr: np.ndarray, block: int = 51, c: float = 14, min_area: int = 12) -> np.ndarray:
    """Return a boolean ink mask (True = ink).

    Uses the minimum colour channel: brown iron-gall ink is darkest in blue,
    grey pencil is equally dark in all three, so both survive.
    """
    chan = bgr.min(axis=2)
    flat = flatten(chan)
    th = cv2.adaptiveThreshold(
        flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, c
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


@dataclass
class Page:
    """A photographed sheet, cropped to the paper, with its ink masks."""

    bgr: np.ndarray  # colour crop
    ink: np.ndarray  # boolean ink mask used by every detector
    faint: np.ndarray  # more sensitive mask; keeps pale dry-pen strokes, and more noise
    contrast: np.ndarray  # float32 darkness of each pixel relative to the local paper (0 = paper)


def load(path: str, faint_c: float = 6) -> Page:
    """Load an image, crop it to the sheet and binarise it."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    crop, sheet, _ = crop_page(bgr)
    ink = drop_border_touching(cut_fold(binarise(crop) & sheet))
    # the fold test is only reliable on the clean mask, so reuse its verdict
    gone = cv2.dilate(((binarise(crop) & sheet) & ~ink).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    faint = binarise(crop, c=faint_c, min_area=8) & sheet & ~gone
    cols = np.where(ink.any(axis=0))[0]
    if len(cols):  # nothing outside the text block (facing page, desk edge)
        faint[:, : cols.min()] = False
        faint[:, cols.max() + 1 :] = False
    faint = drop_border_touching(faint) | ink
    contrast = 1.0 - flatten(crop.min(axis=2)).astype(np.float32) / 255.0
    return Page(crop, ink, faint, contrast)
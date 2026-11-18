"""Full-page, box-free line polygons for two-column manuscript registers.

For each page this finds
  * columns           (gutter = emptiest vertical band near the page centre)
  * text lines        (deskewed projection peaks; boundaries are minimum-ink seams, so
                       ascenders, descenders and superscripts are not cut)
  * brackets          (tall, thin strokes)
  * bracket labels    (text to the right of a bracket, e.g. "tutti Inglesi")
  * bracket groups    (lines spanned by each bracket, linked to its label)

Output: <stem>_polygons.png (overlay), <stem>_regions.json, and optionally one masked
crop per line.

Usage
  python page_polygons.py                 # uses page.png next to this script
  python page_polygons.py my_scan.jpg --crops
or inline:
  from page_polygons import run
  res = run("page.png")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
from scipy.signal import find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "page.png"
DEFAULT_OUT = SCRIPT_DIR / "out_page"


# ======================================================================
# data
# ======================================================================
@dataclass
class Comp:
    """Axis-aligned bounding box of an ink component or region."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int: return self.x2 - self.x1
    @property
    def h(self) -> int: return self.y2 - self.y1
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2


@dataclass
class Line:
    id: str
    column: int
    polygon: np.ndarray                      # (N, 2) int, page coordinates
    centre_y: int
    x_start: int
    x_end: int
    mask: np.ndarray | None = field(default=None, repr=False)


@dataclass
class Bracket:
    id: str
    column: int
    box: Comp
    members: list[str] = field(default_factory=list)   # line ids
    label_id: str | None = None


@dataclass
class Label:
    id: str
    column: int
    polygons: list[np.ndarray]               # one tight polygon per label line
    bracket_id: str


# ======================================================================
# 1. ink
# ======================================================================
def flatten_illumination(gray: np.ndarray, k: int) -> np.ndarray:
    """Divide by a smooth background estimate so shadows/gradients in photos disappear."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    return np.clip(gray.astype(float) / np.maximum(bg, 1) * 255, 0, 255)


def ink_mask(rgb: np.ndarray, rel_thresh: float = 190, bg_kernel: int = 41) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    norm = flatten_illumination(gray, bg_kernel)
    ink = (norm < rel_thresh).astype(np.uint8)
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)


def page_mask(rgb: np.ndarray, min_sat: int = 35, max_hue: int = 18,
              frac: float = 0.5) -> np.ndarray:
    """Page rectangle: trims table/background strips along the image edges.

    A pixel looks like background (e.g. a wooden table) when it is saturated and reddish
    (OpenCV hue < 18). Aged or shadowed paper can be as saturated, but it is yellowish and,
    more importantly, never fills a whole edge strip. So whole columns/rows are trimmed
    from each image edge while more than `frac` of their pixels look like background.
    """
    hsv = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), (0, 0), 5)
    bg = (hsv[..., 1] > min_sat) & (hsv[..., 0] < max_hue)
    H, W = bg.shape

    def trim(profile: np.ndarray) -> tuple[int, int]:
        lo, hi = 0, len(profile)
        while lo < hi and profile[lo] > frac:
            lo += 1
        while hi > lo and profile[hi - 1] > frac:
            hi -= 1
        return lo, hi

    x0, x1 = trim(bg.mean(0))
    y0, y1 = trim(bg[:, x0:x1].mean(1))
    out = np.zeros((H, W), bool)
    out[y0:y1, x0:x1] = True
    return out


def components(mask: np.ndarray):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return lab, st[1:], np.arange(1, n)


def estimate_char_height(ink: np.ndarray) -> float:
    _, st, _ = components(ink)
    st = st[st[:, 4] >= 15]
    if len(st) == 0:
        return 20.0
    big = st[:, 4] >= np.percentile(st[:, 4], 50)
    return float(np.median(st[big, 3]))


def clean_page(ink: np.ndarray, char_h: float, border: int = 3) -> np.ndarray:
    """Drop specks, border-touching junk (page edge, table/wood texture) and long rules."""
    H, W = ink.shape
    lab, st, ids = components(ink)
    x, y, w, h, a = st.T
    keep = a >= max(10, 0.05 * char_h ** 2)
    keep &= ~((x <= border) | (y <= border) | (x + w >= W - border) | (y + h >= H - border))
    keep &= ~((w > 10 * char_h) & (h < 1.2 * char_h))            # horizontal rules
    keep &= ~((h > 0.5 * H) | (w > 0.6 * W))                      # page-long structures
    lut = np.zeros(len(ids) + 1, bool)
    lut[ids] = keep
    return lut[lab]


def drop_edge_verticals(ink: np.ndarray, x0: int, char_h: float) -> np.ndarray:
    """Remove tall strokes hugging the column's left edge (page fold, binding shadow)."""
    lab, st, ids = components(ink)
    lut = np.ones(len(ids) + 1, bool)
    lut[0] = False
    for i, (x, y, w, h, a) in zip(ids, st):
        if x < x0 + 4 * char_h and h > 3 * char_h and h > 2.5 * w:
            lut[i] = False
    return lut[lab] & ink


# ======================================================================
# 2. columns
# ======================================================================
def find_columns(ink: np.ndarray, char_h: float, search=(0.35, 0.65),
                 split_depth: float = 0.8) -> list[tuple[int, int]]:
    """Split at the emptiest vertical band near the centre; trim each column to its text."""
    H, W = ink.shape
    prof = gaussian_filter1d(ink.sum(axis=0).astype(float), 2 * char_h)
    lo, hi = int(search[0] * W), int(search[1] * W)
    gutter = lo + int(np.argmin(prof[lo:hi]))
    # Only split if the gutter is genuinely emptier than the text on both sides.
    if prof[gutter] > split_depth * min(prof[:gutter].max(), prof[gutter:].max()):
        spans = [(0, W)]
    else:
        spans = [(0, gutter), (gutter, W)]
    out = []
    for a, b in spans:
        cols = np.nonzero(prof[a:b] > 0.05 * prof[a:b].max())[0]
        out.append((a + int(cols.min()), a + int(cols.max()) + 1) if cols.size else (a, b))
    return out


# ======================================================================
# 3. brackets and their labels
# ======================================================================
def find_brackets(ink: np.ndarray, x0: int, x1: int, char_h: float, pitch: float,
                  min_lines: float = 1.9, min_aspect: float = 3.0, bridge: float = 0.8,
                  min_x_frac: float = 0.25) -> list[Comp]:
    """Tall thin strokes spanning at least ~2 line pitches, with text on their left.

    * a short vertical closing re-joins brackets broken by faint ink;
    * height is measured in line pitches, which rejects tall single letters ("C", "f");
    * candidates in the left quarter of the column are rejected: there, stacked
      capitals of consecutive lines ("Sl", "Co:") can join into tall thin shapes.
    """
    k = max(3, int(bridge * char_h))
    joined = cv2.morphologyEx(ink[:, x0:x1].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, 1), np.uint8))
    lab, st, _ = components(joined)
    out = []
    for x, y, w, h, _ in st:
        if x < min_x_frac * (x1 - x0):
            continue
        if h >= min_lines * pitch and h / max(w, 1) >= min_aspect:
            if x + w > (x1 - x0) - 2 * char_h:          # no room for a label: page edge etc.
                continue
            # Refine: keep only long original fragments (the closing may have glued
            # neighbouring letters onto the bracket's tips).
            sub = ink[y:y + h, x0 + x:x0 + x + w]
            sl, sst, _ = components(sub)
            long_ = sst[sst[:, 3] >= 1.0 * pitch]
            # A bracket stroke has (almost) blank paper directly to its right; a tall letter
            # glued on by the closing (e.g. the "S" of "Stato") is followed by the rest of
            # its word.
            gap, reach = 2, max(3, int(0.6 * char_h))
            def right_density(f):
                fx, fy, fw, fh = int(x0 + x + f[0]), int(y + f[1]), int(f[2]), int(f[3])
                return ink[fy:fy + fh, fx + fw + gap:fx + fw + gap + reach].mean()
            clean = np.array([f for f in long_ if right_density(f) < 0.06]) if len(long_) else long_
            if len(clean):
                long_ = clean
            if len(long_):
                rx1, ry1 = long_[:, 0].min(), long_[:, 1].min()
                rx2, ry2 = (long_[:, 0] + long_[:, 2]).max(), (long_[:, 1] + long_[:, 3]).max()
                x, y, w, h = x + rx1, y + ry1, rx2 - rx1, ry2 - ry1
                if h < min_lines * pitch * 0.9:
                    continue
            left_ink = ink[y:y + h, x0:x0 + x].sum()
            if left_ink > 5 * char_h ** 2:
                out.append(Comp(int(x0 + x), int(y), int(x0 + x + w), int(y + h)))
    return sorted(out, key=lambda c: c.y1)


def label_mask(ink: np.ndarray, br: Comp, x_end: int, char_h: float,
               core: float = 0.6, gap: float = 2.0) -> np.ndarray:
    """Text to the right of a bracket that forms its label.

    * components whose centre lies in the middle `core` part of the bracket's height
      (keeps continuation text level with the bracket's tip, e.g. "partirono disse", out);
    * of those, the cluster nearest the bracket: components are chained left-to-right
      while the horizontal gap stays under `gap` character heights.
    """
    lab, st, ids = components(ink[:, br.x2:x_end])
    m = (1 - core) / 2
    lo, hi = br.y1 + m * br.h, br.y2 - m * br.h
    cand = [(int(x), int(x + w), i) for i, (x, y, w, h, a) in zip(ids, st)
            if lo <= y + h / 2 <= hi]
    keep = np.zeros(len(ids) + 1, bool)
    if cand:
        cand.sort()
        reach = cand[0][0]
        if reach <= 6 * char_h:                          # must start near the bracket
            for x, xe, i in cand:
                if x - reach > gap * char_h:
                    break
                keep[i] = True
                reach = max(reach, xe)
    out = np.zeros_like(ink)
    out[:, br.x2:x_end] = keep[lab]
    return out


# ======================================================================
# 4. lines (deskew -> peaks -> seams -> polygons)
# ======================================================================
def estimate_skew(ink: np.ndarray, max_deg: float = 4, step: float = 0.25) -> float:
    h, w = ink.shape
    src = ink.astype(np.uint8)
    best, best_s = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + step, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        p = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_NEAREST).sum(1).astype(float)
        s = np.sum(np.diff(p) ** 2)
        if s > best_s:
            best, best_s = a, s
    return float(best)


def line_centres(ink: np.ndarray, char_h: float, prominence: float = 0.08) -> np.ndarray:
    prof = gaussian_filter1d(ink.sum(1).astype(float), 0.25 * char_h)
    if prof.max() == 0:
        return np.array([], int)
    peaks, _ = find_peaks(prof, distance=max(3, int(1.0 * char_h)),
                          prominence=prominence * prof.max())
    return peaks


def min_ink_seam(cost: np.ndarray, y_lo: int, y_hi: int, smooth: float = 0.6) -> np.ndarray:
    y_lo, y_hi = max(0, y_lo), min(cost.shape[0], max(y_hi, y_lo + 1))
    band = cost[y_lo:y_hi]
    H, W = band.shape
    acc = band[:, 0].copy()
    back = np.zeros((H, W), np.int8)
    for x in range(1, W):
        up = np.r_[np.inf, acc[:-1]] + smooth
        down = np.r_[acc[1:], np.inf] + smooth
        st = np.vstack([up, acc, down])
        idx = st.argmin(0)
        acc = st[idx, np.arange(H)] + band[:, x]
        back[:, x] = idx - 1
    path = np.empty(W, int)
    path[-1] = int(acc.argmin())
    for x in range(W - 1, 0, -1):
        path[x - 1] = path[x] + back[path[x], x]
    return path + y_lo


def column_lines(ink: np.ndarray, char_h: float, margin_frac: float = 0.15):
    """Return [(top_seam, bottom_seam)] in this (already deskewed) column image."""
    centres = line_centres(ink, char_h)
    if len(centres) == 0:
        return [], centres
    dist = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)
    cost = ink * 50.0 + 1.0 / (1.0 + dist)
    m = max(1, int(margin_frac * char_h))
    gap = int(np.median(np.diff(centres))) if len(centres) > 1 else int(2 * char_h)
    seams = [min_ink_seam(cost, centres[0] - gap, centres[0] - m)]
    for a, b in zip(centres, centres[1:]):
        seams.append(min_ink_seam(cost, a + m, b - m))
    seams.append(min_ink_seam(cost, centres[-1] + m, centres[-1] + gap))
    return seams, centres


def band_polygon(ink: np.ndarray, top: np.ndarray, bot: np.ndarray, pad: int):
    H, W = ink.shape
    rows = np.arange(H)[:, None]
    region = (rows >= top[None]) & (rows <= bot[None])
    cols = np.nonzero((ink & region).any(0))[0]
    if cols.size == 0:
        return None
    x0, x1 = max(0, cols.min() - pad), min(W, cols.max() + 1 + pad)
    xs = np.arange(x0, x1)
    return np.r_[np.c_[xs, top[xs]], np.c_[xs[::-1], bot[xs[::-1]]]].astype(float)


def rotate(img: np.ndarray, angle: float):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST), M


def to_page(pts: np.ndarray, M: np.ndarray, x0: int) -> np.ndarray:
    Minv = cv2.invertAffineTransform(M)
    p = np.c_[pts, np.ones(len(pts))] @ Minv.T
    p[:, 0] += x0
    return p


def polygon_mask(poly: np.ndarray, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def hull_polygon(mask: np.ndarray, char_h: float) -> np.ndarray | None:
    """Outline around a small region (labels): close gaps, take the outer contour."""
    k = max(3, int(0.6 * char_h))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, 2 * k), np.uint8))
    closed = cv2.dilate(closed, np.ones((3, 3), np.uint8), iterations=max(1, int(0.15 * char_h)))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    pts = np.vstack([c.reshape(-1, 2) for c in cnts])
    return cv2.convexHull(pts).reshape(-1, 2) if len(cnts) > 1 else cnts[0].reshape(-1, 2)


# ======================================================================
# 4b. tight, ink-following polygons (eScriptorium-style)
# ======================================================================
def band_index(seams: list[np.ndarray], ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Band number for each pixel: band i lies between seams[i] and seams[i+1]; -1 outside."""
    S = np.stack(seams)                                   # (n_bands + 1, W)
    idx = (S[:, xs] <= ys[None, :]).sum(0) - 1
    idx[(idx < 0) | (idx >= len(seams) - 1)] = -1
    return idx


def assign_components(ink: np.ndarray, seams: list[np.ndarray]) -> np.ndarray:
    """Owner map: every ink component goes, whole, to the band holding most of its pixels.

    Keeping components whole is what stops ascenders/descenders being cut at a line boundary.
    """
    owner = np.full(ink.shape, -1, np.int32)
    n, lab = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(lab)
    if ys.size == 0:
        return owner
    comp = lab[ys, xs]
    b = band_index(seams, ys, xs)
    valid = b >= 0
    votes = np.zeros((n, len(seams) - 1), np.int64)
    np.add.at(votes, (comp[valid], b[valid]), 1)
    best = np.where(votes.sum(1) > 0, votes.argmax(1), -1)
    owner[ys, xs] = best[comp]
    return owner


def split_segments(own: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Horizontal runs of a line's ink separated by more than `gap` blank columns."""
    occ = own.any(0)
    cols = np.nonzero(occ)[0]
    if cols.size == 0:
        return []
    breaks = np.nonzero(np.diff(cols) > gap)[0]
    starts = np.r_[cols[0], cols[breaks + 1]]
    ends = np.r_[cols[breaks], cols[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))


def envelope_profile(own: np.ndarray, foreign: np.ndarray, centre_y: float, char_h: float,
                     pad: int, bridge: float = 3.0):
    """Top/bottom outline (x0, top, bottom) that follows the line's own ink.

    * per column, top = highest own pixel, bottom = lowest;
    * a grey opening (top) / closing (bottom) with a ~3-character element bridges
      small notches between letters without ever cutting into the ink;
    * the outline is padded, then pulled back wherever it would touch another line's ink.
    """
    H, W = own.shape
    cols = np.nonzero(own.any(0))[0]
    if cols.size == 0:
        return None
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    sub = own[:, x0:x1]
    rows = np.arange(H)[:, None]
    has = sub.any(0)
    t0 = np.where(has, np.where(sub, rows, H).min(0), H).astype(float)
    b0 = np.where(has, np.where(sub, rows, -1).max(0), -1).astype(float)

    win = max(3, int(bridge * char_h))
    top = maximum_filter1d(minimum_filter1d(t0, win, mode="nearest"), win, mode="nearest")
    bot = minimum_filter1d(maximum_filter1d(b0, win, mode="nearest"), win, mode="nearest")
    top = np.where(top >= H, np.nan, top)
    bot = np.where(bot < 0, np.nan, bot)
    idx = np.arange(len(top))
    for v in (top, bot):
        ok = ~np.isnan(v)
        v[~ok] = np.interp(idx[~ok], idx[ok], v[ok])
    top, bot = top - pad, bot + pad

    fsub = foreign[:, x0:x1]
    for j in range(x1 - x0):
        lo, hi = int(max(0, top[j])), int(min(H, bot[j] + 1))
        fy = np.nonzero(fsub[lo:hi, j])[0]
        if fy.size:
            fy += lo
            above, below = fy[fy < centre_y], fy[fy >= centre_y]
            if above.size:
                top[j] = max(top[j], above.max() + 1)
            if below.size:
                bot[j] = min(bot[j], below.min() - 1)
    return x0, top, bot


def profile_to_polygon(x0: int, top: np.ndarray, bot: np.ndarray, W: int, H: int,
                       pad: int) -> np.ndarray:
    xs = np.arange(x0, x0 + len(top))
    xl, xr = max(0, x0 - pad), min(W - 1, xs[-1] + pad)
    upper = np.c_[np.r_[xl, xs, xr], np.r_[top[0], top, top[-1]]]
    lower = np.c_[np.r_[xr, xs[::-1], xl], np.r_[bot[-1], bot[::-1], bot[0]]]
    poly = np.r_[upper, lower]
    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
    return cv2.approxPolyDP(np.round(poly).astype(np.int32), 1.0, True).reshape(-1, 2).astype(float)


def resolve_overlaps(profiles: list) -> None:
    """Where the outline of line i dips below the outline of line i+1, meet halfway."""
    for pa in profiles:
        for pb in profiles:
            if pa["band"] >= pb["band"]:
                continue
            if True:
                lo = max(pa["x0"], pb["x0"])
                hi = min(pa["x0"] + len(pa["top"]), pb["x0"] + len(pb["top"]))
                if lo >= hi:
                    continue
                ia = slice(lo - pa["x0"], hi - pa["x0"])
                ib = slice(lo - pb["x0"], hi - pb["x0"])
                bot_a, top_b = pa["bot"][ia], pb["top"][ib]
                clash = bot_a > top_b - 1
                if clash.any():
                    mid = (bot_a + top_b) / 2
                    bot_a[clash] = np.floor(mid[clash])
                    top_b[clash] = np.floor(mid[clash]) + 1


def fit_baselines(owner: np.ndarray, n: int, char_h: float) -> list:
    """Robust straight baseline y = a*x + b for each band, from the bottoms of its ink.

    Descenders are ignored by refitting on points that are not far below the fit.
    """
    H, W = owner.shape
    rows = np.arange(H)[:, None]
    fits = []
    for i in range(n):
        own = owner == i
        cols = np.nonzero(own.any(0))[0]
        if cols.size < 2 * char_h:
            fits.append(None)
            continue
        bottoms = np.where(own[:, cols], rows, -1).max(0).astype(float)
        x, y = cols.astype(float), bottoms
        a, b = 0.0, float(np.median(y))
        for _ in range(3):
            keep = y <= a * x + b + 0.25 * char_h          # drop descenders
            if keep.sum() < 10:
                break
            a, b = np.polyfit(x[keep], y[keep], 1)
        fits.append((a, b))
    return fits


def reassign_by_baseline(ink: np.ndarray, owner: np.ndarray, char_h: float,
                         max_h: float = 3.0) -> None:
    """Give each small component to the line whose baseline is closest to its bottom.

    Handles prefixes written half a line high (e.g. "Sl" before "Cavalier"): most of their
    pixels lie in the band above, but they end just above their own line's baseline.
    Tall components (> max_h character heights) are left where they are.
    """
    n = owner.max() + 1
    if n < 2:
        return
    fits = fit_baselines(owner, n, char_h)
    lab, st, ids = components(ink)
    for cid, (x, y, w, h, a) in zip(ids, st):
        if h > max_h * char_h:
            continue
        m = lab[y:y + h, x:x + w] == cid
        cur = owner[y:y + h, x:x + w][m]
        cur = int(np.bincount(cur[cur >= 0]).argmax()) if (cur >= 0).any() else -1
        if cur < 0:
            continue
        cx, bottom = x + w / 2, y + h - 1
        best, best_d = cur, np.inf
        for j in (cur - 1, cur, cur + 1):
            if 0 <= j < n and fits[j] is not None:
                fa, fb = fits[j]
                d = bottom - (fa * cx + fb)
                # distance from the stroke's bottom to each candidate baseline
                d = -d if d < 0 else 0.6 * d   # descenders may end a little below
                if d < best_d:
                    best, best_d = j, d
        if best != cur:
            region = owner[y:y + h, x:x + w]
            region[m] = best


def merge_small_segments(owner: np.ndarray, gap: int, min_w: float) -> None:
    """Move tiny segments (e.g. a raised "Sl" prefix) into the neighbouring line they belong to.

    A segment narrower than `min_w` is reassigned to the band directly above or below whose
    ink, within `gap` columns of the segment, is vertically closest to the segment's ink.
    """
    n = owner.max() + 1
    H, W = owner.shape
    for i in range(n):
        own = owner == i
        for sx0, sx1 in split_segments(own, gap):
            if sx1 - sx0 >= min_w:
                continue
            ys = np.nonzero(own[:, sx0:sx1].any(1))[0]
            s_top, s_bot = ys.min(), ys.max()
            wx0, wx1 = max(0, sx0 - gap), min(W, sx1 + gap)
            best, best_d = None, np.inf
            for j in (i - 1, i + 1):
                if j < 0 or j >= n:
                    continue
                jy = np.nonzero((owner[:, wx0:wx1] == j).any(1))[0]
                if jy.size == 0:
                    continue
                d = max(0, jy.min() - s_bot) if j > i else max(0, s_top - jy.max())
                if d < best_d:
                    best, best_d = j, d
            if best is not None:
                seg = np.zeros_like(own)
                seg[:, sx0:sx1] = own[:, sx0:sx1]
                owner[seg] = best


def straight_seams(centres: np.ndarray, width: int, height: int, char_h: float) -> list[np.ndarray]:
    """Flat boundaries halfway between line centres (used for small regions such as labels)."""
    edges = [max(0, centres[0] - 1.5 * char_h)]
    edges += [(a + b) / 2 for a, b in zip(centres, centres[1:])]
    edges += [min(height, centres[-1] + 1.5 * char_h)]
    return [np.full(width, int(e)) for e in edges]


def lines_from_ink(ink: np.ndarray, seams: list[np.ndarray], centres: np.ndarray,
                   char_h: float, pad: int, min_ink: int):
    """Tight polygons for every (band, horizontal segment) of `ink`, in ink's own frame."""
    H, W = ink.shape
    owner = assign_components(ink, seams)
    reassign_by_baseline(ink, owner, char_h)
    gap = max(3, int(2.5 * char_h))
    merge_small_segments(owner, gap, min_w=3 * char_h)
    profiles = []
    for i, c in enumerate(centres):
        own_all = owner == i
        foreign = (owner >= 0) & ~own_all
        for sx0, sx1 in split_segments(own_all, gap):
            own = np.zeros_like(own_all)
            own[:, sx0:sx1] = own_all[:, sx0:sx1]
            if own.sum() < min_ink:
                continue
            prof = envelope_profile(own, foreign, float(np.median(np.nonzero(own)[0])), char_h, pad)
            if prof is not None:
                x0, top, bot = prof
                profiles.append({"band": i, "x0": x0, "top": top, "bot": bot, "own": own})
    resolve_overlaps(profiles)
    return [(p["band"], profile_to_polygon(p["x0"], p["top"], p["bot"], W, H, pad), p["own"])
            for p in profiles]


# ======================================================================
# 5. page
# ======================================================================
def segment_page(rgb: np.ndarray, align_tol: float = 0.35) -> dict:
    H, W = rgb.shape[:2]
    raw = ink_mask(rgb)
    char_h = estimate_char_height(raw)
    paper = cv2.erode(page_mask(rgb).astype(np.uint8), np.ones((int(char_h), int(char_h)), np.uint8))
    ink = clean_page(raw & paper.astype(bool), char_h)
    columns = find_columns(ink, char_h)
    pad = max(2, int(0.3 * char_h))
    min_line_ink = int(0.5 * char_h ** 2)

    lines: list[Line] = []
    brackets: list[Bracket] = []
    labels: list[Label] = []

    for ci, (x0, x1) in enumerate(columns):
        col_ink = np.zeros_like(ink)
        col_ink[:, x0:x1] = ink[:, x0:x1]
        col_ink = drop_edge_verticals(col_ink, x0, char_h)

        # brackets + labels are removed from the text before line finding
        cs = line_centres(col_ink[:, x0:x1], char_h)
        pitch = float(np.median(np.diff(cs))) if len(cs) > 2 else 2.5 * char_h
        col_brackets = find_brackets(col_ink, x0, x1, char_h, pitch)
        text = col_ink.copy()
        for br in col_brackets:
            b = Bracket(f"c{ci}_b{len(brackets) + 1:02d}", ci, br)
            # remove only the bracket's own stroke, not everything in its bounding box
            bl, _, _ = components(col_ink[br.y1:br.y2, br.x1:br.x2])
            if bl.max() > 0:
                big = np.bincount(bl.ravel())[1:].argmax() + 1
                text[br.y1:br.y2, br.x1:br.x2] &= ~(bl == big)
            lm = label_mask(col_ink, br, x1, char_h)
            if lm.any():
                ys, xs = np.nonzero(lm)
                ly0, ly1, lx0, lx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                sub = lm[ly0:ly1, lx0:lx1]
                cs_l = line_centres(sub, char_h)
                if len(cs_l) == 0:
                    cs_l = np.array([sub.shape[0] // 2])
                seams_l = straight_seams(cs_l, sub.shape[1], sub.shape[0], char_h)
                polys = [p + [lx0, ly0] for _, p, _ in
                         lines_from_ink(sub, seams_l, cs_l, char_h, pad, min_line_ink // 4)]
                if polys:
                    lab = Label(f"c{ci}_lab{len(labels) + 1:02d}", ci,
                                [np.round(p).astype(np.int32) for p in polys], b.id)
                    labels.append(lab)
                    b.label_id = lab.id
                text &= ~lm
            brackets.append(b)

        # deskew the column, find line bands, map polygons back to the page
        sub = text[:, x0:x1]
        angle = estimate_skew(sub)
        rot, M = rotate(sub.astype(np.uint8), angle)
        rot = rot.astype(bool)
        seams, centres = column_lines(rot, char_h)
        if not seams:
            continue
        for band, poly_r, own_r in lines_from_ink(rot, seams, centres, char_h, pad, min_line_ink):
            poly = to_page(poly_r, M, x0)
            mask = polygon_mask(poly, (H, W))
            own = text & mask
            if own.sum() < min_line_ink // 2:
                continue
            ys, xs = np.nonzero(own)
            lines.append(Line(f"c{ci}_l{sum(l.column == ci for l in lines) + 1:03d}", ci,
                              np.round(poly).astype(np.int32), int(np.median(ys)),
                              int(np.percentile(xs, 2)), int(np.percentile(xs, 98)), mask))

    # bracket groups: lines in the same column, left of the bracket, centred within its span
    for b in brackets:
        col_lines = [l for l in lines if l.column == b.column]
        cs = sorted(l.centre_y for l in col_lines)
        pitch = float(np.median(np.diff(cs))) if len(cs) > 1 else 2 * char_h
        lo, hi = b.box.y1 - 0.3 * pitch, b.box.y2 + 0.3 * pitch
        # Lines must start near the column's left margin: drops interlinear dates such as
        # "23 Mag." that sit inside the bracket's height but begin far to the right.
        margin = float(np.percentile([l.x_start for l in col_lines], 10))
        max_start = margin + align_tol * (b.box.x1 - margin)
        b.members = [l.id for l in col_lines
                     if lo <= l.centre_y <= hi and l.x_start <= max_start]

    return {"char_h": char_h, "ink": ink, "columns": columns,
            "lines": lines, "brackets": brackets, "labels": labels}


# ======================================================================
# 6. outputs
# ======================================================================
def draw(rgb: np.ndarray, res: dict, alpha: float = 0.22, show_brackets: bool = True) -> np.ndarray:
    """eScriptorium-like overlay: light fill + thin outline per line; groups/labels emphasised."""
    fill = rgb.copy()
    grouped = {m for b in res["brackets"] for m in b.members}
    label_polys = [p for lab in res["labels"] for p in lab.polygons]
    for l in res["lines"]:
        cv2.fillPoly(fill, [l.polygon], (150, 175, 225))
    for p in label_polys:
        cv2.fillPoly(fill, [p], (150, 215, 170))
    vis = cv2.addWeighted(fill, alpha, rgb, 1 - alpha, 0)
    t = max(1, int(round(res["char_h"] / 14)))
    for l in res["lines"]:
        grp = l.id in grouped
        cv2.polylines(vis, [l.polygon], True, (40, 70, 170) if grp else (95, 120, 185),
                      t + 1 if grp else t, cv2.LINE_AA)
    for p in label_polys:
        cv2.polylines(vis, [p], True, (0, 130, 60), t + 1, cv2.LINE_AA)
    if show_brackets:
        for b in res["brackets"]:
            c = b.box
            cv2.rectangle(vis, (c.x1, c.y1), (c.x2, c.y2), (200, 40, 40), t)
    return vis


def crop_masked(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    m = mask[y0:y1, x0:x1]
    out[m] = rgb[y0:y1, x0:x1][m]
    return out


def to_json(res: dict, stem: str) -> dict:
    simp = lambda p: cv2.approxPolyDP(p.astype(np.int32), 1.5, True).reshape(-1, 2).tolist()
    return {
        "image": stem,
        "char_height": float(res["char_h"]),
        "columns": [[int(a), int(b)] for a, b in res["columns"]],
        "lines": [{"id": l.id, "column": l.column, "centre_y": l.centre_y,
                   "polygon": simp(l.polygon)} for l in res["lines"]],
        "brackets": [{"id": b.id, "column": b.column,
                      "box": [b.box.x1, b.box.y1, b.box.x2, b.box.y2],
                      "lines": b.members, "label": b.label_id} for b in res["brackets"]],
        "labels": [{"id": lab.id, "bracket": lab.bracket_id,
                    "polygons": [simp(p) for p in lab.polygons]} for lab in res["labels"]],
    }


def resolve_image(image: str | Path) -> Path:
    p = Path(image)
    for cand in (p, SCRIPT_DIR / p):
        if cand.is_file():
            return cand
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    matches = [m for m in sorted(SCRIPT_DIR.glob(f"{p.stem}.*")) if m.suffix.lower() in exts]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"'{image}' not found (looked in {Path.cwd()} and {SCRIPT_DIR})")


def run(image: str | Path = DEFAULT_IMAGE, out_dir: str | Path = DEFAULT_OUT,
        crops: bool = False, verbose: bool = True) -> dict:
    path = resolve_image(image)
    rgb = np.array(Image.open(path).convert("RGB"))
    res = segment_page(rgb)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    Image.fromarray(draw(rgb, res)).save(out_dir / f"{stem}_polygons.png")
    (out_dir / f"{stem}_regions.json").write_text(json.dumps(to_json(res, stem), indent=1))
    if crops:
        for l in res["lines"]:
            Image.fromarray(crop_masked(rgb, l.mask)).save(out_dir / f"{stem}_{l.id}.png")
    if verbose:
        print(f"[{stem}] {rgb.shape[1]}x{rgb.shape[0]} | char height {res['char_h']:.0f}px "
              f"| columns {res['columns']}")
        print(f"[{stem}] lines {len(res['lines'])} | brackets {len(res['brackets'])} "
              f"| labels {len(res['labels'])}")
        for b in res["brackets"]:
            print(f"  {b.id}: {len(b.members)} lines -> label {b.label_id}")
        print(f"[{stem}] wrote {out_dir}")
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Full-page line polygons with bracket groups.")
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--crops", action="store_true", help="also save one masked crop per line")
    a = ap.parse_args()
    run(a.image, a.out, a.crops)
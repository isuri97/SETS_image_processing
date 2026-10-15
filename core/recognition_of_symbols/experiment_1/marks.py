"""Detectors for the non-brace marks: rules, slashes, dashes, circles, fillers.

Each detector is a pixel-level test on the binary ink mask and returns `Mark`
objects carrying a full-page mask, so marks can be erased from, or isolated
on, the page image. Detectors run in a fixed order and every one works on the
ink left over by the previous ones:

    rules -> slashes -> dashes -> circles -> fillers

All size thresholds are in line spacings (LS) or pen stroke widths (SW).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from preprocess import Page
from strokes import path_length

Box = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class Mark:
    """One detected mark."""

    kind: str  # "rule", "slash", "dash", "circle" or "filler"
    bbox: Box
    mask: np.ndarray  # full-page boolean image of its pixels
    attrs: dict = field(default_factory=dict)


def _bbox(mask: np.ndarray) -> Box:
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _components(mask: np.ndarray):
    """Yield (component mask crop, x, y, w, h, area) for each 8-connected component."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        yield lab[y : y + h, x : x + w] == i, int(x), int(y), int(w), int(h), int(area)


def _solid(ink: np.ndarray, sw: float) -> np.ndarray:
    """Ink without full stops and specks (components under 3 SW^2)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= 3 * sw * sw
    big[0] = False
    return big[lab]


def _paste(shape: tuple[int, int], crop: np.ndarray, x: int, y: int) -> np.ndarray:
    full = np.zeros(shape, bool)
    full[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
    return full


# --------------------------------------------------------------------------
# rules
# --------------------------------------------------------------------------
def detect_rules(ink: np.ndarray, ls: int, sw: float, min_len: float = 5.0) -> list[Mark]:
    """Long horizontal rules (e.g. under a column heading).

    A horizontal path opening keeps pixels that lie on a near-horizontal ink
    path of at least `min_len` LS; no word is that long. Descenders that touch
    the rule are trimmed by keeping only pixels close to a smooth fit of it.
    """
    long_px = path_length(ink, "horizontal") >= min_len * ls
    marks = []
    for comp, x, y, w, h, _ in _components(long_px):
        if w < min_len * ls:
            continue
        cols = np.where(comp.any(axis=0))[0]
        mid = np.array([np.median(np.where(comp[:, c])[0]) for c in cols])
        fit = np.polyval(np.polyfit(cols, mid, 2), np.arange(w))
        rows = np.arange(h)[:, None]
        near = np.abs(rows - fit[None, :]) <= 1.5 * sw + 1
        mask = _paste(ink.shape, comp & near, x, y)
        # widen to the full stroke thickness around the fitted centre line
        band = _paste(ink.shape, np.ones_like(comp) & near, x, y) & ink
        mask |= band
        marks.append(Mark("rule", _bbox(mask), mask, {"length_ls": round(w / ls, 1)}))
    return marks


# --------------------------------------------------------------------------
# slashes
# --------------------------------------------------------------------------
def _shear(mask: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Shift row y right by round(k*y) so strokes with dx/dy = -k become vertical."""
    h, w = mask.shape
    shift = np.round(k * np.arange(h)).astype(int)
    out = np.zeros((h, w + int(shift.max()) + 1), bool)
    out[np.arange(h)[:, None], np.arange(w)[None, :] + shift[:, None]] = mask
    return out, shift


def _unshear(arr: np.ndarray, shift: np.ndarray, w: int) -> np.ndarray:
    return arr[np.arange(arr.shape[0])[:, None], np.arange(w)[None, :] + shift[:, None]]


SLASH_ANGLES = (28.0, 68.0)  # degrees above horizontal accepted for a slash
MAX_RMS = 0.65  # largest perpendicular scatter of a straight stroke, in stroke widths


@dataclass
class _Stroke:
    """A straight stroke, stored as a crop of the page plus its offset."""

    crop: np.ndarray
    x: int
    y: int
    centre: np.ndarray  # (x, y) in page coordinates
    direction: np.ndarray  # unit vector pointing up-right
    length: float
    angle: float  # degrees above horizontal
    rms: float  # perpendicular scatter in pixels
    iso: tuple[float, float] = (0.0, 0.0)

    def stamp(self, target: np.ndarray) -> None:
        """OR the stroke into a full-page mask, in place."""
        h, w = self.crop.shape
        target[self.y : self.y + h, self.x : self.x + w] |= self.crop


def _fit_stroke(crop: np.ndarray, x: int, y: int) -> _Stroke:
    ys, xs = np.where(crop)
    pts = np.stack([xs + x, ys + y], 1).astype(np.float64)
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    d = vt[0] if vt[0][1] < 0 else -vt[0]  # image y grows downwards: point the vector up
    along = (pts - centre) @ d
    perp = (pts - centre) @ np.array([-d[1], d[0]])
    angle = float(np.degrees(np.arctan2(-d[1], d[0])))
    return _Stroke(crop, x, y, centre, d, float(np.ptp(along)), angle, float(perp.std()))


def _split_parallel(crop: np.ndarray, x: int, y: int, sw: float, max_rms: float) -> list[_Stroke]:
    """Fit one stroke, or two if the crop holds a pair of strokes that touch."""
    st = _fit_stroke(crop, x, y)
    if st.rms <= max_rms:
        return [st]
    ys, xs = np.where(crop)
    side = (np.stack([xs + x, ys + y], 1) - st.centre) @ np.array([-st.direction[1], st.direction[0]]) < 0
    halves = []
    for sel in (side, ~side):
        if sel.sum() > 4 * sw:
            half = np.zeros_like(crop)
            half[ys[sel], xs[sel]] = True
            halves.append(_fit_stroke(half, x, y))
    return halves


def _side_isolation(st: _Stroke, other: np.ndarray, near: float, far: float) -> tuple[float, float]:
    """Share of the stroke length with no ink beside it, for each side.

    Samples points along the stroke axis and looks perpendicular to it, at
    distances `near`..`far` pixels, in the mask `other`.
    """
    h, w = other.shape
    n = np.array([-st.direction[1], st.direction[0]])
    ts = np.linspace(-0.5, 0.5, max(8, int(st.length / 3))) * st.length
    ds = np.arange(near, far + 1, 1.0)
    out = []
    for sign in (-1, 1):
        px = st.centre[0] + ts[:, None] * st.direction[0] + sign * ds[None, :] * n[0]
        py = st.centre[1] + ts[:, None] * st.direction[1] + sign * ds[None, :] * n[1]
        xi = np.clip(np.round(px).astype(int), 0, w - 1)
        yi = np.clip(np.round(py).astype(int), 0, h - 1)
        out.append(float(1.0 - other[yi, xi].any(axis=1).mean()))
    return out[0], out[1]


def _bridge(mask: np.ndarray, ls: int) -> np.ndarray:
    """Close gaps of up to 0.15 LS along the slash directions (35, 45, 55 degrees)."""
    n = max(5, int(0.15 * ls)) | 1
    out = mask.copy()
    for deg in (35, 45, 55):
        k = np.zeros((n, n), np.uint8)
        c, t = n // 2, np.radians(deg)
        cv2.line(k, (int(round(c - c * np.cos(t))), int(round(c + c * np.sin(t)))),
                 (int(round(c + c * np.cos(t))), int(round(c - c * np.sin(t)))), 1, 1)
        out |= cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return out


def slash_candidates(ink: np.ndarray, source: np.ndarray, ls: int, sw: float,
                     min_len: float = 0.9) -> tuple[list[_Stroke], float]:
    """Straight slanted strokes of at least `min_len` LS, and the writing slant.

    `source` is the mask the strokes are traced in (the sensitive mask, so
    pale dry-pen strokes are not broken up); `ink` is the clean mask used to
    judge whether a stroke stands in white space. The writing slant is the
    median angle of the short straight strokes, which are nearly all letter
    stems.
    """
    bridged = _bridge(source, ls)
    sheared, shift = _shear(bridged, 1.0)  # a 45-degree slash becomes vertical
    # the path opening counts rows, so the threshold is the height of the flattest slash
    min_rows = min_len * ls * np.sin(np.radians(SLASH_ANGLES[0]))
    long_px = _unshear(path_length(sheared) >= min_rows, shift, source.shape[1]) & bridged
    pieces, stems = [], []
    for comp, x, y, w, h, area in _components(long_px):
        if max(w, h) < 0.3 * ls:
            continue
        for st in _split_parallel(comp, x, y, sw, MAX_RMS * sw):
            if st.rms > MAX_RMS * sw or st.length < 0.4 * ls:
                continue
            if st.length <= 1.2 * ls and 40 <= st.angle <= 88:
                stems.append(st.angle)
            if SLASH_ANGLES[0] <= st.angle <= SLASH_ANGLES[1]:
                pieces.append(st)
    slant = float(np.median(stems)) if len(stems) >= 20 else 65.0
    # isolation is judged against everything that is not itself a straight slanted stroke,
    # so the second stroke of a double slash does not count as a neighbour
    own = np.zeros_like(ink)
    for st in pieces:
        st.stamp(own)
    other = ink & ~cv2.dilate(own.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    strokes = [st for st in pieces if st.length >= min_len * ls]
    for st in strokes:
        st.iso = _side_isolation(st, other, near=1.5 * sw, far=0.25 * ls)
    return strokes, slant


def detect_slashes(ink: np.ndarray, faint: np.ndarray, ls: int, sw: float, min_iso: float = 0.6,
                   long_len: float = 1.6, flatter_by: float = 15.0, lone_len: float = 1.3,
                   min_support: float = 0.3) -> list[Mark]:
    """Slash signs: parallel slanted strokes standing in white space.

    Letter stems are slanted and straight too. Three tests separate them:

      isolation  a slash is free of other ink on both sides for at least
                 `min_iso` of its length; a stem has its letter beside it;
      support    at least `min_support` of the stroke is in the clean ink
                 mask, so marks seen only in the sensitive mask are dropped;
      shape      a sign is either longer than any letter stem (`long_len`
                 LS) or clearly flatter than the writing slant (`flatter_by`
                 degrees), which is measured on the page. A stroke without
                 a partner must also reach `lone_len` LS.

    Parallel strokes that overlap are reported as one sign (the double slash
    before a date, or the closing strokes that fill the blank foot of a column).
    The shape test is applied to the sign as a whole.
    """
    cands, slant = slash_candidates(ink, faint, ls, sw)
    strokes = []
    for s in cands:
        if min(s.iso) < min_iso:
            continue
        h, w = s.crop.shape
        if (s.crop & ink[s.y : s.y + h, s.x : s.x + w]).sum() < min_support * s.crop.sum():
            continue
        strokes.append(s)
    # ---- group parallel neighbours
    parent = list(range(len(strokes)))

    def find(a: int) -> int:
        while parent[a] != a:
            a = parent[a]
        return a

    for i, a in enumerate(strokes):
        for j in range(i + 1, len(strokes)):
            b = strokes[j]
            if abs(a.angle - b.angle) > 15:
                continue
            delta = b.centre - a.centre
            along = abs(float(delta @ a.direction))
            perp = abs(float(delta @ np.array([-a.direction[1], a.direction[0]])))
            if perp <= 0.6 * ls and along <= 0.5 * (a.length + b.length) - 0.3 * min(a.length, b.length):
                parent[find(j)] = find(i)
    groups: dict[int, list[_Stroke]] = {}
    for i, s in enumerate(strokes):
        groups.setdefault(find(i), []).append(s)


    def is_slash(members: list[_Stroke]) -> bool:
        longest = max(s.length for s in members)
        if len(members) == 1 and longest < lone_len * ls:
            return False
        return longest >= long_len * ls or np.mean([s.angle for s in members]) <= slant - flatter_by

    groups = {k: v for k, v in groups.items() if is_slash(v)}
    all_slashes = np.zeros_like(ink)
    for members in groups.values():
        for s in members:
            s.stamp(all_slashes)
    not_slash = _solid(ink, sw) & ~cv2.dilate(all_slashes.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    marks = []
    for members in groups.values():
        mask = np.zeros_like(ink)
        for s in members:
            s.stamp(mask)
        x, y, w, h = _bbox(mask)
        beside = not_slash[y + int(0.2 * h) : y + int(0.8 * h), x + w : x + w + int(1.2 * ls)]
        marks.append(Mark("slash", (x, y, w, h), mask, {
            "strokes": len(members),
            "length_ls": round(max(s.length for s in members) / ls, 2),
            "angle": round(float(np.mean([s.angle for s in members]))),
            "writing_slant": round(slant),
            "_text_beside": bool(beside.sum() >= 8 * sw * sw),
        }))
    # Role (a guess from layout only). Closing strokes come several to a row and
    # fill blank paper; an entry separator is alone on its row with a date or
    # "d.to" written straight after it.
    for m in marks:
        x, y, w, h = m.bbox
        in_row = [o for o in marks if o is not m and abs(o.bbox[0] - x) < 8 * ls
                  and min(y + h, o.bbox[1] + o.bbox[3]) - max(y, o.bbox[1]) > 0.3 * min(h, o.bbox[3])]
        m.attrs["role"] = "separator" if m.attrs.pop("_text_beside") and not in_row else "closing"
    return marks


# --------------------------------------------------------------------------
# dashes
# --------------------------------------------------------------------------
def detect_dashes(ink: np.ndarray, contrast: np.ndarray, ls: int, sw: float) -> list[Mark]:
    """Free-standing short horizontal strokes.

    A dash is a whole ink component that is flat, straight and thin, with
    nothing directly below it and a clear gap on either side. The "below"
    test separates it from abbreviation strokes and letter bars, which always
    sit on top of a letter. It must also be about as dark as the writing.
    """
    ink_level = float(np.median(contrast[ink]))
    marks = []
    for comp, x, y, w, h, area in _components(ink):
        if not (0.22 * ls <= w <= 2.2 * ls and h <= 0.3 * ls and w >= 1.7 * h):
            continue
        cols = np.where(comp.any(axis=0))[0]
        top = np.array([np.argmax(comp[:, c]) for c in cols])
        bot = np.array([comp.shape[0] - 1 - np.argmax(comp[::-1, c]) for c in cols])
        mid, thick = (top + bot) / 2, bot - top + 1
        slope, icpt = np.polyfit(cols, mid, 1)
        wavy = float(np.sqrt(np.mean((slope * cols + icpt - mid) ** 2)))
        # one pen stroke: straight, and nowhere thicker than about two stroke widths
        if abs(slope) > 0.35 or wavy > 0.5 * sw or thick.max() > 2.2 * sw:
            continue
        full = _paste(ink.shape, comp, x, y)
        other = ink & ~full
        reach = int(0.3 * ls)
        below = other[y + h : y + h + reach, x : x + w].any(axis=0).mean()
        above = other[max(0, y - reach) : y, x : x + w].any(axis=0).mean()
        if below > 0.2:
            continue
        # a dash stands clear of its neighbours; a broken-off piece of a letter does not
        side = int(0.12 * ls)
        if other[y : y + h, max(0, x - side) : x].any() or other[y : y + h, x + w : x + w + side].any():
            continue
        darkness = float(np.median(contrast[full])) / max(ink_level, 1e-6)
        if darkness < 0.6:
            continue  # show-through or a stain, paler than the writing
        marks.append(Mark("dash", (x, y, w, h), full, {
            "length_ls": round(w / ls, 2),
            "under_flourish": bool(above > 0.5),
            "darkness": round(darkness, 2),
        }))
    return marks


# --------------------------------------------------------------------------
# circles
# --------------------------------------------------------------------------
def _ellipse_fit(comp: np.ndarray) -> dict | None:
    """Fit an ellipse to the pixels of `comp` and measure how ring-like it is.

    Returns the semi-axes in pixels and three scores, or None if no sensible
    ellipse fits:

      error     mean distance of the pixels from the ellipse outline, in pixels;
      coverage  share of the outline that has pixels on it (36 sectors of 10
                degrees), so 1.0 is a full ring and 0.75 a ring with a quarter
                missing;
      centre    share of the pixels in the inner half of the ellipse; a ring
                has none, a filled blob has about a quarter.
    """
    ys, xs = np.where(comp)
    if len(xs) < 5:
        return None
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(np.stack([xs, ys], 1).astype(np.float32))
    a, b = d1 / 2, d2 / 2
    if not (np.isfinite(a) and np.isfinite(b)) or min(a, b) < 1:
        return None
    t = np.radians(angle)
    dx, dy = xs - cx, ys - cy
    u = (dx * np.cos(t) + dy * np.sin(t)) / a  # coordinates in which the ellipse is the unit circle
    v = (-dx * np.sin(t) + dy * np.cos(t)) / b
    r = np.hypot(u, v)
    sectors = np.unique((np.degrees(np.arctan2(v, u)) % 360 // 10).astype(int))
    return {
        "axes": (float(min(a, b)), float(max(a, b))),
        "error": float(np.abs(r - 1).mean() * (a + b) / 2),
        "coverage": len(sectors) / 36,
        "centre": float((r < 0.5).mean()),
    }


def detect_circles(bgr: np.ndarray, ink: np.ndarray, contrast: np.ndarray, ls: int, sw: float,
                   min_coverage: float = 0.8, max_error: float = 0.5, max_centre: float = 0.05,
                   min_darkness: float = 0.45) -> list[Mark]:
    """Small rings that stand apart from the writing, closed or not.

    A ring drawn in pencil is often open in the ink mask, where the pencil
    skipped or the stroke is too pale to binarise, so the test does not ask
    for an enclosed hole. Instead an ellipse is fitted to each small ink
    component (see `_ellipse_fit`), and the component is a circle if

      - its pixels lie on the ellipse (error at most `max_error` SW),
      - they cover at least `min_coverage` of its outline,
      - the middle of the ellipse is empty (at most `max_centre` of the pixels),
      - it has white space all around it (0.3 LS), which removes the letter
        "o" and the superscript "o" of abbreviations,
      - it is at least `min_darkness` times as dark as the writing, which
        removes round letters showing through from the back of the sheet.

    Each circle is tagged "pencil" or "ink" from the colour saturation of its
    stroke; in a shadowed corner of the photograph this tag is unreliable.
    """
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    ink_sat = float(np.median(sat[ink]))
    paper_sat = float(np.median(sat[~ink]))
    ink_level = float(np.median(contrast[ink]))
    marks = []
    for comp, x, y, w, h, area in _components(ink):
        if not (0.15 * ls <= w <= 0.8 * ls and 0.15 * ls <= h <= 0.8 * ls and 0.6 <= w / h <= 1.6):
            continue
        r = int(0.3 * ls)
        around = ink[max(0, y - r) : y + h + r, max(0, x - r) : x + w + r].sum() - area
        if around > 2 * sw * sw:
            continue
        fit = _ellipse_fit(comp)
        if fit is None or fit["axes"][0] < 0.5 * fit["axes"][1]:
            continue
        if fit["error"] > max_error * sw or fit["coverage"] < min_coverage or fit["centre"] > max_centre:
            continue
        full = _paste(ink.shape, comp, x, y)
        darkness = float(np.median(contrast[full])) / max(ink_level, 1e-6)
        if darkness < min_darkness:
            continue
        s = float(np.median(sat[full]))
        medium = "pencil" if s < 0.5 * (ink_sat + paper_sat) else "ink"
        marks.append(Mark("circle", (x, y, w, h), full, {
            "medium": medium, "saturation": round(s, 1), "diameter_ls": round((w + h) / 2 / ls, 2),
            "coverage": round(fit["coverage"], 2), "fit_error_sw": round(fit["error"] / sw, 2),
            "darkness": round(darkness, 2),
        }))
    return marks


# --------------------------------------------------------------------------
# fillers
# --------------------------------------------------------------------------
def detect_fillers(ink: np.ndarray, contrast: np.ndarray, taken: np.ndarray, gutter_x: int,
                   ls: int, sw: float, min_gap: float = 0.35) -> list[Mark]:
    """Line fillers: the low squiggles ("re re", "ee") that pad a line.

    A blob is a filler if it is no taller than the x-height, has nothing after
    it in its column except other such blobs, up to the next brace or mark
    (`taken`), and the run it belongs to is separated from the text on its
    left by a gap wider than a word space (`min_gap` LS).

    This is the weakest of the detectors: a short final word such as "suo"
    differs from a filler only by the width of the gap in front of it.
    """
    h, w = ink.shape
    free = ink & ~taken
    solid = _solid(ink, sw)  # full stops and specks neither end a gap nor count as following text
    ink_level = float(np.median(contrast[ink]))
    # join the letters of one squiggle ("r" + "e") without bridging word gaps
    k = np.ones((3, max(3, int(0.2 * ls)) | 1), np.uint8)
    blobs = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    cands = []
    owner = np.zeros((h, w), np.int32)  # index + 1 of the candidate owning each pixel
    for comp, x, y, bw, bh, area in _components(blobs):
        if not (bh <= 0.48 * ls and 0.3 * ls <= bw <= 3.0 * ls and area >= 6 * sw * sw):
            continue
        mask = _paste(ink.shape, comp, x, y) & free
        if not mask.any() or np.median(contrast[mask]) < 0.6 * ink_level:
            continue  # show-through, paler than the writing
        cands.append((x, y, bw, bh, mask))
        owner[mask] = len(cands)

    def describe(i: int) -> tuple[bool, float, int]:
        """(nothing but candidates follows, gap to the left in LS, candidate on the left or -1)."""
        x, y, bw, bh, mask = cands[i]
        col_lo, col_hi = (0, gutter_x) if x + bw / 2 < gutter_x else (gutter_x, w)
        band = slice(max(0, y - int(0.1 * ls)), y + bh + int(0.1 * ls))
        stop_cols = np.where(taken[band, x + bw : col_hi].any(axis=0))[0]
        limit = x + bw + int(stop_cols.min()) if len(stop_cols) else col_hi
        after = solid[band, x + bw : limit] & (owner[band, x + bw : limit] == 0)
        left = (solid & ~mask)[band, col_lo:x]
        cols = np.where(left.any(axis=0))[0]
        if len(cols) == 0:
            return not after.any(), np.inf, -1
        nearest = col_lo + int(cols.max())
        ids = owner[band, nearest][solid[band, nearest]]
        return not after.any(), (x - nearest - 1) / ls, int(ids.max()) - 1 if len(ids) else -1

    info = [describe(i) for i in range(len(cands))]

    def is_filler(i: int, depth: int = 0) -> bool:
        clear, gap, left = info[i]
        if not clear or gap > 3.0 or depth > 8:
            return False  # text follows it, or it starts a line of its own
        if gap >= min_gap:
            return True
        return left >= 0 and is_filler(left, depth + 1)  # part of a run of fillers

    accepted = {i for i in range(len(cands)) if is_filler(i)}
    # Fillers repeat ("re re"). A blob of the same size directly left of an
    # accepted filler belongs to the run even if it sits close to the name.
    grew = True
    while grew:
        grew = False
        for i in list(accepted):
            left = info[i][2]
            if left < 0 or left in accepted or not info[left][0]:
                continue
            (_, _, w0, h0, _), (_, _, w1, h1, _) = cands[i], cands[left]
            if abs(w1 - w0) <= 0.3 * w0 and abs(h1 - h0) <= 0.3 * h0:
                accepted.add(left)
                grew = True
    marks = []
    for i in sorted(accepted):
        x, y, bw, bh, mask = cands[i]
        marks.append(Mark("filler", (x, y, bw, bh), mask,
                          {"gap_ls": round(float(info[i][1]), 2), "width_ls": round(bw / ls, 2)}))
    return marks


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def detect_marks(page: Page, brace_mask: np.ndarray, gutter_x: int, ls: int, sw: float) -> list[Mark]:
    """Run all detectors in order on the ink not already claimed by braces."""
    taken = cv2.dilate(brace_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    marks: list[Mark] = []

    def claim(found: list[Mark]) -> None:
        nonlocal taken
        marks.extend(found)
        for m in found:
            taken |= m.mask

    claim(detect_rules(page.ink & ~taken, ls, sw))
    claim(detect_slashes(page.ink & ~taken, page.faint & ~taken, ls, sw))
    claim(detect_dashes(page.ink & ~taken, page.contrast, ls, sw))
    claim(detect_circles(page.bgr, page.ink & ~taken, page.contrast, ls, sw))
    claim(detect_fillers(page.ink, page.contrast, taken, gutter_x, ls, sw))
    return marks
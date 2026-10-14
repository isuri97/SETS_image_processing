"""Brace-to-text association.

Everything here is geometric: no recognition is involved, so the output is
boxes, not strings. For each brace:

  members  text lines to its left that the brace spans (the grouped items);
  note     the short text block to its right, at the height of the cusp.

Lines are found *locally*, in a band next to the brace, and then followed
sideways window by window. A page-wide line model fails on these photographs
because the lines climb by up to half a line spacing across a column.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from braces import Brace

Box = tuple[int, int, int, int]  # x, y, w, h

CORE = 0.25  # half-height of the x-height zone, in line spacings, used as a fallback


@dataclass
class TextLine:
    """One text line next to a brace."""

    centre: int  # row of the line centre at the brace
    box: Box
    coverage: float  # share of the line's x-height zone that the brace spans (0..1)


@dataclass
class BraceGroup:
    """A brace with the lines it groups (left) and the note it points to (right)."""

    brace: Brace
    column: int
    members: list[TextLine] = field(default_factory=list)
    borderline: list[TextLine] = field(default_factory=list)  # touched, not spanned
    note: list[TextLine] = field(default_factory=list)

    @property
    def note_box(self) -> Box | None:
        return union([n.box for n in self.note])


def union(boxes: list[Box]) -> Box | None:
    """Smallest box containing all `boxes`."""
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def gutter(ink: np.ndarray, ls: int) -> int:
    """x of the gap between the two columns (emptiest band near the centre)."""
    w = ink.shape[1]
    prof = gaussian_filter1d(ink.sum(axis=0).astype(float), 0.3 * ls)
    lo, hi = int(0.35 * w), int(0.65 * w)
    return lo + int(np.argmin(prof[lo:hi]))


def local_lines(text: np.ndarray, x0: int, x1: int, y0: int, y1: int, ls: int) -> list[int]:
    """Rows of the text-line centres inside the window [x0, x1) x [y0, y1)."""
    h, w = text.shape
    x0, x1, y0, y1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return []
    prof = gaussian_filter1d(text[y0:y1, x0:x1].sum(axis=1).astype(float), 0.12 * ls)
    if prof.max() <= 0:
        return []
    peaks, _ = find_peaks(prof, distance=int(0.6 * ls), prominence=0.15 * prof.max())
    return [int(p) + y0 for p in peaks]


def track_line(text: np.ndarray, y: int, x_start: int, step_dir: int, x_limit: int,
               ls: int, max_gap: float) -> Box | None:
    """Follow one text line sideways from `x_start` and return its box.

    The line centre is re-estimated in every window of 0.5 LS, so the box
    follows a sloping line. Tracking stops at `x_limit` or after an empty
    stretch longer than `max_gap` line spacings.
    """
    h, w = text.shape
    step, half = max(4, int(0.5 * ls)), int(0.5 * ls)
    yc, gap, x = float(y), 0.0, x_start
    xs_lo, xs_hi, ys_lo, ys_hi = [], [], [], []
    while (x > x_limit) if step_dir < 0 else (x < x_limit):
        xa, xb = (max(x_limit, x - step), x) if step_dir < 0 else (x, min(x_limit, x + step))
        r0, r1 = max(0, int(yc) - half), min(h, int(yc) + half + 1)
        win = text[r0:r1, xa:xb]
        if win.any():
            rows = np.where(win.any(axis=1))[0]
            cols = np.where(win.any(axis=0))[0]
            xs_lo.append(xa + cols.min()); xs_hi.append(xa + cols.max())
            ys_lo.append(r0 + rows.min()); ys_hi.append(r0 + rows.max())
            c0 = max(0, int(yc - 0.3 * ls))
            core = text[c0 : int(yc + 0.3 * ls) + 1, xa:xb].sum(axis=1)
            if core.sum() > 0:
                centroid = c0 + float((core * np.arange(len(core))).sum() / core.sum())
                yc = 0.7 * yc + 0.3 * centroid
            gap = 0.0
        else:
            gap += (xb - xa) / ls
            if gap > max_gap:
                break
        x = xa if step_dir < 0 else xb
    if not xs_lo:
        return None
    return (int(min(xs_lo)), int(min(ys_lo)), int(max(xs_hi) - min(xs_lo) + 1), int(max(ys_hi) - min(ys_lo) + 1))


def core_band(text: np.ndarray, c: int, x0: int, x1: int, ls: int) -> tuple[int, int]:
    """Rows of the x-height zone of the line at row `c`, measured in [x0, x1).

    The zone is the run of rows around the densest row whose ink count stays
    above half of the maximum. Measuring it next to the brace, rather than
    from a page-wide line model, keeps it right on sloping lines.
    """
    h, w = text.shape
    r0, r1 = max(0, int(c - 0.45 * ls)), min(h, int(c + 0.45 * ls) + 1)
    prof = gaussian_filter1d(text[r0:r1, max(0, x0) : min(w, x1)].sum(axis=1).astype(float), 2)
    if prof.max() <= 0:
        return int(c - CORE * ls), int(c + CORE * ls)
    k = int(np.argmax(prof))
    lo = hi = k
    while lo > 0 and prof[lo - 1] >= 0.5 * prof[k]:
        lo -= 1
    while hi < len(prof) - 1 and prof[hi + 1] >= 0.5 * prof[k]:
        hi += 1
    return r0 + lo, r0 + hi


def _coverage(band: tuple[int, int], span: tuple[int, int]) -> float:
    """Share of the rows of `band` that lie inside `span`."""
    lo, hi = band
    return float(max(0, min(hi, span[1]) - max(lo, span[0]) + 1) / (hi - lo + 1))


def _without_specks(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Drop components smaller than `min_area` (dots, show-through specks)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


def _trim_to(box: Box, text: np.ndarray, band: tuple[int, int], ls: int) -> Box | None:
    """Shrink a tracked line box sideways to the words it holds.

    The left end is the first column with text anywhere in the box. The right
    end, next to the brace, is the last column with text in the rows of the
    line's own x-height zone (`band`, slightly widened): the box of a tracked
    line also covers the descenders of the line above, which may run further
    right than the line itself.
    """
    x, y, w, h = box
    cols = np.where(text[y : y + h, x : x + w].any(axis=0))[0]
    r0, r1 = max(y, int(band[0] - 0.1 * ls)), min(y + h, int(band[1] + 0.1 * ls) + 1)
    own = np.where(text[r0:r1, x : x + w].any(axis=0))[0]
    if len(cols) == 0 or len(own) == 0:
        return None
    return (x + int(cols.min()), y, int(own.max() - cols.min() + 1), h)


def associate(braces: list[Brace], ink: np.ndarray, ls: int, sw: float, exclude: np.ndarray | None = None,
              fillers: np.ndarray | None = None, min_cover: float = 0.8, low_cover: float = 0.25,
              reach: float = 6.0) -> list[BraceGroup]:
    """Attach grouped lines and the side note to every brace.

    `exclude` is an optional mask of other non-text marks (slashes, rules...)
    that must not be mistaken for text. `fillers` is the mask of line fillers:
    they are not text, but they sit on the writing line between a name and
    its brace, so they are used to *follow* the line and to measure where it
    meets the brace, and are then cut from the reported box.

    A line is a member if the brace spans at least `min_cover` of its core
    band. Lines in which the brace merely ends (coverage `low_cover` to
    `min_cover`) are returned as `borderline`: geometry alone cannot tell a
    short brace arm from a decorative tail there.
    """
    h, w = ink.shape
    gx = gutter(ink, ls)
    bounds = [(0, gx), (gx, w)]
    text = ink.copy()
    for b in braces:
        text &= ~b.mask
    if exclude is not None:
        text &= ~exclude
    if fillers is not None:
        text &= ~fillers
    text = _without_specks(text, 3 * sw * sw)
    line_ink = text | fillers if fillers is not None else text  # what a writing line consists of

    groups = []
    for b in braces:
        x, y, bw, bh = b.bbox
        col = int(x + bw / 2 > gx)
        cx0, cx1 = bounds[col]
        span = b.span
        pad = int(0.6 * ls)
        g = BraceGroup(b, col)

        # ---- left: lines whose core band the brace spans
        for c in local_lines(line_ink, max(cx0, x - int(reach * ls)), x + bw, y - pad, y + bh + pad, ls):
            start = b.right_edge(int(c - CORE * ls), int(c + CORE * ls))
            tracked = track_line(line_ink, c, start, -1, cx0, ls, max_gap=2.5)
            if tracked is None:
                continue
            # x-height zone of the line where it meets the brace
            band = core_band(line_ink, c, max(tracked[0], start - int(2.5 * ls)), start, ls)
            box = _trim_to(tracked, text, band, ls)  # the words only, without the fillers
            if box is None or box[2] < 0.5 * ls:
                continue
            cov = _coverage(band, span)
            if cov >= low_cover:
                line = TextLine(sum(band) // 2, box, round(cov, 2))
                (g.members if cov >= min_cover else g.borderline).append(line)

        # ---- right: the lines nearest the cusp, grown to adjacent spanned lines
        ys, xs = np.where(b.mask)
        cusp_y = int(np.median(ys[xs >= xs.max() - 2]))
        right = []
        for c in local_lines(text, x + bw, min(cx1, x + bw + int(4 * ls)), y - pad, y + bh + pad, ls):
            start = b.right_edge(int(c - CORE * ls), int(c + CORE * ls)) + 1
            box = track_line(text, c, start, +1, cx1, ls, max_gap=1.0)
            if box is None or box[0] - start > 2.5 * ls or box[2] < 0.5 * ls:
                continue
            band = core_band(text, c, box[0], min(box[0] + box[2], box[0] + int(2.5 * ls)), ls)
            cov = _coverage(band, span)
            if cov >= min_cover:
                right.append(TextLine(sum(band) // 2, box, round(cov, 2)))
        if right:
            seed = min(right, key=lambda t: abs(t.centre - cusp_y))
            if abs(seed.centre - cusp_y) <= 1.2 * ls:
                chosen, grew = [seed], True
                while grew:
                    grew = False
                    for t in right:
                        if t not in chosen and min(abs(t.centre - u.centre) for u in chosen) <= 1.4 * ls:
                            chosen.append(t)
                            grew = True
                g.note = sorted(chosen, key=lambda t: t.centre)
        groups.append(g)
    return groups